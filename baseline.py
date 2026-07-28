"""Per-subject baseline engine.

Computes a rolling resting-pulse baseline from quality-gated sessions and
scores how far today's reading sits from it. This is the core of every
Tier-2 alert: deviation from *her* normal, never a population threshold.
"""
import os, pathlib, datetime
from statistics import mean, stdev
import numpy as np
from supabase import create_client

HERE = pathlib.Path(__file__).parent
for line in open(HERE / ".env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

Q_MIN            = 0.35    # autocorrelation confidence floor
RATE_MIN         = 90.0    # samples/sec floor
MIN_WINDOWS      = 60      # a session needs 60s of usable data to count
BASELINE_DAYS    = 14      # rolling window
MIN_SESSIONS     = 2       # below this, no baseline is published
REST_CONTEXTS    = ("rest", "synth-rest")


def fetch(subject_id, days=BASELINE_DAYS, contexts=REST_CONTEXTS):
    """Paged fetch - PostgREST caps each response at 1000 rows."""
    since = (datetime.datetime.now(datetime.timezone.utc)
             - datetime.timedelta(days=days)).isoformat()
    rows, page, SIZE = [], 0, 1000
    while True:
        batch = (sb.table("readings").select("*")
                   .eq("subject_id", subject_id)
                   .gte("ts", since)
                   .in_("context", list(contexts))
                   .order("ts")
                   .range(page * SIZE, (page + 1) * SIZE - 1)
                   .execute().data)
        rows.extend(batch)
        if len(batch) < SIZE:
            break
        page += 1
    return rows


def session_value(rows):
    """Collapse one session into a single quality-weighted resting pulse."""
    good = [r for r in rows
            if r.get("heart_rate_bpm")
            and r.get("finger_present")
            and (r.get("signal_quality") or 0) >= Q_MIN
            and (r.get("sample_rate") or 0) >= RATE_MIN]
    if len(good) < MIN_WINDOWS:
        return None

    hr = np.array([r["heart_rate_bpm"] for r in good], float)
    w  = np.array([r["signal_quality"] for r in good], float)

    # trim the top and bottom deciles before weighting — kills residual outliers
    lo, hi = np.percentile(hr, [10, 90])
    keep = (hr >= lo) & (hr <= hi)
    if keep.sum() < 20:
        keep = np.ones_like(hr, bool)

    value = float(np.average(hr[keep], weights=w[keep]))
    spo2 = [r["spo2_pct"] for r in good if r.get("spo2_pct")]

    return {
        "session_id": good[0]["session_id"],
        "posture":    good[0].get("posture"),
        "ts":         good[0]["ts"],
        "hr":         round(value, 1),
        "spo2":       round(float(np.median(spo2)), 1) if spo2 else None,
        "n_windows":  len(good),
        "mean_q":     round(float(w.mean()), 2),
    }


def sessions_for(subject_id, **kw):
    rows = fetch(subject_id, **kw)
    by_session = {}
    for r in rows:
        by_session.setdefault(r["session_id"], []).append(r)
    out = [session_value(v) for v in by_session.values()]
    out = [s for s in out if s]
    return sorted(out, key=lambda s: s["ts"])


def baseline(subject_id, posture=None, **kw):
    """Rolling baseline. Pass posture to avoid pooling sitting with lying."""
    sess = sessions_for(subject_id, **kw)
    if posture:
        sess = [s for s in sess if s["posture"] == posture]
    if len(sess) < MIN_SESSIONS:
        return {"ready": False, "n_sessions": len(sess), "sessions": sess}

    vals = [s["hr"] for s in sess]
    sd = stdev(vals) if len(vals) > 1 else 0.0
    return {
        "ready":       True,
        "subject_id":  subject_id,
        "posture":     posture or "all",
        "hr_mean":     round(mean(vals), 1),
        "hr_sd":       round(max(sd, 1.5), 1),
        "n_sessions":  len(sess),
        "window_days": kw.get("days", BASELINE_DAYS),
        "sessions":    sess,
    }


def deviation(subject_id, current_hr, posture=None, **kw):
    b = baseline(subject_id, posture=posture, **kw)
    if not b["ready"]:
        return {"ready": False,
                "reason": f"needs {MIN_SESSIONS} clean sessions, have {b['n_sessions']}"}

    delta = current_hr - b["hr_mean"]
    z = delta / b["hr_sd"]

    if z >= 3.0 or delta >= 15:
        level = "marked"
    elif z >= 2.0 or delta >= 10:
        level = "moderate"
    elif z >= 1.5:
        level = "slight"
    else:
        level = "usual"

    return {
        "ready":      True,
        "current_hr": round(current_hr, 1),
        "baseline":   b["hr_mean"],
        "sd":         b["hr_sd"],
        "delta":      round(delta, 1),
        "z":          round(z, 2),
        "level":      level,
        "n_sessions": b["n_sessions"],
    }


if __name__ == "__main__":
    import sys
    subj = sys.argv[1] if len(sys.argv) > 1 else "s01"

    b = baseline(subj)
    print(f"\n=== baseline · {subj} ===")
    if not b["ready"]:
        print(f"not ready — {b['n_sessions']} clean session(s)")
    else:
        print(f"resting pulse {b['hr_mean']} bpm  (sd {b['hr_sd']}, "
              f"{b['n_sessions']} sessions over {b['window_days']}d)\n")
        for s in b["sessions"]:
            print(f"  {s['ts'][:16]}  {s['posture']:<8} "
                  f"{s['hr']:>6} bpm   q={s['mean_q']}  n={s['n_windows']}")

        print("\n=== deviation tests ===")
        for hr in [b["hr_mean"], b["hr_mean"] + 8,
                   b["hr_mean"] + 14, b["hr_mean"] + 25]:
            d = deviation(subj, hr)
            print(f"  {hr:6.1f} bpm  ->  {d['level']:<9} "
                  f"(delta {d['delta']:+.1f}, z {d['z']:+.2f})")