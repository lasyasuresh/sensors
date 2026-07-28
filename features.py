"""Feature extraction for model training.

Pulls windows from Supabase and engineers per-window and per-session
features. Everything here is derived from what the sensor actually
measured - no imputation, no smoothing across session boundaries.
"""
import os, pathlib, datetime
import numpy as np
import pandas as pd
from supabase import create_client

HERE = pathlib.Path(__file__).parent
for line in open(HERE / ".env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def fetch_all(days=60):
    """Paged fetch of every reading in the window."""
    since = (datetime.datetime.now(datetime.timezone.utc)
             - datetime.timedelta(days=days)).isoformat()
    rows, page, SIZE = [], 0, 1000
    while True:
        batch = (sb.table("readings").select("*")
                   .gte("ts", since).order("ts")
                   .range(page * SIZE, (page + 1) * SIZE - 1)
                   .execute().data)
        rows.extend(batch)
        if len(batch) < SIZE:
            break
        page += 1
    return pd.DataFrame(rows)


def window_features(df):
    """Per-window features. Target: is this window trustworthy?"""
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    d = d.sort_values(["session_id", "ts"]).reset_index(drop=True)

    # raw signal descriptors
    d["ir_mean"] = d["ir_mean"].fillna(0)
    d["ir_ac"] = d["ir_ac"].fillna(0)
    d["signal_quality"] = d["signal_quality"].fillna(0)
    d["sample_rate"] = d["sample_rate"].fillna(0)

    # perfusion index: pulsatile amplitude over DC. classic PPG quality metric
    d["perfusion"] = np.where(d["ir_mean"] > 0,
                              d["ir_ac"] / d["ir_mean"] * 100, 0)

    # local stability within each session
    g = d.groupby("session_id")
    d["ir_roll_sd"] = g["ir_mean"].transform(
        lambda s: s.rolling(10, min_periods=2).std()).fillna(0)
    d["q_roll_mean"] = g["signal_quality"].transform(
        lambda s: s.rolling(10, min_periods=1).mean())
    d["ir_delta"] = g["ir_mean"].transform(lambda s: s.diff().abs()).fillna(0)

    # time into the session - early windows are less settled
    d["t_into"] = g.cumcount()

    # label: did the pipeline produce a usable HR?
    d["usable"] = d["heart_rate_bpm"].notna().astype(int)

    # a coarse ground-truth label from the recording context
    d["bad_context"] = d["context"].fillna("").str.contains("poor").astype(int)

    cols = ["ir_mean", "ir_ac", "perfusion", "signal_quality", "sample_rate",
            "ir_roll_sd", "q_roll_mean", "ir_delta", "t_into"]
    return d, cols


def session_features(df):
    """Per-session aggregates. Target: resting pulse and its reliability."""
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    good = d[(d["heart_rate_bpm"].notna()) & (d["signal_quality"] >= 0.35)]

    out = []
    for sid, g in good.groupby("session_id"):
        hr = g["heart_rate_bpm"].values
        if len(hr) < 30:
            continue
        lo, hi = np.percentile(hr, [10, 90])
        core = hr[(hr >= lo) & (hr <= hi)]
        first = g.iloc[0]
        out.append({
            "session_id": sid,
            "subject_id": first["subject_id"],
            "context": first["context"],
            "posture": first["posture"],
            "ts": first["ts"],
            "hour": first["ts"].hour,
            "n_windows": len(g),
            "coverage": len(g) / max(len(d[d["session_id"] == sid]), 1),
            "hr_mean": float(core.mean()),
            "hr_sd": float(core.std()),
            "hr_p10": float(lo),
            "hr_p90": float(hi),
            "hr_range": float(hi - lo),
            "q_mean": float(g["signal_quality"].mean()),
            "q_min": float(g["signal_quality"].min()),
            "perfusion_mean": float((g["ir_ac"] / g["ir_mean"] * 100).mean()),
            "spo2_median": float(g["spo2_pct"].median())
                           if g["spo2_pct"].notna().any() else np.nan,
            "amb_temp": float(g["temperature_c"].mean())
                        if g["temperature_c"].notna().any() else np.nan,
            "amb_hum": float(g["humidity_pct"].mean())
                       if g["humidity_pct"].notna().any() else np.nan,
        })
    return pd.DataFrame(out).sort_values("ts").reset_index(drop=True)


if __name__ == "__main__":
    print("fetching...")
    raw = fetch_all()
    print(f"  {len(raw)} rows, {raw['session_id'].nunique()} sessions")

    win, cols = window_features(raw)
    sess = session_features(raw)

    win.to_csv(HERE / "features_windows.csv", index=False)
    sess.to_csv(HERE / "features_sessions.csv", index=False)

    print(f"\nwindow features: {len(win)} rows, {len(cols)} predictors")
    print(f"  usable {win['usable'].mean()*100:.0f}%  "
          f"poor-contact {win['bad_context'].mean()*100:.0f}%")

    print(f"\nsession features: {len(sess)} sessions")
    print(sess[["session_id", "context", "posture", "hr_mean",
                "hr_sd", "q_mean", "coverage"]].to_string(index=False))
    print("\nwrote features_windows.csv, features_sessions.csv")