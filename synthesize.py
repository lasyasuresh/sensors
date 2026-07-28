"""Generate labelled synthetic sessions from a real recording.

Synthetic rows are marked subject_id='synth*' and context='synth-*' so they
can never be confused with real data. Use for pipeline development only —
never for validating detection performance.
"""
import os, json, pathlib, datetime, random
import numpy as np
from supabase import create_client

HERE = pathlib.Path(__file__).parent
for line in open(HERE / ".env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
DEVICE_ID = os.environ["DEVICE_ID"]

SOURCE = "s01-rest-0728-0343"      # your real session

src = (sb.table("readings").select("*")
         .eq("session_id", SOURCE).order("ts")
         .limit(2000).execute().data)
if not src:
    raise SystemExit(f"no rows found for {SOURCE}")
print(f"template: {len(src)} rows from {SOURCE}")

real_hr = [r["heart_rate_bpm"] for r in src if r["heart_rate_bpm"]]
base_hr = float(np.mean(real_hr))
print(f"baseline HR {base_hr:.1f}")


def make(session, subject, context, posture, hr_shift, hr_noise,
         q_scale, ir_scale, drop_rate, n=None, start_offset_h=0):
    """Build one synthetic session from the template's temporal structure."""
    rows = []
    t0 = (datetime.datetime.now(datetime.timezone.utc)
          - datetime.timedelta(hours=start_offset_h))
    tmpl = src[:n] if n else src

    for i, r in enumerate(tmpl):
        drop = random.random() < drop_rate
        q = (r["signal_quality"] or 0.5) * q_scale
        q = float(np.clip(q + random.gauss(0, 0.05), 0.01, 0.99))

        if drop or q < 0.35 or not r["heart_rate_bpm"]:
            hr = None
        else:
            hr = float(r["heart_rate_bpm"] + hr_shift + random.gauss(0, hr_noise))
            hr = float(np.clip(hr, 45, 190))

        ir = (r["ir_mean"] or 50000) * ir_scale * random.uniform(0.97, 1.03)
        present = ir > 15000

        spo2 = None
        if present and r["spo2_pct"]:
            spo2 = float(np.clip(r["spo2_pct"] + random.gauss(0, 1.5), 70, 100))

        rows.append({
            "device_id": DEVICE_ID,
            "session_id": session,
            "subject_id": subject,
            "posture": posture,
            "context": context,
            "ts": (t0 + datetime.timedelta(seconds=i)).isoformat(),
            "temperature_c": round(random.uniform(26, 31), 1),
            "humidity_pct": round(random.uniform(62, 74), 0),
            "heart_rate_bpm": round(hr, 1) if hr else None,
            "spo2_pct": round(spo2, 1) if spo2 else None,
            "finger_present": bool(present),
            "ir_mean": round(ir, 0),
            "ir_ac": round((r["ir_ac"] or 700) * random.uniform(.8, 1.2), 0),
            "signal_quality": round(q, 3),
            "sample_rate": round(random.uniform(99.4, 99.9), 1),
        })

    for i in range(0, len(rows), 400):
        sb.table("readings").insert(rows[i:i + 400]).execute()
    usable = sum(1 for r in rows if r["heart_rate_bpm"])
    print(f"  {session:34} {len(rows):>4} rows, {usable:>4} usable "
          f"({100*usable/len(rows):.0f}%)")


print("\ngenerating...")

# --- extra resting sessions, same subject, different times of day ---
make("synth01-rest-morning",   "synth01", "synth-rest", "sitting",
     hr_shift=-3, hr_noise=1.5, q_scale=1.00, ir_scale=1.00,
     drop_rate=0.05, start_offset_h=30)
make("synth01-rest-afternoon", "synth01", "synth-rest", "sitting",
     hr_shift=+4, hr_noise=2.0, q_scale=0.95, ir_scale=0.98,
     drop_rate=0.08, start_offset_h=24)
make("synth01-rest-evening",   "synth01", "synth-rest", "sitting",
     hr_shift=+1, hr_noise=1.8, q_scale=0.98, ir_scale=1.01,
     drop_rate=0.06, start_offset_h=18)
make("synth01-rest-lying",     "synth01", "synth-rest", "lying",
     hr_shift=-7, hr_noise=1.4, q_scale=1.02, ir_scale=1.03,
     drop_rate=0.04, start_offset_h=12)

# --- elevated: the case your Tier-2 alert must catch ---
make("synth01-exertion-1", "synth01", "synth-post-exertion", "sitting",
     hr_shift=+27, hr_noise=4.5, q_scale=0.88, ir_scale=0.97,
     drop_rate=0.15, start_offset_h=8)
make("synth01-exertion-2", "synth01", "synth-post-exertion", "sitting",
     hr_shift=+21, hr_noise=4.0, q_scale=0.90, ir_scale=0.99,
     drop_rate=0.12, start_offset_h=6)

# --- poor contact: the quality gate must reject these ---
make("synth01-poor-1", "synth01", "synth-poor-contact", "sitting",
     hr_shift=0, hr_noise=14, q_scale=0.28, ir_scale=0.22,
     drop_rate=0.72, n=180, start_offset_h=4)
make("synth01-poor-2", "synth01", "synth-poor-contact", "sitting",
     hr_shift=0, hr_noise=18, q_scale=0.22, ir_scale=0.15,
     drop_rate=0.80, n=180, start_offset_h=3)

# --- second subject: proves baselines are per-person ---
make("synth02-rest-1", "synth02", "synth-rest", "sitting",
     hr_shift=-13, hr_noise=2.2, q_scale=0.96, ir_scale=1.05,
     drop_rate=0.07, start_offset_h=20)
make("synth02-rest-2", "synth02", "synth-rest", "sitting",
     hr_shift=-10, hr_noise=2.5, q_scale=0.94, ir_scale=1.02,
     drop_rate=0.09, start_offset_h=14)

print("\ndone. synthetic rows are subject_id synth01/synth02, context synth-*")