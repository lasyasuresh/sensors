"""Session-based sensor recording with per-window quality scoring.

Usage:
    python record_v2.py                 # interactive prompts
    python record_v2.py --calibrate     # also prompt for reference oximeter
"""
import os, sys, json, time, sqlite3, datetime, pathlib, threading
from collections import deque
import numpy as np
from supabase import create_client
from sensors import MAX30100, read_dht, analyze, reset_hr_history, FS

HERE = pathlib.Path(__file__).parent
for line in open(HERE / ".env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v

DEVICE_ID  = os.environ["DEVICE_ID"]
WINDOW_SEC = 5
BATCH_SEC  = 10
CALIBRATE  = "--calibrate" in sys.argv

# ---------- session metadata ----------
def ask(prompt, options=None, default=None):
    if options:
        prompt += f" [{'/'.join(options)}]"
    if default:
        prompt += f" ({default})"
    v = input(prompt + ": ").strip()
    return v or default or ""

print("=" * 52)
print("  NEW RECORDING SESSION")
print("=" * 52)
subject  = ask("Subject id", default="s01")
posture  = ask("Posture", ["sitting", "lying", "standing"], "sitting")
context  = ask("Context", ["rest", "post-exertion", "poor-contact", "other"], "rest")
minutes  = float(ask("Minutes to record", default="5"))

SESSION_ID = f"{subject}-{context}-{datetime.datetime.now():%m%d-%H%M}"
print(f"\nsession: {SESSION_ID}")
if CALIBRATE:
    print("CALIBRATION MODE — you'll be asked for reference values every 30 s")
print("finger on sensor, light steady pressure, hold still.")
print("Ctrl+C to stop early.\n")

# ---------- storage ----------
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
buf = sqlite3.connect(HERE / "buffer.db", check_same_thread=False)
buf.execute("create table if not exists pending ("
            "id integer primary key autoincrement, payload text)")
buf.commit()
lock = threading.Lock()

def queue_row(row):
    with lock:
        buf.execute("insert into pending (payload) values (?)", (json.dumps(row),))
        buf.commit()

def flush():
    with lock:
        rows = buf.execute("select id, payload from pending "
                           "order by id limit 500").fetchall()
    if not rows:
        return 0
    sb.table("readings").insert([json.loads(r[1]) for r in rows]).execute()
    with lock:
        buf.execute("delete from pending where id <= ?", (rows[-1][0],))
        buf.commit()
    return len(rows)

def uploader():
    while True:
        time.sleep(BATCH_SEC)
        try:
            n = flush()
            if n:
                print(f"    ^ uploaded {n}")
        except Exception as e:
            print(f"    ! offline, buffering ({type(e).__name__})")

threading.Thread(target=uploader, daemon=True).start()

# ---------- reference oximeter capture ----------
ref = {"hr": None, "spo2": None}

def ref_prompt():
    while True:
        time.sleep(30)
        try:
            raw = input("\n  >> reference reading 'HR SPO2' (blank to skip): ")
            parts = raw.split()
            if len(parts) == 2:
                ref["hr"], ref["spo2"] = float(parts[0]), float(parts[1])
                print(f"  >> logged ref {ref['hr']:.0f} bpm / {ref['spo2']:.0f}%\n")
        except Exception:
            pass

if CALIBRATE:
    threading.Thread(target=ref_prompt, daemon=True).start()

# ---------- main loop ----------
ox = MAX30100()
ir_buf  = deque(maxlen=FS * WINDOW_SEC)
red_buf = deque(maxlen=FS * WINDOW_SEC)

t0 = time.time()
next_row = t0 + 1.0
n_samples = 0
was_present = False
kept = 0
total = 0

try:
    while time.time() - t0 < minutes * 60:
        ir, red = ox.read_samples()
        ir_buf.extend(ir)
        red_buf.extend(red)
        n_samples += len(ir)

        if time.time() >= next_row:
            next_row += 1.0
            elapsed = time.time() - t0
            rate = n_samples / elapsed

            a = analyze(list(ir_buf), list(red_buf))
            if not a["finger"] and was_present:
                reset_hr_history()
            was_present = a["finger"]

            temp, hum = read_dht()
            total += 1
            if a["hr"] is not None:
                kept += 1

            queue_row({
                "device_id": DEVICE_ID,
                "session_id": SESSION_ID,
                "subject_id": subject,
                "posture": posture,
                "context": context,
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "temperature_c": float(temp) if temp is not None else None,
                "humidity_pct": float(hum) if hum is not None else None,
                "heart_rate_bpm": a["hr"],
                "spo2_pct": a["spo2"],
                "finger_present": a["finger"],
                "ir_mean": a["ir_mean"],
                "ir_ac": a["ir_ac"],
                "signal_quality": a["signal_quality"],
                "sample_rate": round(rate, 1),
                "ref_hr": ref["hr"],
                "ref_spo2": ref["spo2"],
            })

            q  = a["signal_quality"]
            qs = f"{q:.2f}" if q is not None else " -- "
            hr = f"{a['hr']:5.1f}" if a["hr"] is not None else "  -- "
            sp = f"{a['spo2']:5.1f}" if a["spo2"] is not None else "  -- "
            ac = f"{a['ir_ac']:6.0f}" if a["ir_ac"] is not None else "    --"
            ir_m = int(a["ir_mean"]) if a["ir_mean"] is not None else 0
            bar = "#" * int((q or 0) * 10)
            print(f"{elapsed:5.0f}s {'FINGER' if a['finger'] else '  --  '} | "
                  f"IR {ir_m:>6} | AC {ac} | q {qs} {bar:<10} | "
                  f"HR {hr} | SpO2 {sp} | {rate:5.1f}/s")

        time.sleep(0.005)

except KeyboardInterrupt:
    print("\nstopped early")

print(f"\n{'=' * 52}")
print(f"  session {SESSION_ID}")
print(f"  {total} windows, {kept} with usable HR ({100*kept/max(total,1):.0f}%)")
print(f"  mean sample rate {n_samples/(time.time()-t0):.1f}/s")
print("  flushing buffer...")
for _ in range(15):
    try:
        if flush() == 0:
            break
    except Exception:
        time.sleep(2)
print("  done")
print("=" * 52)