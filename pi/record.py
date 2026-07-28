import os, sys, json, time, sqlite3, datetime, pathlib, threading
from collections import deque
import numpy as np
from supabase import create_client
from sensors import (MAX30100, read_dht, compute_hr, compute_spo2,
                     finger_present, reset_hr_history, FS)

HERE = pathlib.Path(__file__).parent
for line in open(HERE / ".env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v

DEVICE_ID  = os.environ["DEVICE_ID"]
SESSION_ID = sys.argv[1] if len(sys.argv) > 1 else \
             datetime.datetime.now().strftime("s%Y%m%d-%H%M%S")
WINDOW_SEC = 5
BATCH_SEC  = 10

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

buf = sqlite3.connect(HERE / "buffer.db", check_same_thread=False)
buf.execute("create table if not exists pending ("
            "id integer primary key autoincrement, payload text)")
buf.commit()
lock = threading.Lock()


def queue_row(row):
    with lock:
        buf.execute("insert into pending (payload) values (?)",
                    (json.dumps(row),))
        buf.commit()


def flush():
    with lock:
        rows = buf.execute(
            "select id, payload from pending order by id limit 500").fetchall()
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

ox = MAX30100()
ir_buf  = deque(maxlen=FS * WINDOW_SEC)
red_buf = deque(maxlen=FS * WINDOW_SEC)

print(f"session: {SESSION_ID}")
print("finger on sensor, light steady pressure, hold still. Ctrl+C to stop.\n")

next_row = time.time() + 1.0
was_present = False

try:
    while True:
        ir, red = ox.read_samples()
        ir_buf.extend(ir)
        red_buf.extend(red)

        if time.time() >= next_row:
            next_row += 1.0
            i, r = list(ir_buf), list(red_buf)
            present = finger_present(i)

            if not present and was_present:
                reset_hr_history()
            was_present = present

            hr   = compute_hr(i)        if present else None
            spo2 = compute_spo2(i, r)   if present else None
            temp, hum = read_dht()

            queue_row({
                "device_id": DEVICE_ID,
                "session_id": SESSION_ID,
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "temperature_c": float(temp) if temp is not None else None,
                "humidity_pct": float(hum) if hum is not None else None,
                "heart_rate_bpm": float(hr) if hr is not None else None,
                "spo2_pct": float(spo2) if spo2 is not None else None,
                "finger_present": bool(present),
            })

            ir_mean = int(np.mean(i)) if i else 0
            hr_s   = f"{hr:5.1f}"   if hr   is not None else "  -- "
            spo2_s = f"{spo2:5.1f}" if spo2 is not None else "  -- "
            t_s    = f"{temp:.1f}"  if temp is not None else "--"
            h_s    = f"{hum:.0f}"   if hum  is not None else "--"
            print(f"{'FINGER' if present else '  --  '} | "
                  f"IR {ir_mean:>6} | HR {hr_s} | SpO2 {spo2_s} | "
                  f"{t_s}C {h_s}%")

        time.sleep(0.005)

except KeyboardInterrupt:
    print("\nflushing buffer...")
    for _ in range(10):
        try:
            if flush() == 0:
                break
        except Exception:
            time.sleep(2)
    print("done")