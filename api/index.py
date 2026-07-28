"""Garbha Sethu - public API (Vercel serverless).

Deliberately dependency-light: fastapi + httpx only. No numpy, no pandas,
no scikit-learn, and above all no smbus2 - there is no I2C bus here. All
sensor work stays on the Pi; this tier reads what the Pi already wrote to
Supabase and applies the clinical rules.

Env vars required on Vercel:
  SUPABASE_URL, SUPABASE_KEY
Optional:
  PI_URL          e.g. http://192.168.50.2:8000  (live sessions, LAN only)
  SMS_PROVIDER    'demo' (default) or a real gateway
  MSG91_AUTHKEY, MSG91_TEMPLATE_ID
"""
import os, json, time, random, secrets, datetime, statistics, pathlib
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
PI_URL = os.environ.get("PI_URL", "").rstrip("/")
SMS_PROVIDER = os.environ.get("SMS_PROVIDER", "demo")

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

app = FastAPI(title="Garbha Sethu API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

Q_MIN = 0.35
RATE_MIN = 90.0
MIN_WINDOWS = 30
MIN_SESSIONS = 2
REST = ("rest", "synth-rest")


# ══════════════════════════════════════════════════ supabase (REST)
def _headers():
    if not SUPABASE_KEY:
        raise HTTPException(500, "SUPABASE_KEY not configured")
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def sb_get(table: str, params: Dict[str, Any], limit: int = 5000) -> List[dict]:
    """Paged select. PostgREST caps each response at 1000 rows."""
    out, offset, PAGE = [], 0, 1000
    with httpx.Client(timeout=6) as c:
        while offset < limit:
            h = dict(_headers())
            h["Range"] = f"{offset}-{offset + PAGE - 1}"
            r = c.get(f"{SUPABASE_URL}/rest/v1/{table}", params=params, headers=h)
            if r.status_code >= 400:
                raise HTTPException(502, f"supabase: {r.text[:200]}")
            batch = r.json()
            out.extend(batch)
            if len(batch) < PAGE:
                break
            offset += PAGE
    return out


def sb_insert(table: str, rows) -> Any:
    with httpx.Client(timeout=20) as c:
        r = c.post(f"{SUPABASE_URL}/rest/v1/{table}",
                   headers={**_headers(), "Prefer": "return=representation"},
                   json=rows)
        if r.status_code >= 400:
            raise HTTPException(502, f"supabase: {r.text[:200]}")
        return r.json()


def since_iso(days: int) -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=days)).isoformat()


# ══════════════════════════════════════════════════ clinical rules
LOG, SELFCARE, CONFIRM, CONTACT = 0, 1, 2, 3
TIER_NAME = {0: "log", 1: "self-care", 2: "confirm it", 3: "call your provider"}


def _r(tier, title, detail, evidence=None, action=None, module=None):
    return {"module": module, "tier": tier, "tier_name": TIER_NAME[tier],
            "title": title, "detail": detail,
            "evidence": evidence or [], "action": action}


def gestosis_score(**k):
    items = []

    def add(pts, label):
        if pts:
            items.append((label, pts))

    add(2 if k.get("age", 0) > 35 else 0, "Age over 35")
    add(2 if k.get("bmi", 0) > 30 else 0, "BMI over 30")
    add(1 if k.get("primigravida") else 0, "First pregnancy")
    add(2 if k.get("map_mmhg", 0) > 85 else 0, "MAP above 85 mmHg")
    add(3 if k.get("prior_pe") else 0, "Previous pre-eclampsia")
    add(3 if k.get("chronic_htn") else 0, "Chronic hypertension")
    add(2 if k.get("multiple") else 0, "Multiple pregnancy")
    add(2 if k.get("art") else 0, "Assisted reproduction")
    add(2 if k.get("diabetes") else 0, "Diabetes")
    add(2 if k.get("renal") else 0, "Renal disease")
    add(2 if k.get("autoimmune") else 0, "Autoimmune disease")
    add(1 if k.get("family_pe") else 0, "Family history")
    add(1 if k.get("interval_over_10y") else 0, "Interval over 10 years")

    total = sum(p for _, p in items)
    high = total >= 3
    return {
        "score": total, "high_risk": high, "factors": items,
        "result": _r(
            CONTACT if high else LOG,
            "Higher risk of pre-eclampsia" if high else "Standard risk",
            ("Your score is 3 or above. Show this to your doctor - women in this "
             "group are usually offered low-dose aspirin before 16 weeks and "
             "closer monitoring.") if high else
            "No additional risk factors recorded. Rescore at each visit.",
            evidence=[f"{l} (+{p})" for l, p in items],
            action="Discuss aspirin prophylaxis at your next visit" if high else None,
            module="gestosis")}


def vitals_check(temp_c=None, hr=None, sbp=None, dbp=None, spo2=None):
    red, yellow = [], []
    if temp_c is not None:
        if temp_c >= 38.0 or temp_c < 35.0:
            red.append(f"Temperature {temp_c} C")
        elif temp_c < 36.0:
            yellow.append(f"Temperature {temp_c} C")
    if hr is not None:
        if hr > 120 or hr < 40:
            red.append(f"Pulse {hr:.0f} bpm")
        elif hr > 100 or hr < 50:
            yellow.append(f"Pulse {hr:.0f} bpm")
    if sbp is not None:
        if sbp >= 160 or sbp < 90:
            red.append(f"Systolic {sbp} mmHg")
        elif sbp >= 150 or sbp <= 100:
            yellow.append(f"Systolic {sbp} mmHg")
    if dbp is not None:
        if dbp > 100:
            red.append(f"Diastolic {dbp} mmHg")
        elif dbp >= 90:
            yellow.append(f"Diastolic {dbp} mmHg")
    if spo2 is not None and spo2 < 95:
        red.append(f"Oxygen {spo2}%")

    if red or len(yellow) >= 2:
        return _r(CONTACT, "Call your provider today",
                  "One or more readings are outside the range used in obstetric "
                  "early-warning charts. This needs a clinician, not an app.",
                  red + yellow, "Call now", "vitals")
    if yellow:
        return _r(CONFIRM, "Worth rechecking",
                  "One reading is borderline. Rest quietly for ten minutes and "
                  "take it again.", yellow, "Recheck in 10 minutes", "vitals")
    return _r(LOG, "Recorded", "Readings logged.", module="vitals")


def fever_screen(dev, symptoms=None, days_elevated=1):
    symptoms = symptoms or []
    if not dev.get("ready") or dev["level"] in ("usual", "slight"):
        return _r(LOG, "Nothing to flag",
                  "Resting pulse is in your usual range.", module="fever")
    ev = [f"Resting pulse {dev['current_hr']} bpm, your usual is {dev['baseline']}",
          f"Elevated on {days_elevated} of the last {days_elevated} reading(s)"]
    if symptoms:
        ev.append("You reported: " + ", ".join(symptoms))
    ev.append("The belt cannot measure body temperature")
    return _r(CONFIRM, "Your resting pulse has been higher than usual",
              "That happens for ordinary reasons - heat, poor sleep, a busy few "
              "days. It can also happen with a fever. Take your temperature with "
              "a thermometer so we know which.",
              ev, "Take your temperature and log it", "fever")


def anaemia_screen(dev, symptoms=None, prior_hb=None, ifa_missed_days=0,
                   short_birth_interval=False, teenage=False, vegetarian=False):
    symptoms = symptoms or []
    pts, ev = 0, []
    if dev.get("ready") and dev["level"] in ("moderate", "marked"):
        pts += 2
        ev.append(f"Resting pulse {dev['current_hr']} bpm vs usual {dev['baseline']}")
    hits = [s for s in symptoms if s in
            ("tiredness", "breathlessness", "dizziness", "pica", "palpitations")]
    if len(hits) >= 2:
        pts += 2
        ev.append("Reported: " + ", ".join(hits))
    elif hits:
        pts += 1
        ev.append("Reported: " + ", ".join(hits))
    if prior_hb is not None and prior_hb < 11.0:
        pts += 2
        ev.append(f"Last haemoglobin {prior_hb} g/dL")
    if ifa_missed_days >= 7:
        pts += 1
        ev.append(f"Iron tablets missed on {ifa_missed_days} days")
    if short_birth_interval:
        pts += 1
        ev.append("Short interval since last birth")
    if teenage:
        pts += 1
        ev.append("Under 20")
    if vegetarian:
        pts += 1
        ev.append("Vegetarian diet")

    if pts < 3:
        return _r(LOG, "Nothing to flag",
                  "No anaemia risk pattern right now.", module="anaemia")
    return _r(CONFIRM, "Worth checking your haemoglobin",
              "A few things together suggest a blood test is worth asking for. "
              "Anaemia in pregnancy is common in India and treatable - a simple "
              "Hb test at your next visit will settle it.",
              ev, "Ask for an Hb test at your next visit", "anaemia")


def contractions_rule(gestational_weeks, intervals_min, duration_sec, hours_ongoing):
    if not intervals_min:
        return _r(LOG, "Timer running",
                  "Keep tapping with each contraction.", module="contractions")
    avg = sum(intervals_min) / len(intervals_min)
    regular = (len(intervals_min) >= 4 and
               max(intervals_min) - min(intervals_min) <= avg * 0.5)

    if gestational_weeks < 37:
        if regular or len(intervals_min) >= 6:
            return _r(CONTACT, "Call your provider now",
                      f"You are {gestational_weeks} weeks. Regular tightenings "
                      "before 37 weeks need checking straight away - the 5-1-1 "
                      "rule does not apply before term.",
                      [f"About every {avg:.0f} minutes",
                       f"{len(intervals_min)} recorded"],
                      "Call now", "contractions")
        return _r(CONFIRM, "Keep timing",
                  f"At {gestational_weeks} weeks any regular pattern matters. "
                  "Drink water, lie on your left side, and keep the timer running.",
                  [f"About every {avg:.0f} minutes"],
                  "Continue timing for 30 minutes", "contractions")

    if avg <= 5 and duration_sec >= 60 and hours_ongoing >= 1:
        return _r(CONTACT, "Time to go in",
                  "Five minutes apart, lasting a minute, for an hour. That is the "
                  "point at which you head to hospital.",
                  [f"Every {avg:.0f} min", f"Lasting {duration_sec:.0f} s",
                   f"Going {hours_ongoing:.1f} h"],
                  "Go to hospital", "contractions")

    return _r(LOG, "Not yet 5-1-1",
              "Keep timing. Practice contractions are irregular, do not get "
              "stronger, and ease off with rest and water.",
              [f"Every {avg:.0f} min, lasting {duration_sec:.0f} s"],
              module="contractions")


def fluid_leak_rule(gestational_weeks, dev=None, temp_c=None, fever_reported=False):
    infection = (fever_reported or (temp_c is not None and temp_c >= 38.0)
                 or (dev and dev.get("ready") and dev["level"] == "marked"))
    if infection:
        return _r(CONTACT, "Go in now",
                  "Fluid leaking together with a raised temperature or pulse can "
                  "mean an infection in the waters. Go to hospital rather than "
                  "waiting.",
                  ["Fluid leak reported",
                   f"Temperature {temp_c} C" if temp_c else "Pulse well above your usual"],
                  "Go to hospital now", "prom")
    return _r(CONTACT, "Get examined today",
              f"At {gestational_weeks} weeks, any leaking fluid should be checked "
              "the same day. It is hard to tell amniotic fluid from urine at home.",
              ["Fluid leak reported"], "Call your provider today", "prom")


PE_SYMPTOMS = {
    "headache": "Bad headache that will not go",
    "vision": "Blurred vision, flashing lights or spots",
    "epigastric": "Pain under the ribs on the right",
    "swelling": "Sudden swelling of face or hands",
    "reduced_urine": "Passing much less urine",
    "vomiting": "Vomiting in late pregnancy",
}


def symptom_check(reported, sbp=None, dbp=None):
    hits = [PE_SYMPTOMS[s] for s in reported if s in PE_SYMPTOMS]
    if not hits:
        return _r(LOG, "Recorded", "No red-flag symptoms today.", module="symptoms")
    high_bp = (sbp is not None and sbp >= 140) or (dbp is not None and dbp >= 90)
    if len(hits) >= 2 or high_bp:
        return _r(CONTACT, "Call your provider now",
                  "These are the warning signs of pre-eclampsia. Do not wait for "
                  "your next appointment.",
                  hits + ([f"BP {sbp}/{dbp}"] if high_bp else []),
                  "Call now", "symptoms")
    return _r(CONFIRM, "Check your blood pressure",
              "This can be a warning sign when it comes with raised blood "
              "pressure. Take a reading now and log it.",
              hits, "Take your blood pressure", "symptoms")


def weight_check(kg_gained_this_week, sbp=None, dbp=None):
    if kg_gained_this_week is None or kg_gained_this_week < 2.0:
        return _r(LOG, "Recorded", "Weight logged.", module="weight")
    high_bp = (sbp is not None and sbp >= 140) or (dbp is not None and dbp >= 90)
    tier = CONTACT if high_bp else CONFIRM
    return _r(tier, "Call your provider" if high_bp else "Check your blood pressure",
              "Gaining more than 2 kg in a week usually means fluid, not fat. "
              "With raised blood pressure it matters more.",
              [f"Up {kg_gained_this_week:.1f} kg this week"]
              + ([f"BP {sbp}/{dbp}"] if high_bp else []),
              "Call now" if high_bp else "Take your blood pressure", "weight")


def heat_index(temp_c, humidity_pct):
    T = temp_c * 9 / 5 + 32
    R = humidity_pct
    if T < 80:
        hi = 0.5 * (T + 61 + (T - 68) * 1.2 + R * 0.094)
    else:
        hi = (-42.379 + 2.04901523 * T + 10.14333127 * R - 0.22475541 * T * R
              - 0.00683783 * T * T - 0.05481717 * R * R + 0.00122874 * T * T * R
              + 0.00085282 * T * R * R - 0.00000199 * T * T * R * R)
        if R < 13 and 80 <= T <= 112:
            hi -= ((13 - R) / 4) * ((17 - abs(T - 95)) / 17) ** 0.5
    return round((hi - 32) * 5 / 9, 1)


def heat_check(temp_c, humidity_pct, forecast_peak_c=None):
    hi = heat_index(temp_c, humidity_pct)
    ev = [f"Feels like {hi} C", f"{temp_c} C at {humidity_pct:.0f}% humidity"]
    if forecast_peak_c:
        ev.append(f"Peak today around {forecast_peak_c} C")
    if hi >= 40:
        return _r(SELFCARE, "Too hot to be outside",
                  "Stay indoors, drink water often, and use a wet cloth on your "
                  "neck and wrists. Heat is harder on you in pregnancy.",
                  ev, "Stay indoors and hydrate", "heat")
    if hi >= 33:
        return _r(SELFCARE, "Take care in the heat",
                  "Drink a glass of water now and avoid being outdoors between "
                  "noon and four. Loose cotton over the belt lets trapped air escape.",
                  ev, "Drink water now", "heat")
    return _r(LOG, "Comfortable", "Conditions are fine today.", ev, module="heat")


# ══════════════════════════════════════════════════ baseline (pure python)
def session_rollup(rows: List[dict]) -> List[dict]:
    by: Dict[str, List[dict]] = {}
    for r in rows:
        by.setdefault(r["session_id"], []).append(r)

    out = []
    for sid, g in by.items():
        good = [r for r in g
                if r.get("heart_rate_bpm")
                and r.get("finger_present")
                and (r.get("signal_quality") or 0) >= Q_MIN
                and (r.get("sample_rate") or 0) >= RATE_MIN]
        first = g[0]
        hr = [r["heart_rate_bpm"] for r in good]
        spo2 = [r["spo2_pct"] for r in good if r.get("spo2_pct")]
        q = [r.get("signal_quality") or 0 for r in g]
        out.append({
            "session_id": sid,
            "subject_id": first.get("subject_id"),
            "context": first.get("context"),
            "posture": first.get("posture"),
            "ts": first["ts"],
            "windows": len(g),
            "usable": len(good),
            "pct_usable": round(100 * len(good) / max(len(g), 1)),
            "hr_mean": round(statistics.fmean(hr), 1) if hr else None,
            "hr_min": round(min(hr), 1) if hr else None,
            "hr_max": round(max(hr), 1) if hr else None,
            "spo2_median": round(statistics.median(spo2), 1) if spo2 else None,
            "q_mean": round(statistics.fmean(q), 2) if q else 0,
            "n_good": len(good),
        })
    return sorted(out, key=lambda s: s["ts"])


def fetch_readings(subject_id: str, days: int = 60) -> List[dict]:
    return sb_get("readings", {
        "subject_id": f"eq.{subject_id}",
        "ts": f"gte.{since_iso(days)}",
        "order": "ts.asc",
        "select": "*",
    })


def baseline(subject_id: str, posture: Optional[str] = None, days: int = 14):
    rows = fetch_readings(subject_id, days)
    sess = [s for s in session_rollup(rows)
            if (s["context"] or "") in REST and s["n_good"] >= MIN_WINDOWS]
    if posture:
        sess = [s for s in sess if s["posture"] == posture]
    if len(sess) < MIN_SESSIONS:
        return {"ready": False, "n_sessions": len(sess), "sessions": sess}

    vals = [s["hr_mean"] for s in sess]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "ready": True, "subject_id": subject_id, "posture": posture or "all",
        "hr_mean": round(statistics.fmean(vals), 1),
        "hr_sd": round(max(sd, 1.5), 1),
        "n_sessions": len(sess), "window_days": days, "sessions": sess,
    }


def deviation(subject_id: str, current_hr: float, posture=None, days: int = 14):
    b = baseline(subject_id, posture, days)
    if not b["ready"]:
        return {"ready": False,
                "reason": f"needs {MIN_SESSIONS} clean sessions, "
                          f"have {b['n_sessions']}"}
    delta = current_hr - b["hr_mean"]
    z = delta / b["hr_sd"]
    level = ("marked" if (z >= 3.0 or delta >= 15) else
             "moderate" if (z >= 2.0 or delta >= 10) else
             "slight" if z >= 1.5 else "usual")
    return {"ready": True, "current_hr": round(current_hr, 1),
            "baseline": b["hr_mean"], "sd": b["hr_sd"],
            "delta": round(delta, 1), "z": round(z, 2),
            "level": level, "n_sessions": b["n_sessions"]}


# ══════════════════════════════════════════════════ models
class Profile(BaseModel):
    age: int = 28
    bmi: float = 24.0
    map_mmhg: float = 82.0
    gestational_weeks: int = 28
    primigravida: bool = False
    multiple: bool = False
    art: bool = False
    prior_pe: bool = False
    chronic_htn: bool = False
    diabetes: bool = False
    renal: bool = False
    autoimmune: bool = False
    family_pe: bool = False
    interval_over_10y: bool = False


class LogEntry(BaseModel):
    subject_id: str
    kind: str
    value_num: Optional[float] = None
    value_num2: Optional[float] = None
    value_text: Optional[str] = None
    gestational_weeks: Optional[int] = None


class ContractionLog(BaseModel):
    gestational_weeks: int
    intervals_min: List[float]
    duration_sec: float
    hours_ongoing: float


class OtpSend(BaseModel):
    phone: str


class OtpVerify(BaseModel):
    token: str
    code: str


# ══════════════════════════════════════════════════ endpoints
@app.get("/api/health")
def health():
    return {"ok": True, "supabase": bool(SUPABASE_URL and SUPABASE_KEY),
            "pi": PI_URL or None, "tier": "vercel"}


@app.get("/api/subjects")
def subjects():
    rows = sb_get("readings", {"select": "subject_id",
                               "subject_id": "not.is.null"}, limit=2000)
    seen: Dict[str, int] = {}
    for r in rows:
        seen[r["subject_id"]] = seen.get(r["subject_id"], 0) + 1
    return [{"subject_id": k, "rows": v}
            for k, v in sorted(seen.items(), key=lambda t: -t[1])]


@app.get("/api/sessions/{subject_id}")
def sessions(subject_id: str):
    return session_rollup(fetch_readings(subject_id))


@app.get("/api/session/{session_id}")
def session_detail(session_id: str):
    rows = sb_get("readings", {
        "session_id": f"eq.{session_id}", "order": "ts.asc",
        "select": "ts,heart_rate_bpm,spo2_pct,signal_quality,ir_mean,ir_ac,"
                  "temperature_c,humidity_pct,finger_present"})
    if not rows:
        raise HTTPException(404, "session not found")
    t0 = datetime.datetime.fromisoformat(rows[0]["ts"].replace("Z", "+00:00"))
    for r in rows:
        t = datetime.datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
        r["t"] = round((t - t0).total_seconds())
    return rows


@app.get("/api/baseline/{subject_id}")
def get_baseline(subject_id: str, posture: Optional[str] = None):
    return baseline(subject_id, posture)


@app.get("/api/status/{subject_id}")
def status(subject_id: str, posture: Optional[str] = "sitting"):
    sess = session_rollup(fetch_readings(subject_id))
    rest = [s for s in sess if (s["context"] or "") in REST and s["hr_mean"]]
    if not rest:
        return {"ready": False, "reason": "no resting sessions recorded"}
    latest = rest[-1]
    dev = deviation(subject_id, latest["hr_mean"], posture)
    out = {"ready": True, "subject_id": subject_id, "latest": latest,
           "deviation": dev, "n_sessions": len(rest), "cards": []}
    if dev.get("ready"):
        out["cards"] = [fever_screen(dev), anaemia_screen(dev)]
    return out


@app.post("/api/log")
def add_log(e: LogEntry):
    row = e.model_dump()
    row["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    sb_insert("logs", row)
    return {"ok": True, "kind": e.kind}


@app.get("/api/logs/{subject_id}")
def get_logs(subject_id: str, kind: Optional[str] = None, days: int = 28):
    params = {"subject_id": f"eq.{subject_id}",
              "ts": f"gte.{since_iso(days)}",
              "order": "ts.desc", "select": "*"}
    if kind:
        params["kind"] = f"eq.{kind}"
    return sb_get("logs", params, limit=1000)


@app.get("/api/today/{subject_id}")
def today(subject_id: str):
    start = datetime.datetime.now(datetime.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0).isoformat()
    rows = sb_get("logs", {"subject_id": f"eq.{subject_id}",
                           "ts": f"gte.{start}", "select": "kind"}, limit=500)
    done = {r["kind"]: True for r in rows}
    if session_rollup(fetch_readings(subject_id, days=1)):
        done["reading"] = True
    return {"done": done, "date": start[:10]}


@app.get("/api/assess/{subject_id}")
def assess(subject_id: str, weeks: int = 28):
    rows = get_logs(subject_id)
    latest: Dict[str, dict] = {}
    for r in rows:
        latest.setdefault(r["kind"], r)

    sbp = latest["bp"]["value_num"] if "bp" in latest else None
    dbp = latest["bp"]["value_num2"] if "bp" in latest else None
    temp = latest["temp"]["value_num"] if "temp" in latest else None
    hb = latest["hb"]["value_num"] if "hb" in latest else None
    syms = json.loads(latest["symptoms"]["value_text"] or "[]") \
        if "symptoms" in latest else []
    missed = sum(1 for r in rows if r["kind"] == "iron" and not r.get("value_num"))

    dev = {"ready": False}
    try:
        sess = session_rollup(fetch_readings(subject_id))
        rest = [s for s in sess if (s["context"] or "") in REST and s["hr_mean"]]
        if rest:
            dev = deviation(subject_id, rest[-1]["hr_mean"], "sitting")
    except Exception:
        pass

    cards = []
    if dev.get("ready"):
        cards.append(fever_screen(dev, symptoms=syms))
        cards.append(anaemia_screen(dev, symptoms=syms, prior_hb=hb,
                                    ifa_missed_days=missed))
    if any(v is not None for v in (temp, sbp, dbp)):
        cards.append(vitals_check(temp_c=temp, sbp=sbp, dbp=dbp))
    if syms:
        cards.append(symptom_check(syms, sbp=sbp, dbp=dbp))

    weights = sorted([r for r in rows if r["kind"] == "weight"],
                     key=lambda r: r["ts"])
    if len(weights) >= 2:
        cards.append(weight_check(
            weights[-1]["value_num"] - weights[-2]["value_num"], sbp, dbp))

    if "leak" in latest:
        t = datetime.datetime.fromisoformat(
            latest["leak"]["ts"].replace("Z", "+00:00"))
        age_h = (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds() / 3600
        if age_h < 48:
            cards.append(fluid_leak_rule(weeks, dev=dev, temp_c=temp))

    cards = [c for c in cards if c["tier"] > 0]
    cards.sort(key=lambda c: -c["tier"])
    return {"cards": cards, "max_tier": max([c["tier"] for c in cards], default=0)}


@app.post("/api/gestosis")
def gestosis(p: Profile):
    return gestosis_score(**p.model_dump())


@app.post("/api/contractions")
def contractions(c: ContractionLog):
    return contractions_rule(c.gestational_weeks, c.intervals_min,
                             c.duration_sec, c.hours_ongoing)


@app.get("/api/heat")
def heat(lat: float = 12.9716, lon: float = 77.5946, place: str = "Bengaluru"):
    out = None
    try:
        with httpx.Client(timeout=12) as c:
            r = c.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature",
                "daily": "temperature_2m_max,apparent_temperature_max",
                "timezone": "auto", "forecast_days": 5})
            r.raise_for_status()
            d = r.json()
        cur, daily = d["current"], d["daily"]
        out = {"temp_c": cur["temperature_2m"],
               "humidity": cur["relative_humidity_2m"],
               "feels_c": cur["apparent_temperature"],
               "forecast": [{"date": daily["time"][i],
                             "max_c": daily["temperature_2m_max"][i],
                             "feels_max_c": daily["apparent_temperature_max"][i]}
                            for i in range(len(daily["time"]))]}
    except Exception as e:
        return {"error": f"weather unavailable: {type(e).__name__}", "place": place}

    result = {"place": place, "outdoor": out, "belt": None,
              "guidance": None, "ahead": []}

    # under-belt microclimate: newest reading the Pi wrote
    try:
        rows = sb_get("readings", {
            "order": "ts.desc", "limit": "1",
            "select": "temperature_c,humidity_pct,ts",
            "temperature_c": "not.is.null"}, limit=1)
        if rows:
            t, h = rows[0]["temperature_c"], rows[0]["humidity_pct"]
            result["belt"] = {"temp_c": t, "humidity": h,
                              "feels_c": heat_index(t, h), "ts": rows[0]["ts"]}
    except Exception:
        pass

    src_t, src_h = out["temp_c"], out["humidity"]
    if result["belt"] and result["belt"]["feels_c"] > heat_index(src_t, src_h):
        src_t, src_h = result["belt"]["temp_c"], result["belt"]["humidity"]
    peak = out["forecast"][0]["feels_max_c"]
    result["guidance"] = heat_check(src_t, src_h, round(peak, 1))

    for f in out["forecast"][1:5]:
        if f["feels_max_c"] >= 38:
            day = datetime.date.fromisoformat(f["date"]).strftime("%A")
            result["ahead"].append({"day": day,
                                    "feels_max_c": round(f["feels_max_c"], 1)})
    return result


# ── live sessions proxy to the Pi (LAN only) ──────────────────────
@app.get("/api/live/available")
def live_available():
    if not PI_URL:
        return {"available": False, "reason": "no PI_URL configured"}
    try:
        with httpx.Client(timeout=3) as c:
            c.get(f"{PI_URL}/api/live/state")
        return {"available": True, "pi": PI_URL}
    except Exception:
        return {"available": False,
                "reason": "the belt is not reachable from here"}


# ── OTP ───────────────────────────────────────────────────────────
_otp: Dict[str, dict] = {}


def _send_sms(phone: str, code: str) -> bool:
    if SMS_PROVIDER == "demo":
        print(f"[OTP] {phone} -> {code}")
        return True
    if SMS_PROVIDER == "msg91":
        try:
            with httpx.Client(timeout=10) as c:
                c.post("https://control.msg91.com/api/v5/otp", params={
                    "authkey": os.environ["MSG91_AUTHKEY"],
                    "template_id": os.environ["MSG91_TEMPLATE_ID"],
                    "mobile": f"91{phone}", "otp": code})
            return True
        except Exception:
            return False
    return False


@app.post("/api/otp/send")
def otp_send(r: OtpSend):
    digits = "".join(ch for ch in r.phone if ch.isdigit())
    if len(digits) < 10:
        raise HTTPException(400, "invalid phone number")
    code = f"{random.randint(0, 999999):06d}"
    token = secrets.token_urlsafe(16)
    _otp[token] = {"phone": digits, "code": code, "tries": 0,
                   "expires": time.time() + 300}
    _send_sms(digits, code)
    out = {"ok": True, "token": token, "expires_in": 300}
    if SMS_PROVIDER == "demo":
        out["demo_code"] = code
    return out


@app.post("/api/otp/verify")
def otp_verify(r: OtpVerify):
    rec = _otp.get(r.token)
    if not rec or time.time() > rec["expires"]:
        _otp.pop(r.token, None)
        return {"ok": False, "reason": "Code expired. Request a new one."}
    rec["tries"] += 1
    if rec["tries"] > 5:
        _otp.pop(r.token, None)
        return {"ok": False, "reason": "Too many attempts. Request a new code."}
    if r.code != rec["code"]:
        return {"ok": False, "reason": "That code did not match."}
    _otp.pop(r.token, None)
    return {"ok": True, "phone": rec["phone"]}


# ── frontend ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
def frontend():
    for p in (ROOT / "public" / "index.html", ROOT / "index.html"):
        if p.exists():
            return HTMLResponse(p.read_text())
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)