"""API layer for the maternal monitoring app.

Serves the full pipeline - readings, baselines, deviations, clinical rules,
heat assessment and the doctor report - plus the dashboard itself.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Then: http://192.168.50.2:8000
"""
import os, io, pathlib, datetime
from typing import Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from supabase import create_client

import rules
import baseline as bl

HERE = pathlib.Path(__file__).parent
for line in open(HERE / ".env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

app = FastAPI(title="Maternal belt API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

REST = ("rest", "synth-rest")
Q_MIN = 0.35


# ------------------------------------------------------------ helpers
def paged(query_fn, size=1000):
    rows, page = [], 0
    while True:
        batch = query_fn(page * size, (page + 1) * size - 1)
        rows.extend(batch)
        if len(batch) < size:
            break
        page += 1
    return rows


def fetch_subject(subject_id, days=60):
    since = (datetime.datetime.now(datetime.timezone.utc)
             - datetime.timedelta(days=days)).isoformat()
    return paged(lambda a, b: (
        sb.table("readings").select("*")
          .eq("subject_id", subject_id).gte("ts", since)
          .order("ts").range(a, b).execute().data))


def session_rollup(rows):
    """Collapse windows into one row per session."""
    by = {}
    for r in rows:
        by.setdefault(r["session_id"], []).append(r)

    out = []
    for sid, g in by.items():
        good = [r for r in g
                if r.get("heart_rate_bpm")
                and (r.get("signal_quality") or 0) >= Q_MIN]
        first = g[0]
        hr = [r["heart_rate_bpm"] for r in good]
        spo2 = [r["spo2_pct"] for r in good if r.get("spo2_pct")]
        q = [r["signal_quality"] or 0 for r in g]
        out.append({
            "session_id": sid,
            "subject_id": first.get("subject_id"),
            "context": first.get("context"),
            "posture": first.get("posture"),
            "ts": first["ts"],
            "windows": len(g),
            "usable": len(good),
            "pct_usable": round(100 * len(good) / max(len(g), 1)),
            "hr_mean": round(float(np.mean(hr)), 1) if hr else None,
            "hr_min": round(float(np.min(hr)), 1) if hr else None,
            "hr_max": round(float(np.max(hr)), 1) if hr else None,
            "spo2_median": round(float(np.median(spo2)), 1) if spo2 else None,
            "q_mean": round(float(np.mean(q)), 2),
        })
    return sorted(out, key=lambda s: s["ts"])


# ------------------------------------------------------------- models
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


class ManualEntry(BaseModel):
    temp_c: Optional[float] = None
    sbp: Optional[int] = None
    dbp: Optional[int] = None
    weight_gain_kg: Optional[float] = None
    prior_hb: Optional[float] = None
    ifa_missed_days: int = 0
    symptoms: List[str] = []
    fluid_leak: bool = False


class ContractionLog(BaseModel):
    gestational_weeks: int
    intervals_min: List[float]
    duration_sec: float
    hours_ongoing: float


# ---------------------------------------------------------- endpoints
@app.get("/api/subjects")
def subjects():
    rows = paged(lambda a, b: (
        sb.table("readings").select("subject_id")
          .not_.is_("subject_id", "null").range(a, b).execute().data))
    seen = {}
    for r in rows:
        seen[r["subject_id"]] = seen.get(r["subject_id"], 0) + 1
    return [{"subject_id": k, "rows": v}
            for k, v in sorted(seen.items(), key=lambda t: -t[1])]


@app.get("/api/sessions/{subject_id}")
def sessions(subject_id: str):
    return session_rollup(fetch_subject(subject_id))


@app.get("/api/session/{session_id}")
def session_detail(session_id: str):
    rows = paged(lambda a, b: (
        sb.table("readings").select(
            "ts,heart_rate_bpm,spo2_pct,signal_quality,ir_mean,ir_ac,"
            "temperature_c,humidity_pct,finger_present")
          .eq("session_id", session_id).order("ts").range(a, b).execute().data))
    if not rows:
        raise HTTPException(404, "session not found")
    t0 = pd.to_datetime(rows[0]["ts"])
    for i, r in enumerate(rows):
        r["t"] = round((pd.to_datetime(r["ts"]) - t0).total_seconds())
    return rows


@app.get("/api/baseline/{subject_id}")
def get_baseline(subject_id: str, posture: Optional[str] = None):
    return bl.baseline(subject_id, posture=posture)


@app.get("/api/heat")
def heat():
    try:
        import heat as heat_mod
        return heat_mod.assess()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/api/status/{subject_id}")
def status(subject_id: str, posture: Optional[str] = "sitting"):
    """Full assessment: latest reading vs baseline, routed to a tier."""
    sess = session_rollup(fetch_subject(subject_id))
    rest = [s for s in sess if (s["context"] or "") in REST]
    if not rest:
        return {"ready": False, "reason": "no resting sessions recorded"}

    latest = rest[-1]
    dev = bl.deviation(subject_id, latest["hr_mean"], posture=posture)

    out = {
        "ready": True,
        "subject_id": subject_id,
        "latest": latest,
        "deviation": dev,
        "n_sessions": len(rest),
        "cards": [],
    }
    if dev.get("ready"):
        out["cards"].append({"module": "fever", **rules.fever_screen(dev)})
        out["cards"].append({"module": "anaemia", **rules.anaemia_screen(dev)})
    return out


@app.post("/api/gestosis")
def gestosis(p: Profile):
    g = rules.gestosis_score(
        age=p.age, bmi=p.bmi, map_mmhg=p.map_mmhg,
        primigravida=p.primigravida, multiple=p.multiple, art=p.art,
        prior_pe=p.prior_pe, chronic_htn=p.chronic_htn,
        diabetes=p.diabetes, renal=p.renal, autoimmune=p.autoimmune,
        family_pe=p.family_pe, interval_over_10y=p.interval_over_10y)
    return g


@app.post("/api/assess/{subject_id}")
def assess(subject_id: str, m: ManualEntry, gestational_weeks: int = 28):
    """Run every manual-entry rule and return the cards."""
    cards = []

    if any(v is not None for v in (m.temp_c, m.sbp, m.dbp)):
        cards.append({"module": "vitals", **rules.vitals_check(
            temp_c=m.temp_c, sbp=m.sbp, dbp=m.dbp)})

    if m.symptoms:
        cards.append({"module": "symptoms", **rules.symptom_check(
            m.symptoms, sbp=m.sbp, dbp=m.dbp)})

    if m.weight_gain_kg is not None:
        cards.append({"module": "weight", **rules.weight_check(
            kg_gained_this_week=m.weight_gain_kg, sbp=m.sbp, dbp=m.dbp)})

    if m.fluid_leak:
        dev = None
        try:
            sess = session_rollup(fetch_subject(subject_id))
            rest = [s for s in sess if (s["context"] or "") in REST]
            if rest:
                dev = bl.deviation(subject_id, rest[-1]["hr_mean"])
        except Exception:
            pass
        cards.append({"module": "prom", **rules.fluid_leak(
            gestational_weeks=gestational_weeks, dev=dev, temp_c=m.temp_c)})

    if m.prior_hb is not None or m.ifa_missed_days:
        dev = {"ready": False}
        try:
            sess = session_rollup(fetch_subject(subject_id))
            rest = [s for s in sess if (s["context"] or "") in REST]
            if rest:
                dev = bl.deviation(subject_id, rest[-1]["hr_mean"])
        except Exception:
            pass
        cards.append({"module": "anaemia", **rules.anaemia_screen(
            dev, symptoms=m.symptoms, prior_hb=m.prior_hb,
            ifa_missed_days=m.ifa_missed_days)})

    cards.sort(key=lambda c: -c["tier"])
    return {"cards": cards, "max_tier": max([c["tier"] for c in cards], default=0)}


@app.post("/api/contractions")
def contractions(c: ContractionLog):
    return rules.contractions(
        gestational_weeks=c.gestational_weeks,
        intervals_min=c.intervals_min,
        duration_sec=c.duration_sec,
        hours_ongoing=c.hours_ongoing)


@app.get("/api/report/{subject_id}")
def report(subject_id: str):
    import report as report_mod
    data = report_mod.gather(subject_id)
    out = HERE / f"report_{subject_id}.pdf"
    report_mod.build(data, out, name=subject_id)
    return FileResponse(out, media_type="application/pdf",
                        filename=f"antenatal_summary_{subject_id}.pdf")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    p = HERE / "dashboard.html"
    if not p.exists():
        return HTMLResponse("<h1>dashboard.html not found in ~/sensors</h1>")
    return HTMLResponse(p.read_text())


@app.get("/api/quality/{session_id}")
def quality_model(session_id: str):
    """Score every window with the trained classifier, not the heuristic."""
    import pickle, numpy as np
    with open(HERE / "model_quality.pkl", "rb") as fh:
        bundle = pickle.load(fh)

    rows = paged(lambda a, b: (
        sb.table("readings").select("*")
          .eq("session_id", session_id).order("ts").range(a, b).execute().data))
    if not rows:
        raise HTTPException(404, "session not found")

    df = pd.DataFrame(rows)
    df["perfusion"] = np.where(df["ir_mean"] > 0,
                               df["ir_ac"] / df["ir_mean"] * 100, 0)
    df["ir_roll_sd"] = df["ir_mean"].rolling(10, min_periods=2).std().fillna(0)
    df["ir_delta"] = df["ir_mean"].diff().abs().fillna(0)
    df["t_into"] = range(len(df))

    X = df[bundle["features"]].fillna(0).values
    p = bundle["model"].predict_proba(X)[:, 1]

    return {
        "session_id": session_id,
        "n": len(p),
        "mean_confidence": round(float(p.mean()), 3),
        "predicted_usable": int((p >= 0.5).sum()),
        "actual_usable": int(df["heart_rate_bpm"].notna().sum()),
        "agreement": round(float(((p >= 0.5) == df["heart_rate_bpm"].notna()).mean()), 3),
    }

@app.get("/app", response_class=HTMLResponse)
def mobile_app():
    p = HERE / "garbha.html"
    if not p.exists():
        return HTMLResponse("<h1>garbha.html not found in ~/sensors</h1>")
    return HTMLResponse(p.read_text())


@app.get("/logo.mp4")
def logo():
    p = HERE / "logo.mp4"
    if not p.exists():
        raise HTTPException(404, "logo.mp4 not found")
    return FileResponse(p, media_type="video/mp4")


@app.get("/app", response_class=HTMLResponse)
def mobile_app():
    p = HERE / "garbha.html"
    if not p.exists():
        return HTMLResponse("<h1>garbha.html not found in ~/sensors</h1>")
    return HTMLResponse(p.read_text())


@app.get("/logo.mp4")
def logo():
    p = HERE / "logo.mp4"
    if not p.exists():
        raise HTTPException(404, "logo.mp4 not found")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/quality/{session_id}")
def quality_model(session_id: str):
    """Score every window with the trained classifier, not the heuristic."""
    import pickle
    with open(HERE / "model_quality.pkl", "rb") as fh:
        bundle = pickle.load(fh)

    rows = paged(lambda a, b: (
        sb.table("readings").select("*")
          .eq("session_id", session_id).order("ts").range(a, b).execute().data))
    if not rows:
        raise HTTPException(404, "session not found")

    df = pd.DataFrame(rows)
    df["perfusion"] = np.where(df["ir_mean"] > 0,
                               df["ir_ac"] / df["ir_mean"] * 100, 0)
    df["ir_roll_sd"] = df["ir_mean"].rolling(10, min_periods=2).std().fillna(0)
    df["ir_delta"] = df["ir_mean"].diff().abs().fillna(0)
    df["t_into"] = range(len(df))

    X = df[bundle["features"]].fillna(0).values
    p = bundle["model"].predict_proba(X)[:, 1]
    actual = df["heart_rate_bpm"].notna()

    return {
        "session_id": session_id,
        "n": int(len(p)),
        "mean_confidence": round(float(p.mean()), 3),
        "predicted_usable": int((p >= 0.5).sum()),
        "actual_usable": int(actual.sum()),
        "agreement": round(float(((p >= 0.5) == actual).mean()), 3),
    }


# ═══════════════ manual logs ═══════════════
class LogEntry(BaseModel):
    subject_id: str
    kind: str
    value_num: Optional[float] = None
    value_num2: Optional[float] = None
    value_text: Optional[str] = None
    gestational_weeks: Optional[int] = None


@app.post("/api/log")
def add_log(e: LogEntry):
    row = e.model_dump()
    row["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    sb.table("logs").insert(row).execute()
    return {"ok": True, "kind": e.kind}


@app.get("/api/logs/{subject_id}")
def get_logs(subject_id: str, kind: Optional[str] = None, days: int = 28):
    since = (datetime.datetime.now(datetime.timezone.utc)
             - datetime.timedelta(days=days)).isoformat()
    q = (sb.table("logs").select("*")
           .eq("subject_id", subject_id).gte("ts", since)
           .order("ts", desc=True).limit(500))
    if kind:
        q = q.eq("kind", kind)
    return q.execute().data


@app.get("/api/today/{subject_id}")
def today(subject_id: str):
    start = datetime.datetime.now(datetime.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0).isoformat()
    rows = (sb.table("logs").select("kind")
              .eq("subject_id", subject_id).gte("ts", start)
              .limit(200).execute().data)
    done = {r["kind"]: True for r in rows}

    sess = session_rollup(fetch_subject(subject_id, days=1))
    if sess:
        done["reading"] = True
    return {"done": done, "date": start[:10]}


def _latest_logs(subject_id, days=28):
    since = (datetime.datetime.now(datetime.timezone.utc)
             - datetime.timedelta(days=days)).isoformat()
    rows = (sb.table("logs").select("*")
              .eq("subject_id", subject_id).gte("ts", since)
              .order("ts", desc=True).limit(500).execute().data)
    latest = {}
    for r in rows:
        latest.setdefault(r["kind"], r)
    return latest, rows


@app.get("/api/assess/{subject_id}")
def assess_stored(subject_id: str, weeks: int = 28):
    """Run every rule against what the person has actually logged."""
    import json as _json
    L, all_rows = _latest_logs(subject_id)
    cards = []

    sbp = L["bp"]["value_num"] if "bp" in L else None
    dbp = L["bp"]["value_num2"] if "bp" in L else None
    temp = L["temp"]["value_num"] if "temp" in L else None
    hb = L["hb"]["value_num"] if "hb" in L else None
    syms = _json.loads(L["symptoms"]["value_text"] or "[]") if "symptoms" in L else []

    missed = sum(1 for r in all_rows
                 if r["kind"] == "iron" and not r.get("value_num"))

    dev = {"ready": False}
    try:
        rest = [s for s in session_rollup(fetch_subject(subject_id))
                if (s["context"] or "") in REST]
        if rest:
            dev = bl.deviation(subject_id, rest[-1]["hr_mean"], posture="sitting")
    except Exception:
        pass

    if dev.get("ready"):
        cards.append({"module": "fever", **rules.fever_screen(dev, symptoms=syms)})
        cards.append({"module": "anaemia", **rules.anaemia_screen(
            dev, symptoms=syms, prior_hb=hb, ifa_missed_days=missed)})

    if any(v is not None for v in (temp, sbp, dbp)):
        cards.append({"module": "vitals", **rules.vitals_check(
            temp_c=temp, sbp=sbp, dbp=dbp)})

    if syms:
        cards.append({"module": "symptoms", **rules.symptom_check(
            syms, sbp=sbp, dbp=dbp)})

    weights = sorted([r for r in all_rows if r["kind"] == "weight"],
                     key=lambda r: r["ts"])
    if len(weights) >= 2:
        gain = weights[-1]["value_num"] - weights[-2]["value_num"]
        cards.append({"module": "weight", **rules.weight_check(
            kg_gained_this_week=gain, sbp=sbp, dbp=dbp)})

    if "leak" in L:
        age_h = (datetime.datetime.now(datetime.timezone.utc)
                 - pd.to_datetime(L["leak"]["ts"]).to_pydatetime()).total_seconds() / 3600
        if age_h < 48:
            cards.append({"module": "prom", **rules.fluid_leak(
                gestational_weeks=weeks, dev=dev, temp_c=temp)})

    cards = [c for c in cards if c["tier"] > 0]
    cards.sort(key=lambda c: -c["tier"])
    return {"cards": cards, "max_tier": max([c["tier"] for c in cards], default=0)}
