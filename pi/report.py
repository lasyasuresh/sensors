"""One-page antenatal summary PDF.

Everything self-recorded between visits, with an explicit limitations
footer. Designed to be printed and handed over, not to replace a chart.
"""
import os, sys, pathlib, datetime
import numpy as np
from supabase import create_client
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

HERE = pathlib.Path(__file__).parent
for line in open(HERE / ".env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

INK, MUTED, RULE = (0.14, 0.12, 0.21), (0.48, 0.46, 0.56), (0.85, 0.83, 0.89)


def gather(subject_id, days=28):
    since = (datetime.datetime.now(datetime.timezone.utc)
             - datetime.timedelta(days=days)).isoformat()
    rows, page, SIZE = [], 0, 1000
    while True:
        batch = (sb.table("readings").select("*")
                   .eq("subject_id", subject_id).gte("ts", since)
                   .order("ts")
                   .range(page * SIZE, (page + 1) * SIZE - 1)
                   .execute().data)
        rows.extend(batch)
        if len(batch) < SIZE:
            break
        page += 1
    good = [r for r in rows
            if r.get("heart_rate_bpm")
            and (r.get("signal_quality") or 0) >= 0.35
            and r.get("context") in ("rest", "synth-rest")]
    hr = [r["heart_rate_bpm"] for r in good]
    spo2 = [r["spo2_pct"] for r in good if r.get("spo2_pct")]
    temps = [r["temperature_c"] for r in rows if r.get("temperature_c")]
    hums = [r["humidity_pct"] for r in rows if r.get("humidity_pct")]

    sessions = {}
    for r in good:
        sessions.setdefault(r["session_id"], []).append(r["heart_rate_bpm"])
    trend = [float(np.mean(v)) for _, v in sorted(sessions.items())]

    return {
        "subject_id": subject_id, "days": days,
        "n_sessions": len(sessions), "n_windows": len(good),
        "hr_mean": np.mean(hr) if hr else None,
        "hr_min": min(hr) if hr else None,
        "hr_max": max(hr) if hr else None,
        "hr_first": trend[0] if trend else None,
        "hr_last": trend[-1] if trend else None,
        "spo2_mean": np.mean(spo2) if spo2 else None,
        "spo2_sd": np.std(spo2) if len(spo2) > 1 else None,
        "amb_t": (min(temps), max(temps)) if temps else None,
        "amb_h": (min(hums), max(hums)) if hums else None,
        "trend": trend,
    }


def build(data, out_path, *, name="", weeks=None, manual=None):
    manual = manual or {}
    c = canvas.Canvas(str(out_path), pagesize=A4)
    W, H = A4
    L, R = 20 * mm, W - 20 * mm
    y = H - 20 * mm

    # header
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(L, y, "Antenatal summary")
    y -= 6 * mm
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(*MUTED)
    end = datetime.date.today()
    start = end - datetime.timedelta(days=data["days"])
    sub = f"Self-recorded at home  |  {start:%d %b} to {end:%d %b %Y}"
    if name:
        sub = f"{name}  |  " + sub
    if weeks:
        sub += f"  |  {weeks} weeks"
    c.drawString(L, y, sub)
    y -= 4 * mm
    c.setStrokeColorRGB(*INK)
    c.setLineWidth(1.2)
    c.line(L, y, R, y)
    y -= 9 * mm

    def section(title):
        nonlocal y
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(L, y, title.upper())
        y -= 5.5 * mm

    def row(k, v, flag=False):
        nonlocal y
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica", 9.5)
        c.drawString(L + 2 * mm, y, k)
        c.setFillColorRGB(0.72, 0.25, 0.25) if flag else c.setFillColorRGB(*INK)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawRightString(R - 2 * mm, y, str(v))
        c.setStrokeColorRGB(*RULE)
        c.setLineWidth(0.4)
        c.line(L, y - 2 * mm, R, y - 2 * mm)
        y -= 7 * mm

    # ---- pulse
    section("Resting pulse (optical sensor, quality filtered)")
    if data["hr_mean"]:
        row("Average", f"{data['hr_mean']:.0f} bpm")
        row("Range", f"{data['hr_min']:.0f} - {data['hr_max']:.0f} bpm")
        if data["hr_first"] and data["hr_last"]:
            d = data["hr_last"] - data["hr_first"]
            row("Change over period", f"{data['hr_first']:.0f} -> "
                                      f"{data['hr_last']:.0f} bpm ({d:+.0f})",
                flag=abs(d) >= 10)
        row("Sessions / windows", f"{data['n_sessions']} / {data['n_windows']}")
    else:
        row("Average", "no usable data")
    y -= 2 * mm

    # ---- sparkline
    if len(data["trend"]) > 1:
        t = data["trend"]
        lo, hi = min(t), max(t)
        span = max(hi - lo, 1)
        bw = (R - L) / len(t)
        base, height = y - 16 * mm, 14 * mm
        c.setFillColorRGB(0.87, 0.84, 0.95)
        for i, v in enumerate(t):
            h = 2 + (v - lo) / span * height
            c.rect(L + i * bw, base, bw * 0.75, h, fill=1, stroke=0)
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica", 7)
        c.drawString(L, base - 4 * mm, f"per-session mean, {lo:.0f}-{hi:.0f} bpm")
        y = base - 10 * mm

    # ---- manual entries
    if manual:
        section("Recorded by the patient")
        for k, v in manual.items():
            flag = any(w in str(v).lower() for w in ("high", "yes", "reported"))
            row(k, v, flag=flag)
        y -= 2 * mm

    # ---- oxygen
    section("Oxygen saturation (uncalibrated, trend only)")
    if data["spo2_mean"]:
        row("Average", f"{data['spo2_mean']:.1f} %")
        if data["spo2_sd"]:
            row("Variability (SD)", f"{data['spo2_sd']:.1f} points",
                flag=data["spo2_sd"] > 3)
    else:
        row("Average", "no usable data")
    y -= 2 * mm

    # ---- environment
    section("Environment (ambient air, not body temperature)")
    if data["amb_t"]:
        row("Air temperature", f"{data['amb_t'][0]:.1f} - {data['amb_t'][1]:.1f} C")
    if data["amb_h"]:
        row("Humidity", f"{data['amb_h'][0]:.0f} - {data['amb_h'][1]:.0f} %")

    # ---- limitations footer
    fy = 26 * mm
    c.setStrokeColorRGB(*RULE)
    c.setLineWidth(0.6)
    c.line(L, fy + 14 * mm, R, fy + 14 * mm)
    c.setFillColorRGB(*MUTED)
    c.setFont("Helvetica-Oblique", 7.2)
    notes = [
        "Limitations. Pulse and oxygen saturation come from an uncalibrated optical sensor (MAX30100) worn at home; readings below a signal-quality",
        "threshold are excluded rather than reported. Temperature and humidity are ambient air measurements from a DHT11 and are NOT body temperature.",
        "Nothing in this summary is a diagnosis. It is intended to support, not replace, clinical assessment at an antenatal visit.",
    ]
    for i, n in enumerate(notes):
        c.drawString(L, fy + 9 * mm - i * 3.6 * mm, n)

    c.setFont("Helvetica", 6.5)
    c.drawString(L, fy - 4 * mm,
                 f"Generated {datetime.datetime.now():%d %b %Y, %H:%M}  |  subject {data['subject_id']}")

    c.showPage()
    c.save()
    return out_path


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "synth01"

    data = gather(subj)
    out = HERE / f"report_{subj}.pdf"

    build(data, out,
          name="Patient A", weeks="28+3",
          manual={
              "Gestosis score": "4 - high risk",
              "Blood pressure (avg of 7)": "128/84 mmHg",
              "Highest blood pressure": "142/91 mmHg",
              "Weight change": "+2.4 kg over 4 weeks",
              "Iron tablets taken": "24 of 28 days",
              "Fluid leak reported": "none",
              "Symptoms reported": "headache x2, swelling x3",
          })

    print(f"\nwrote {out}")
    print(f"  {data['n_sessions']} sessions, {data['n_windows']} usable windows")
    if data["hr_mean"]:
        print(f"  resting pulse {data['hr_mean']:.0f} bpm "
              f"({data['hr_min']:.0f}-{data['hr_max']:.0f})")