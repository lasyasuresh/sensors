"""Clinical rules engine.

Every function returns a dict with a tier and a plain-language message.
Nothing here diagnoses. Sensor findings can reach tier 2 at most; only
reported symptoms and confirmed instrument readings can reach tier 3.
"""

LOG, SELFCARE, CONFIRM, CONTACT = 0, 1, 2, 3

TIER_NAME = {0: "log", 1: "self-care", 2: "confirm it", 3: "call your provider"}


def _r(tier, title, detail, evidence=None, action=None):
    return {"tier": tier, "tier_name": TIER_NAME[tier], "title": title,
            "detail": detail, "evidence": evidence or [], "action": action}


# ---------------------------------------------------------------- gestosis
def gestosis_score(*, age, bmi, map_mmhg, primigravida=False, multiple=False,
                   art=False, prior_pe=False, chronic_htn=False,
                   diabetes=False, renal=False, autoimmune=False,
                   family_pe=False, interval_over_10y=False):
    """FOGSI HDP-Gestosis score. Total >=3 = high risk."""
    items = []
    def add(pts, label):
        if pts:
            items.append((label, pts))

    add(2 if age > 35 else 0,            "Age over 35")
    add(2 if bmi > 30 else 0,            "BMI over 30")
    add(1 if primigravida else 0,        "First pregnancy")
    add(2 if map_mmhg > 85 else 0,       "MAP above 85 mmHg")
    add(3 if prior_pe else 0,            "Previous pre-eclampsia")
    add(3 if chronic_htn else 0,         "Chronic hypertension")
    add(2 if multiple else 0,            "Multiple pregnancy")
    add(2 if art else 0,                 "Assisted reproduction")
    add(2 if diabetes else 0,            "Diabetes")
    add(2 if renal else 0,               "Renal disease")
    add(2 if autoimmune else 0,          "Autoimmune disease")
    add(1 if family_pe else 0,           "Family history")
    add(1 if interval_over_10y else 0,   "Interval over 10 years")

    total = sum(p for _, p in items)
    high = total >= 3

    return {
        "score": total, "high_risk": high, "factors": items,
        "result": _r(
            CONTACT if high else LOG,
            "Higher risk of pre-eclampsia" if high else "Standard risk",
            ("Your score is 3 or above. Show this to your doctor — women in "
             "this group are usually offered low-dose aspirin before 16 weeks "
             "and closer monitoring.")
            if high else
            "No additional risk factors recorded. Rescore at each visit.",
            evidence=[f"{l} (+{p})" for l, p in items],
            action="Discuss aspirin prophylaxis at your next visit" if high else None,
        )}


# ------------------------------------------------- confirmed vitals (MEOWS)
def vitals_check(*, temp_c=None, hr=None, sbp=None, dbp=None, spo2=None):
    """Applied ONLY to instrument-confirmed values, never to belt output."""
    red, yellow = [], []

    if temp_c is not None:
        if temp_c >= 38.0 or temp_c < 35.0: red.append(f"Temperature {temp_c} C")
        elif temp_c < 36.0:                 yellow.append(f"Temperature {temp_c} C")
    if hr is not None:
        if hr > 120 or hr < 40:             red.append(f"Pulse {hr:.0f} bpm")
        elif hr > 100 or hr < 50:           yellow.append(f"Pulse {hr:.0f} bpm")
    if sbp is not None:
        if sbp >= 160 or sbp < 90:          red.append(f"Systolic {sbp} mmHg")
        elif sbp >= 150 or sbp <= 100:      yellow.append(f"Systolic {sbp} mmHg")
    if dbp is not None:
        if dbp > 100:                       red.append(f"Diastolic {dbp} mmHg")
        elif dbp >= 90:                     yellow.append(f"Diastolic {dbp} mmHg")
    if spo2 is not None and spo2 < 95:      red.append(f"Oxygen {spo2}%")

    if red or len(yellow) >= 2:
        return _r(CONTACT, "Call your provider today",
                  "One or more readings are outside the range used in "
                  "obstetric early-warning charts. This needs a clinician to "
                  "look at it, not an app.",
                  evidence=red + yellow, action="Call now")
    if yellow:
        return _r(CONFIRM, "Worth rechecking",
                  "One reading is borderline. Rest quietly for ten minutes "
                  "and take it again.",
                  evidence=yellow, action="Recheck in 10 minutes")
    return _r(LOG, "Recorded", "Readings logged.", evidence=[])


# ------------------------------------------------------ fever screen (belt)
def fever_screen(dev, symptoms=None, days_elevated=1):
    """dev = output of baseline.deviation(). Can never exceed tier 2."""
    symptoms = symptoms or []
    if not dev.get("ready") or dev["level"] in ("usual", "slight"):
        return _r(LOG, "Nothing to flag", "Resting pulse is in your usual range.")

    ev = [f"Resting pulse {dev['current_hr']} bpm, your usual is {dev['baseline']}",
          f"Elevated on {days_elevated} of the last {days_elevated} morning(s)"]
    if symptoms:
        ev.append("You reported: " + ", ".join(symptoms))
    ev.append("The belt cannot measure body temperature")

    return _r(CONFIRM, "Your resting pulse has been higher than usual",
              "That happens for ordinary reasons — heat, poor sleep, a busy "
              "few days. It can also happen with a fever. Take your "
              "temperature with a thermometer so we know which.",
              evidence=ev, action="Take your temperature and log it")


# ---------------------------------------------------- anaemia risk (no Hb)
def anaemia_screen(dev, *, symptoms=None, prior_hb=None, ifa_missed_days=0,
                   short_birth_interval=False, teenage=False, vegetarian=False):
    """Compensatory tachycardia + symptoms + risk factors -> ask for an Hb test."""
    symptoms = symptoms or []
    points, ev = 0, []

    if dev.get("ready") and dev["level"] in ("moderate", "marked"):
        points += 2
        ev.append(f"Resting pulse {dev['current_hr']} bpm vs usual {dev['baseline']}")

    hits = [s for s in symptoms
            if s in ("tiredness", "breathlessness", "dizziness", "pica", "palpitations")]
    if len(hits) >= 2:
        points += 2; ev.append("Reported: " + ", ".join(hits))
    elif hits:
        points += 1; ev.append("Reported: " + ", ".join(hits))

    if prior_hb is not None and prior_hb < 11.0:
        points += 2; ev.append(f"Last haemoglobin {prior_hb} g/dL")
    if ifa_missed_days >= 7:
        points += 1; ev.append(f"Iron tablets missed on {ifa_missed_days} days")
    if short_birth_interval:
        points += 1; ev.append("Short interval since last birth")
    if teenage:
        points += 1; ev.append("Under 20")
    if vegetarian:
        points += 1; ev.append("Vegetarian diet")

    if points < 3:
        return _r(LOG, "Nothing to flag", "No anaemia risk pattern right now.")

    return _r(CONFIRM, "Worth checking your haemoglobin",
              "A few things together suggest it's worth a blood test. Anaemia "
              "in pregnancy is common in India and treatable — a simple Hb "
              "test at your next visit will settle it.",
              evidence=ev, action="Ask for an Hb test at your next visit")


# ------------------------------------------------------------- contractions
def contractions(*, gestational_weeks, intervals_min, duration_sec, hours_ongoing):
    """Gestational age gates everything. 5-1-1 is a TERM rule only."""
    if not intervals_min:
        return _r(LOG, "Timer running", "Keep tapping with each contraction.")

    avg_gap = sum(intervals_min) / len(intervals_min)
    regular = len(intervals_min) >= 4 and \
              max(intervals_min) - min(intervals_min) <= avg_gap * 0.5

    if gestational_weeks < 37:
        if regular or len(intervals_min) >= 6:
            return _r(CONTACT, "Call your provider now",
                      f"You are {gestational_weeks} weeks. Regular tightenings "
                      "before 37 weeks need to be checked straight away — the "
                      "5-1-1 rule does not apply before term.",
                      evidence=[f"About every {avg_gap:.0f} minutes",
                                f"{len(intervals_min)} recorded"],
                      action="Call now")
        return _r(CONFIRM, "Keep timing",
                  f"At {gestational_weeks} weeks any regular pattern matters. "
                  "Drink water, lie on your left side, and keep the timer "
                  "running. If they settle, they were likely practice "
                  "contractions.",
                  evidence=[f"About every {avg_gap:.0f} minutes"],
                  action="Continue timing for 30 minutes")

    if avg_gap <= 5 and duration_sec >= 60 and hours_ongoing >= 1:
        return _r(CONTACT, "Time to go in",
                  "Five minutes apart, lasting a minute, for an hour. That is "
                  "the point at which you head to hospital.",
                  evidence=[f"Every {avg_gap:.0f} min",
                            f"Lasting {duration_sec:.0f} s",
                            f"Going {hours_ongoing:.1f} h"],
                  action="Go to hospital")

    return _r(LOG, "Not yet 5-1-1",
              "Keep timing. Practice contractions are irregular, don't get "
              "stronger, and ease off with rest and water.",
              evidence=[f"Every {avg_gap:.0f} min, lasting {duration_sec:.0f} s"])


# -------------------------------------------------------- fluid leak (PROM)
def fluid_leak(*, gestational_weeks, dev=None, temp_c=None, fever_reported=False):
    """Any suspected rupture is tier 3. No algorithm, by design."""
    infection = fever_reported or (temp_c is not None and temp_c >= 38.0) or \
                (dev and dev.get("ready") and dev["level"] == "marked")

    if infection:
        return _r(CONTACT, "Go in now",
                  "Fluid leaking together with a raised temperature or pulse "
                  "can mean an infection in the waters. This is urgent — go to "
                  "the hospital rather than waiting.",
                  evidence=["Fluid leak reported",
                            f"Temperature {temp_c} C" if temp_c else "Pulse well above your usual"],
                  action="Go to hospital now")

    return _r(CONTACT, "Get examined today",
              f"At {gestational_weeks} weeks, any leaking fluid should be "
              "checked the same day. It is hard to tell amniotic fluid from "
              "urine at home, and only an examination can say for certain.",
              evidence=["Fluid leak reported"],
              action="Call your provider today")


# ----------------------------------------------------- pre-eclampsia flags
PE_SYMPTOMS = {
    "headache":      "Bad headache that won't go",
    "vision":        "Blurred vision, flashing lights or spots",
    "epigastric":    "Pain under the ribs on the right",
    "swelling":      "Sudden swelling of face or hands",
    "reduced_urine": "Passing much less urine",
    "vomiting":      "Vomiting in late pregnancy",
}

def symptom_check(reported, *, sbp=None, dbp=None):
    hits = [PE_SYMPTOMS[s] for s in reported if s in PE_SYMPTOMS]
    if not hits:
        return _r(LOG, "Recorded", "No red-flag symptoms today.")

    high_bp = (sbp is not None and sbp >= 140) or (dbp is not None and dbp >= 90)

    if len(hits) >= 2 or high_bp:
        return _r(CONTACT, "Call your provider now",
                  "These are the warning signs of pre-eclampsia. Do not wait "
                  "for your next appointment.",
                  evidence=hits + ([f"BP {sbp}/{dbp}"] if high_bp else []),
                  action="Call now")

    return _r(CONFIRM, "Check your blood pressure",
              "This can be a warning sign when it comes with raised blood "
              "pressure. Take a reading now and log it.",
              evidence=hits, action="Take your blood pressure")


# ----------------------------------------------------------- weight & heat
def weight_check(*, kg_gained_this_week, sbp=None, dbp=None):
    if kg_gained_this_week is None or kg_gained_this_week < 2.0:
        return _r(LOG, "Recorded", "Weight logged.")
    high_bp = (sbp is not None and sbp >= 140) or (dbp is not None and dbp >= 90)
    tier = CONTACT if high_bp else CONFIRM
    return _r(tier,
              "Call your provider" if high_bp else "Check your blood pressure",
              "Gaining more than 2 kg in a week usually means fluid, not fat. "
              "With raised blood pressure it matters more.",
              evidence=[f"Up {kg_gained_this_week:.1f} kg this week"]
                       + ([f"BP {sbp}/{dbp}"] if high_bp else []),
              action="Call now" if high_bp else "Take your blood pressure")


def heat_index(temp_c, humidity_pct):
    """NOAA heat index, metric. Ambient measurement only."""
    T = temp_c * 9 / 5 + 32
    R = humidity_pct
    if T < 80:
        hi_f = 0.5 * (T + 61 + (T - 68) * 1.2 + R * 0.094)
    else:
        hi_f = (-42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R
                - 0.00683783*T*T - 0.05481717*R*R + 0.00122874*T*T*R
                + 0.00085282*T*R*R - 0.00000199*T*T*R*R)
        if R < 13 and 80 <= T <= 112:
            hi_f -= ((13 - R) / 4) * ((17 - abs(T - 95)) / 17) ** 0.5
    return round((hi_f - 32) * 5 / 9, 1)


def heat_check(temp_c, humidity_pct, *, forecast_peak_c=None):
    hi = heat_index(temp_c, humidity_pct)
    ev = [f"Feels like {hi} C", f"{temp_c} C at {humidity_pct:.0f}% humidity"]
    if forecast_peak_c:
        ev.append(f"Peak today around {forecast_peak_c} C")

    if hi >= 40:
        return _r(SELFCARE, "Too hot to be outside",
                  "Stay indoors, drink water often, and use a wet cloth on "
                  "your neck and wrists. Heat is harder on you in pregnancy.",
                  evidence=ev, action="Stay indoors and hydrate")
    if hi >= 33:
        return _r(SELFCARE, "Take care in the heat",
                  "Drink a glass of water now and avoid being outdoors "
                  "between noon and four. Loose cotton over the belt lets "
                  "trapped air escape.",
                  evidence=ev, action="Drink water now")
    return _r(LOG, "Comfortable", "Conditions are fine today.", evidence=ev)


# ------------------------------------------------------------------- demo
if __name__ == "__main__":
    def show(res):
        print(f"\n[{res['tier']}] {res['tier_name'].upper():<18} {res['title']}")
        print(f"    {res['detail']}")
        for e in res["evidence"]:
            print(f"      · {e}")
        if res["action"]:
            print(f"    -> {res['action']}")

    print("=" * 66)
    g = gestosis_score(age=32, bmi=31.2, map_mmhg=88, primigravida=True)
    print(f"GESTOSIS SCORE: {g['score']}  high_risk={g['high_risk']}")
    show(g["result"])

    show(vitals_check(temp_c=38.4, hr=104, sbp=138, dbp=88))
    show(fever_screen({"ready": True, "level": "moderate",
                       "current_hr": 96, "baseline": 82},
                      symptoms=["tiredness"], days_elevated=3))
    show(anaemia_screen({"ready": True, "level": "moderate",
                         "current_hr": 96, "baseline": 82},
                        symptoms=["tiredness", "breathlessness"],
                        prior_hb=10.2, ifa_missed_days=9))
    show(contractions(gestational_weeks=31, intervals_min=[9, 10, 9, 11, 10, 9],
                      duration_sec=45, hours_ongoing=1.5))
    show(contractions(gestational_weeks=39, intervals_min=[5, 4, 5, 5, 4],
                      duration_sec=62, hours_ongoing=1.2))
    show(fluid_leak(gestational_weeks=33, temp_c=38.2))
    show(symptom_check(["headache", "vision"], sbp=148, dbp=94))
    show(weight_check(kg_gained_this_week=2.6, sbp=144, dbp=92))
    show(heat_check(36.0, 68))
    print("\n" + "=" * 66)