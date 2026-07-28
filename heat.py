"""Heat exposure module.

Combines live outdoor conditions from Open-Meteo with the belt's
under-clothing microclimate reading. Ambient measurements only - nothing
here is a body measurement.
"""
import os, pathlib, datetime
import requests
from supabase import create_client
from rules import heat_index, heat_check
from sensors import read_dht

HERE = pathlib.Path(__file__).parent
for line in open(HERE / ".env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v

LAT = float(os.environ.get("LAT", "12.9716"))    # Bengaluru
LON = float(os.environ.get("LON", "77.5946"))
PLACE = os.environ.get("PLACE", "Bengaluru")

API = "https://api.open-meteo.com/v1/forecast"


def outdoor(lat=LAT, lon=LON):
    """Current conditions + 4-day peaks. Returns None if offline."""
    try:
        r = requests.get(API, timeout=10, params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature",
            "daily": "temperature_2m_max,apparent_temperature_max",
            "timezone": "auto", "forecast_days": 5,
        })
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        print(f"  ! weather unavailable ({type(e).__name__})")
        return None

    cur = d["current"]
    daily = d["daily"]
    return {
        "temp_c":    cur["temperature_2m"],
        "humidity":  cur["relative_humidity_2m"],
        "feels_c":   cur["apparent_temperature"],
        "forecast": [
            {"date": daily["time"][i],
             "max_c": daily["temperature_2m_max"][i],
             "feels_max_c": daily["apparent_temperature_max"][i]}
            for i in range(len(daily["time"]))
        ],
    }


def assess(lat=LAT, lon=LON, place=PLACE):
    """Full heat picture: outdoor, under-belt, guidance, forecast warnings."""
    out = outdoor(lat, lon)
    belt_t, belt_h = read_dht()

    result = {"place": place,
              "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "outdoor": out, "belt": None, "guidance": None, "ahead": []}

    if belt_t is not None and belt_h is not None:
        result["belt"] = {"temp_c": belt_t, "humidity": belt_h,
                          "feels_c": heat_index(belt_t, belt_h)}

    # guidance uses whichever environment is harsher
    if out:
        src_t, src_h = out["temp_c"], out["humidity"]
        if result["belt"] and result["belt"]["feels_c"] > heat_index(src_t, src_h):
            src_t, src_h = belt_t, belt_h
        peak = max(f["feels_max_c"] for f in out["forecast"][:1])
        result["guidance"] = heat_check(src_t, src_h, forecast_peak_c=round(peak, 1))

        for f in out["forecast"][1:5]:
            if f["feels_max_c"] >= 38:
                day = datetime.date.fromisoformat(f["date"]).strftime("%A")
                result["ahead"].append(
                    {"day": day, "feels_max_c": round(f["feels_max_c"], 1)})
    elif result["belt"]:
        result["guidance"] = heat_check(belt_t, belt_h)

    return result


if __name__ == "__main__":
    a = assess()
    print(f"\n=== heat - {a['place']} ===")

    if a["outdoor"]:
        o = a["outdoor"]
        print(f"  outdoor    {o['temp_c']:.1f} C  {o['humidity']:.0f}% RH  "
              f"feels {heat_index(o['temp_c'], o['humidity']):.1f} C")
    if a["belt"]:
        b = a["belt"]
        print(f"  under belt {b['temp_c']:.1f} C  {b['humidity']:.0f}% RH  "
              f"feels {b['feels_c']:.1f} C")

    g = a["guidance"]
    if g:
        print(f"\n  [{g['tier']}] {g['tier_name'].upper()}  {g['title']}")
        print(f"      {g['detail']}")
        for e in g["evidence"]:
            print(f"        - {e}")

    if a["ahead"]:
        print("\n  hot days ahead:")
        for d in a["ahead"]:
            print(f"    {d['day']:<10} feels {d['feels_max_c']} C")
    print()