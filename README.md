# Garbha Sethu — Maternal Health Monitoring System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?style=flat-square&logo=vercel)
![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-4B-red?style=flat-square&logo=raspberrypi)
![Status](https://img.shields.io/badge/Production-Live-green?style=flat-square)

A real-time maternal health wearable system built on Raspberry Pi 4B — continuously acquiring physiological signals, running on-device risk analysis, and streaming alerts to a cloud dashboard. Built in a single hackathon.

**Live Dashboard:** [garbhasethu.vercel.app](https://garbhasethu.vercel.app)  
**Authors:** Lasya N S | BIT Bangalore, VTU

---

## System Architecture

Raspberry Pi 4B (Edge Device)
├── sensors.py — multi-channel physiological signal acquisition
├── features.py — on-device signal conditioning + feature extraction
├── baseline.py — patient baseline computation
├── rules.py — deterministic clinical rules engine (obstetric thresholds)
├── record.py — continuous data recording pipeline
└── report.py — risk report generation
│
▼ (HTTP — low bandwidth optimized)
Vercel Serverless API
├── index.py — main API endpoint
├── model.py — ML risk prediction model
├── heat.py — heatmap generation
└── synthesize.py — data synthesis for testing
│
▼
Dashboard (garbha.html / dashboard.html)
└── Real-time risk alerts + physiological data visualization


---

## What It Does

1. **Sensor Acquisition** — Raspberry Pi 4B continuously reads multi-channel physiological signals from wearable sensors
2. **On-Device Processing** — Signal conditioning, noise filtering, artifact rejection, and windowed feature aggregation — all on-device to minimize payload size
3. **Clinical Rules Engine** — Deterministic obstetric threshold logic converts raw sensor streams into triaged risk alerts with high clinical auditability
4. **Cloud Pipeline** — Optimized data transmission to Vercel serverless API over low-bandwidth connections
5. **Real-Time Dashboard** — Near-instant device-to-dashboard alert propagation via Cloudflare edge routing

---

## Files

### Pi (Edge Device)
| File | Description |
|------|-------------|
| `sensors.py` | Multi-channel physiological sensor acquisition |
| `features.py` | Signal conditioning + feature extraction |
| `baseline.py` | Patient baseline computation |
| `rules.py` | Clinical rules engine — obstetric risk thresholds |
| `record.py` | Continuous data recording pipeline |
| `record_v2.py` | Optimized recording with buffer management |
| `report.py` | Risk report generation |

### API (Vercel Serverless)
| File | Description |
|------|-------------|
| `index.py` | Main API endpoint |
| `model.py` | ML risk prediction model |
| `heat.py` | Heatmap data generation |
| `synthesize.py` | Synthetic data generation for testing |

---

## Deployment

3 production deployments on Vercel:
- **Production** — main dashboard
- **Production – sensors** — sensor data API
- **Production – garbhasethu** — full system

```bash
# Install dependencies
pip install -r requirements.txt

# Run on Raspberry Pi
cd pi/
python record.py

# Run API locally
cd api/
python index.py
```

---

## Built At

Hackathon project — full device-to-dashboard pipeline built in under 48 hours on Raspberry Pi 4B.

**Tech Stack:** Python • Raspberry Pi 4B • Sensor Fusion • Vercel • Cloudflare • HTML/CSS/JS
