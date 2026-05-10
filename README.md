# 🏗️ Singly Reinforced RCC Beam Designer

**Design of Singly Reinforced RCC Beams as per IS 456:2000 (Limit State Method)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## Features

- Fully code-compliant with **IS 456:2000**
- Simply supported beam under uniformly distributed load (UDL)
- Detailed step-by-step design report with:
  - Every formula cited to the IS 456 clause
  - Formula expression with numerical values substituted
  - Final result in SI units (N, mm, N/mm², kN·m)
- Downloadable `.txt` design report
- Instant KPI dashboard (Ast, utilisation ratio, shear, deflection)

## Design Steps Covered

| Step | Description | IS 456:2000 Reference |
|------|-------------|----------------------|
| 1 | Effective depth | Cl. 22.2 |
| 2 | Factored bending moment | Table 18 |
| 3 | Limiting moment of resistance (xu,max/d) | Annex G, Cl. G-1.1 |
| 4 | Section adequacy (singly vs doubly reinforced) | Annex G |
| 5 | Tension steel area — quadratic solution | Annex G, Eq. G-1.1a |
| 6 | Minimum and maximum steel checks | Cl. 26.5.1.1 |
| 7 | Bar selection (number and diameter) | — |
| 8 | Actual neutral axis depth | Annex G |
| 9 | Actual moment of resistance | Annex G, Eq. G-1.1a |
| 10 | Shear design (τv, τc, τc,max, stirrup spacing) | Cl. 40 |
| 11 | Deflection check (basic L/d × MF) | Cl. 23.2 |
| 12 | Detailing — minimum width, side face steel, development length | Cl. 26 |

## Local Setup

```bash
git clone https://github.com/<your-username>/rcc-beam-designer.git
cd rcc-beam-designer
pip install -r requirements.txt
streamlit run app.py
```

## Deployment on Streamlit Community Cloud

1. Push this repository to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set **Main file path** to `app.py`.
4. Click **Deploy**. Your app will be live in ~1–2 minutes.

## Repository Structure

```
rcc-beam-designer/
├── app.py            ← Streamlit application (single file)
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

## Inputs

| Parameter | Unit | Description |
|-----------|------|-------------|
| L | m | Effective span |
| w_DL | kN/m | Dead load (UDL) |
| w_LL | kN/m | Live load (UDL) |
| fck | N/mm² | Concrete grade (M15–M40) |
| fy | N/mm² | Steel grade (Fe250/415/500/550) |
| b | mm | Beam width |
| D | mm | Total depth |
| c' | mm | Clear cover |
| φ | mm | Main bar diameter |
| φv | mm | Stirrup diameter |

## Reference

IS 456:2000 — *Plain and Reinforced Concrete — Code of Practice*,
Bureau of Indian Standards, New Delhi.

---
Developed for academic and professional use in Structural Engineering.
