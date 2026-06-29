# CERN Relevance

CERN operates and maintains some of the world's largest and most complex particle accelerators, including:

- **LHC** (Large Hadron Collider) — 27 km circumference, 7 TeV beams
- **SPS** (Super Proton Synchrotron) — 7 km injector
- **PS** (Proton Synchrotron) — 628 m
- **LINAC4** — 160 m H⁻ linear accelerator

## Techniques Used at CERN Relevant to This Project

### Beam Diagnostics
CERN's accelerators use hundreds of Beam Position Monitors (BPMs) to continuously measure beam orbit. This project simulates BPM measurements with realistic noise models.

### Orbit Correction
Real machines like the LHC routinely perform orbit correction using response matrices and least-squares or SVD-based solvers. The SVD truncation technique shown in this project is directly analogous to the method used by CERN's orbit correction software (e.g., `ORBIT_CORRECTION` in the LHC control system).

### Feedback Systems
CERN uses slow orbit feedback loops that gradually steer the beam toward the reference orbit over multiple iterations — exactly what the iterative correction algorithm in this project demonstrates.

### Control-Room Visualization
The CERN control rooms (CCC, PCR) feature real-time dashboards showing beam orbit, BPM readings, corrector settings, and performance metrics. This project's Streamlit dashboard mirrors that concept.

### Numerical Methods
Linear algebra, singular value decomposition, and pseudo-inverse computation are fundamental tools in accelerator physics and are used daily by CERN's BE-ABP and BE-OP groups.

## How This Project Connects

| Project Feature | CERN Counterpart |
|---|---|
| FODO lattice simulation | MAD-X / MAD-NG lattice design for LHC, SPS |
| BPM measurement with noise | Real BPM electronics and calibration |
| Response matrix computation | Response matrix measurement campaigns |
| SVD-based orbit correction | LHC orbit correction software |
| Iterative feedback | Slow orbit feedback in SPS/LHC |
| Interactive dashboard | CERN control room displays (JCOP, FESA) |
| Data export (CSV/JSON) | CERN logging and archiving systems |

## Extensions Toward Real-World Complexity

Future versions could incorporate:
- Coupled horizontal/vertical dynamics
- Chromatic effects and sextupole fields
- Comparison with MAD-X simulation output
- Real BPM data processing (turn-by-turn)
- Integration with CERN's `pyJAPC` or `jpype` for live machine access
