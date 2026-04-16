---
title: March Madness 2026
emoji: 🏀
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Nine bracket models vs the 2026 NCAA tournament.
---

# March Madness 2026 — Bracket Retrospective

> Eight prediction models walked into a bracket. Only one of them knew how it ended.

A portfolio project built for EMBA 693R Advanced Machine Learning. Eight independent models — from seeding chalk to a Claude-judged mascot fight — predicted the 2026 NCAA Tournament. This Space is the retrospective: side-by-side brackets, round-by-round accuracy, and a deliberately overfit "Hindsight" model that looks back at the 67 games to ask *which variables actually mattered this year*.

**Why it's here**
- The assignment asked for pre-tournament predictions. This Space adds the one thing notebooks can't: the post-mortem, wrapped in an editorial UI you can click through in a presentation.
- Pre-computed predictions for all eight models are baked into the build, so response time is instant and no live API calls happen at request time.
- Source code and the write-up ship alongside the app — see the **Docs** tab.

## The eight models

| Model | What it does |
|---|---|
| **Seeding Only** | Higher seed wins. The chalk baseline. |
| **Comparative Metrics** | Regression on Barttorvik efficiency diffs + coach tenure + strength of schedule. |
| **Greg_v1** | Tuned Ridge regression, 28 features, recency-weighted samples. |
| **Lean GB (Sampled)** | Gradient-boosted classifier, stochastic draw from each game's win probability. |
| **Lean GB (Tiered)** | Same classifier, deterministic thresholds. |
| **Lean GB (MC Consensus)** | 10k Monte Carlo sims per slot, majority winner. |
| **Animal Kingdom** | Claude judges: if these two mascots fought, who wins? |
| **Vegas Odds** | Real sportsbook lines where available; AI-estimated where not. |
| **Hindsight (overfit)** | Trained *on* the 67 tournament games themselves. Near-perfect accuracy, by construction — useful for surfacing which features explained this year best. |

## Stack

- **Data pipeline & models:** pandas, numpy, scikit-learn, joblib
- **Pre-compute:** `hf_space/prepare_data.py` runs every model through the bracket engine, freezes the output to JSON, trains the Hindsight model, and writes the retrospective stats and documentation to `web/public/data/`.
- **Runtime:** FastAPI serving static HTML/CSS/JS. Plotly.js for charts.
- **Container:** Python 3.11 slim, single process on port 7860.

## Running locally

```bash
pip install -r requirements.txt
python prepare_data.py          # one-time: pre-compute all model brackets into web/public/data/
uvicorn inference.main:app --reload --port 8001
```

Then open http://localhost:8001.

## Credits

Source data: Kaggle *March Machine Learning Mania*, Barttorvik, Massey Consensus. Editorial inspiration: FiveThirtyEight, WSJ, The Economist.
