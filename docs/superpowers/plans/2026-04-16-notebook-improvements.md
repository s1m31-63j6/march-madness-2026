# Notebook Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire class-required outlier handling into the real pipeline, calibrate the lean GB model, fix McKay's geographic feature bug, and update the future improvements cell.

**Architecture:** All changes are notebook cell edits/inserts in `march_madness.ipynb`. Cells are located by their JSON `id` attribute, not positional index. A Python helper script modifies the notebook JSON directly since notebook cells can't be edited with standard text tools.

**Tech Stack:** Python, scikit-learn (CalibratedClassifierCV, SimpleImputer), scipy.stats, numpy, pandas, SHAP

---

## File Map

- **Modify:** `march_madness.ipynb` — 4 cell edits + 2 cell insertions
  - Cell `id=d9dba56a` (McKay's geo feature) — remove zeroing bug
  - Cell `id=feat_variant_code` (variant comparison) — add geo variants
  - Cell `id=18708365` (SHAP) — already handles calibrated models (no change needed, already patched)
  - Cell `id=future_improvements` (future improvements) — replace content
  - **Insert** new cell after `id=7ed7e987` (1.13b Sync Seed/Round Context) — outlier capping on `matchups_clean`
  - **Insert** new cell after `id=b63ca19e` — lean GB calibration

---

### Task 1: Fix McKay's Geographic Feature Bug

**Files:**
- Modify: `march_madness.ipynb` cell `id=d9dba56a`

- [ ] **Step 1: Remove the two zeroing lines**

In cell `id=d9dba56a`, find and remove these two lines:

```python
    matchups_clean['teamA_home_state_adv'] = 0
```

and:

```python
    matchups_clean['ast_vs_stl_clash_adv'] = 0
```

These appear AFTER the features are computed and BEFORE the `new_features` registration block. They zero out the features McKay built, making them useless.

- [ ] **Step 2: Add NaN fill for geographic feature**

In the same cell, after the line `matchups_clean = matchups_clean.drop(columns=['MinTeam', 'MaxTeam', 'Season', 'DayNum', 'GameState', 'teamA_home_state', 'teamB_home_state'])`, add:

```python
    # Fill NaN (games where city data missing) with 0 — no advantage
    matchups_clean['teamA_home_state_adv'] = matchups_clean['teamA_home_state_adv'].fillna(0).astype(int)
```

- [ ] **Step 3: Verify by running the cell**

Expected output should still show "Found ~107 games" and correlations should now be real (not NaN for point-biserial). The correlation with `score_margin` should be ~0.133 for `teamA_home_state_adv`.

- [ ] **Step 4: Commit**

```bash
git add march_madness.ipynb
git commit -m "fix: remove zeroing bug in geographic/stylistic features

McKay's cell computed teamA_home_state_adv and ast_vs_stl_clash_adv
correctly, then unconditionally set them to 0. Removed the zeroing
lines so features carry real values. Added NaN fill for missing city data."
```

---

### Task 2: Add Geo Variants to Feature Comparison

**Files:**
- Modify: `march_madness.ipynb` cell `id=feat_variant_code`

- [ ] **Step 1: Add two new variants to the variants dict**

In cell `id=feat_variant_code`, after the line `KITCHEN_10 = CURRENT_8 + ['adj_o_diff', 'adj_d_diff']`, add:

```python
# Geographic advantage (McKay's feature, previously zeroed out — now fixed)
GEO_9 = CURRENT_8 + ['teamA_home_state_adv']

# Geographic + stylistic clash
GEO_STYLE_10 = CURRENT_8 + ['teamA_home_state_adv', 'ast_vs_stl_clash_adv']
```

Then in the `variants` dict, add the two new entries:

```python
    'Current 8 + geo':       GEO_9,
    'Current 8 + geo+style': GEO_STYLE_10,
```

- [ ] **Step 2: Verify the full variants dict looks like this**

```python
variants = {
    'Current 8':              CURRENT_8,
    'OLS Sig 9':              OLS_SIG_9,
    'Hybrid 9 (+def)':        HYBRID_9,
    'Lean 7 (-min_seed)':     LEAN_7,
    'Kitchen sink 10':        KITCHEN_10,
    'Current 8 + geo':        GEO_9,
    'Current 8 + geo+style':  GEO_STYLE_10,
}
```

- [ ] **Step 3: Commit**

```bash
git add march_madness.ipynb
git commit -m "feat: add geographic/stylistic variants to feature comparison

Tests McKay's fixed features against the lean 8 baseline.
If they improve AUC by >0.002, the pipeline auto-adopts them."
```

---

### Task 3: Wire Outlier Handling Into Real Pipeline

**Files:**
- Modify: `march_madness.ipynb` — insert new cell after `id=28ac32d2`

- [ ] **Step 1: Insert a new code cell after cell `id=28ac32d2`**

Use a Python script to insert the cell into the notebook JSON:

```python
import json

with open('march_madness.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find insertion point: after cell id=7ed7e987 (1.13b Sync Seed/Round Context)
# This runs AFTER geo features (d9dba56a) and McKay's features (df0cd7d9) are computed,
# so outlier capping sees all features including the newly fixed ones.
insert_after = None
for i, c in enumerate(nb['cells']):
    if c.get('id') == '7ed7e987':
        insert_after = i
        break

new_cell = {
    "cell_type": "code",
    "id": "outlier_capping_real",
    "metadata": {},
    "outputs": [],
    "source": [
        "# 1.13a Apply skewness-branched outlier capping to matchups_clean\n",
        "# Class requirement: |skewness| <= 1.0 -> Z-score (Empirical Rule), > 1.0 -> Tukey IQR\n",
        "# This runs on the REAL pipeline data, not a demo copy.\n",
        "import scipy.stats as sp_stats\n",
        "import numpy as np\n",
        "\n",
        "# Identify numeric modeling features (exclude binary flags)\n",
        "BINARY_FLAGS = ['is_big_gap', 'is_late_round']\n",
        "numeric_feat = [c for c in MODEL_FEATURE_COLS if c in matchups_clean.columns and c not in BINARY_FLAGS]\n",
        "\n",
        "# Store capping thresholds so we can apply the same transform to season_2026 later\n",
        "outlier_thresholds = {}\n",
        "\n",
        "print(f\"{'Feature':<30} | {'Skew':>8} | {'Method':<10} | {'Capped':>7} | {'New Skew':>8}\")\n",
        "print('-' * 75)\n",
        "\n",
        "for col in numeric_feat:\n",
        "    skew = matchups_clean[col].skew()\n",
        "    n_before = len(matchups_clean)\n",
        "\n",
        "    if abs(skew) <= 1.0:\n",
        "        # Z-score / Empirical Rule: cap at mean +/- 3 std\n",
        "        method = 'Z-Score'\n",
        "        mean = matchups_clean[col].mean()\n",
        "        std = matchups_clean[col].std()\n",
        "        lower = mean - 3 * std\n",
        "        upper = mean + 3 * std\n",
        "    else:\n",
        "        # Tukey IQR: cap at Q1 - 1.5*IQR / Q3 + 1.5*IQR\n",
        "        method = 'Tukey IQR'\n",
        "        q1 = matchups_clean[col].quantile(0.25)\n",
        "        q3 = matchups_clean[col].quantile(0.75)\n",
        "        iqr = q3 - q1\n",
        "        lower = q1 - 1.5 * iqr\n",
        "        upper = q3 + 1.5 * iqr\n",
        "\n",
        "    outlier_thresholds[col] = {'lower': lower, 'upper': upper, 'method': method}\n",
        "\n",
        "    n_capped = ((matchups_clean[col] < lower) | (matchups_clean[col] > upper)).sum()\n",
        "    matchups_clean[col] = matchups_clean[col].clip(lower=lower, upper=upper)\n",
        "    new_skew = matchups_clean[col].skew()\n",
        "\n",
        "    print(f\"{col:<30} | {skew:>8.3f} | {method:<10} | {n_capped:>7} | {new_skew:>8.3f}\")\n",
        "\n",
        "print(f\"\\nOutlier capping applied to {len(numeric_feat)} features in matchups_clean.\")\n",
        "print(f\"Thresholds saved for consistent 2026 prediction preprocessing.\")\n",
        "\n",
        "# Apply same thresholds to season_2026 features (if they exist)\n",
        "if 'season_2026' in dir():\n",
        "    for col, bounds in outlier_thresholds.items():\n",
        "        # season_2026 uses raw column names (e.g. 'adj_em'), not diff names\n",
        "        # Diffs are computed at prediction time, so we skip diff columns here\n",
        "        if col in season_2026.columns:\n",
        "            season_2026[col] = season_2026[col].clip(lower=bounds['lower'], upper=bounds['upper'])\n",
        "    print(f\"Same thresholds applied to season_2026 where columns match.\")\n"
    ]
}

nb['cells'].insert(insert_after + 1, new_cell)

with open('march_madness.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Inserted outlier capping cell after index {insert_after}")
```

- [ ] **Step 2: Verify the cell is in the right position**

Run a quick check that the new cell sits between "1.13 Data Cleaning" (id=28ac32d2) and the next cell:

```python
import json
with open('march_madness.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, c in enumerate(nb['cells']):
    if c.get('id') in ['28ac32d2', 'outlier_capping_real']:
        print(f"idx={i} id={c.get('id')}")
```

Expected: two consecutive indices.

- [ ] **Step 3: Commit**

```bash
git add march_madness.ipynb
git commit -m "feat: wire skewness-branched outlier capping into real pipeline

Applies Z-score (|skew| <= 1.0) or Tukey IQR (|skew| > 1.0) capping
to matchups_clean before model training. Stores thresholds for
consistent 2026 prediction preprocessing. Class rubric Phase 3 requirement."
```

---

### Task 4: Calibrate the Lean GB Model

**Files:**
- Modify: `march_madness.ipynb` — insert new cell after `id=b63ca19e` (final model selection)

- [ ] **Step 1: Insert calibration cell after cell `id=b63ca19e`**

```python
import json

with open('march_madness.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

insert_after = None
for i, c in enumerate(nb['cells']):
    if c.get('id') == 'b63ca19e':
        insert_after = i
        break

new_cell = {
    "cell_type": "code",
    "id": "lean_gb_calibration",
    "metadata": {},
    "outputs": [],
    "source": [
        "# 3.2b-cal Calibrate the final lean GB model\n",
        "# Raw GBM probabilities (max_depth=2, lr=0.02) tend to cluster near 0.5,\n",
        "# making upset detection unreliable. Isotonic calibration corrects this\n",
        "# so downstream Monte Carlo sims and bracket strategies get trustworthy P(win).\n",
        "from sklearn.calibration import CalibratedClassifierCV, calibration_curve\n",
        "from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "# Save uncalibrated model for SHAP (CalibratedClassifierCV wraps the estimator)\n",
        "final_model_uncalibrated = final_model\n",
        "\n",
        "# ── Raw model metrics ─────────────────────────────────────────────\n",
        "raw_probs = final_model.predict_proba(val_df[MODEL_FEATURE_COLS].values)[:, 1]\n",
        "raw_preds = (raw_probs > 0.5).astype(int)\n",
        "raw_acc   = accuracy_score(y_val, raw_preds)\n",
        "raw_auc   = roc_auc_score(y_val, raw_probs)\n",
        "raw_brier = brier_score_loss(y_val, raw_probs)\n",
        "\n",
        "# ── Calibrate ─────────────────────────────────────────────────────\n",
        "cal_model = CalibratedClassifierCV(final_model, method='isotonic', cv=5)\n",
        "fit_kw_cal = {'sample_weight': weights_train} if USE_SAMPLE_WEIGHTS else {}\n",
        "cal_model.fit(train_df[MODEL_FEATURE_COLS].values, y_train, **fit_kw_cal)\n",
        "\n",
        "cal_probs = cal_model.predict_proba(val_df[MODEL_FEATURE_COLS].values)[:, 1]\n",
        "cal_preds = (cal_probs > 0.5).astype(int)\n",
        "cal_acc   = accuracy_score(y_val, cal_preds)\n",
        "cal_auc   = roc_auc_score(y_val, cal_probs)\n",
        "cal_brier = brier_score_loss(y_val, cal_probs)\n",
        "\n",
        "# ── Comparison table ──────────────────────────────────────────────\n",
        "print('Calibration Comparison (validation set 2023-2025)')\n",
        "print(f\"{'Metric':<20} {'Raw GB':>10} {'Calibrated':>12} {'Delta':>10}\")\n",
        "print('-' * 55)\n",
        "print(f\"{'Accuracy':<20} {raw_acc:>10.3f} {cal_acc:>12.3f} {cal_acc - raw_acc:>+10.3f}\")\n",
        "print(f\"{'ROC AUC':<20} {raw_auc:>10.3f} {cal_auc:>12.3f} {cal_auc - raw_auc:>+10.3f}\")\n",
        "print(f\"{'Brier Score':<20} {raw_brier:>10.3f} {cal_brier:>12.3f} {cal_brier - raw_brier:>+10.3f}\")\n",
        "print('(Brier: lower is better)')\n",
        "\n",
        "# ── Reliability diagram ───────────────────────────────────────────\n",
        "fig, ax = plt.subplots(1, 1, figsize=(6, 5))\n",
        "for label, probs in [('Raw GB', raw_probs), ('Calibrated', cal_probs)]:\n",
        "    frac_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)\n",
        "    ax.plot(mean_pred, frac_pos, 's-', label=label)\n",
        "ax.plot([0, 1], [0, 1], 'k--', label='Perfect')\n",
        "ax.set_xlabel('Predicted probability')\n",
        "ax.set_ylabel('Observed frequency')\n",
        "ax.set_title('Reliability Diagram: Raw vs Calibrated')\n",
        "ax.legend()\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# ── Adopt calibrated model if AUC didn't drop ─────────────────────\n",
        "if cal_auc >= raw_auc - 0.005:\n",
        "    final_model = cal_model\n",
        "    print(f'\\n>>> Adopted calibrated model (AUC {cal_auc:.3f} >= {raw_auc:.3f} - 0.005)')\n",
        "else:\n",
        "    print(f'\\n>>> Keeping raw model (calibrated AUC {cal_auc:.3f} dropped too much vs {raw_auc:.3f})')\n"
    ]
}

nb['cells'].insert(insert_after + 1, new_cell)

with open('march_madness.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Inserted calibration cell after index {insert_after}")
```

- [ ] **Step 2: Verify SHAP cell (id=18708365) handles calibrated model**

The SHAP cell already has this logic (we can confirm it's present):

```python
if hasattr(final_model, 'named_steps'):
    base_pipe = final_model
elif hasattr(final_model, 'estimator'):
    base_pipe = final_model.estimator
else:
    base_pipe = final_model_uncalibrated
```

This handles all three cases: raw Pipeline, CalibratedClassifierCV (has `.estimator`), or fallback to `final_model_uncalibrated`. Confirm cell `id=18708365` contains this logic. If not, update it.

- [ ] **Step 3: Commit**

```bash
git add march_madness.ipynb
git commit -m "feat: add isotonic calibration for lean GB model

Wraps final_model in CalibratedClassifierCV(isotonic, cv=5).
Prints accuracy/AUC/Brier comparison and reliability diagram.
Adopts calibrated model if AUC doesn't drop by >0.005.
All downstream cells (MC, Options A-F, export) get better
probabilities automatically via predict_proba()."
```

---

### Task 5: Update Future Improvements Cell

**Files:**
- Modify: `march_madness.ipynb` cell `id=future_improvements`

- [ ] **Step 1: Replace the cell source**

Use a Python script to find cell `id=future_improvements` and replace its source:

```python
import json

with open('march_madness.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb['cells']:
    if c.get('id') == 'future_improvements':
        c['source'] = [
            "---\n",
            "## 6. Future Improvement Options\n",
            "\n",
            "Prioritized by expected impact on prediction accuracy, based on analysis of where\n",
            "the current model's signal breaks down.\n",
            "\n",
            "### 1. Integrate Real Vegas Closing Lines (Highest Impact)\n",
            "\n",
            "Historical closing lines are the single strongest predictor of tournament outcomes.\n",
            "The line incorporates everything: efficiency metrics, matchup specifics, injuries,\n",
            "public and sharp money, and market consensus. Free historical data is available from\n",
            "sports-reference.com. **Benchmark use:** if the model can't beat the closing line,\n",
            "we know the ceiling. If it beats the line on specific game types (e.g., 5-vs-12\n",
            "matchups), that's where real edge lives.\n",
            "\n",
            "### 2. Quality-Opponent Efficiency Splits\n",
            "\n",
            "Replace season-average Barttorvik metrics with performance against top-50\n",
            "defenses/offenses. This directly addresses the **Sweet 16 accuracy collapse**\n",
            "(58.3% — matching seed baseline). By Sweet 16, all remaining teams are good;\n",
            "season averages lose discriminating power. Barttorvik publishes opponent-quality\n",
            "splits that we don't currently use.\n",
            "\n",
            "### 3. Ensemble Model Probabilities Before Monte Carlo\n",
            "\n",
            "Current MC simulates 10k brackets from a single model's P(win). This captures\n",
            "bracket-path uncertainty but not **model uncertainty**. Blending multiple models'\n",
            "probabilities (e.g., lean GB + logistic regression, which have identical 0.824 AUC\n",
            "but different error profiles) reduces single-model error compounding through 63\n",
            "games. A simple average of calibrated probabilities is a strong starting point.\n",
            "\n",
            "### 4. Matchup Interaction Features\n",
            "\n",
            "Current features are marginal averages (team A's steal rate, team B's assist rate\n",
            "as independent columns). The model can't learn that Team A's pressing defense\n",
            "specifically disrupts Team B's ball-handling. **Tempo mismatches** (fast vs slow),\n",
            "**3-point reliance vs rim protection**, and **turnover forcing vs ball security**\n",
            "all require conditional interaction terms, not simple diffs.\n",
            "\n",
            "### 5. In-Tournament Dynamics\n",
            "\n",
            "The biggest gap vs what Vegas uses. Injury data alone moves lines 3-5 points.\n",
            "Rest days between games, travel distance, and momentum (win streak entering\n",
            "tournament) all matter. Would require external data sources updated in real-time\n",
            "during the tournament window — a fundamentally different data pipeline than\n",
            "pre-tournament feature assembly.\n"
        ]
        break

with open('march_madness.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Updated future improvements cell")
```

- [ ] **Step 2: Commit**

```bash
git add march_madness.ipynb
git commit -m "docs: replace future improvements with evidence-backed priorities

Ordered by expected impact based on expert analysis of where
the model's signal breaks down (Sweet 16 collapse, calibration,
single-model MC, missing market data)."
```

---

### Task 6: Final Verification

- [ ] **Step 1: Verify all cell IDs are present and in correct order**

```python
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('march_madness.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

expected_order = [
    '28ac32d2',              # 1.13 Data Cleaning (feature cols)
    'd9dba56a',              # McKay's geo feature (bug fixed)
    'outlier_capping_real',  # NEW: outlier capping on matchups_clean (after all features)
    'e9cd49b3',              # Train/val split
    'b63ca19e',              # Final model selection (3.2b)
    'lean_gb_calibration',   # NEW: calibration
    'feat_variant_code',     # Feature variant comparison (with geo variants)
    '18708365',              # SHAP (handles calibrated model)
    '1f4fc7cd',              # Export (prob_model.pkl)
    'future_improvements',   # Updated future improvements
]

positions = {}
for i, c in enumerate(nb['cells']):
    cid = c.get('id', '')
    if cid in expected_order:
        positions[cid] = i

print("Cell positions (should be in ascending order):")
for cid in expected_order:
    idx = positions.get(cid, 'MISSING')
    status = '✓' if cid in positions else '✗ MISSING'
    print(f"  {status} idx={idx:<4} id={cid}")

# Verify order
idxs = [positions[c] for c in expected_order if c in positions]
if idxs == sorted(idxs):
    print("\n✓ All cells in correct order")
else:
    print("\n✗ ORDER VIOLATION — cells are not in expected sequence")
```

- [ ] **Step 2: Run the notebook end-to-end and verify no errors**

Open the notebook in Jupyter/Colab and run all cells. Key checkpoints:
- Outlier capping cell prints the before/after table
- Geographic feature shows real correlations (not NaN)
- Calibration cell prints comparison table and reliability diagram
- SHAP cell renders without errors
- Monte Carlo and Options A-F run with calibrated probabilities
- Export cell saves `prob_model.pkl` (calibrated version)
- Feature variant comparison includes geo variants

- [ ] **Step 3: Final commit**

```bash
git add march_madness.ipynb
git commit -m "verify: all notebook improvements in place and ordered correctly"
```
