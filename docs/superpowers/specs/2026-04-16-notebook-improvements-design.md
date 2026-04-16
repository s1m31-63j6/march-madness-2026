# Notebook Improvements: Class Requirements + Prediction Accuracy

**Date:** 2026-04-16
**Goal:** Shore up class rubric gaps (Phase 3/4), then layer in high-impact accuracy improvements. Keep scope tight — UI work follows immediately.

---

## Context

The March Madness 2026 notebook has a working 8-feature lean GB model (80.3% accuracy, 0.824 AUC on 2023-2025 validation). A classmate's rubric review identified Phase 3 (data prep) and Phase 4 (bivariate analysis) gaps. A recent merge added McKay's cells that demonstrate these techniques on a `df_prep` copy, but they don't feed into the real modeling pipeline.

Separately, an expert review identified that the final model's probabilities are uncalibrated (affecting Monte Carlo and bracket strategies) and that McKay's geographic feature (`teamA_home_state_adv`) is computed then zeroed out due to a bug.

## Changes

### 1. Wire Outlier Handling Into Real Pipeline

**Problem:** The class requires skewness-branched outlier capping (Z-score if |skew| <= 1.0, Tukey IQR if > 1.0). McKay's cell 71 demonstrates this on `df_prep`, but the actual model trains on `matchups_clean` without it.

**Solution:** New code cell after cell 57 (feature column definition), before cell 83 (train/val split):

1. Loop over numeric columns in `MODEL_FEATURE_COLS` (excluding binary flags: `is_big_gap`, `is_late_round`)
2. Compute skewness for each column
3. If |skewness| <= 1.0: cap at mean +/- 3 standard deviations (Z-score / Empirical Rule)
4. If |skewness| > 1.0: cap at Q1 - 1.5*IQR / Q3 + 1.5*IQR (Tukey)
5. Print summary table: feature, initial skewness, method used, values capped, final skewness
6. Apply the same capping thresholds (computed on training data) to `season_2026` for consistent 2026 predictions

**Risk:** May shift model accuracy. Print before/after AUC comparison. Keep the capping regardless (professor wants to see it), but document impact.

**Does not replace McKay's demo cells.** Those remain as the "show your work" demonstration. This cell is the "actually use it" step.

### 2. Calibrate the Lean GB Model

**Problem:** The lean GB model's raw probabilities feed directly into Monte Carlo simulations and all 6 bracket strategies (Options A-F) without calibration. GBMs with shallow trees (`max_depth=2`) tend to produce conservative probabilities that cluster near 0.5, making upset detection unreliable.

**Solution:** New code cell between cell 103 (final model selection) and cell 105 (feature variant comparison):

1. Wrap `final_model` in `CalibratedClassifierCV(method='isotonic', cv=5)`
2. Fit on training data with sample weights (same as original training)
3. Evaluate on validation set: print accuracy, AUC, and Brier score for raw vs calibrated
4. Plot reliability diagram (predicted probability vs observed frequency, 10 bins)
5. Replace `final_model` with calibrated version if AUC doesn't drop by more than 0.005
6. All downstream consumers (`final_model.predict_proba()`) get calibrated probabilities automatically

**SHAP compatibility:** `CalibratedClassifierCV` wraps the base estimator. SHAP (cell 111) needs the inner model. Extract via `final_model.calibrated_classifiers_[0].estimator` for SHAP explanation only. Add a comment explaining why.

**Propagation check — `final_model` is used in:**
- Cell 107 (stacking comparison): `.predict_proba()` — compatible
- Cell 111 (SHAP): needs inner model extraction — fix required
- Cell 117 (Monte Carlo): `.predict_proba()` — compatible
- Cells 119-125 (Options A-F): `.predict_proba()` — compatible
- Cell 129 (export to `prob_model.pkl`): `joblib.dump()` — compatible, engine loads with `.predict_proba()`

### 3. Fix McKay's Geographic Feature Bug

**Problem:** Cell 59 computes `teamA_home_state_adv` correctly (found 107 games with home-state advantage, correlation 0.133 with score_margin), then unconditionally sets it to 0: `matchups_clean['teamA_home_state_adv'] = 0`. Same for `ast_vs_stl_clash_adv`.

**Solution:**
1. Remove the two zeroing lines in cell 59
2. Fill NaN values with 0 (no advantage when city data is missing)
3. Do NOT force these features into the lean model
4. Add one new variant in cell 105 (feature variant comparison): `"Current 8 + geo" = CURRENT_8 + ['teamA_home_state_adv']` and `"Current 8 + geo + style" = CURRENT_8 + ['teamA_home_state_adv', 'ast_vs_stl_clash_adv']`
5. Let the variant comparison decide. If it beats Current 8 on AUC: adopt it. If not: document as "tested, no improvement" (stronger academic story than ignoring it).

**Propagation:** If a new variant wins, `MODEL_FEATURE_COLS` and `LEAN_FEATURES` are updated in cell 105 already. All downstream cells reference these dynamically. The 2026 prediction cell (cell 115) calls `compute_matchup_features()` which would need these columns — verify they're available in `season_2026.csv` or computed at prediction time.

**Likely outcome:** The geographic feature has weak signal (r=0.133 univariate, NaN for point-biserial because the column was zeroed). Even with real values, it probably won't beat the lean 8. But we'll know for sure.

### 4. Update Future Improvements Cell

**Problem:** Cell 140 (future improvements) is generic. Replace with prioritized, evidence-backed improvements from the expert review.

**Content (ordered by expected impact):**

1. **Integrate real Vegas closing lines** — Historical closing lines are the single strongest predictor of tournament outcomes (incorporates efficiency, matchups, injuries, market consensus). Free historical data available from sports-reference.com. Benchmark: if the model can't beat the closing line, we know the ceiling. If it can beat it on specific game types (e.g., 5-vs-12 matchups), that's where real edge lives.

2. **Quality-opponent efficiency splits** — Replace season-average Barttorvik metrics with performance against top-50 defenses/offenses. Directly addresses the Sweet 16 accuracy collapse (58.3% — matching seed baseline). By Sweet 16, all remaining teams are good; season averages lose discriminating power.

3. **Ensemble model probabilities before Monte Carlo** — Current MC simulates 10k brackets from a single model's P(win). This captures bracket-path uncertainty but not model uncertainty. Blending multiple models' probabilities (e.g., lean GB + logistic regression, which have identical 0.824 AUC but different error profiles) reduces single-model error compounding through 63 games.

4. **Matchup interaction features** — Tempo mismatches (fast-paced team vs slow-paced team), pressing defense vs turnover-prone offense. Current features are marginal averages; the model can't learn that Team A's steal rate specifically disrupts Team B's ball-handling. Requires conditional interaction terms, not simple diffs.

5. **In-tournament dynamics** — Injury data, rest days between games, travel distance, and momentum (win streak entering tournament). The biggest gap vs what Vegas uses. Injury news alone moves lines 3-5 points. Requires external data sources updated in real-time.

## Cell Number Note

Cell numbers reference the current merged notebook as of 2026-04-16. Inserting new cells will shift downstream indices. Implementation should locate cells by `id` attribute (e.g., `id=28ac32d2` for feature column definition) rather than positional index.

## What This Does NOT Change

- McKay's demo cells (Phase 3/4) remain as-is — they serve as "show your work"
- The model architecture (HistGradientBoosting, 8 features) stays unless the geo variant wins
- The dashboard code is untouched
- The engine code is untouched (calibrated model has the same `.predict_proba()` API)

## Success Criteria

- All Phase 3 rubric requirements met in the real pipeline (not just demonstrated)
- Calibrated probabilities improve or maintain AUC (>= 0.822)
- Geographic feature tested as a variant with clear result documented
- Future improvements cell reflects expert analysis, not boilerplate
- No downstream cells break (SHAP, Monte Carlo, Options A-F, exports all work)
