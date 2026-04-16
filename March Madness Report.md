# Univariate Statistics

This section summarises the pre-cleaning exploration of every column in the assembled matchups dataset (924 rows × 67 columns). The analysis identifies data-quality issues that inform the cleaning decisions applied before modeling.

## Column Classification

| **Type** | **Count** | **Representative Columns** |
|----|----|----|
| Continuous numeric | 61 | adj_em_diff, barthag_diff, wab_diff, seed_diff, score_margin, … |
| Boolean (0/1) | 3 | game_result, is_big_gap, is_late_round |
| Categorical / ID | 6 | season, teamA_id, teamB_id (+ 3 derived) |

## Summary Statistics Table — Continuous Numeric Features

*All 61 continuous columns have 924 complete observations except the three tempo columns (78.6% missing). The table below shows a representative subset; the full 61-row table is available in the notebook (§1.10b).*

| **Column** | **Mean** | **Median** | **Std** | **Min** | **Max** | **Skewness** | **Kurtosis** | **Missing %** |
|----|----|----|----|----|----|----|----|----|
| teamA_seed | 4.197 | 3.000 | 3.385 | 1.000 | 16.000 | +1.56 | 2.59 | 0.0% |
| teamB_seed | 10.197 | 11.000 | 3.896 | 1.000 | 16.000 | −0.39 | −0.67 | 0.0% |
| seed_diff | −6.000 | −5.000 | 4.335 | −15.000 | 0.000 | −0.44 | −0.77 | 0.0% |
| score_margin | 6.956 | 7.000 | 13.048 | −34.000 | 47.000 | +0.05 | 0.17 | 0.0% |
| total_points | 139.486 | 139.000 | 18.664 | 90.000 | 202.000 | +0.22 | −0.03 | 0.0% |
| adj_em_diff | 9.027 | 7.249 | 9.460 | −12.650 | 46.030 | +0.99 | 1.03 | 0.0% |
| barthag_diff | 0.121 | 0.067 | 0.155 | −0.217 | 0.690 | +1.52 | 1.96 | 0.0% |
| wab_diff | 5.095 | 4.221 | 4.816 | −8.650 | 25.304 | +1.07 | 1.36 | 0.0% |
| a_barthag | 0.902 | 0.925 | 0.099 | 0.253 | 0.984 | −4.15 | 20.08 | 0.0% |
| a_adj_t | 67.592 | 67.757 | 2.764 | 61.323 | 75.064 | −0.03 | −0.05 | 78.6% |
| b_adj_t | 67.298 | 67.455 | 2.564 | 59.136 | 75.103 | −0.04 | +1.20 | 78.6% |
| adj_t_diff | 0.294 | 0.295 | 3.796 | −9.100 | 13.181 | +0.16 | +0.11 | 78.6% |
| a_coach_tourn_wins | 18.424 | 10.000 | 21.380 | 0.000 | 97.000 | +1.60 | 2.17 | 0.0% |
| a_coach_final_fours | 0.948 | 0.000 | 1.833 | 0.000 | 9.000 | +2.67 | 7.71 | 0.0% |
| b_coach_tourn_wins | 6.978 | 1.000 | 12.903 | 0.000 | 91.000 | +2.84 | 8.89 | 0.0% |
| b_coach_final_fours | 0.249 | 0.000 | 0.825 | 0.000 | 9.000 | +5.18 | 38.61 | 0.0% |
| seed_disagreement | 0.000 | −0.219 | 5.605 | −19.919 | 21.179 | +0.21 | 0.39 | 0.0% |
| sos_diff | 0.038 | 0.032 | 0.061 | −0.118 | 0.232 | +0.34 | −0.30 | 0.0% |
| ast_rate_diff | 0.010 | 0.011 | 0.073 | −0.185 | 0.238 | +0.04 | −0.10 | 0.0% |

## Summary Statistics Table — Boolean Features

| **Column**    | **Count** | **Missing** | **Mean** | **% Positive** |
|---------------|-----------|-------------|----------|----------------|
| game_result   | 924       | 0           | 0.725    | 72.5%          |
| is_big_gap    | 924       | 0           | 0.374    | 37.4%          |
| is_late_round | 924       | 0           | 0.227    | 22.7%          |

## Key Univariate Visualizations

Rather than reproducing every histogram from the 61 continuous columns, the six plots below highlight the patterns and issues most relevant to modeling:

**1. teamA_seed histogram  
**Right-skewed distribution (skew = +1.56). Lower seeds (1–4) dominate because Team A is always the lower (favored) seed by construction. This is a structural property of bracket assembly, not a recording error.

> <img src="/media/image1.png" style="width:4.08187in;height:2.34882in" />
>
> **2. a_barthag boxplot  
> **Severely left-skewed (skew = −4.15, kurtosis = 20.08) with 27 extreme outliers beyond the 3×IQR fence. Elite teams cluster near 1.0, creating a long lower tail from the few weak tournament qualifiers. These are legitimate observations.
>
> <img src="/media/image2.png" style="width:3.27809in;height:1.89926in" />
>
> **3. adj_t_diff histogram (missing-value annotation)  
> **78.6% of values are missing — the dominant data-quality issue in the dataset. Barttorvik tempo data is absent from pre-2019 CSV files. The 198 non-missing rows are roughly normally distributed around zero, but the near-total absence renders this column unusable for modeling.
>
> <img src="/media/image3.png" style="width:3.25509in;height:1.92906in" />
>
> **4. score_margin histogram  
> **Near-perfectly symmetric (skew = +0.05), approximating a normal distribution centred at +6.96 (Team A wins by ~7 points on average). No transformation required. This is the primary regression target.
>
> <img src="/media/image4.png" style="width:4.23086in;height:2.37941in" />
>
> **5. b_coach_final_fours boxplot  
> **Extreme right skew (skew = +5.18, kurtosis = 38.61) with 113 outliers. The distribution is zero-inflated: most Team B coaches have zero Final Four appearances, while a small cohort (e.g., Coach K, Roy Williams) have 5–9. A meaningful signal, but requires attention in linear models.
>
> <img src="/media/image5.png" style="width:3.01492in;height:1.75097in" />
>
> **6. game_result count plot  
> **72.5% of games are won by Team A (the lower seed), confirming the bracket seeding structure and flagging a modest class imbalance for binary classification tasks. Stratified cross-validation will be used during modeling.
>
> <img src="/media/image6.png" style="width:3.31123in;height:2.77493in" />

## Data Quality Issues and Phase 2 Remediation Plan

The automated quality scan flagged 24 issues across 67 columns, grouped into five categories:

### \[1\] Missing Values

| **Column** | **Missing %** | **Remediation in Phase 2**                        |
|------------|---------------|---------------------------------------------------|
| a_adj_t    | 78.6%         | Exclude from model feature set — majority missing |
| b_adj_t    | 78.6%         | Exclude from model feature set — majority missing |
| adj_t_diff | 78.6%         | Exclude from model feature set — majority missing |

*All other columns are 100% complete. No imputation is required beyond dropping the three tempo columns from the feature set.*

### \[2\] High Skewness (\|skew\| \> 1.5)

| **Column** | **Skewness** | **Direction** | **Remediation** |
|----|----|----|----|
| teamA_seed | +1.56 | Right | Structural — tree models handle naturally; no transform needed |
| daynum | +1.71 | Right | Structural (round-progression); treat as ordinal or exclude |
| a_barthag | −4.15 | Left | Note for OLS; tree-based models (GBM/RF) are robust |
| a_coach_tourn_wins | +1.60 | Right | Consider log(x+1) transform if used in OLS |
| a_coach_final_fours | +2.67 | Right | Consider log(x+1) or cap at 95th percentile for OLS |
| b_coach_appearances | +1.58 | Right | Consider log(x+1) transform if used in OLS |
| b_coach_tourn_wins | +2.84 | Right | Consider log(x+1) transform if used in OLS |
| b_coach_final_fours | +5.18 | Right | Consider log(x+1) or exclude — extreme skew and kurtosis |
| barthag_diff | +1.52 | Right | Note for OLS; tree models handle naturally |
| coach_final_fours_diff | +1.74 | Right | Consider log transform or cap at 95th percentile |

### \[3\] Extreme Outliers (\> 3 observations beyond 3×IQR fence)

| **Column** | **\# Outliers** | **Remediation** |
|----|----|----|
| b_coach_final_fours | 113 | Zero-inflated; tree models robust; cap for OLS |
| b_coach_tourn_wins | 65 | Elite-coach spikes; cap at 95th pct for OLS |
| coach_final_fours_diff | 45 | Same root cause; cap or log(x+1) for OLS |
| daynum | 42 | By-design tournament round structure — not a defect |
| a_coach_final_fours | 42 | Zero-inflated coach statistic; cap or log for OLS |
| barthag_diff | 19 | Large seed-gap matchups (1 vs 16); legitimate values |
| coach_tourn_wins_diff | 13 | Elite-coach effect; tree models robust |
| a_adj_em | 8 | Extreme efficiency margins; legitimate values |
| a_wab | 7 | Wins Above Bubble extremes; legitimate values |
| a_sos | 5 | Unusual strength-of-schedule; investigate per season |

### \[4\] Class Imbalance (Boolean Features)

No column failed the 25%/75% threshold for severe imbalance. game_result is 72.5% positive (Team A wins) — acceptable, but stratified cross-validation will be applied during all classification modeling to ensure balanced evaluation across folds.

### \[5\] Near-Zero Variance

No features with standard deviation \< 0.01 were detected. All 61 continuous columns carry meaningful variance and are candidates for the feature set.

### \[6\] Structural Multicollinearity Notes

- teamA_seed, teamB_seed, and seed_diff are algebraically redundant. Only seed_diff (or the composite seed_disagreement) will enter the model feature set.

- All a\_\* and b\_\* raw columns have a corresponding \_diff column. Raw team-level columns will be excluded to avoid multicollinearity; only the differential features will be used for modeling.

- teamA_id and teamB_id are high-cardinality ID columns with no direct predictive signal. They will be excluded from all feature sets.

## Summary

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>24 issues flagged across 67 columns.<br />
</strong>The dominant issue is 78.6% missingness in all three tempo columns (a_adj_t, b_adj_t, adj_t_diff), resolved by exclusion from the feature set. The remaining issues — skewness and outliers in coach statistics and barthag — are concentrated in columns used within tree-based models (GBM / Random Forest), which are robust to these distributions. Linear model variants (OLS) will apply log(x+1) transforms where flagged before Phase 2 modeling.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>
