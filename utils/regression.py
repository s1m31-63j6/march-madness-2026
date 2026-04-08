import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt


def run_regression(df, label, exclude_cols=None, cat_cols=None,
                   max_cardinality=50, test_size=0.2, random_state=42,
                   vif_threshold=10.0, methods=None, nan_strategy='median',
                   cv_folds=5):
    """
    General-purpose OLS / Ridge / Lasso comparison pipeline.

    Methodology
    -----------
    - Label NaN rows dropped first; all other NaNs imputed on train only
    - Split BEFORE any preprocessing to prevent data leakage
    - Categorical encoding fit on train, test aligned to same columns
    - Zero-variance columns computed on train only
    - K-fold CV for stable, comparable metrics across all models
    - Baseline (mean predictor) included for context
    - OLS diagnostics: Breusch-Pagan, Durbin-Watson, Jarque-Bera, Cook's Distance, VIF

    Parameters
    ----------
    nan_strategy : 'median' | 'mean' — imputation strategy for numeric NaNs
    cv_folds     : number of folds for cross-validation
    methods      : list of 'ols', 'ridge', 'lasso' (default: all three)
    """
    if exclude_cols is None:
        exclude_cols = []
    if methods is None:
        methods = ['ols', 'ridge', 'lasso']

    # ── 1. Drop rows where label is NaN, warn about selection bias ────────────
    df_clean = df[df[label].notna()].copy().reset_index(drop=True)
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        pct = n_dropped / len(df)
        print(f'WARNING: Dropped {n_dropped} rows ({pct:.1%}) with missing label.')
        print('  If missingness is non-random, results may be biased toward observed cases.\n')

    # ── 2. Identify feature columns ───────────────────────────────────────────
    num_cols = [c for c in df_clean.select_dtypes(include='number').columns
                if c != label and c not in exclude_cols]

    if cat_cols is None:
        cat_cols = [c for c in df_clean.select_dtypes(include='object').columns
                    if c not in exclude_cols and df_clean[c].nunique() <= max_cardinality]

    print(f'Numeric features ({len(num_cols)}): {num_cols}')
    print(f'Categorical features ({len(cat_cols)}): {cat_cols}\n')

    X_raw = df_clean[num_cols + cat_cols]
    y     = df_clean[label]

    # ── 3. Split FIRST (no preprocessing before this line) ───────────────────
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state
    )
    X_tr_raw = X_tr_raw.reset_index(drop=True)
    X_te_raw = X_te_raw.reset_index(drop=True)
    y_train  = y_train.reset_index(drop=True)
    y_test   = y_test.reset_index(drop=True)

    # ── 4. Impute numeric (fit on train only) ─────────────────────────────────
    if num_cols:
        num_imputer = SimpleImputer(strategy=nan_strategy)
        X_tr_num = pd.DataFrame(num_imputer.fit_transform(X_tr_raw[num_cols]), columns=num_cols)
        X_te_num = pd.DataFrame(num_imputer.transform(X_te_raw[num_cols]),     columns=num_cols)
    else:
        X_tr_num = X_te_num = pd.DataFrame()

    # ── 5. Encode categorical (fit on train, align test to same columns) ──────
    tr_dummies, te_dummies = [], []

    for col in cat_cols:
        tr_col = X_tr_raw[col].fillna('MISSING')
        te_col = X_te_raw[col].fillna('MISSING')
        is_multival = tr_col.str.contains(',', na=False).any()

        if is_multival:
            tr_d = tr_col.str.get_dummies(sep=',').add_prefix(f'{col}_')
            te_d = te_col.str.get_dummies(sep=',').add_prefix(f'{col}_')
        else:
            tr_d = pd.get_dummies(tr_col, prefix=col, drop_first=True, dtype=int)
            te_d = pd.get_dummies(te_col, prefix=col,                   dtype=int)

        # Unseen categories in test → 0; categories only in test → dropped
        te_d = te_d.reindex(columns=tr_d.columns, fill_value=0)
        tr_dummies.append(tr_d.reset_index(drop=True))
        te_dummies.append(te_d.reset_index(drop=True))

    # ── 6. Assemble full feature matrices ─────────────────────────────────────
    X_train = pd.concat([X_tr_num] + tr_dummies, axis=1)
    X_test  = pd.concat([X_te_num] + te_dummies, axis=1)

    # ── 7. Drop zero-variance columns (computed on train only) ────────────────
    zero_var = X_train.columns[X_train.std() == 0].tolist()
    if zero_var:
        print(f'Dropping zero-variance columns (train): {zero_var}')
        X_train = X_train.drop(columns=zero_var)
        X_test  = X_test.drop(columns=zero_var, errors='ignore')

    print(f'Rows: train={len(X_train)}, test={len(X_test)}  |  Features: {X_train.shape[1]}')

    # ── 8. Standardize ───────────────────────────────────────────────────────
    # Ridge/Lasso: scale everything (regularization is scale-sensitive)
    # OLS: scale numeric only — dummies are already [0,1] and scaling them
    #      distorts coefficient interpretation without improving the model
    features     = X_train.columns.tolist()
    num_in_X = [c for c in num_cols if c in X_train.columns]

    scaler     = StandardScaler()
    scaler_ols = StandardScaler()

    # Full scale (Ridge / Lasso)
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=features)

    # Numeric-only scale (OLS)
    X_train_ols = X_train.copy()
    X_test_ols  = X_test.copy()
    if num_in_X:
        X_train_ols[num_in_X] = scaler_ols.fit_transform(X_train[num_in_X])
        X_test_ols[num_in_X]  = scaler_ols.transform(X_test[num_in_X])

    # ── 9. Fit models + CV ───────────────────────────────────────────────────
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}

    # Baseline
    baseline_cv = cross_val_score(DummyRegressor(strategy='mean'), X_train_s, y_train, cv=kf, scoring='r2')
    results['baseline'] = {
        'coefs'  : pd.Series(dtype=float),
        'metrics': {'r2': float(baseline_cv.mean()), 'mae': np.nan, 'rmse': np.nan},
        'cv_r2'  : baseline_cv,
    }

    # OLS (numeric-only scaling)
    if 'ols' in methods:
        X_tr_c = X_train_ols.copy(); X_tr_c.insert(0, 'const', 1.0)
        X_te_c = X_test_ols.copy();  X_te_c.insert(0, 'const', 1.0)
        ols        = sm.OLS(y_train, X_tr_c).fit()
        y_pred_ols = ols.predict(X_te_c)
        ols_cv     = cross_val_score(LinearRegression(), X_train_ols, y_train, cv=kf, scoring='r2')

        results['ols'] = {
            'model'  : ols,
            'coefs'  : ols.params.drop('const'),
            'pvals'  : ols.pvalues.drop('const'),
            'metrics': _metrics(y_test, y_pred_ols),
            'cv_r2'  : ols_cv,
        }

    # Ridge (alpha path from data via cross-validation)
    ridge_alphas = np.logspace(-3, 3, 100)
    if 'ridge' in methods:
        ridge_cv_model = RidgeCV(alphas=ridge_alphas, cv=kf)
        ridge_cv_model.fit(X_train_s, y_train)
        y_pred_ridge = ridge_cv_model.predict(X_test_s)
        ridge_cv     = cross_val_score(Ridge(alpha=ridge_cv_model.alpha_), X_train_s, y_train, cv=kf, scoring='r2')
        _plot_alpha_search('ridge', ridge_alphas, X_train_s, y_train, kf, ridge_cv_model.alpha_)

        results['ridge'] = {
            'model'  : ridge_cv_model,
            'coefs'  : pd.Series(ridge_cv_model.coef_, index=features),
            'alpha'  : ridge_cv_model.alpha_,
            'metrics': _metrics(y_test, y_pred_ridge),
            'cv_r2'  : ridge_cv,
        }

    # Lasso (alpha path computed from data — no hardcoded range)
    if 'lasso' in methods:
        lasso_cv_model = LassoCV(cv=kf, random_state=random_state, max_iter=10000)
        lasso_cv_model.fit(X_train_s, y_train)
        y_pred_lasso = lasso_cv_model.predict(X_test_s)
        lasso_cv     = cross_val_score(Lasso(alpha=lasso_cv_model.alpha_, max_iter=10000),
                                       X_train_s, y_train, cv=kf, scoring='r2')
        _plot_alpha_search('lasso', lasso_cv_model.alphas_, X_train_s, y_train, kf, lasso_cv_model.alpha_)

        results['lasso'] = {
            'model'  : lasso_cv_model,
            'coefs'  : pd.Series(lasso_cv_model.coef_, index=features),
            'alpha'  : lasso_cv_model.alpha_,
            'metrics': _metrics(y_test, y_pred_lasso),
            'cv_r2'  : lasso_cv,
            'n_zero' : int((lasso_cv_model.coef_ == 0).sum()),
        }

    # ── 10. Metrics table ────────────────────────────────────────────────────
    print()
    print('=' * 65)
    print('MODEL COMPARISON')
    print('=' * 65)

    metric_rows = {}
    for name, r in results.items():
        row = {**r['metrics']}
        if 'cv_r2' in r:
            row['cv_r2_mean'] = r['cv_r2'].mean()
            row['cv_r2_std']  = r['cv_r2'].std()
        metric_rows[name] = row

    print(pd.DataFrame(metric_rows).T.to_string(float_format='{:.4f}'.format))

    if 'ridge' in results:
        print(f"\n  Ridge best alpha: {results['ridge']['alpha']:.5f}")
    if 'lasso' in results:
        print(f"  Lasso best alpha: {results['lasso']['alpha']:.5f}  "
              f"|  Features zeroed: {results['lasso']['n_zero']}/{len(features)}")

    # ── 11. OLS assumption tests ─────────────────────────────────────────────
    if 'ols' in results:
        ols_model = results['ols']['model']
        resid     = ols_model.resid
        X_tr_c_vals = X_tr_c.values

        print()
        print('OLS ASSUMPTION TESTS')
        print('-' * 65)

        # Breusch-Pagan (heteroscedasticity)
        bp_lm, bp_pval, _, _ = het_breuschpagan(resid, X_tr_c_vals)
        _print_test('Breusch-Pagan (heteroscedasticity)', bp_lm, bp_pval,
                    'Heteroscedasticity detected — OLS SEs unreliable. '
                    'Consider log-transforming the label.')

        # Durbin-Watson (autocorrelation)
        dw = durbin_watson(resid)
        dw_flag = dw < 1.5 or dw > 2.5
        flag = '***' if dw_flag else 'OK '
        print(f'  [{flag}] Durbin-Watson (autocorrelation):   {dw:.4f}  (ideal ~2.0)')
        if dw_flag:
            print('         Autocorrelation present — check if data has temporal/spatial ordering.')

        # Jarque-Bera (normality of residuals)
        jb_stat, jb_pval, skew, kurt = jarque_bera(resid)
        _print_test('Jarque-Bera (normality of residuals)', jb_stat, jb_pval,
                    f'Non-normal residuals (skew={skew:.2f}, kurt={kurt:.2f}). '
                    'P-values and CIs may be unreliable with small samples.')

        # Cook's Distance
        influence = OLSInfluence(ols_model)
        cooks_d   = influence.cooks_distance[0]
        threshold = 4 / len(y_train)
        n_inf     = int((cooks_d > threshold).sum())
        flag = '***' if n_inf > 0 else 'OK '
        print(f"  [{flag}] Cook's Distance:                    "
              f"{n_inf} influential obs (threshold={threshold:.4f})")
        results['cooks_d'] = cooks_d

        # VIF
        vif_df = pd.DataFrame({
            'feature': features,
            'VIF'    : [variance_inflation_factor(X_train_s.values, i)
                        for i in range(X_train_s.shape[1])],
        }).sort_values('VIF', ascending=False)

        print()
        print(f'VIF (multicollinearity, threshold={vif_threshold}):')
        print(vif_df.to_string(index=False, float_format='{:.2f}'.format))
        high_vif = vif_df[vif_df['VIF'] > vif_threshold]
        if not high_vif.empty:
            print(f'  *** High VIF — consider removing: {high_vif["feature"].tolist()}')

        results['vif'] = vif_df

    # ── 12. Plots ────────────────────────────────────────────────────────────
    _plot_comparison(results, features, label)

    if 'ols' in results:
        _plot_ols_diagnostics(results['ols']['model'], results.get('cooks_d'), y_train)

    results['scaler']       = scaler
    results['feature_cols'] = features
    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred):
    return {
        'r2'  : r2_score(y_true, y_pred),
        'mae' : mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
    }


def _print_test(name, stat, pval, warning):
    flag = '***' if pval < 0.05 else 'OK '
    print(f'  [{flag}] {name}: stat={stat:.3f}, p={pval:.4f}')
    if pval < 0.05:
        print(f'         {warning}')


def _plot_comparison(results, features, label):
    """Coefficient comparison bar chart + CV R2 boxplot."""
    model_keys = [k for k in results if k not in ('baseline', 'scaler', 'feature_cols', 'vif', 'cooks_d')
                  and len(results[k].get('coefs', [])) > 0]
    colors = {'ols': '#4C72B0', 'ridge': '#DD8452', 'lasso': '#55A868'}

    coef_df = (pd.DataFrame({k: results[k]['coefs'] for k in model_keys}, index=features)
               .fillna(0)
               .pipe(lambda d: d.reindex(d.abs().max(axis=1).sort_values(ascending=False).index)))

    n     = len(coef_df)
    x     = np.arange(n)
    width = 0.8 / len(model_keys)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n * 0.35 + 1)))

    for i, name in enumerate(model_keys):
        axes[0].barh(x + i * width, coef_df[name].values, width,
                     label=name.upper(), color=colors.get(name, f'C{i}'))

    axes[0].set_yticks(x + width * (len(model_keys) - 1) / 2)
    axes[0].set_yticklabels(coef_df.index, fontsize=8)
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel('Coefficient (standardized features)')
    axes[0].set_title('Coefficient Comparison')
    axes[0].legend()
    axes[0].invert_yaxis()

    cv_keys  = [k for k in results if 'cv_r2' in results[k]]
    cv_data  = [results[k]['cv_r2'] for k in cv_keys]
    bp = axes[1].boxplot(cv_data, labels=[k.upper() for k in cv_keys], patch_artist=True)
    for patch, key in zip(bp['boxes'], cv_keys):
        patch.set_facecolor(colors.get(key, 'lightblue'))
    axes[1].axhline(0, color='red', linewidth=0.8, linestyle='--', label='R2=0 (trivial)')
    axes[1].set_ylabel('R2')
    axes[1].set_title(f'{len(cv_data[0])}-Fold CV R2 Distribution')
    axes[1].legend(fontsize=8)

    plt.suptitle(f'Regression Comparison  |  label={label}', fontsize=12)
    plt.tight_layout()
    plt.show()


def _plot_ols_diagnostics(ols_model, cooks_d, y_train):
    """Residuals vs Fitted, Q-Q, Cook's Distance."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].scatter(ols_model.fittedvalues, ols_model.resid, alpha=0.4, s=20)
    axes[0].axhline(0, color='red', linewidth=1)
    axes[0].set_xlabel('Fitted values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted')

    sm.qqplot(ols_model.resid, line='s', ax=axes[1], alpha=0.4)
    axes[1].set_title('Normal Q-Q')

    if cooks_d is not None:
        threshold = 4 / len(y_train)
        axes[2].stem(range(len(cooks_d)), cooks_d, markerfmt=',', linefmt='C0-', basefmt='k-')
        axes[2].axhline(threshold, color='red', linestyle='--', linewidth=1,
                        label=f'4/n = {threshold:.4f}')
        axes[2].set_xlabel('Observation')
        axes[2].set_ylabel("Cook's Distance")
        axes[2].set_title("Cook's Distance")
        axes[2].legend(fontsize=8)

    plt.suptitle('OLS Diagnostics', fontsize=12)
    plt.tight_layout()
    plt.show()


def _plot_alpha_search(method, alphas, X_train, y_train, kf, best_alpha):
    """Plot CV R2 vs alpha with confidence band and best-alpha marker."""
    alphas = np.sort(alphas)
    rows = []
    for a in alphas:
        if method == 'ridge':
            m = Ridge(alpha=a)
        else:
            m = Lasso(alpha=a, max_iter=10000)
        scores = cross_val_score(m, X_train, y_train, cv=kf, scoring='r2')
        rows.append({'alpha': a, 'mean_r2': scores.mean(), 'std_r2': scores.std()})

    res = pd.DataFrame(rows)

    plt.figure(figsize=(8, 4))
    plt.plot(res['alpha'], res['mean_r2'], marker='o', markersize=3)
    plt.fill_between(res['alpha'],
                     res['mean_r2'] - res['std_r2'],
                     res['mean_r2'] + res['std_r2'], alpha=0.2)
    plt.axvline(best_alpha, color='red', linestyle='--',
                label=f'Best alpha = {best_alpha:.4f}')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Mean CV R2')
    plt.title(f'{method.upper()} — CV R2 vs Alpha')
    plt.legend()
    plt.tight_layout()
    plt.show()
