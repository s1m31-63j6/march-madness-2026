"""One-time pre-compute: run every model through the bracket, freeze JSON.

Run from the repo root:

    cd march_madness
    python hf_space/prepare_data.py

Writes:
    hf_space/web/public/data/manifest.json
    hf_space/web/public/data/brackets/<slug>.json        (one per model)
    hf_space/web/public/data/retrospective.json
    hf_space/web/public/data/hindsight.json
    hf_space/web/public/docs/<slug>.md                   (README, report, etc.)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.db import TeamDB
from engine.bracket import Bracket, ROUND_LABELS
from engine.actuals import load_actuals
from engine.evaluation import (
    accuracy_table,
    games_graded_count,
    merge_tournament_results_into_bracket_dfs,
    overall_pick_accuracy,
    spread_accuracy_table,
    truth_dataframe_from_tournament_csv,
)
from engine.models.seeding import SeedingModel
from engine.models.advanced_metrics import AdvancedMetricsModel
from engine.models.greg_v1 import GregV1Model
from engine.models.probability import (
    SampledProbabilityModel,
    ThresholdProbabilityModel,
    MonteCarloConsensusModel,
)

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "hf_space" / "web" / "public" / "data"
DOCS_OUT_DIR = REPO_ROOT / "hf_space" / "web" / "public" / "docs"

REGION_NAMES = {"W": "East", "X": "South", "Y": "Midwest", "Z": "West"}

MODEL_COLORS = {
    "Seeding Only": "#6b7280",
    "Comparative Metrics": "#1f77b4",
    "Greg_v1": "#d97706",
    "Lean GB (Sampled)": "#7c3aed",
    "Lean GB (Tiered Threshold)": "#a855f7",
    "Lean GB (MC Consensus)": "#6366f1",
    "Animal Kingdom": "#dc2626",
    "Vegas Odds": "#059669",
    "Hindsight (Overfit)": "#f59e0b",
}

MODEL_BLURBS = {
    "Seeding Only": "Higher seed wins. Ties broken by regular-season win %.",
    "Comparative Metrics": "Regression on Barttorvik efficiency diffs, coach tenure, and strength of schedule.",
    "Greg_v1": "Tuned Ridge regression, 28 features, recency-weighted samples from 2011–2025 tournaments.",
    "Lean GB (Sampled)": "Gradient-boosted classifier. Each game's winner drawn stochastically from its win probability.",
    "Lean GB (Tiered Threshold)": "Same classifier, round-aware deterministic thresholds for picking upsets.",
    "Lean GB (MC Consensus)": "10,000 full-bracket Monte Carlo sims; the per-slot majority winner advances.",
    "Animal Kingdom": "Claude judges a hypothetical fight between the two mascots. Pure entertainment.",
    "Vegas Odds": "Real sportsbook lines when available; Claude estimates them when not.",
    "Hindsight (Overfit)": "Trained on the 67 tournament games themselves — shows which variables explained THIS year.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    import re
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def _to_py(v):
    """Make a value JSON-safe (convert numpy types, NaN → None)."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if np.isnan(f) else f
    if isinstance(v, float):
        return None if np.isnan(v) else v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.ndarray,)):
        return [_to_py(x) for x in v.tolist()]
    return v


def _row_to_game(row: pd.Series) -> dict[str, Any]:
    """Flatten a bracket dataframe row to a JSON-ready game dict."""
    return {
        "slot_id": row["slot_id"],
        "round_num": int(row["round_num"]),
        "round_label": row["round_label"],
        "region": row["region"],
        "strong_team_id": _to_py(row.get("strong_team_id")),
        "strong_team": row.get("strong_team") or None,
        "strong_seed": _to_py(row.get("strong_seed")),
        "weak_team_id": _to_py(row.get("weak_team_id")),
        "weak_team": row.get("weak_team") or None,
        "weak_seed": _to_py(row.get("weak_seed")),
        "pred_winner_id": _to_py(row.get("pred_winner_id")),
        "pred_winner": row.get("pred_winner") or None,
        "strong_pred_score": _to_py(row.get("strong_pred_score")),
        "weak_pred_score": _to_py(row.get("weak_pred_score")),
        "confidence": _to_py(row.get("confidence")),
        "result_winner_id": _to_py(row.get("result_winner_id")),
        "result_winner": row.get("result_winner") or None,
        "result_strong_score": _to_py(row.get("result_strong_score")),
        "result_weak_score": _to_py(row.get("result_weak_score")),
        "is_actual": bool(row.get("is_actual", False)),
    }


def _normalize_importances(pairs: list[tuple[str, float]]) -> list[dict]:
    """Normalize (feature, score) pairs to sum to 1.0 and sort descending."""
    pairs = [(f, max(0.0, float(s))) for f, s in pairs]
    total = sum(s for _, s in pairs) or 1.0
    out = [{"feature": f, "importance": s / total} for f, s in pairs]
    out.sort(key=lambda d: -d["importance"])
    return out


def collect_comparison_importances() -> dict[str, dict]:
    """Extract feature importances + per-model training stats for side-by-side UI."""
    import joblib
    import json

    models_dir = DATA_DIR / "models"
    out: dict[str, dict] = {}
    retrain_stats = globals().get("_RETRAIN_STATS", {})

    # Greg_v1 — Ridge regression. Use |coef| as importance proxy.
    try:
        greg_model = joblib.load(models_dir / "greg_v1_margin_model.pkl")
        greg_cols = joblib.load(models_dir / "greg_v1_feature_cols.pkl")
        coefs = np.abs(getattr(greg_model, "coef_", np.zeros(len(greg_cols))))
        greg_stats = {}
        summary_path = models_dir / "greg_v1_summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            # Greg_v1 was trained on 924 tournament games (2011-2025) with recency
            # weighting (recent seasons weighted 4x). The `holdout_*` metrics in the
            # summary are from a cross-validation holdout inside that dataset, not a
            # year-based split — 2024 and 2025 games are in training, too.
            greg_stats = {
                "games": 924,
                "win_acc": s.get("holdout_win_acc"),
                "margin_mae": s.get("holdout_margin_mae"),
                "total_mae": s.get("holdout_total_mae"),
                "data_note": "2011-2025 tournament games, recency-weighted (recent seasons 4x). Metrics shown are cross-validation holdout.",
            }
        out["Greg_v1"] = {
            "feature_cols": list(greg_cols),
            "importances": _normalize_importances(list(zip(greg_cols, coefs))),
            "method": "|β| from Ridge regression (normalized)",
            "stats": greg_stats,
        }
    except Exception as exc:
        print(f"    skipped Greg_v1 importances: {exc}")

    # Comparative Metrics — retrained GB regressor for score margin.
    try:
        cm_model = joblib.load(models_dir / "score_margin_model.pkl")
        cm_cols = joblib.load(models_dir / "feature_cols.pkl")
        imp = getattr(cm_model, "feature_importances_", np.zeros(len(cm_cols)))
        out["Comparative Metrics"] = {
            "feature_cols": list(cm_cols),
            "importances": _normalize_importances(list(zip(cm_cols, imp))),
            "method": "gradient-boosting gain on score-margin (normalized)",
            "stats": retrain_stats.get("comparative_metrics", {}),
        }
    except Exception as exc:
        print(f"    skipped Comparative Metrics importances: {exc}")

    # Lean GB — classifier on game_result.
    try:
        prob_model = joblib.load(models_dir / "prob_model.pkl")
        prob_cols = joblib.load(models_dir / "prob_feature_cols.pkl")
        imp = getattr(prob_model, "feature_importances_", np.zeros(len(prob_cols)))
        out["Lean GB"] = {
            "feature_cols": list(prob_cols),
            "importances": _normalize_importances(list(zip(prob_cols, imp))),
            "method": "gradient-boosting gain on win classifier (normalized)",
            "stats": retrain_stats.get("lean_gb", {}),
        }
    except Exception as exc:
        print(f"    skipped Lean GB importances: {exc}")

    return out


def retrain_pickled_models() -> dict:
    """Retrain the 3 regressor/classifier pickles with the current sklearn.

    Returns a dict of per-artifact training stats so the UI can show the same
    metrics (games, win_acc, margin_mae, total_mae) for every comparison model.
    """
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    models_dir = DATA_DIR / "models"
    matchups = pd.read_csv(DATA_DIR / "matchup_dataset.csv", low_memory=False)

    # Exclude any 2026 rows that might have leaked in (training data only).
    matchups = matchups[matchups["season"] < 2026].copy()

    score_cols = list(dict.fromkeys(joblib.load(models_dir / "feature_cols.pkl")))  # dedup
    prob_cols = joblib.load(models_dir / "prob_feature_cols.pkl")

    # Save the deduplicated score_cols back (fixes the duplicate seed_diff in original pkl)
    joblib.dump(score_cols, models_dir / "feature_cols.pkl")

    # ── Score regressors ─────────────────────────────────────────
    score_df = matchups.dropna(subset=score_cols + ["score_margin", "total_points"])
    Xs = score_df[score_cols].fillna(0.0).to_numpy()
    y_margin = score_df["score_margin"].to_numpy()
    y_total = score_df["total_points"].to_numpy()

    margin_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
    ).fit(Xs, y_margin)
    total_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
    ).fit(Xs, y_total)
    joblib.dump(margin_model, models_dir / "score_margin_model.pkl")
    joblib.dump(total_model, models_dir / "total_points_model.pkl")

    margin_pred = margin_model.predict(Xs)
    total_pred = total_model.predict(Xs)
    cm_margin_mae = float(np.mean(np.abs(margin_pred - y_margin)))
    cm_total_mae = float(np.mean(np.abs(total_pred - y_total)))
    cm_win_acc = float(((margin_pred >= 0) == (y_margin >= 0)).mean())
    print(f"    score regressors: trained on {len(score_df):,} games, "
          f"margin train-MAE={cm_margin_mae:.2f}")

    # ── Probability classifier ───────────────────────────────────
    prob_df = matchups.dropna(subset=prob_cols + ["game_result"])
    Xp = prob_df[prob_cols].fillna(0.0).to_numpy()
    yp = prob_df["game_result"].astype(int).to_numpy()
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
    ).fit(Xp, yp)
    joblib.dump(clf, models_dir / "prob_model.pkl")
    lgb_train_acc = float((clf.predict(Xp) == yp).mean())
    print(f"    prob classifier:  trained on {len(prob_df):,} games, "
          f"train-acc={lgb_train_acc:.1%}")

    return {
        "comparative_metrics": {
            "games": len(score_df),
            "win_acc": cm_win_acc,
            "margin_mae": cm_margin_mae,
            "total_mae": cm_total_mae,
            "data_note": "Training set (2008-2025 tournament games, excludes 2026).",
        },
        "lean_gb": {
            "games": len(prob_df),
            "win_acc": lgb_train_acc,
            "margin_mae": None,
            "total_mae": None,
            "data_note": "Classifier — predicts P(win), not scores. Training set.",
        },
    }


def build_models(db: TeamDB) -> dict:
    """Instantiate every model that can be built from this environment."""
    models: dict = {}

    models["Seeding Only"] = SeedingModel(db)
    try:
        models["Comparative Metrics"] = AdvancedMetricsModel(str(DATA_DIR / "models"))
    except Exception as exc:
        print(f"  skipped Comparative Metrics: {exc}")
    try:
        models["Greg_v1"] = GregV1Model(str(DATA_DIR / "models"))
    except Exception as exc:
        print(f"  skipped Greg_v1: {exc}")
    try:
        models["Lean GB (Sampled)"] = SampledProbabilityModel(
            str(DATA_DIR / "models"), random_seed=12345
        )
        models["Lean GB (Tiered Threshold)"] = ThresholdProbabilityModel(
            str(DATA_DIR / "models")
        )
        models["Lean GB (MC Consensus)"] = MonteCarloConsensusModel(
            str(DATA_DIR / "models"), n_sims=10_000, random_seed=12345
        )
    except Exception as exc:
        print(f"  skipped Probability models: {exc}")

    has_api = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_api:
        try:
            from engine.models.animal_kingdom import AnimalKingdomModel
            models["Animal Kingdom"] = AnimalKingdomModel()
        except Exception as exc:
            print(f"  skipped Animal Kingdom: {exc}")
        try:
            from engine.models.vegas_odds import VegasOddsModel
            models["Vegas Odds"] = VegasOddsModel(lines_path=DATA_DIR / "vegas_lines.csv")
        except Exception as exc:
            print(f"  skipped Vegas Odds: {exc}")
    else:
        print("  ANTHROPIC_API_KEY not set — skipping Animal Kingdom & Vegas Odds")

    return models


# ---------------------------------------------------------------------------
# Hindsight model (overfit on 2026 tournament games)
# ---------------------------------------------------------------------------

def train_hindsight(db: TeamDB) -> dict[str, Any]:
    """Train a deliberately overfit regressor on the 67 tournament games.

    Uses the same engine features (`compute_matchup_features`) so importances are
    directly comparable to Comparative Metrics.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    seeds_df = pd.read_csv(DATA_DIR / "kaggle" / "MNCAATourneySeeds.csv")
    slots_df = pd.read_csv(DATA_DIR / "kaggle" / "MNCAATourneySlots.csv")
    b = Bracket(seeds_df, slots_df, season=2026)
    load_actuals(DATA_DIR / "actuals.csv", b, db)
    truth_df = b.to_dataframe(db)

    train_rows = []
    for _, r in truth_df.iterrows():
        if not r["is_actual"]:
            continue
        if pd.isna(r["strong_team_id"]) or pd.isna(r["weak_team_id"]):
            continue
        # engine convention: team_a = favored (lower seed) by ID ordering via Bracket._order_by_seed
        a_id, b_id = int(r["strong_team_id"]), int(r["weak_team_id"])
        sa, sb = db.get_seed(a_id), db.get_seed(b_id)
        if not np.isnan(sa) and not np.isnan(sb) and sa > sb:
            a_id, b_id = b_id, a_id  # keep "team_a = lower seed" convention
        feats = db.compute_matchup_features(a_id, b_id, round_num=int(r["round_num"]))

        # Actuals: strong_score / weak_score are relative to strong/weak, not a/b.
        # Convert to (team_a, team_b) perspective.
        if pd.isna(r.get("actual_strong_score")) or pd.isna(r.get("actual_weak_score")):
            continue
        strong_id = int(r["strong_team_id"])
        if a_id == strong_id:
            score_a = float(r["actual_strong_score"])
            score_b = float(r["actual_weak_score"])
        else:
            score_a = float(r["actual_weak_score"])
            score_b = float(r["actual_strong_score"])

        row = dict(feats)
        row["__margin"] = score_a - score_b
        row["__total"] = score_a + score_b
        row["__round_num"] = int(r["round_num"])
        row["__slot_id"] = r["slot_id"]
        row["__team_a_id"] = a_id
        row["__team_b_id"] = b_id
        train_rows.append(row)

    if not train_rows:
        raise RuntimeError("No training rows for Hindsight — actuals not loaded?")

    df = pd.DataFrame(train_rows)
    feature_cols = [c for c in df.columns if not c.startswith("__")]
    X = df[feature_cols].fillna(0.0).to_numpy()
    y_margin = df["__margin"].to_numpy()
    y_total = df["__total"].to_numpy()

    # Deliberately overfit: deep trees, no pruning, fit hard to training data.
    margin_model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=1,
        learning_rate=0.05,
        random_state=7,
    ).fit(X, y_margin)
    total_model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=1,
        learning_rate=0.05,
        random_state=7,
    ).fit(X, y_total)

    # Rank features by importance on margin (the winner-deciding target).
    importances = sorted(
        [(c, float(i)) for c, i in zip(feature_cols, margin_model.feature_importances_)],
        key=lambda x: -x[1],
    )

    # Training-set predictions (these will be near-perfect by construction).
    pred_margin = margin_model.predict(X)
    pred_total = total_model.predict(X)
    train_wins = int(((pred_margin >= 0) == (y_margin >= 0)).sum())

    return {
        "feature_cols": feature_cols,
        "margin_model": margin_model,
        "total_model": total_model,
        "train_games": len(df),
        "train_win_acc": train_wins / len(df),
        "importances": importances,
        "train_margin_mae": float(np.mean(np.abs(pred_margin - y_margin))),
        "train_total_mae": float(np.mean(np.abs(pred_total - y_total))),
    }


class HindsightModel:
    """PredictionModel that calls the overfit regressor above."""

    name = "Hindsight (Overfit)"

    def __init__(self, trained: dict):
        self._trained = trained

    def predict(self, team_a_id, team_b_id, db, round_num=1, slot_id=None):
        from engine.models.base import Prediction

        feats = db.compute_matchup_features(team_a_id, team_b_id, round_num=round_num)
        vec = np.array(
            [[feats.get(c, 0.0) for c in self._trained["feature_cols"]]]
        )
        margin = float(self._trained["margin_model"].predict(vec)[0])
        total = float(self._trained["total_model"].predict(vec)[0])

        score_a = max((total + margin) / 2, 40.0)
        score_b = max((total - margin) / 2, 40.0)
        winner = team_a_id if margin >= 0 else team_b_id
        conf = min(abs(margin) / 30.0, 1.0) * 0.5 + 0.5
        return Prediction(
            team_a_score=round(score_a, 1),
            team_b_score=round(score_b, 1),
            winner_id=winner,
            confidence=round(conf, 3),
        )


# ---------------------------------------------------------------------------
# Bracket simulation + JSON output
# ---------------------------------------------------------------------------

def simulate_all(db, seeds_df, slots_df, models: dict) -> dict[str, pd.DataFrame]:
    """Run each model through the bracket WITHOUT actuals (pure pre-tournament prediction)."""
    bracket_dfs: dict[str, pd.DataFrame] = {}
    for name, model in models.items():
        print(f"  simulating {name}…")
        try:
            b = Bracket(seeds_df, slots_df, season=2026)
            b.simulate(model, db)
            bracket_dfs[name] = b.to_dataframe(db)
        except Exception as exc:
            print(f"    failed: {exc}")
    return bracket_dfs


def write_brackets(bracket_dfs: dict[str, pd.DataFrame], db: TeamDB):
    """Write one JSON per model."""
    out = OUT_DIR / "brackets"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in bracket_dfs.items():
        games = [_row_to_game(r) for _, r in df.iterrows()]
        payload = {
            "model": name,
            "slug": _slug(name),
            "champion_id": None,
            "champion": None,
            "games": games,
        }
        # Champion = winner of the sole round-6 slot
        ch = df[df["round_num"] == 6]
        if not ch.empty:
            row = ch.iloc[0]
            payload["champion_id"] = _to_py(row.get("pred_winner_id"))
            payload["champion"] = row.get("pred_winner") or None
        with (out / f"{_slug(name)}.json").open("w") as f:
            json.dump(payload, f)
        print(f"    wrote brackets/{_slug(name)}.json  (champion: {payload['champion']})")


# ---------------------------------------------------------------------------
# Retrospective stats
# ---------------------------------------------------------------------------

def build_retrospective(bracket_dfs: dict[str, pd.DataFrame], truth_df: pd.DataFrame, db: TeamDB) -> dict:
    """Accuracy tables, round-by-round, upsets, champion picks."""
    acc_df = accuracy_table(bracket_dfs)
    spread_df = spread_accuracy_table(bracket_dfs)

    # Headline accuracy per model (overall + per round)
    summary_rows = []
    for name, df in bracket_dfs.items():
        acc = overall_pick_accuracy(df)
        n = games_graded_count(df)
        ch_row = df[df["round_num"] == 6]
        champ_correct = False
        champ_name = None
        if not ch_row.empty:
            r = ch_row.iloc[0]
            champ_name = r.get("pred_winner") or None
            if pd.notna(r.get("pred_winner_id")) and pd.notna(r.get("result_winner_id")):
                champ_correct = int(r["pred_winner_id"]) == int(r["result_winner_id"])
        summary_rows.append({
            "model": name,
            "games_graded": int(n),
            "pick_accuracy": _to_py(acc),
            "predicted_champion": champ_name,
            "champion_correct": bool(champ_correct),
            "color": MODEL_COLORS.get(name, "#888"),
            "blurb": MODEL_BLURBS.get(name, ""),
        })

    # Per-round accuracy: flat table
    round_rows = []
    ref_df = next(iter(bracket_dfs.values()))
    rounds = sorted(ref_df["round_num"].unique())
    for rnd in rounds:
        label = ROUND_LABELS.get(int(rnd), f"R{rnd}")
        row = {"round_num": int(rnd), "round_label": label}
        for name, df in bracket_dfs.items():
            slc = df[df["round_num"] == rnd]
            if "result_winner_id" in slc.columns:
                mask = slc["result_winner_id"].notna() & slc["pred_winner_id"].notna()
                if mask.any():
                    correct = (slc.loc[mask, "pred_winner_id"] == slc.loc[mask, "result_winner_id"]).sum()
                    row[name] = float(correct / mask.sum())
                    continue
            row[name] = None
        round_rows.append(row)

    # Upsets: games where the winner had a HIGHER numeric seed than the loser.
    upsets = []
    if not truth_df.empty:
        for _, r in truth_df.iterrows():
            if not r["is_actual"]:
                continue
            rw_id = r.get("actual_winner_id") or r.get("winner_id")
            if pd.isna(rw_id):
                continue
            ss = r.get("strong_seed")
            ws = r.get("weak_seed")
            if pd.isna(ss) or pd.isna(ws):
                continue
            # The "strong" side is always the lower seed in the engine. So an upset
            # is when the winner is the weak side.
            if int(rw_id) == int(r.get("weak_team_id", -1)):
                # Check which models called it
                called_by = []
                for name, df in bracket_dfs.items():
                    m = df[df["slot_id"] == r["slot_id"]]
                    if not m.empty:
                        pm = m.iloc[0]
                        if pd.notna(pm.get("pred_winner_id")) and int(pm["pred_winner_id"]) == int(rw_id):
                            called_by.append(name)
                upsets.append({
                    "slot_id": r["slot_id"],
                    "round_num": int(r["round_num"]),
                    "round_label": ROUND_LABELS.get(int(r["round_num"]), ""),
                    "winner": r.get("actual_winner") or "",
                    "winner_seed": int(ws),
                    "loser": r.get("strong_team") or "",
                    "loser_seed": int(ss),
                    "winner_score": _to_py(r.get("actual_weak_score")),
                    "loser_score": _to_py(r.get("actual_strong_score")),
                    "seed_gap": int(ws) - int(ss),
                    "called_by": called_by,
                })
        upsets.sort(key=lambda u: (-u["seed_gap"], u["round_num"]))

    # Acc tables to JSON-safe rows
    def _df_to_rows(df: pd.DataFrame) -> list[dict]:
        out = []
        for _, row in df.iterrows():
            out.append({k: _to_py(v) for k, v in row.to_dict().items()})
        return out

    return {
        "summary": summary_rows,
        "per_round": round_rows,
        "accuracy_table": _df_to_rows(acc_df),
        "spread_table": _df_to_rows(spread_df),
        "upsets": upsets,
    }


# ---------------------------------------------------------------------------
# Docs conversion (.docx → .md)
# ---------------------------------------------------------------------------

def docx_to_markdown(docx_path: Path, media_dir: Path | None = None) -> str | None:
    """Convert .docx to markdown. Prefers pandoc; falls back to python-docx (paragraph dump).

    When ``media_dir`` is provided, embedded images are extracted there and
    the generated markdown references ``media/<file>`` paths relative to the
    web root.
    """
    import shutil
    import subprocess

    if shutil.which("pandoc"):
        try:
            args = ["pandoc", str(docx_path), "-t", "gfm", "--wrap=none"]
            if media_dir is not None:
                media_dir.mkdir(parents=True, exist_ok=True)
                args += ["--extract-media", str(media_dir)]
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as exc:
            print(f"  pandoc failed: {exc}")

    try:
        from docx import Document  # python-docx
        doc = Document(str(docx_path))
        lines = []
        for p in doc.paragraphs:
            txt = p.text.strip()
            if not txt:
                lines.append("")
                continue
            sty = (p.style.name or "").lower()
            if "heading 1" in sty:
                lines.append(f"# {txt}")
            elif "heading 2" in sty:
                lines.append(f"## {txt}")
            elif "heading 3" in sty:
                lines.append(f"### {txt}")
            elif "heading 4" in sty:
                lines.append(f"#### {txt}")
            else:
                lines.append(txt)
        return "\n\n".join(lines)
    except Exception as exc:
        print(f"  python-docx fallback failed: {exc}")
        return None


def build_docs():
    DOCS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # README.md
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        (DOCS_OUT_DIR / "readme.md").write_text(readme.read_text())
        print("    wrote docs/readme.md")

    # UPDATING_ACTUALS.md
    ua = REPO_ROOT / "UPDATING_ACTUALS.md"
    if ua.exists():
        (DOCS_OUT_DIR / "updating-actuals.md").write_text(ua.read_text())
        print("    wrote docs/updating-actuals.md")

    # March Madness Report.docx → .md (extract images to web/public/media/)
    docx_path = REPO_ROOT / "March Madness Report.docx"
    web_root = REPO_ROOT / "hf_space" / "web" / "public"
    if docx_path.exists():
        md = docx_to_markdown(docx_path, media_dir=web_root)
        if md:
            # Rewrite absolute media paths to root-relative web paths
            md = md.replace(str(web_root) + "/", "/")
            md_parallel = REPO_ROOT / "March Madness Report.md"
            md_parallel.write_text(md)  # save parallel to .docx at repo root
            (DOCS_OUT_DIR / "report.md").write_text(md)
            print("    wrote docs/report.md and parallel March Madness Report.md")
        else:
            print("    !! could not convert March Madness Report.docx")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _use_pre_tournament_snapshot() -> Path | None:
    """Swap the frozen pre-tournament season file into place.

    Without this, ``TeamDB`` loads whatever is currently in ``data/season_2026.csv``,
    which at some point got refreshed with post-tournament Barttorvik stats (record,
    adj_em, wab all reflect tournament wins). Using post-tournament features to
    "predict" tournament outcomes is circular — each tournament win already inflates
    the feature values we're feeding the model.

    Returns a path to the working-copy backup so ``main`` can restore it after.
    """
    import shutil

    pre = DATA_DIR / "season_2026_pre_tournament.csv"
    live = DATA_DIR / "season_2026.csv"
    if not pre.exists() or not live.exists():
        print("  no pre-tournament snapshot found; using live season file as-is")
        return None

    backup = DATA_DIR / ".season_2026_working_copy.csv"
    shutil.copy(live, backup)
    shutil.copy(pre, live)
    print(f"  swapped in pre-tournament snapshot; working copy → {backup.name}")
    return backup


def _restore_working_copy(backup: Path | None) -> None:
    if backup is None or not backup.exists():
        return
    import shutil

    shutil.copy(backup, DATA_DIR / "season_2026.csv")
    backup.unlink()


def main():
    print("Preparing March Madness 2026 retrospective data …")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[pre] freezing pre-tournament feature snapshot")
    backup = _use_pre_tournament_snapshot()

    try:
        _main_body()
    finally:
        _restore_working_copy(backup)


def _main_body():
    print("[0/6] retraining pickled models with current sklearn")
    retrain_stats = retrain_pickled_models()
    globals()["_RETRAIN_STATS"] = retrain_stats

    print("[1/6] loading engine & seeds")
    db = TeamDB(str(DATA_DIR))
    seeds_df = pd.read_csv(DATA_DIR / "kaggle" / "MNCAATourneySeeds.csv")
    db.load_seeds(seeds_df, season=2026)
    slots_df = pd.read_csv(DATA_DIR / "kaggle" / "MNCAATourneySlots.csv")

    print("[2/6] training Hindsight model on 67 tournament games")
    actuals_path = DATA_DIR / "actuals.csv"
    hindsight = train_hindsight(db)
    print(
        f"    trained on {hindsight['train_games']} games, "
        f"train_win_acc={hindsight['train_win_acc']:.1%}, "
        f"train_margin_mae={hindsight['train_margin_mae']:.2f}"
    )

    print("[3/6] building models & simulating pre-tournament brackets")
    models = build_models(db)
    models["Hindsight (Overfit)"] = HindsightModel(hindsight)
    bracket_dfs = simulate_all(db, seeds_df, slots_df, models)

    print("[4/6] merging actual tournament results for comparison")
    truth_df = truth_dataframe_from_tournament_csv(actuals_path, seeds_df, slots_df, db)
    bracket_dfs = merge_tournament_results_into_bracket_dfs(bracket_dfs, truth_df)

    write_brackets(bracket_dfs, db)

    print("[5/6] computing retrospective stats")
    retro = build_retrospective(bracket_dfs, truth_df, db)
    (OUT_DIR / "retrospective.json").write_text(json.dumps(retro))

    # Hindsight JSON (feature importances + comparison to pre-tournament models)
    greg_imp_path = DATA_DIR / "models" / "greg_v1_summary.json"
    greg_imp = None
    if greg_imp_path.exists():
        greg_imp = json.loads(greg_imp_path.read_text()).get("features", [])

    # Narratives — basketball-framed "what this model is thinking"
    MODEL_NARRATIVES = {
        "Hindsight": {
            "headline": "What the 2026 tournament actually rewarded",
            "body": (
                "Because Hindsight is trained on the 67 tournament games themselves, its importance chart is a read-out of the variables that separated this year's winners from losers. "
                "<strong>seed_disagreement</strong> tops the list — the gap between where the committee seeded a team and where their underlying efficiency said they belonged. "
                "In basketball terms: teams the bracket committee under-rated won games they were statistically supposed to win but were slotted as underdogs. "
                "<strong>wab_diff</strong> (Wins Above Bubble) and <strong>adj_em_diff</strong> round out the top three — résumé depth and points-per-possession dominance."
            ),
        },
        "Greg_v1": {
            "headline": "Greg_v1 is a résumé model",
            "body": (
                "Greg_v1 spreads 28 features across a Ridge regression and leans hardest on <strong>strength-of-schedule</strong> signals — overall SOS, conference SOS, and how teams fared against top-Barthag opponents. "
                "Its basketball theory: tournaments reward teams forged by tough competition. Who you beat all winter matters more than how prettily you beat them. "
                "That worked well for this year (86.6% pick accuracy, Michigan as champion), but notice how little weight it placed on <strong>seed_disagreement</strong> — the one feature Hindsight says mattered most. "
                "Greg_v1 trusts the committee more than the bracket rewarded this year."
            ),
        },
        "Comparative Metrics": {
            "headline": "Comparative Metrics picks the better team, full stop",
            "body": (
                "Over half of this model's weight goes to a single number: <strong>adj_em_diff</strong> (adjusted efficiency margin). "
                "It's asking one question: which team has been flat-out better per possession all season? Seed, coach tenure, and schedule quality get cameo roles. "
                "In basketball terms, it's the Ken Pomeroy / Barttorvik view: don't overthink it, pick the team with the bigger scoring gap. "
                "Respectable against the field, but notice: where Hindsight sees nuance (seed_disagreement, tempo), this model sees one big blunt instrument."
            ),
        },
        "Lean GB": {
            "headline": "Lean GB hunts for mispriced teams",
            "body": (
                "The lean gradient-boosted classifier's top feature is <strong>seed_disagreement</strong> — the same one Hindsight says this tournament rewarded. "
                "In basketball terms: Lean GB is looking for teams the committee got wrong. A 5-seed whose efficiency numbers belong to a 2-seed is a green light; a 2-seed whose numbers look like a 6-seed is a trap. "
                "It's a contrarian approach — find value where the bracket structure and underlying stats disagree. "
                "The Monte Carlo Consensus and Tiered Threshold variants run this same classifier through different bracket strategies (stochastic draw, thresholded picks, 10k-sim majority)."
            ),
        },
    }

    # Normalize Hindsight importances the same way so bars are comparable
    hindsight_importances_norm = _normalize_importances(hindsight["importances"])
    comparison = collect_comparison_importances()

    # Attach narrative per comparison model
    for name, block in comparison.items():
        block["narrative"] = MODEL_NARRATIVES.get(name, {"headline": name, "body": ""})

    hindsight_payload = {
        "model": "Hindsight (Overfit)",
        "train_games": hindsight["train_games"],
        "train_win_acc": hindsight["train_win_acc"],
        "train_margin_mae": hindsight["train_margin_mae"],
        "train_total_mae": hindsight["train_total_mae"],
        "importances": hindsight_importances_norm,
        "feature_cols": hindsight["feature_cols"],
        "method": "gradient-boosting gain on 2026 tournament games (normalized)",
        "narrative": MODEL_NARRATIVES["Hindsight"],
        "comparison_models": comparison,
        "pretournament_feature_order": greg_imp,
    }
    (OUT_DIR / "hindsight.json").write_text(json.dumps(hindsight_payload))

    print("[6/6] manifest + docs")
    manifest = {
        "tournament": "2026 NCAA Men's Basketball Tournament",
        "champion": None,
        "runner_up": None,
        "models": [],
    }
    # Champion from actuals
    if not truth_df.empty:
        ch = truth_df[truth_df["round_num"] == 6]
        if not ch.empty:
            row = ch.iloc[0]
            manifest["champion"] = row.get("actual_winner")
            # Identify runner-up (loser of championship)
            loser_id = (row["strong_team_id"] if int(row["actual_winner_id"]) == int(row["weak_team_id"])
                        else row["weak_team_id"])
            if pd.notna(loser_id):
                manifest["runner_up"] = db.get_team_name(int(loser_id))
    for name, df in bracket_dfs.items():
        ch_row = df[df["round_num"] == 6]
        predicted_ch = ch_row.iloc[0]["pred_winner"] if not ch_row.empty else None
        manifest["models"].append({
            "name": name,
            "slug": _slug(name),
            "blurb": MODEL_BLURBS.get(name, ""),
            "color": MODEL_COLORS.get(name, "#888"),
            "predicted_champion": predicted_ch,
        })
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest))

    build_docs()

    print("\nDone.  Output in:", OUT_DIR)


if __name__ == "__main__":
    main()
