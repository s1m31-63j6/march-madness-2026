"""Greg_v1 model — tuned Ridge regression with 28 features.

Trained in model_lab.ipynb on 924 tournament games (2011-2025) with
recency-weighted samples and permutation-importance feature selection.
"""

from pathlib import Path

import joblib
import numpy as np

from engine.db import TeamDB
from engine.models.base import Prediction, PredictionModel

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Maps feature name (as trained) → source column key in TeamDB.get_team() dicts.
# "diff" features use (team_a_value - team_b_value).
# "raw" features use one team's value directly.
_DIFF_FEATURES: dict[str, str] = {
    "adj_em_diff": "adj_em",
    "seed_diff": "__seed__",
    "wab_diff": "wab",
    "conf_winpct_diff": "Conf Win%",
    "sos_diff": "sos",
    "qual_games_diff": "Qual Games",
    "adj_o_diff": "adj_o",
    "qual_barthag_diff": "Qual Barthag",
    "qual_o_diff": "Qual O",
    "coach_tourn_wins_diff": "coach_tourn_wins",
    "adj_d_diff": "adj_d",
    "ncsos_diff": "ncsos",
    "consos_diff": "consos",
    "qual_d_diff": "Qual D",
    "coach_appearances_diff": "coach_appearances",
    "elite_sos_diff": "elite SOS",
    "coach_final_fours_diff": "coach_final_fours",
    "con_adj_oe_diff": "Con Adj OE",
    "con_adj_de_diff": "Con Adj DE",
}

_RAW_FEATURES: dict[str, tuple[str, str]] = {
    # feature_name → (team_prefix "a"/"b", source column)
    "b_adj_em": ("b", "adj_em"),
    "a_adj_em": ("a", "adj_em"),
    "a_adj_o": ("a", "adj_o"),
    "b_adj_o": ("b", "adj_o"),
    "b_barthag": ("b", "barthag"),
    "a_adj_d": ("a", "adj_d"),
}

_SEED_FEATURES = ["seed_product", "seed_sum"]
_OTHER_FEATURES = ["is_late_round"]


class GregV1Model(PredictionModel):
    name = "Greg_v1"

    def __init__(self, models_dir: Path | str | None = None):
        self._models_dir = Path(models_dir) if models_dir else DATA_DIR / "models"
        self._margin_model = None
        self._total_model = None
        self._feature_cols: list[str] | None = None

    def _load(self) -> None:
        if self._margin_model is not None and self._total_model is not None and self._feature_cols is not None:
            return
        try:
            self._margin_model = joblib.load(self._models_dir / "greg_v1_margin_model.pkl")
            self._total_model = joblib.load(self._models_dir / "greg_v1_total_model.pkl")
            self._feature_cols = joblib.load(self._models_dir / "greg_v1_feature_cols.pkl")
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Greg_v1 artifacts from "
                f"{self._models_dir}. Re-run `model_lab.ipynb` to re-export "
                "the `greg_v1_*.pkl` files in a compatible environment."
            ) from exc

    def predict(
        self,
        team_a_id: int,
        team_b_id: int,
        db: TeamDB,
        round_num: int = 1,
        slot_id: str | None = None,
    ) -> Prediction:
        self._load()
        assert self._feature_cols is not None
        features = self._compute_features(team_a_id, team_b_id, db, round_num)
        vec = np.array([[features.get(c, 0.0) for c in self._feature_cols]])

        margin = float(self._margin_model.predict(vec)[0])
        total = float(self._total_model.predict(vec)[0])

        score_a = max((total + margin) / 2, 40.0)
        score_b = max((total - margin) / 2, 40.0)

        winner = team_a_id if margin >= 0 else team_b_id
        confidence = min(abs(margin) / 30.0, 1.0) * 0.5 + 0.5

        return Prediction(
            team_a_score=round(score_a, 1),
            team_b_score=round(score_b, 1),
            winner_id=winner,
            confidence=round(confidence, 3),
        )

    @staticmethod
    def _safe(val, default: float = 0.0) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _compute_features(
        self, team_a_id: int, team_b_id: int, db: TeamDB, round_num: int
    ) -> dict[str, float]:
        a = db.get_team(team_a_id)
        b = db.get_team(team_b_id)
        a_seed = db.get_seed(team_a_id)
        b_seed = db.get_seed(team_b_id)
        sf = self._safe

        feats: dict[str, float] = {}

        for fname, col in _DIFF_FEATURES.items():
            if col == "__seed__":
                feats[fname] = sf(a_seed) - sf(b_seed)
            else:
                feats[fname] = sf(a.get(col)) - sf(b.get(col))

        teams = {"a": a, "b": b}
        for fname, (prefix, col) in _RAW_FEATURES.items():
            feats[fname] = sf(teams[prefix].get(col))

        feats["seed_product"] = sf(a_seed, 8) * sf(b_seed, 8)
        feats["seed_sum"] = sf(a_seed, 8) + sf(b_seed, 8)
        feats["is_late_round"] = float(round_num >= 3)

        return feats
