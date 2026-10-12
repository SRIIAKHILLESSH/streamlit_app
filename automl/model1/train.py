"""AutoML 1 model fitting: positional XGBoost (3 x 10-class) and an optional direct 1,000-class model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from automl.common import combine_positional

SEED = 2026


def _xgb_params(cfg: dict, num_class: int) -> dict:
    return {
        "objective": "multi:softprob", "num_class": num_class, "eval_metric": "mlogloss",
        "tree_method": "hist", "max_depth": cfg["max_depth"], "learning_rate": cfg["learning_rate"],
        "subsample": cfg["subsample"], "colsample_bytree": cfg.get("colsample_bytree", 0.8),
        "reg_lambda": cfg["reg_lambda"], "min_child_weight": cfg.get("min_child_weight", 1.0),
        "max_bin": 64, "seed": SEED, "nthread": -1,
    }


def _fit_one(X: np.ndarray, y: np.ndarray, cfg: dict, num_class: int) -> tuple[xgb.Booster, int]:
    """Fit with early stopping on the chronologically last 10% of the training rows."""
    cut = int(len(X) * 0.9)
    dtr = xgb.DMatrix(X[:cut], label=y[:cut])
    dva = xgb.DMatrix(X[cut:], label=y[cut:])
    booster = xgb.train(_xgb_params(cfg, num_class), dtr, num_boost_round=cfg.get("max_rounds", 400),
                        evals=[(dva, "val")], early_stopping_rounds=cfg.get("early_stop", 25), verbose_eval=False)
    return booster, booster.best_iteration + 1


def fit_predict_positional(Xtr: pd.DataFrame, Ytr: np.ndarray, Xte: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, dict]:
    """Ytr: (n,3) digits. Returns (rows x 1000) joint probabilities and per-position round counts."""
    probs, rounds = [], {}
    Xtr_, Xte_ = Xtr.to_numpy(np.float32), Xte.to_numpy(np.float32)
    for j in range(3):
        booster, r = _fit_one(Xtr_, Ytr[:, j], cfg, 10)
        rounds[f"d{j + 1}"] = r
        probs.append(booster.predict(xgb.DMatrix(Xte_), iteration_range=(0, r)))
    return combine_positional(*probs), rounds


def fit_predict_direct(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, dict]:
    """Direct 1,000-class softprob. Classes never seen in training get a tiny floor, then rows renormalise."""
    booster, r = _fit_one(Xtr.to_numpy(np.float32), ytr, cfg, 1000)
    p = booster.predict(xgb.DMatrix(Xte.to_numpy(np.float32)), iteration_range=(0, r))
    p = np.clip(p, 1e-9, None)
    return p / p.sum(axis=1, keepdims=True), {"direct": r}
