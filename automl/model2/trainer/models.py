"""AutoML 2 candidate families. Every family returns exact 1,000-number probabilities (finite, >= 0, rows sum to 1).

  uniform        0.001 each (reference)
  freq_exact     Laplace-smoothed frequency of each exact number in the training rows
  freq_positional product of smoothed per-position digit frequencies
  xgb_positional three 10-class XGBoost models (hundreds/tens/ones) on calendar features
  lgbm_positional same with LightGBM
  xgb_direct     one 1,000-class XGBoost (rounds capped for cost)
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from automl.common import combine_positional
from automl.model2.config import SEED
from automl.model2.trainer.features import calendar


def _digits(numbers: pd.Series) -> np.ndarray:
    n = numbers.astype(int).to_numpy()
    return np.stack([n // 100, (n // 10) % 10, n % 10], axis=1)


class Model:
    def __init__(self, family: str, cfg: dict):
        self.family, self.cfg = family, cfg
        self.state: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> "Model":
        f, cfg = self.family, self.cfg
        D = _digits(train["number"])
        n = D[:, 0] * 100 + D[:, 1] * 10 + D[:, 2]
        if f == "freq_exact":
            cnt = np.bincount(n, minlength=1000) + cfg.get("alpha", 1.0)
            self.state["p"] = cnt / cnt.sum()
        elif f == "freq_positional":
            ps = [(np.bincount(D[:, j], minlength=10) + cfg.get("alpha", 1.0)) for j in range(3)]
            ps = [p / p.sum() for p in ps]
            self.state["p"] = combine_positional(*[p[None, :] for p in ps])[0]
        elif f in ("xgb_positional", "lgbm_positional", "xgb_direct"):
            X = calendar(train, cfg["recipe"]).to_numpy(np.float32)
            cut = int(len(X) * 0.9)  # chronological early-stopping slice (last 10% of training rows)
            if f == "xgb_direct":
                self.state["boosters"] = [self._fit_xgb(X, n, cut, 1000)]
            elif f == "xgb_positional":
                self.state["boosters"] = [self._fit_xgb(X, D[:, j], cut, 10) for j in range(3)]
            else:
                self.state["boosters"] = [self._fit_lgbm(X, D[:, j], cut) for j in range(3)]
        elif f != "uniform":
            raise ValueError(f"unknown family {f}")
        return self

    def _fit_xgb(self, X, y, cut, k):
        c = self.cfg
        params = {"objective": "multi:softprob", "num_class": k, "eval_metric": "mlogloss", "tree_method": "hist",
                  "max_depth": c["max_depth"], "learning_rate": c["learning_rate"], "subsample": c["subsample"],
                  "reg_lambda": c["reg_lambda"], "min_child_weight": c["min_child_weight"], "seed": SEED, "nthread": -1}
        rounds = c.get("max_rounds", 400) if k == 10 else min(c.get("max_rounds", 40), 40)
        b = xgb.train(params, xgb.DMatrix(X[:cut], label=y[:cut]), num_boost_round=rounds,
                      evals=[(xgb.DMatrix(X[cut:], label=y[cut:]), "val")], early_stopping_rounds=20, verbose_eval=False)
        return b

    def _fit_lgbm(self, X, y, cut):
        c = self.cfg
        params = {"objective": "multiclass", "num_class": 10, "learning_rate": c["learning_rate"],
                  "num_leaves": c["num_leaves"], "min_data_in_leaf": c["min_data_in_leaf"],
                  "lambda_l2": c["reg_lambda"], "bagging_fraction": c["subsample"], "bagging_freq": 1,
                  "feature_fraction": 1.0, "seed": SEED, "verbosity": -1, "num_threads": 0, "deterministic": True}
        return lgb.train(params, lgb.Dataset(X[:cut], label=y[:cut]), num_boost_round=c.get("max_rounds", 400),
                         valid_sets=[lgb.Dataset(X[cut:], label=y[cut:])],
                         callbacks=[lgb.early_stopping(20, verbose=False)])

    # ------------------------------------------------------------------ predict
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        f = self.family
        if f == "uniform":
            p = np.full((len(frame), 1000), 1e-3)
        elif f in ("freq_exact", "freq_positional"):
            p = np.tile(self.state["p"], (len(frame), 1))
        else:
            X = calendar(frame, self.cfg["recipe"]).to_numpy(np.float32)
            bs = self.state["boosters"]
            if f == "xgb_direct":
                b = bs[0]
                p = np.clip(b.predict(xgb.DMatrix(X), iteration_range=(0, b.best_iteration + 1)), 1e-9, None)
            elif f == "xgb_positional":
                p = combine_positional(*[b.predict(xgb.DMatrix(X), iteration_range=(0, b.best_iteration + 1)) for b in bs])
            else:
                p = combine_positional(*[b.predict(X, num_iteration=b.best_iteration if b.best_iteration > 0 else None)
                                         for b in bs])
        p = np.asarray(p, dtype=np.float64)
        return p / p.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ persistence (model freeze)
    def save(self, folder: Path) -> list[Path]:
        folder.mkdir(parents=True, exist_ok=True)
        files = [folder / "model_meta.json"]
        files[0].write_text(json.dumps({"family": self.family, "cfg": self.cfg}, indent=2))
        if "p" in self.state:
            np.save(folder / "probs.npy", self.state["p"]); files.append(folder / "probs.npy")
        for i, b in enumerate(self.state.get("boosters", [])):
            path = folder / (f"booster{i}.json" if self.family.startswith("xgb") else f"booster{i}.txt")
            b.save_model(str(path)); files.append(path)
        return files

    @classmethod
    def load(cls, folder: Path) -> "Model":
        meta = json.loads((folder / "model_meta.json").read_text())
        m = cls(meta["family"], meta["cfg"])
        if (folder / "probs.npy").exists():
            m.state["p"] = np.load(folder / "probs.npy")
        boosters = []
        for path in sorted(folder.glob("booster*")):
            if path.suffix == ".json":
                b = xgb.Booster(); b.load_model(str(path))
                boosters.append(b)
            else:
                boosters.append(lgb.Booster(model_file=str(path)))
        if boosters:
            m.state["boosters"] = boosters
        return m
