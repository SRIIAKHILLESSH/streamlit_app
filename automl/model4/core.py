"""AutoML 4 — 2025 blind reconstruction (single-number predictions, scored True/False).

Contract
- Train/tune: draws up to 2024-12-31. Validation (model selection): 2024, trained on <=2023 only, 50% of 2024 draws masked.
- Blind test: calendar year 2025; 50% of each session's draws drawn at random (fixed seed) are hidden. The trainer receives
  only (date, session) for those cases, never a value, and never any 2025 value at all.
- The evaluator returns True/False per case internally and publishes only the aggregate; it runs once and refuses to re-run.
- Winning numbers stay locked unless blind accuracy >= 80%.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "automl4"
TRAIN_END = pd.Timestamp("2024-12-31")
VAL_START = pd.Timestamp("2024-01-01")
TEST_START, TEST_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
MASK_SEED = 2025
TARGET = 0.80


def long_draws(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or ROOT / "data" / "daily3.csv", dtype=str, keep_default_na=False)
    df["date"] = pd.to_datetime(df["date"])
    out = pd.concat([df[["date", s]].rename(columns={s: "num"}).assign(session=s) for s in ("midday", "evening")])
    out = out[out["num"].str.len() == 3].copy()
    out["n"] = out["num"].astype(int)
    return out.sort_values(["date", "session"]).reset_index(drop=True)


def mask_cases(draws: pd.DataFrame, start, end, seed: int) -> pd.DataFrame:
    """Random 50% of each session's draws in [start, end] (seeded)."""
    parts = []
    for i, s in enumerate(("midday", "evening")):
        d = draws[(draws.session == s) & (draws.date >= start) & (draws.date <= end)]
        parts.append(d.sample(frac=0.5, random_state=seed + i))
    return pd.concat(parts).sort_values(["date", "session"]).reset_index(drop=True)


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---- candidate predictors: history (date,session,n) -> one number per case -------------------------------------
def predict(cfg: dict, hist: pd.DataFrame, cases: pd.DataFrame) -> np.ndarray:
    """Return one predicted number per case using only `hist`."""
    kind = cfg["kind"]
    if kind == "random":
        return np.random.default_rng(cfg["seed"]).integers(0, 1000, len(cases))
    end = hist.date.max()
    preds = []
    for _, c in cases.iterrows():
        h = hist[hist.session == c.session] if cfg.get("by_session") else hist
        w = np.exp(-np.log(2) * (end - h.date).dt.days.to_numpy() / cfg["half_life"]) if cfg["half_life"] else np.ones(len(h))
        if kind == "weekday":
            w = w * np.where(h.date.dt.dayofweek.to_numpy() == c.date.dayofweek, 1.0, cfg["off_weight"])
        cnt = np.bincount(h.n.to_numpy(), weights=w, minlength=1000)
        if kind == "positional":  # product of smoothed per-position digit frequencies
            p = np.ones(1000)
            for pos, div in enumerate((100, 10, 1)):
                dc = np.bincount((h.n.to_numpy() // div) % 10, weights=w, minlength=10) + cfg["alpha"]
                p *= (dc / dc.sum())[(np.arange(1000) // div) % 10]
            cnt = p
        preds.append(int(np.argmax(cnt)))
    return np.array(preds)


def accuracy(cfg, hist, cases) -> tuple[int, int]:
    p = predict(cfg, hist, cases)
    return int((p == cases.n.to_numpy()).sum()), len(cases)
