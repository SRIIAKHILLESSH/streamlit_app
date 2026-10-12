"""Leakage-safe feature engine for AutoML 1.

Row t describes draw t using ONLY draws 0..t-1 (plus advance-known metadata of draw t itself:
its session, weekday, month). Every block is computed once for the whole stream with prefix sums,
then a config selects columns. `assert_no_leakage` verifies row t is unchanged when draws >= t are altered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_CHOICES = (5, 10, 20, 30)
FREQ_CHOICES = (30, 100, 500)
SINCE_CAP = 200


def _onehot_digits(D: np.ndarray) -> np.ndarray:
    """N x 30 one-hot: position j digit v at column 10*j+v."""
    oh = np.zeros((len(D), 30), dtype=np.float32)
    for j in range(3):
        oh[np.arange(len(D)), 10 * j + D[:, j]] = 1
    return oh


def build_blocks(s: pd.DataFrame) -> dict[str, pd.DataFrame]:
    D = s[["d1", "d2", "d3"]].to_numpy()
    n = s["n"].to_numpy()
    N = len(s)
    blocks: dict[str, pd.DataFrame] = {}

    # lags: digits of draw t-l (missing = -1), largest window built once
    lag = {}
    for l in range(1, max(LAG_CHOICES) + 1):
        for j in range(3):
            col = np.full(N, -1, dtype=np.int16)
            col[l:] = D[:-l, j]
            lag[f"lag{l}_d{j + 1}"] = col
    blocks["lags"] = pd.DataFrame(lag)

    # rolling digit frequency per position over the previous W draws (bucket = hundreds digit = position 1)
    oh = _onehot_digits(D)
    C = np.vstack([np.zeros((1, 30), np.float32), np.cumsum(oh, axis=0)])  # C[t] = counts in draws < t
    t = np.arange(N)
    for W in FREQ_CHOICES:
        lo = np.maximum(0, t - W)
        cnt = C[t] - C[lo]
        denom = np.maximum(t - lo, 1)[:, None]
        freq = cnt / denom
        blocks[f"freq{W}"] = pd.DataFrame(freq, columns=[f"f{W}_p{c // 10 + 1}_{c % 10}" for c in range(30)])

    # time since each position-digit last appeared (capped)
    since = np.full((N, 30), SINCE_CAP, dtype=np.int16)
    last = np.full(30, -1)
    for i in range(N):
        seen = last >= 0
        since[i, seen] = np.minimum(i - last[seen], SINCE_CAP)
        for j in range(3):
            last[10 * j + D[i, j]] = i
    blocks["since"] = pd.DataFrame(since, columns=[f"since_p{c // 10 + 1}_{c % 10}" for c in range(30)])

    # advance-known metadata of draw t
    d = pd.to_datetime(s["date"])
    blocks["meta"] = pd.DataFrame({
        "is_evening": (s["draw_type"] == "evening").astype(np.int8).to_numpy(),
        "weekday": d.dt.weekday.astype(np.int8).to_numpy(),
        "month": d.dt.month.astype(np.int8).to_numpy(),
    })

    # hypothesis features: recent 900-999 / 000-099 counts (the High-Low idea)
    hi = np.concatenate([[0], np.cumsum(n >= 900)])
    lo_ = np.concatenate([[0], np.cumsum(n < 100)])
    hyp = {}
    for w in (7, 10):
        a = np.maximum(0, t - w)
        hyp[f"high_last{w}"] = (hi[t] - hi[a]).astype(np.int8)
        hyp[f"low_last{w}"] = (lo_[t] - lo_[a]).astype(np.int8)
    blocks["hyp"] = pd.DataFrame(hyp)
    return blocks


def select(blocks: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    cols = [blocks["lags"][[f"lag{l}_d{j}" for l in range(1, cfg["lag_window"] + 1) for j in (1, 2, 3)]]]
    if cfg.get("freq_window"):
        cols.append(blocks[f"freq{cfg['freq_window']}"])
    if cfg.get("use_since"):
        cols.append(blocks["since"])
    if cfg.get("use_meta", True):
        cols.append(blocks["meta"])
    if cfg.get("use_hyp"):
        cols.append(blocks["hyp"])
    return pd.concat(cols, axis=1)


def assert_no_leakage(s: pd.DataFrame, cfg: dict, probe: int = 5000) -> None:
    """Changing draws >= probe must not change any feature row <= probe."""
    base = select(build_blocks(s), cfg).iloc[: probe + 1]
    s2 = s.copy()
    rng = np.random.default_rng(1)
    s2.loc[probe:, "n"] = rng.integers(0, 1000, len(s2) - probe)
    s2["d1"], s2["d2"], s2["d3"] = s2["n"] // 100, (s2["n"] // 10) % 10, s2["n"] % 10
    alt = select(build_blocks(s2), cfg).iloc[: probe + 1]
    pd.testing.assert_frame_equal(base.reset_index(drop=True), alt.reset_index(drop=True))
