"""Randomness tests for a stream of 3-digit draws, plus a simulation of truly random draws of the same length."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

METRICS = {
    "digit_chi": "Digit balance (χ², 0–9 across all digits)",
    "pos_chi": "Digit balance by position (χ², 3 positions)",
    "pattern_chi": "All-different / pair / triple mix (χ²)",
    "distinct": "Different numbers used (of 1,000)",
    "serial": "Serial correlation, draw to next draw",
    "runs_z": "High/low runs test (z, split at 500)",
    "repeats": "Same number twice in a row",
}


def metrics(n: np.ndarray) -> dict:
    n = np.asarray(n)
    N = len(n)
    dg = np.stack([n // 100, (n // 10) % 10, n % 10], axis=1)
    exp = N * 3 / 10
    digit_chi = ((np.bincount(dg.ravel(), minlength=10) - exp) ** 2 / exp).sum()
    e2 = N / 10
    pos_chi = sum(((np.bincount(dg[:, j], minlength=10) - e2) ** 2 / e2).sum() for j in range(3))
    u = np.array([len(set(r)) for r in dg.tolist()])
    obs = np.array([(u == 3).sum(), (u == 2).sum(), (u == 1).sum()])
    ex = N * np.array([0.72, 0.27, 0.01])
    pattern_chi = ((obs - ex) ** 2 / ex).sum()
    hi = n >= 500
    n1, n0 = hi.sum(), (~hi).sum()
    runs = 1 + (hi[1:] != hi[:-1]).sum()
    mu = 2 * n1 * n0 / N + 1
    var = (mu - 1) * (mu - 2) / (N - 1)
    runs_z = (runs - mu) / np.sqrt(var) if var > 0 else 0.0
    serial = float(np.corrcoef(n[:-1], n[1:])[0, 1])
    return {"digit_chi": digit_chi, "pos_chi": pos_chi, "pattern_chi": pattern_chi,
            "distinct": float(len(set(n.tolist()))), "serial": serial, "runs_z": float(runs_z),
            "repeats": float((n[1:] == n[:-1]).sum())}


def null_runs(N: int, sims: int = 2000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    out = {k: [] for k in METRICS}
    for _ in range(sims):
        m = metrics(rng.integers(0, 1000, N))
        for k in out:
            out[k].append(m[k])
    return {k: np.array(v) for k, v in out.items()}


def two_sided_p(value: float, null: np.ndarray) -> float:
    lo = (null <= value).mean()
    hi = (null >= value).mean()
    return float(min(1.0, 2 * min(lo, hi)))
