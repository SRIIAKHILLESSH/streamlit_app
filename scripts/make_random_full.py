"""Build data/random_full.csv: computer-random Daily 3 numbers with exactly the same draws-per-day layout as
data/daily3.csv, nudged so each randomness test's p-value is close to the real history's p-value.
Run:  python scripts/make_random_full.py   (seeded, so the result is reproducible)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lib import randomyear as ry  # noqa: E402

KEYS = list(ry.METRICS)


def fast(n):
    n = n.astype(np.int64)
    N = len(n)
    d = np.stack([n // 100, (n // 10) % 10, n % 10], 1)
    e = N * 0.3
    digit = ((np.bincount(d.ravel(), minlength=10) - e) ** 2 / e).sum()
    e2 = N / 10
    pos = sum(((np.bincount(d[:, j], minlength=10) - e2) ** 2 / e2).sum() for j in range(3))
    a, b, c = d[:, 0], d[:, 1], d[:, 2]
    n3 = ((a == b) & (b == c)).sum()
    n1 = (((a == b) | (b == c) | (a == c))).sum() - n3
    obs = np.array([N - n1 - n3, n1, n3])
    ex = N * np.array([0.72, 0.27, 0.01])
    pat = ((obs - ex) ** 2 / ex).sum()
    hi = n >= 500
    h, l = hi.sum(), (~hi).sum()
    runs = 1 + (hi[1:] != hi[:-1]).sum()
    mu = 2 * h * l / N + 1
    var = (mu - 1) * (mu - 2) / (N - 1)
    z = (runs - mu) / np.sqrt(var)
    return {"digit_chi": digit, "pos_chi": pos, "pattern_chi": pat,
            "distinct": float((np.bincount(n, minlength=1000) > 0).sum()),
            "serial": float(np.corrcoef(n[:-1], n[1:])[0, 1]), "runs_z": float(z),
            "repeats": float((n[1:] == n[:-1]).sum())}


def cdf(v, null):  # mid-rank percentile, 0..1
    return ((null < v).sum() + 0.5 * (null == v).sum()) / len(null)


def main():
    df = pd.read_csv(ROOT / "data/daily3.csv", dtype=str, keep_default_na=False)
    mask = np.stack([(df["midday"] != "").to_numpy(), (df["evening"] != "").to_numpy()], 1)  # row-major = chronological
    N = int(mask.sum())
    real = np.array([int(x) for x in df[["midday", "evening"]].to_numpy()[mask]])
    rng = np.random.default_rng(2026)
    null = {k: [] for k in KEYS}
    for _ in range(1000):
        m = fast(rng.integers(0, 1000, N))
        for k in KEYS:
            null[k].append(m[k])
    null = {k: np.array(v) for k, v in null.items()}
    rm = fast(real)
    target = {k: cdf(rm[k], null[k]) for k in KEYS}

    def loss(m):
        g = [abs(cdf(m[k], null[k]) - target[k]) for k in KEYS]
        return max(g) + 0.01 * sum(g) / len(g)

    best, bl = None, 9
    for _ in range(300):
        x = rng.integers(0, 1000, N)
        l = loss(fast(x))
        if l < bl:
            best, bl = x, l
    x, cur = best.copy(), bl
    for it in range(400000):
        if cur < 0.004:
            break
        i = rng.integers(N)
        old = x[i]
        x[i] = rng.integers(0, 1000)
        l = loss(fast(x))
        if l <= cur:
            cur = l
        else:
            x[i] = old
    out = np.full(mask.shape, "", dtype=object)
    out[mask] = [f"{v:03d}" for v in x]
    res = pd.DataFrame({"date": df["date"], "midday": out[:, 0], "evening": out[:, 1]})
    res.to_csv(ROOT / "data/random_full.csv", index=False)
    print("draws", N, "max percentile gap", round(cur, 4))
    for k in KEYS:
        print(k, "real p", round(ry.two_sided_p(rm[k], null[k]), 3), "random p", round(ry.two_sided_p(fast(x)[k], null[k]), 3))


main()
