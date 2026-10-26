"""K-means / DBSCAN clustering of the 1,000 Daily 3 numbers, with a random-draw null for comparison."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

KS = list(range(2, 11))                       # k range: 10 values wide (2..10)
EPS = [round(0.3 + 0.1 * i, 1) for i in range(10)]  # 10 DBSCAN radii, 0.3..1.2
MIN_SAMPLES = 5
FEATURES = ["Times drawn", "Mean gap", "Longest gap", "Draws since last", "Drawn in last 2,000 draws"]


def number_features(nums: np.ndarray) -> np.ndarray:
    """One row per number 000-999 from a chronological array of draws."""
    n = len(nums)
    idx = np.arange(n)
    out = np.zeros((1000, 5))
    order = np.argsort(nums, kind="stable")
    sn, si = nums[order], idx[order]
    bounds = np.searchsorted(sn, np.arange(1001))
    for v in range(1000):
        pos = si[bounds[v]:bounds[v + 1]]
        if len(pos) == 0:
            out[v] = [0, n, n, n, 0]
            continue
        gaps = np.diff(np.concatenate([[-1], pos]))
        out[v] = [len(pos), gaps.mean(), max(gaps.max(), n - 1 - pos[-1]), n - 1 - pos[-1], (pos >= n - 2000).sum()]
    return out


def scale(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def kmeans_scan(Z: np.ndarray) -> dict:
    sil, inertia = {}, {}
    for k in KS:
        km = KMeans(k, n_init=5, random_state=0).fit(Z)
        sil[k], inertia[k] = float(silhouette_score(Z, km.labels_)), float(km.inertia_)
    return {"sil": sil, "inertia": inertia}


def dbscan_scan(Z: np.ndarray) -> dict:
    res = {}
    for e in EPS:
        lab = DBSCAN(eps=e, min_samples=MIN_SAMPLES).fit_predict(Z)
        res[e] = {"clusters": int(len(set(lab)) - (-1 in lab)), "noise": int((lab == -1).sum())}
    return res


def null_scan(n: int, sims: int = 20, seed: int = 1) -> dict:
    """Same scans on genuinely random draws with the same number of draws."""
    rng = np.random.default_rng(seed)
    sil = {k: [] for k in KS}
    noise = {e: [] for e in EPS}
    for _ in range(sims):
        Z = scale(number_features(rng.integers(0, 1000, n)))
        for k, v in kmeans_scan(Z)["sil"].items():
            sil[k].append(v)
        for e, v in dbscan_scan(Z).items():
            noise[e].append(v["noise"])
    return {"sil": {k: np.array(v) for k, v in sil.items()}, "noise": {e: np.array(v) for e, v in noise.items()}}


def fit(Z: np.ndarray, k: int, eps: float):
    km = KMeans(k, n_init=10, random_state=0).fit(Z)
    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(Z)
    return km.labels_, db, PCA(2, random_state=0).fit_transform(Z)
