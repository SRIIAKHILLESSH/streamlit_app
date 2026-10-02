"""Analysis shared by the pages: per-session stats and the High-Low rule."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

HL_HIGH, HL_LOW, HL_WINDOW = 900, 100, 7
HL_CHANCE = 1 - 0.9 ** HL_WINDOW
FWD_START = "2026-10-02"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def session_numbers(df: pd.DataFrame, col: str) -> list[str]:
    return [v for v in df[col] if v]


def session_stats(nums: list[str]) -> dict:
    n = len(nums)
    vals = np.array([int(x) for x in nums])
    digits = np.array([[int(c) for c in x] for x in nums])
    grid = np.bincount(vals, minlength=1000)
    last = {}
    for i, v in enumerate(vals):
        last[v] = i
    overdue = sorted(((f"{k:03d}", n - 1 - last.get(k, -1)) for k in range(1000)), key=lambda t: -t[1])
    uniq = [len(set(x)) for x in nums]
    return {
        "n": n,
        "grid": grid,
        "digit_freq": np.bincount(digits.ravel(), minlength=10),
        "pos_freq": [np.bincount(digits[:, j], minlength=10) for j in range(3)],
        "sum_hist": np.bincount(digits.sum(axis=1), minlength=28),
        "repeat": {"All unique": uniq.count(3), "Pair": uniq.count(2), "Triple": uniq.count(1)},
        "odd_count": np.bincount((digits % 2).sum(axis=1), minlength=4),
        "overdue": overdue[:10],
    }


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 1.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def high_low(s: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Triggers (900-999) and whether a 000-099 draw appears in the next 7 draws of stream `s`."""
    n = s["n"].to_numpy()
    low = n < HL_LOW
    rows = []
    for i in np.flatnonzero(n >= HL_HIGH):
        nxt = s.iloc[i + 1:i + 1 + HL_WINDOW]
        hits = np.flatnonzero(nxt["n"].to_numpy() < HL_LOW)
        k = int(hits[0]) if len(hits) else -1
        status = "green" if k >= 0 else ("pending" if len(nxt) < HL_WINDOW else "red")
        prev = n[max(0, i - HL_WINDOW):i]
        lows_before = np.flatnonzero(low[:i])
        since = i - 1 - lows_before[-1] if len(lows_before) else i
        rows.append({
            "date": s.at[i, "date"], "session": s.at[i, "session"], "num": s.at[i, "num"], "n": int(n[i]),
            "next": list(zip(nxt["num"], nxt["date"], nxt["session"])), "k": k, "status": status,
            "prev_low": int((prev < HL_LOW).sum()), "prev_high": int((prev >= HL_HIGH).sum()), "since_low": int(since),
        })
    ev = pd.DataFrame(rows)
    if len(ev):
        d = pd.to_datetime(ev["date"])
        ev["year"] = d.dt.year
        ev["weekday"] = d.dt.strftime("%a")
        ev["month"] = d.dt.strftime("%b")
    # rate after ANY draw, for comparison
    win = np.convolve(low.astype(int), np.ones(HL_WINDOW, dtype=int), mode="valid")  # lows in [j, j+7)
    any_rate = float((win[1:] > 0).mean()) if len(win) > 1 else 0.0
    return ev, any_rate


def chi_p(groups: list[tuple[int, int]]) -> float:
    """groups: [(green, total)] -> chi-square p-value (no continuity correction)."""
    tab = [[g, t - g] for g, t in groups if t]
    if len(tab) < 2:
        return 1.0
    return float(chi2_contingency(tab, correction=False)[1])


def why_features(ev: pd.DataFrame) -> list[dict]:
    done = ev[ev.status != "pending"].copy()
    done["green"] = done.status == "green"
    done["decade"] = (done.year // 10 * 10).astype(str) + "s"
    done["band"] = np.where(done.n < 950, "900–949", "950–999")
    done["last"] = (done.n % 10).astype(str)
    cap = lambda v: "2+" if v >= 2 else str(v)
    done["prevLow"] = done.prev_low.map(cap)
    done["prevHigh"] = done.prev_high.map(cap)
    done["since"] = pd.cut(done.since_low, [-1, 4, 9, 19, 10**9], labels=["0–4", "5–9", "10–19", "20+"]).astype(str)
    feats = [
        ("session", "Session", ["Midday", "Evening"]),
        ("weekday", "Weekday", DAYS),
        ("month", "Month", MONTHS),
        ("decade", "Decade", sorted(done.decade.unique())),
        ("band", "Trigger number", ["900–949", "950–999"]),
        ("last", "Trigger last digit", [str(i) for i in range(10)]),
        ("prevLow", "Lows in previous 7", ["0", "1", "2+"]),
        ("prevHigh", "Highs in previous 7", ["0", "1", "2+"]),
        ("since", "Draws since last low", ["0–4", "5–9", "10–19", "20+"]),
    ]
    out = []
    for key, label, order in feats:
        groups = []
        for cat in order:
            sub = done[done[key] == cat]
            if len(sub):
                groups.append((cat, int(sub.green.sum()), len(sub)))
        out.append({"label": label, "groups": groups, "p": chi_p([(g, t) for _, g, t in groups])})
    return sorted(out, key=lambda r: r["p"])
