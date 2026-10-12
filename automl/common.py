"""Shared data layer for both AutoML experiments.

Decisions (logged in every report):
- A *draw stream* is every Daily 3 result in chronological order; on a date with both sessions, midday precedes evening.
- *Dataset years* are explicit indices, not calendar years: year k covers
  [1984-09-01 + (k-1) years, 1984-09-01 + k years). The history starts 1984-09-01, so years 1-42 end 2026-08-31;
  anything later is "post-history" (live / prospective).
- Draw times (America/New_York): midday 12:59, evening 19:29. Used only to check that a live prediction was made before its draw.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "daily3.csv"
ARTIFACTS = ROOT / "artifacts"
ORIGIN = pd.Timestamp("1984-09-01")
SESSIONS = {"midday": 0, "evening": 1}
DRAW_TIME_ET = {"midday": (12, 59), "evening": (19, 29)}

YEAR_DEFINITION = ("dataset-year k = [1984-09-01 + (k-1) years, 1984-09-01 + k years); "
                   "years 1-42 = 1984-09-01..2026-08-31; midday precedes evening on the same date")


def year_bounds(k: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive start, exclusive end of dataset-year k."""
    return ORIGIN + pd.DateOffset(years=k - 1), ORIGIN + pd.DateOffset(years=k)


def dataset_year(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    years = (d.dt.year - ORIGIN.year) - ((d.dt.month < ORIGIN.month)).astype(int) + 1
    return years.to_numpy()


def load_stream(path: Path = DATA_CSV) -> pd.DataFrame:
    """All draws in order: date, draw_type, number (3-char string), n, d1, d2, d3, year."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    recs = []
    for r in raw.itertuples(index=False):
        for col in ("midday", "evening"):
            v = getattr(r, col)
            if isinstance(v, str) and v.strip():
                recs.append((r.date, col, v.strip().zfill(3)))
    s = pd.DataFrame(recs, columns=["date", "draw_type", "number"])
    s["order"] = s["draw_type"].map(SESSIONS)
    s = s.sort_values(["date", "order"], kind="stable").reset_index(drop=True)
    s["n"] = s["number"].astype(int)
    s["d1"], s["d2"], s["d3"] = s["n"] // 100, (s["n"] // 10) % 10, s["n"] % 10
    s["year"] = dataset_year(s["date"])
    return s.drop(columns="order")


def audit(s: pd.DataFrame) -> dict:
    """Step 1 checks: duplicates, invalid values, gaps, session start dates, leading zeros."""
    d = pd.to_datetime(s["date"])
    dup = s.duplicated(["date", "draw_type"]).sum()
    bad = (~s["number"].str.fullmatch(r"\d{3}")).sum()
    dates = d.drop_duplicates().sort_values()
    gaps = dates.diff().dt.days
    big = dates[gaps > 3]
    first = {t: s.loc[s.draw_type == t, "date"].min() for t in SESSIONS}
    sundays = s[d.dt.weekday == 6]
    per_year = s.groupby("year").size()
    return {
        "draws": int(len(s)),
        "first_date": s["date"].iloc[0], "last_date": s["date"].iloc[-1],
        "duplicates_date_drawtype": int(dup),
        "invalid_numbers": int(bad),
        "leading_zero_numbers": int((s["n"] < 100).sum()),
        "first_draw_by_type": first,
        "first_sunday_draw": sundays["date"].min() if len(sundays) else None,
        "gaps_over_3_days": [{"after": str((dates[dates < x].max()).date()), "resumes": str(x.date())} for x in big][:20],
        "draws_per_dataset_year": {int(k): int(v) for k, v in per_year.items()},
        "year_definition": YEAR_DEFINITION,
        "rule_change_notes": [
            f"Midday draws begin {first['midday']} (before that, evening only).",
            f"Sunday draws begin {sundays['date'].min() if len(sundays) else 'n/a'}.",
        ],
    }


def draw_time_utc(date: str, draw_type: str) -> dt.datetime:
    from zoneinfo import ZoneInfo
    h, m = DRAW_TIME_ET[draw_type]
    local = dt.datetime.fromisoformat(date).replace(hour=h, minute=m, tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(dt.timezone.utc)


def next_draw(s: pd.DataFrame, now_utc: dt.datetime | None = None) -> tuple[str, str]:
    """The first scheduled draw after the latest known result (draws happen every day, midday then evening)."""
    last = s.iloc[-1]
    if last.draw_type == "midday":
        return last.date, "evening"
    nxt = (pd.Timestamp(last.date) + pd.Timedelta(days=1)).date().isoformat()
    return nxt, "midday"


# ---------------------------------------------------------------- probability / ranking helpers
ALL = np.array([f"{i:03d}" for i in range(1000)])


def combine_positional(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """(rows x 10) x3 -> (rows x 1000) joint probabilities assuming conditional independence; rows sum to 1."""
    p = p1[:, :, None, None] * p2[:, None, :, None] * p3[:, None, None, :]
    p = p.reshape(len(p1), 1000)
    return p / p.sum(axis=1, keepdims=True)


def rank_order(p: np.ndarray, seed: int = 0) -> np.ndarray:
    """Deterministic full ranking (rows x 1000 of number ids), ties broken by a seeded permutation."""
    rng = np.random.default_rng(seed)
    tiebreak = rng.random(1000) * 1e-12
    return np.argsort(-(p + tiebreak), axis=1, kind="stable")


def topk_hits(order: np.ndarray, truth: np.ndarray, ks=(100, 50, 20, 10, 1)) -> dict[int, np.ndarray]:
    pos = np.argmax(order == truth[:, None], axis=1)  # rank position of the true number
    return {k: pos < k for k in ks}


def log_loss(p: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(truth)), truth], 1e-15, 1.0))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def metric_block(p: np.ndarray, truth: np.ndarray, seed: int = 0) -> dict:
    order = rank_order(p, seed)
    hits = topk_hits(order, truth)
    out = {"n": int(len(truth)), "log_loss": float(log_loss(p, truth).mean()), "uniform_log_loss": float(np.log(1000))}
    for k, h in hits.items():
        lo, hi = wilson(int(h.sum()), len(h))
        out[f"top{k}"] = {"hits": int(h.sum()), "rate": float(h.mean()) if len(h) else 0.0,
                          "ci95": [lo, hi], "baseline": k / 1000}
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
