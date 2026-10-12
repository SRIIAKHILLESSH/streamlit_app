"""Append-only live prediction log shared by both AutoML models.

A prediction row is written BEFORE its draw and never edited afterwards, except for the score columns, which
are filled in once the real result is known. Rows created after the scheduled draw time are kept but marked
`made_before_draw = False` and excluded from the live scorecard.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from automl import common as c

COLUMNS = ["draw_date", "draw_type", "kind", "model_version", "created_utc", "draw_time_utc", "made_before_draw",
           "top100", "ranking_sha256", "result", "result_rank", "hit100", "hit50", "hit20", "hit10", "scored_utc"]


def load(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def already(log: pd.DataFrame, date: str, draw_type: str, kind: str) -> bool:
    return bool(((log.draw_date == date) & (log.draw_type == draw_type) & (log.kind == kind)).any())


def make_row(date: str, draw_type: str, kind: str, version: str, order: np.ndarray) -> dict:
    """order: the full ranking of 1,000 number ids, best first."""
    ranking = " ".join(c.ALL[order])
    created = c.utcnow()
    draw_t = c.draw_time_utc(date, draw_type)
    return {"draw_date": date, "draw_type": draw_type, "kind": kind, "model_version": version,
            "created_utc": created, "draw_time_utc": draw_t.isoformat(),
            "made_before_draw": str(pd.Timestamp(created) < pd.Timestamp(draw_t)),
            "top100": " ".join(c.ALL[order[:100]]), "ranking_sha256": hashlib.sha256(ranking.encode()).hexdigest(),
            "result": "", "result_rank": "", "hit100": "", "hit50": "", "hit20": "", "hit10": "", "scored_utc": "",
            "_ranking": ranking}


def append(path: Path, rows: list[dict]) -> None:
    """Append rows; full rankings go to a sidecar file keyed by the row's sha so the CSV stays small."""
    if not rows:
        return
    log = load(path)
    rank_dir = path.parent / "rankings"
    rank_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (rank_dir / f"{r['ranking_sha256']}.txt").write_text(r.pop("_ranking"))
    log = pd.concat([log, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)


def score(path: Path, s: pd.DataFrame) -> int:
    """Fill score columns for rows whose draw result is now known. Returns rows scored."""
    log = load(path)
    if log.empty:
        return 0
    known = {(r.date, r.draw_type): r.number for r in s.itertuples(index=False)}
    n = 0
    for i, r in log.iterrows():
        if r["result"] or (r["draw_date"], r["draw_type"]) not in known:
            continue
        res = known[(r["draw_date"], r["draw_type"])]
        ranking_file = path.parent / "rankings" / f"{r['ranking_sha256']}.txt"
        ranking = ranking_file.read_text().split() if ranking_file.exists() else r["top100"].split()
        if hashlib.sha256(" ".join(ranking).encode()).hexdigest() != r["ranking_sha256"] and len(ranking) == 1000:
            raise RuntimeError(f"ranking file for row {i} does not match its recorded hash")
        rank = ranking.index(res) + 1 if res in ranking else 1001
        log.loc[i, ["result", "result_rank", "scored_utc"]] = [res, str(rank), c.utcnow()]
        for k in (100, 50, 20, 10):
            log.loc[i, f"hit{k}"] = str(rank <= k)
        n += 1
    log.to_csv(path, index=False)
    return n


def summary(path: Path) -> dict:
    """Live scorecard per prediction kind, counting only rows made before their draw."""
    log = load(path)
    out = {}
    if log.empty:
        return out
    ok = log[(log.made_before_draw == "True") & (log.result != "")]
    for kind, g in ok.groupby("kind"):
        block = {"n": int(len(g))}
        for k in (100, 50, 20, 10):
            h = int((g[f"hit{k}"] == "True").sum())
            lo, hi = c.wilson(h, len(g))
            block[f"top{k}"] = {"hits": h, "rate": h / len(g) if len(g) else 0.0, "ci95": [lo, hi], "baseline": k / 1000}
        out[kind] = block
    return out
