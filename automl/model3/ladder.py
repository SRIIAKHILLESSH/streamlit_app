"""AutoML 3 — Coverage Ladder.

Every draw gets a pre-registered list of K numbers (the top K of a full 1,000-number ranking). K shrinks by 100
each week: week 1 = 900, week 2 = 800, ... week 9 onward = 100. The question is whether the draw lands inside the
list more often than chance, which is exactly K / 1,000.

AutoML step (once per week, at the first run of the week, frozen for that week):
  every candidate ranker is back-tested walk-forward on the last BACKTEST draws (each draw ranked only from draws
  before it) and the one with the most hits at this week's K is selected. Because ~14 candidates are compared on the
  same draws, the winner's back-test rate is optimistic by construction; the live weekly result is the honest number.

Run: python -m automl.model3.ladder
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from automl import common as c

ART = c.ARTIFACTS / "automl3"
LOG = ART / "live" / "predictions.csv"
WEEKS = ART / "weeks.json"
START = pd.Timestamp("2026-10-13")  # week 1 = Tue 2026-10-13 .. Mon 2026-10-19
FIRST_K, STEP, MIN_K = 900, 100, 100
BACKTEST = 730  # draws (~1 year) used to pick each week's ranker

COLUMNS = ["draw_date", "draw_type", "week", "k", "ranker", "created_utc", "draw_time_utc", "made_before_draw",
           "ranking_sha256", "result", "result_rank", "hit", "scored_utc"]


# ---------------------------------------------------------------- schedule
def week_of(date: str) -> int:
    """1-based ladder week of a draw date; 0 = before the ladder starts."""
    d = (pd.Timestamp(date) - START).days
    return d // 7 + 1 if d >= 0 else 0


def k_for_week(week: int) -> int:
    return max(MIN_K, FIRST_K - STEP * (week - 1))


def week_range(week: int) -> tuple[str, str]:
    a = START + pd.Timedelta(days=7 * (week - 1))
    return a.date().isoformat(), (a + pd.Timedelta(days=6)).date().isoformat()


# ---------------------------------------------------------------- candidate rankers
def _tiebreak(date: str, draw_type: str) -> np.ndarray:
    seed = int(hashlib.sha256(f"ladder|{date}|{draw_type}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).random(1000)


def _counts(past: np.ndarray, w: int) -> np.ndarray:
    return np.bincount(past[-w:], minlength=1000).astype(float)


def _gap(past: np.ndarray) -> np.ndarray:
    """Draws since each number last appeared (0 = the latest draw); never seen = len(past)."""
    gap = np.full(1000, float(len(past)))
    u, first = np.unique(past[::-1], return_index=True)
    gap[u] = first
    return gap


def _digit_logp(past: np.ndarray, w: int) -> np.ndarray:
    p = past[-w:]
    d = [p // 100, (p // 10) % 10, p % 10]
    f = [np.log(np.bincount(x, minlength=10) + 1.0) for x in d]
    return (f[0][:, None, None] + f[1][None, :, None] + f[2][None, None, :]).reshape(1000)


def _recent_out(past: np.ndarray, w: int) -> np.ndarray:
    return -(np.bincount(past[-w:], minlength=1000) > 0).astype(float)


RANKERS = {  # score: higher = earlier in the list. Ties broken by the seeded per-draw tiebreak.
    "random": lambda p: np.zeros(1000),
    "hot_30": lambda p: _counts(p, 30),
    "hot_100": lambda p: _counts(p, 100),
    "hot_365": lambda p: _counts(p, 365),
    "hot_1000": lambda p: _counts(p, 1000),
    "cold_365": lambda p: -_counts(p, 365),
    "cold_1000": lambda p: -_counts(p, 1000),
    "overdue": lambda p: _gap(p),
    "recently_drawn_first": lambda p: -_gap(p),
    "skip_last_7": lambda p: _recent_out(p, 7),
    "skip_last_30": lambda p: _recent_out(p, 30),
    "digits_hot_100": lambda p: _digit_logp(p, 100),
    "digits_hot_500": lambda p: _digit_logp(p, 500),
    "digits_cold_500": lambda p: -_digit_logp(p, 500),
}


def ranking(name: str, past: np.ndarray, date: str, draw_type: str) -> np.ndarray:
    """Full ranking of number ids 0..999, best first, from draws strictly before (date, draw_type)."""
    score = RANKERS[name](past) + _tiebreak(date, draw_type) * 1e-9
    return np.argsort(-score, kind="stable")


def backtest(s: pd.DataFrame, k: int, n: int = BACKTEST) -> dict:
    """Walk-forward hits at K for every ranker over the last n known draws."""
    nums = s["n"].to_numpy()
    start = len(s) - n
    out = {}
    for name, fn in RANKERS.items():
        hits = 0
        for t in range(start, len(s)):
            past = nums[:t]
            score = fn(past) + _tiebreak(s.date.iat[t], s.draw_type.iat[t]) * 1e-9
            truth_score = score[nums[t]]
            hits += int((score > truth_score).sum() < k)  # truth's 0-based rank < k
        out[name] = {"hits": hits, "n": n, "rate": hits / n}
    return out


def select_for_week(s: pd.DataFrame, week: int) -> dict:
    k = k_for_week(week)
    bt = backtest(s, k)
    best = max(bt, key=lambda r: (bt[r]["hits"], r == "random"))  # ties -> prefer the honest random baseline
    a, b = week_range(week)
    return {"week": week, "start": a, "end": b, "k": k, "ranker": best, "selected_utc": c.utcnow(),
            "selected_from": f"{BACKTEST} draws ending {s.date.iat[-1]} {s.draw_type.iat[-1]}",
            "chance_rate": k / 1000, "backtest": bt,
            "note": f"best of {len(RANKERS)} rankers on the same draws, so its back-test rate is optimistic"}


# ---------------------------------------------------------------- log
def load_log() -> pd.DataFrame:
    if LOG.exists():
        return pd.read_csv(LOG, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def score_log(s: pd.DataFrame) -> int:
    log = load_log()
    known = {(r.date, r.draw_type): r.number for r in s.itertuples(index=False)}
    n = 0
    for i, r in log.iterrows():
        key = (r["draw_date"], r["draw_type"])
        if r["result"] or key not in known:
            continue
        order = (LOG.parent / "rankings" / f"{r['ranking_sha256']}.txt").read_text().split()
        if hashlib.sha256(" ".join(order).encode()).hexdigest() != r["ranking_sha256"]:
            raise RuntimeError(f"ranking for {key} does not match its recorded hash")
        rank = order.index(known[key]) + 1
        log.loc[i, ["result", "result_rank", "hit", "scored_utc"]] = [known[key], str(rank), str(rank <= int(r["k"])),
                                                                      c.utcnow()]
        n += 1
    if n:
        log.to_csv(LOG, index=False)
    return n


def weekly_scorecard(log: pd.DataFrame) -> list[dict]:
    ok = log[(log.made_before_draw == "True") & (log.result != "")]
    rows = []
    for w, g in ok.groupby(ok.week.astype(int)):
        k, n, h = int(g.k.iat[0]), len(g), int((g.hit == "True").sum())
        lo, hi = c.wilson(h, n)
        rows.append({"week": w, "k": k, "draws": n, "hits": h, "rate": h / n, "ci95": [lo, hi], "chance": k / 1000,
                     "expected_hits": n * k / 1000,
                     "p_better_than_chance": binomtest(h, n, k / 1000, alternative="greater").pvalue})
    return rows


# ---------------------------------------------------------------- main
def main() -> None:
    s = c.load_stream()
    scored = score_log(s) if LOG.exists() else 0
    date, draw_type = c.next_draw(s)
    week = week_of(date)
    weeks = c.read_json(WEEKS) if WEEKS.exists() else {}
    made = None
    if week >= 1:
        if str(week) not in weeks:
            weeks[str(week)] = select_for_week(s, week)
            c.write_json(WEEKS, weeks)
        sel = weeks[str(week)]
        log = load_log()
        if not ((log.draw_date == date) & (log.draw_type == draw_type)).any():
            order = ranking(sel["ranker"], s["n"].to_numpy(), date, draw_type)
            text = " ".join(c.ALL[order])
            sha = hashlib.sha256(text.encode()).hexdigest()
            (LOG.parent / "rankings").mkdir(parents=True, exist_ok=True)
            (LOG.parent / "rankings" / f"{sha}.txt").write_text(text)
            created, draw_t = c.utcnow(), c.draw_time_utc(date, draw_type)
            made = {"draw_date": date, "draw_type": draw_type, "week": str(week), "k": str(sel["k"]),
                    "ranker": sel["ranker"], "created_utc": created, "draw_time_utc": draw_t.isoformat(),
                    "made_before_draw": str(pd.Timestamp(created) < pd.Timestamp(draw_t)), "ranking_sha256": sha,
                    "result": "", "result_rank": "", "hit": "", "scored_utc": ""}
            log = pd.concat([log, pd.DataFrame([made], columns=COLUMNS)], ignore_index=True)
            LOG.parent.mkdir(parents=True, exist_ok=True)
            log.to_csv(LOG, index=False)

    log = load_log()
    up = log[(log.draw_date == date) & (log.draw_type == draw_type)]
    nxt = None
    if len(up):
        r = up.iloc[0]
        order = (LOG.parent / "rankings" / f"{r.ranking_sha256}.txt").read_text().split()
        k = int(r.k)
        nxt = {"date": date, "draw_type": draw_type, "week": int(r.week), "k": k, "ranker": r.ranker,
               "created_utc": r.created_utc, "made_before_draw": r.made_before_draw, "ranking_sha256": r.ranking_sha256,
               "picks": sorted(order[:k]), "left_out": sorted(order[k:])}
    sc = weekly_scorecard(log)
    tot_n = sum(w["draws"] for w in sc)
    tot_h = sum(w["hits"] for w in sc)
    exp = sum(w["expected_hits"] for w in sc)
    var = sum(w["draws"] * w["chance"] * (1 - w["chance"]) for w in sc)
    c.write_json(ART / "live" / "latest.json", {
        "updated_utc": c.utcnow(), "start": START.date().isoformat(),
        "schedule": [{"week": w, "k": k_for_week(w), "dates": week_range(w)} for w in range(1, 10)],
        "current_week": week, "next": nxt, "weeks": weeks, "scorecard": sc,
        "overall": {"draws": tot_n, "hits": tot_h, "expected_hits": exp,
                    "z_vs_chance": (tot_h - exp) / var ** 0.5 if var else None},
    })
    print(f"scored {scored}; week {week}; " + (f"pre-registered {date} {draw_type} K={made['k']} ({made['ranker']})"
                                               if made else "nothing new to pre-register"))


if __name__ == "__main__":
    main()
