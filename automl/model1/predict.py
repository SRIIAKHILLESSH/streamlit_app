"""AutoML 1 live step: score finished predictions, then pre-register a ranking for the next draw.

Two rankings are logged for every draw:
- "champion": the promoted model. While no model has cleared the promotion rule, the champion is the uniform
  forecast, so its ranking is a seeded random permutation (seed = the draw id), i.e. an honest random baseline.
- "best_candidate": the best Optuna configuration so far, refit on every draw before the target
  (refit-every-step, fixed in advance), even though it has not been promoted.

Run: python -m automl.model1.predict
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from automl import common as c
from automl import live
from automl.model1 import features as F
from automl.model1 import train as T

OUT = c.ARTIFACTS / "automl1"
LOG = OUT / "live" / "predictions.csv"


def seeded_random_order(date: str, draw_type: str) -> np.ndarray:
    seed = int(hashlib.sha256(f"{date}|{draw_type}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).permutation(1000)


def model_order(s: pd.DataFrame, cfg: dict, date: str, draw_type: str) -> np.ndarray:
    """Refit the positional model on all known draws and rank the target draw."""
    placeholder = pd.DataFrame([{"date": date, "draw_type": draw_type, "number": "000", "n": 0, "d1": 0, "d2": 0,
                                 "d3": 0, "year": c.dataset_year(pd.Series([date]))[0]}])
    full = pd.concat([s, placeholder], ignore_index=True)  # row -1 only contributes its own date/session
    X = F.select(F.build_blocks(full), cfg)
    tr = np.arange(len(s)) >= 30
    Y = s[["d1", "d2", "d3"]].to_numpy()
    p, _ = T.fit_predict_positional(X.iloc[: len(s)][tr], Y[tr], X.iloc[[len(s)]], cfg)
    return c.rank_order(p, seed=T.SEED)[0]


def main() -> None:
    s = c.load_stream()
    scored = live.score(LOG, s)
    status = c.read_json(OUT / "status.json") if (OUT / "status.json").exists() else None
    date, draw_type = c.next_draw(s)
    log = live.load(LOG)
    rows = []
    if status:
        champ = status["champion"]
        if not live.already(log, date, draw_type, "champion"):
            order = (seeded_random_order(date, draw_type) if champ["kind"] == "uniform"
                     else model_order(s, champ["config"], date, draw_type))
            rows.append(live.make_row(date, draw_type, "champion", champ["version"], order))
        if not live.already(log, date, draw_type, "best_candidate"):
            order = model_order(s, status["best_config"], date, draw_type)
            rows.append(live.make_row(date, draw_type, "best_candidate", f"trial{status['best_trial']}", order))
    live.append(LOG, rows)
    log = live.load(LOG)
    upcoming = log[(log.draw_date == date) & (log.draw_type == draw_type)]
    c.write_json(OUT / "live" / "latest.json", {
        "updated_utc": c.utcnow(), "next_draw": {"date": date, "draw_type": draw_type,
                                                  "draw_time_utc": c.draw_time_utc(date, draw_type).isoformat()},
        "predictions": {r.kind: {"model_version": r.model_version, "created_utc": r.created_utc,
                                 "made_before_draw": r.made_before_draw, "top100": r.top100.split(),
                                 "ranking_sha256": r.ranking_sha256} for r in upcoming.itertuples()},
        "scorecard": live.summary(LOG),
        "champion": status["champion"] if status else None,
    })
    print(f"scored {scored}; predicted {len(rows)} ranking(s) for {date} {draw_type}")


if __name__ == "__main__":
    main()
