"""Stage 3: one-time evaluator. Per-case True/False is computed internally; only aggregates are published.
Refuses to run twice or if the frozen files changed. Answers are exposed only if accuracy >= 80%."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
from scipy.stats import binomtest

from automl.model4 import core as C
from lib.stats import wilson


def main():
    rep = C.ART / "blind_evaluation_report.json"
    if rep.exists():
        raise SystemExit("report exists: blind test already used; refusing to re-run")
    m = json.loads((C.ART / "manifest.json").read_text())
    assert C.file_sha(C.ART / "predictions.csv") == m["predictions_sha256"], "predictions changed after freeze"
    assert dt.datetime.fromisoformat(m["frozen_utc"]) <= dt.datetime.now(dt.timezone.utc)
    draws = C.long_draws()
    test = draws[(draws.date >= C.TEST_START) & (draws.date <= C.TEST_END)]
    key = C.mask_cases(test, C.TEST_START, C.TEST_END, C.MASK_SEED)
    pred = pd.read_csv(C.ART / "predictions.csv", parse_dates=["date"])
    j = key.merge(pred, on=["date", "session"], validate="one_to_one")
    assert len(j) == len(key) == len(pred) == m["blind_cases"]
    ok = (j.n == j.pred)                                 # True/False per case, never published
    hits, n = int(ok.sum()), len(j)
    acc = hits / n
    lo, hi = wilson(hits, n)
    out = {"evaluated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "cases": n, "correct": hits,
           "accuracy": acc, "ci95": [lo, hi], "chance": 0.001, "target": C.TARGET,
           "p_value_vs_chance": float(binomtest(hits, n, 0.001, alternative="greater").pvalue),
           "target_met": acc >= C.TARGET, "answers_unlocked": acc >= C.TARGET,
           "per_session": {s: {"cases": int((j.session == s).sum()), "correct": int(ok[j.session == s].sum())} for s in ("midday", "evening")}}
    if out["answers_unlocked"]:
        out["answers"] = j.assign(date=j.date.dt.strftime("%Y-%m-%d"))[["date", "session", "n"]].to_dict("records")
    rep.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "answers"}, indent=1))


if __name__ == "__main__":
    main()
