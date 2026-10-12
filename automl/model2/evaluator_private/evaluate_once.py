"""AutoML 2 — Step 10: the independent blind evaluator. Runs exactly once per period.

Runs in its own job, after the trainer job has committed the frozen predictions. It:
  1. refuses to run if a report for the period already exists (one-time lock)
  2. verifies every frozen file against the manifest hashes (nothing changed after freezing)
  3. verifies the frozen predictions were committed to git BEFORE this evaluation started
  4. builds the answer key itself from the full history (the trainer never had it)
  5. checks alignment and ranking validity, then scores Top-100/50/20/10, log loss and 95% intervals
The result can never feed back into training: no workflow reads this report to retrain.

Run: python -m automl.model2.evaluator_private.evaluate_once <artifacts/automl2> [blind|confirm ...]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from automl import common as c
from automl.model2 import config as K

PERIOD_YEARS = {"blind": K.BLIND_YEARS, "confirm": K.CONFIRM_YEARS}


def committed_utc(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cI", "--", str(path)], capture_output=True, text=True,
                             check=True).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def score_rankings(top100: pd.Series, truth: np.ndarray) -> dict:
    out = {}
    lists = [t.split() for t in top100]
    for k in (100, 50, 20, 10):
        hits = np.array([str(f"{y:03d}") in lst[:k] for lst, y in zip(lists, truth)])
        h, n = int(hits.sum()), len(hits)
        lo, hi = c.wilson(h, n)
        p = stats.binomtest(h, n, k / 1000, alternative="greater").pvalue if n else 1.0
        out[f"top{k}"] = {"hits": h, "n": n, "rate": h / n if n else 0.0, "ci95": [lo, hi], "baseline": k / 1000,
                          "p_value_vs_baseline_one_sided": float(p)}
    return out


def evaluate(art: Path, period: str) -> dict:
    report_path = art / f"{period}_evaluation_report.json"
    if report_path.exists():
        raise SystemExit(f"{report_path.name} already exists — the {period} evaluation runs only once.")
    manifest = c.read_json(art / "manifest.json")
    frozen = art / "frozen"

    # 2. integrity of frozen files
    for name, digest in manifest["sha256"].items():
        if name.startswith("frozen/"):
            actual = c.sha256_file(art / name)
            if actual != digest:
                raise SystemExit(f"hash mismatch for {name}: frozen files changed after the freeze")

    # 3. predictions must already be committed
    eval_started = c.utcnow()
    commit_time = committed_utc(frozen / "blind_predictions.csv")
    if commit_time is None:
        raise SystemExit("frozen predictions are not committed to git; refusing to evaluate")
    if pd.Timestamp(commit_time) >= pd.Timestamp(eval_started):
        raise SystemExit("predictions were committed after the evaluation started")

    # 4. answer key, built here from the full history
    s = c.load_stream()
    ans = s[s.year.isin(PERIOD_YEARS[period])][["date", "draw_type", "n"]]
    preds = pd.read_csv(frozen / "blind_predictions.csv", dtype=str, keep_default_na=False)
    probs = np.load(frozen / "blind_probabilities.npz")["p"].astype(np.float64)
    preds["row"] = np.arange(len(preds))
    preds = preds[preds.period == period]

    # 5. alignment + ranking validity
    if preds.duplicated(["date", "draw_type"]).any():
        raise SystemExit("more than one prediction for a date/draw type")
    for t in preds.top100:
        lst = t.split()
        if len(lst) != 100 or len(set(lst)) != 100:
            raise SystemExit("a Top-100 list does not contain 100 distinct numbers")
    merged = ans.merge(preds, on=["date", "draw_type"], how="left", validate="one_to_one")
    missing = int(merged.top100.isna().sum())
    merged = merged.dropna(subset=["top100"])
    truth = merged.n.to_numpy()
    p = probs[merged.row.astype(int).to_numpy()]
    p = p / p.sum(axis=1, keepdims=True)

    ll = c.log_loss(p, truth)
    report = {
        "period": period, "years": list(PERIOD_YEARS[period]), "evaluated_utc": c.utcnow(),
        "predictions_committed_utc": commit_time, "frozen_utc": manifest["frozen_utc"],
        "draws_scored": int(len(merged)), "draws_without_prediction": missing,
        "champion": {"family": manifest["champion"]["family"], **score_rankings(merged.top100, truth),
                     "log_loss": float(ll.mean()), "uniform_log_loss": float(np.log(1000)),
                     "log_loss_gain_vs_uniform": float(np.log(1000) - ll.mean())},
        "baseline_seeded_random": score_rankings(merged.random_top100, truth),
        "baseline_frequency_years_1_38": score_rankings(merged.frequency_top100, truth),
    }
    years = c.dataset_year(merged.date)
    report["champion_by_year"] = {int(y): score_rankings(merged.top100[years == y], truth[years == y])["top100"]
                                  for y in np.unique(years)}
    t100 = report["champion"]["top100"]
    report["conclusion"] = (
        f"Top-100 hit rate {t100['rate']:.1%} (95% CI {t100['ci95'][0]:.1%}-{t100['ci95'][1]:.1%}) vs the 10% "
        f"fair-draw baseline; one-sided p = {t100['p_value_vs_baseline_one_sided']:.3f}. "
        + ("This is evidence of an edge on blind data — confirm it on a new future period before trusting it."
           if t100["p_value_vs_baseline_one_sided"] < 0.01 and t100["ci95"][0] > 0.10
           else "No established edge over chance on blind data."))
    c.write_json(report_path, report)
    return report


def main(art: Path, periods: list[str]) -> None:
    for period in periods:
        r = evaluate(art, period)
        print(period, r["conclusion"])


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:] or ["blind"])
