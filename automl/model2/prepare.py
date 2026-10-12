"""AutoML 2 — Step 1-2: normalise the history and build the trainer bundle (the security boundary).

Writes ONLY to the bundle directory:
  train.csv              years 1-38   (date, draw_type, number)
  forecast_schedule.csv  years 40-42  (date, draw_type, period)  -- no results of any kind
  bundle_manifest.json   counts, date ranges, hashes, year definition
Year 39 is embargoed: it appears in neither file. Answers for years 40-42 are never written anywhere by this
script; the evaluator rebuilds them itself in its own job.

Run: python -m automl.model2.prepare <bundle_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

from automl import common as c
from automl.model2.config import BLIND_YEARS, CONFIRM_YEARS, EMBARGO_YEARS, TRAIN_YEARS


def main(bundle: Path) -> None:
    s = c.load_stream()
    audit = c.audit(s)
    bundle.mkdir(parents=True, exist_ok=True)

    train = s[s.year.isin(TRAIN_YEARS)][["date", "draw_type", "number"]]
    sched = s[s.year.isin(BLIND_YEARS + CONFIRM_YEARS)][["date", "draw_type", "year"]].copy()
    sched["period"] = sched["year"].map(lambda y: "blind" if y in BLIND_YEARS else "confirm")
    sched = sched[["date", "draw_type", "period"]]

    train.to_csv(bundle / "train.csv", index=False)
    sched.to_csv(bundle / "forecast_schedule.csv", index=False)
    c.write_json(bundle / "bundle_manifest.json", {
        "created_utc": c.utcnow(), "year_definition": c.YEAR_DEFINITION,
        "train_years": [TRAIN_YEARS[0], TRAIN_YEARS[-1]], "embargo_years": list(EMBARGO_YEARS),
        "blind_years": list(BLIND_YEARS), "confirm_years": list(CONFIRM_YEARS),
        "train_rows": int(len(train)), "train_first": train.date.iloc[0], "train_last": train.date.iloc[-1],
        "schedule_rows": int(len(sched)), "schedule_first": sched.date.iloc[0], "schedule_last": sched.date.iloc[-1],
        "sha256": {"train.csv": c.sha256_file(bundle / "train.csv"),
                   "forecast_schedule.csv": c.sha256_file(bundle / "forecast_schedule.csv")},
        "audit": audit,
    })
    print(f"bundle: {len(train)} train rows (to {train.date.iloc[-1]}), {len(sched)} schedule rows")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "build/trainer_bundle"))
