"""Stage 1+2: trainer. Sees only draws <=2024-12-31 and the (date, session) schedule of hidden 2025 cases."""
from __future__ import annotations

import datetime as dt
import json
import sys

import numpy as np

from automl.model4 import core as C


def main(n_trials: int = 60):
    draws = C.long_draws()
    train = draws[draws.date <= C.TRAIN_END]          # nothing from 2025 is ever passed on from here
    sched = C.mask_cases(draws[(draws.date >= C.TEST_START) & (draws.date <= C.TEST_END)], C.TEST_START, C.TEST_END, C.MASK_SEED)[["date", "session"]]
    # validation: train <=2023, 50% of 2024 masked
    val_hist = train[train.date < C.VAL_START]
    val = C.mask_cases(train[train.date >= C.VAL_START], C.VAL_START, C.TRAIN_END, 7)
    rng = np.random.default_rng(42)
    trials = []
    for t in range(n_trials):
        kind = rng.choice(["random", "freq", "weekday", "positional"])
        cfg = {"kind": str(kind)}
        if kind == "random":
            cfg["seed"] = int(rng.integers(1e6))
        else:
            cfg.update(half_life=float(rng.choice([0, 90, 365, 1095, 3650])), by_session=bool(rng.integers(2)))
            if kind == "weekday":
                cfg["off_weight"] = float(rng.choice([0.0, 0.3, 0.7]))
            if kind == "positional":
                cfg["alpha"] = float(rng.choice([0.5, 5, 50]))
        hit, n = C.accuracy(cfg, val_hist, val)
        trials.append({"trial": t, "config": cfg, "val_hits": hit, "val_n": n, "val_accuracy": hit / n})
    best = max(trials, key=lambda r: (r["val_hits"], -r["trial"]))
    champ = best["config"]
    cases = sched.copy()
    cases["pred"] = C.predict(champ, train, cases)
    C.ART.mkdir(parents=True, exist_ok=True)
    cases.assign(date=cases.date.dt.strftime("%Y-%m-%d")).to_csv(C.ART / "predictions.csv", index=False)
    (C.ART / "trials.json").write_text(json.dumps(trials, indent=1))
    manifest = {
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "champion": champ, "champion_val_accuracy": best["val_accuracy"], "validation_n": best["val_n"],
        "trials": n_trials, "train_rows": int(len(train)), "train_end": "2024-12-31", "blind_cases": int(len(cases)),
        "predictions_sha256": C.file_sha(C.ART / "predictions.csv"), "trials_sha256": C.file_sha(C.ART / "trials.json"),
        "code_sha256": C.sha([C.file_sha(C.ROOT / "automl/model4/core.py"), C.file_sha(C.ROOT / "automl/model4/run.py")]),
        "val_best_accuracy_note": "best of all trials on the same 2024 validation cases, so optimistic",
    }
    (C.ART / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
