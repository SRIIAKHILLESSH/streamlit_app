"""AutoML 2 trainer: baselines -> Optuna search inside years 1-38 -> freeze champion -> blind predictions.

This module reads ONLY the trainer bundle (train.csv + forecast_schedule.csv). It never imports the shared
history loader or opens the repository's data folder. In CI it runs on a runner whose checkout excludes data/.

Run: python -m automl.model2.trainer.run <bundle_dir> <out_dir>
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from automl.common import ALL, log_loss, metric_block, rank_order, sha256_file, utcnow, write_json
from automl.model2 import config as K
from automl.model2.trainer.models import Model

FAMILIES = ("freq_exact", "freq_positional", "xgb_positional", "lgbm_positional", "xgb_direct")


def load_bundle(bundle: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(bundle / "train.csv", dtype=str, keep_default_na=False)
    sched = pd.read_csv(bundle / "forecast_schedule.csv", dtype=str, keep_default_na=False)
    # integrity: exact columns, no targets in the schedule, training ends before the embargo year
    assert list(train.columns) == ["date", "draw_type", "number"], train.columns
    assert list(sched.columns) == ["date", "draw_type", "period"], f"unexpected schedule columns {list(sched.columns)}"
    train["year"] = _year(train["date"])
    assert train["year"].max() <= K.TRAIN_YEARS[-1], "training data reaches beyond year 38"
    assert not sched.duplicated(["date", "draw_type"]).any(), "duplicate schedule rows"
    sy = _year(sched["date"])
    assert set(np.unique(sy)) <= set(K.BLIND_YEARS + K.CONFIRM_YEARS), "schedule outside years 40-42"
    return train, sched


def _year(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    return ((d.dt.year - 1984) - (d.dt.month < 9).astype(int) + 1).to_numpy()


def cv(train: pd.DataFrame, family: str, cfg: dict, trial: optuna.Trial | None = None) -> dict:
    ll, ps, truth = [], [], []
    for i, Y in enumerate(K.CV_YEARS):
        tr, te = train[train.year < Y], train[train.year == Y]
        p = Model(family, cfg).fit(tr).predict(te)
        t = te["number"].astype(int).to_numpy()
        ll.append(log_loss(p, t)); ps.append(p); truth.append(t)
        if trial is not None:
            trial.report(float(np.concatenate(ll).mean()), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    m = metric_block(np.vstack(ps), np.concatenate(truth), seed=K.SEED)
    m["per_year_log_loss"] = {int(Y): float(x.mean()) for Y, x in zip(K.CV_YEARS, ll)}
    return m


def suggest(trial: optuna.Trial) -> tuple[str, dict]:
    family = trial.suggest_categorical("family", list(FAMILIES))
    if family.startswith("freq"):
        return family, {"alpha": trial.suggest_float("alpha", 0.1, 500.0, log=True)}
    recipe = {g: trial.suggest_categorical(f"use_{g}", [False, True]) for g in
              ("weekday", "month", "season", "day_of_year", "timeline")}
    recipe["session"] = trial.suggest_categorical("use_session", [False, True])
    cfg = {"recipe": recipe,
           "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
           "subsample": trial.suggest_float("subsample", 0.6, 1.0),
           "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True)}
    if family == "lgbm_positional":
        cfg.update(num_leaves=trial.suggest_int("num_leaves", 4, 64, log=True),
                   min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 500, log=True))
    else:
        cfg.update(max_depth=trial.suggest_int("max_depth", 2, 7 if family == "xgb_positional" else 3),
                   min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True))
    return family, cfg


def main(bundle: Path, out: Path) -> None:
    if os.environ.get("AUTOML2_SANDBOX") == "1":
        assert not Path("data").exists(), "trainer sandbox must not contain the data folder"
    started = utcnow()
    train, sched = load_bundle(bundle)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Step 3: baselines first
    baselines = {"uniform": cv(train, "uniform", {}),
                 "freq_exact(alpha=1)": cv(train, "freq_exact", {"alpha": 1.0}),
                 "freq_positional(alpha=1)": cv(train, "freq_positional", {"alpha": 1.0})}

    # ---- Step 6/8: Optuna inside the training zone with finite budgets
    n_trials = int(os.environ.get("AUTOML2_TRIALS", K.MAX_TRIALS))
    minutes = float(os.environ.get("AUTOML2_MINUTES", K.MAX_MINUTES))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=K.SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    t_end = time.time() + minutes * 60
    stop = {"reason": f"trial budget ({n_trials}) used"}

    def objective(trial):
        family, cfg = suggest(trial)
        t0 = time.time()
        m = cv(train, family, cfg, trial)
        trial.set_user_attr("family", family)
        trial.set_user_attr("cfg", cfg)
        for k in (100, 50, 20, 10):
            trial.set_user_attr(f"top{k}", m[f"top{k}"]["rate"])
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        return m["log_loss"]

    def callback(st, tr):
        done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not done:
            return
        best = st.best_trial
        # patience counts trials since the last *meaningful* improvement (> MIN_DELTA)
        vals = [(t.number, t.value) for t in done]
        best_meaningful = vals[0]
        for num, v in vals[1:]:
            if v < best_meaningful[1] - K.MIN_DELTA:
                best_meaningful = (num, v)
        since = sum(1 for t in st.trials if t.number > best_meaningful[0])
        if since >= K.PATIENCE:
            stop["reason"] = f"plateau: {since} trials without a > {K.MIN_DELTA} nats improvement"
            st.stop()
        elif time.time() > t_end:
            stop["reason"] = f"time budget ({minutes:.0f} min) used"
            st.stop()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], gc_after_trial=True)

    trial_log = pd.DataFrame([{"number": t.number, "state": t.state.name, "log_loss": t.value,
                               "family": t.user_attrs.get("family", t.params.get("family")),
                               **{f"top{k}": t.user_attrs.get(f"top{k}") for k in (100, 50, 20, 10)},
                               "seconds": t.user_attrs.get("seconds"), "params": t.params} for t in study.trials])
    trial_log.to_csv(out / "trial_log.csv", index=False)

    best = study.best_trial
    family, cfg = best.user_attrs["family"], best.user_attrs["cfg"]
    champion_cv = cv(train, family, cfg)
    uni_ll = baselines["uniform"]["log_loss"]
    edge = uni_ll - champion_cv["log_loss"]

    # ---- Step 9: freeze — fit on ALL of years 1-38, predict every scheduled draw (40-42), sign everything
    frozen = out / "frozen"
    model = Model(family, cfg).fit(train)
    model_files = model.save(frozen / "model")
    p = model.predict(sched)
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1.0), "invalid probabilities"
    order = rank_order(p, seed=K.SEED)
    for row in order[:5]:
        assert len(set(row)) == 1000

    # frozen baselines for context (seeded random ranking + training-zone frequency)
    rnd = np.stack([np.random.default_rng(int(hashlib.sha256(f"{d}|{t}".encode()).hexdigest()[:8], 16)).permutation(1000)
                    for d, t in zip(sched.date, sched.draw_type)])
    freq_order = rank_order(Model("freq_exact", {"alpha": 1.0}).fit(train).predict(sched[:1]), seed=K.SEED)[0]

    preds = sched.copy()
    preds["top100"] = [" ".join(ALL[r[:100]]) for r in order]
    preds["random_top100"] = [" ".join(ALL[r[:100]]) for r in rnd]
    preds["frequency_top100"] = " ".join(ALL[freq_order[:100]])
    preds.to_csv(frozen / "blind_predictions.csv", index=False)
    np.savez_compressed(frozen / "blind_probabilities.npz", p=p.astype(np.float32))
    np.savez_compressed(frozen / "blind_rankings.npz", order=order.astype(np.int16))

    code_dir = Path(__file__).resolve().parent
    manifest = {
        "frozen_utc": utcnow(), "started_utc": started, "git_commit": os.environ.get("GITHUB_SHA", "local"),
        "seed": K.SEED, "mode": "A — strict 2-year blind: calendar features only, no results from years 39-42",
        "contract": {"train_years": [K.TRAIN_YEARS[0], K.TRAIN_YEARS[-1]], "embargo": list(K.EMBARGO_YEARS),
                     "blind": list(K.BLIND_YEARS), "confirm": list(K.CONFIRM_YEARS), "cv_years": list(K.CV_YEARS)},
        "champion": {"family": family, "config": cfg, "trial": best.number, "cv": champion_cv},
        "baselines_cv": baselines,
        "edge_vs_uniform_cv_nats": edge,
        "conclusion_cv": ("candidate beats uniform in training-zone CV by a meaningful margin" if edge > K.MIN_DELTA
                          else "no established edge over uniform in training-zone CV"),
        "search": {"trials": len(study.trials), "stopped": stop["reason"], "max_trials": n_trials,
                   "max_minutes": minutes, "patience": K.PATIENCE, "min_delta": K.MIN_DELTA},
        "schedule_rows": int(len(sched)),
        "sha256": {
            **{f"bundle/{n}": sha256_file(bundle / n) for n in ("train.csv", "forecast_schedule.csv")},
            **{f"frozen/{f.relative_to(frozen).as_posix()}": sha256_file(f) for f in model_files},
            **{f"frozen/{n}": sha256_file(frozen / n) for n in
               ("blind_predictions.csv", "blind_probabilities.npz", "blind_rankings.npz")},
            **{f"code/{f.name}": sha256_file(f) for f in sorted(code_dir.glob("*.py"))},
        },
    }
    write_json(out / "manifest.json", manifest)
    print(f"champion {family} cv log loss {champion_cv['log_loss']:.5f} (uniform {uni_ll:.5f}); "
          f"{len(study.trials)} trials, stopped: {stop['reason']}; froze {len(sched)} predictions")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
