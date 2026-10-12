"""AutoML 1 — Optuna search, champion/challenger promotion, locked-test report.

Protocol (fixed before any run):
- Rows: every draw with at least 30 earlier draws. Targets: the three digits.
- Optimisation folds: dataset-years 33..37. For fold year Y, fit on all rows with year < Y, predict every draw
  of year Y (refit once per year = "yearly refit").
- Objective: mean log loss of the exact 3-digit number over the 5 folds (uniform = ln 1000 = 6.9078).
  Top-100/50/20/10 hit rates are recorded as secondary outcomes.
- Locked test: years 38..42, same yearly walk-forward, evaluated only for the reported models and NEVER used
  to choose anything.
- Promotion: a challenger replaces the champion only if, on identical fold draws, it improves mean log loss by
  > MIN_DELTA and a one-sided paired t-test gives p < ALPHA against BOTH the champion and the uniform forecast.
  The starting champion is the uniform forecast (no model).
- Stopping: per-run trial budget and time budget; study-level patience (no new best for PATIENCE trials) and a
  lifetime cap end the search for good.

Run: python -m automl.model1.optimize   (env: AUTOML1_TRIALS, AUTOML1_MINUTES)
"""
from __future__ import annotations

import os
import time

import numpy as np
import optuna
import pandas as pd
from scipy import stats

from automl import common as c
from automl.model1 import features as F
from automl.model1 import train as T

OUT = c.ARTIFACTS / "automl1"
STUDY_NAME = "daily3_xgboost"
FOLD_YEARS = (33, 34, 35, 36, 37)
LOCKED_YEARS = (38, 39, 40, 41, 42)
MIN_ROW = 30
PATIENCE = 30
LIFETIME_TRIALS = 300
MIN_DELTA = 0.0005
ALPHA = 0.05


# ---------------------------------------------------------------- data
class Data:
    def __init__(self):
        self.s = c.load_stream()
        self.blocks = F.build_blocks(self.s)
        self.Y = self.s[["d1", "d2", "d3"]].to_numpy()
        self.n = self.s["n"].to_numpy()
        self.year = self.s["year"].to_numpy()
        self.valid = np.arange(len(self.s)) >= MIN_ROW


def walk_forward(D: Data, cfg: dict, years, kind: str = "positional", trial: optuna.Trial | None = None,
                 shuffle_labels: bool = False) -> dict:
    X = F.select(D.blocks, cfg)
    per_draw, truth_all, p_all, years_all = [], [], [], []
    rng = np.random.default_rng(7)
    for i, Y in enumerate(years):
        tr = D.valid & (D.year < Y)
        te = D.year == Y
        if not te.any():
            continue
        Ytr = D.Y[tr].copy()
        if shuffle_labels:  # control: destroy any real relation between features and targets
            Ytr = Ytr[rng.permutation(len(Ytr))]
        if kind == "positional":
            p, _ = T.fit_predict_positional(X[tr], Ytr, X[te], cfg)
        else:
            p, _ = T.fit_predict_direct(X[tr], Ytr[:, 0] * 100 + Ytr[:, 1] * 10 + Ytr[:, 2], X[te], cfg)
        ll = c.log_loss(p, D.n[te])
        per_draw.append(ll)
        truth_all.append(D.n[te])
        p_all.append(p)
        years_all.append(np.full(te.sum(), Y))
        if trial is not None:
            trial.report(float(np.concatenate(per_draw).mean()), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    p = np.vstack(p_all)
    truth = np.concatenate(truth_all)
    yrs = np.concatenate(years_all)
    m = c.metric_block(p, truth)
    m["per_year"] = {int(y): c.metric_block(p[yrs == y], truth[yrs == y]) for y in np.unique(yrs)}
    m["per_draw_log_loss"] = np.concatenate(per_draw)
    return m


def uniform_block(D: Data, years) -> dict:
    te = np.isin(D.year, years)
    p = np.full((te.sum(), 1000), 1e-3)
    m = c.metric_block(p, D.n[te])
    m["per_draw_log_loss"] = np.full(te.sum(), np.log(1000))
    return m


def frequency_block(D: Data, years, alpha: float = 1.0) -> dict:
    """Historical-frequency baseline: Laplace-smoothed counts of each exact number in all earlier draws."""
    lls, ps, truths = [], [], []
    for Y in years:
        tr, te = D.year < Y, D.year == Y
        cnt = np.bincount(D.n[tr], minlength=1000) + alpha
        p = np.tile(cnt / cnt.sum(), (te.sum(), 1))
        ps.append(p); truths.append(D.n[te])
    p, truth = np.vstack(ps), np.concatenate(truths)
    m = c.metric_block(p, truth)
    m["per_draw_log_loss"] = c.log_loss(p, truth)
    return m


def paired_p(challenger: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Mean improvement (reference - challenger, nats/draw) and one-sided paired t-test p-value."""
    diff = reference - challenger
    if np.allclose(diff, 0):
        return 0.0, 1.0
    t, p_two = stats.ttest_1samp(diff, 0.0)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return float(diff.mean()), float(p_one)


# ---------------------------------------------------------------- search
def suggest(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "lag_window": trial.suggest_categorical("lag_window", list(F.LAG_CHOICES)),
        "freq_window": trial.suggest_categorical("freq_window", [0, *F.FREQ_CHOICES]),
        "use_since": trial.suggest_categorical("use_since", [False, True]),
        "use_hyp": trial.suggest_categorical("use_hyp", [False, True]),
        "use_meta": trial.suggest_categorical("use_meta", [False, True]),
    }


def clean(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "per_draw_log_loss"}


def main() -> None:
    n_trials = int(os.environ.get("AUTOML1_TRIALS", 100))
    minutes = float(os.environ.get("AUTOML1_MINUTES", 150))
    OUT.mkdir(parents=True, exist_ok=True)
    started = c.utcnow()
    D = Data()
    cfg_probe = {"lag_window": 30, "freq_window": 500, "use_since": True, "use_meta": True, "use_hyp": True}
    F.assert_no_leakage(D.s, cfg_probe)

    status_path = OUT / "status.json"
    prev = c.read_json(status_path) if status_path.exists() else {}
    storage = f"sqlite:///{(OUT / 'experiments.db').as_posix()}"
    study = optuna.create_study(direction="minimize", storage=storage, study_name=STUDY_NAME, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=len(prev.get("runs", []))),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    def stop_reason() -> str | None:
        done_ = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
        if len(done_) >= LIFETIME_TRIALS:
            return f"lifetime cap of {LIFETIME_TRIALS} trials reached"
        complete = [t for t in done_ if t.state == optuna.trial.TrialState.COMPLETE]
        if complete:
            best_no = study.best_trial.number
            since = sum(1 for t in done_ if t.number > best_no)
            if since >= PATIENCE:
                return f"plateau: {since} trials since the last improvement (patience {PATIENCE})"
        return None

    reason = prev.get("stopped") or stop_reason()
    run_trials = 0
    if not reason:
        t_end = time.time() + minutes * 60

        def objective(trial: optuna.Trial) -> float:
            cfg = suggest(trial)
            t0 = time.time()
            m = walk_forward(D, cfg, FOLD_YEARS, trial=trial)
            for k in (100, 50, 20, 10):
                trial.set_user_attr(f"top{k}", m[f"top{k}"]["rate"])
            trial.set_user_attr("seconds", round(time.time() - t0, 1))
            return m["log_loss"]

        def callback(st: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
            nonlocal reason
            r = stop_reason()
            if r or time.time() > t_end:
                reason = r
                st.stop()

        before = len(study.trials)
        study.optimize(objective, n_trials=n_trials, callbacks=[callback], gc_after_trial=True)
        run_trials = len(study.trials) - before
        reason = reason or stop_reason()

    # ---- trial log
    rows = []
    for t in study.trials:
        rows.append({"number": t.number, "state": t.state.name, "log_loss": t.value,
                     **{f"top{k}": t.user_attrs.get(f"top{k}") for k in (100, 50, 20, 10)},
                     "seconds": t.user_attrs.get("seconds"), **t.params,
                     "finished": t.datetime_complete})
    pd.DataFrame(rows).to_csv(OUT / "trials.csv", index=False)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise SystemExit("no completed trials")
    best = study.best_trial
    best_cfg = dict(best.params)

    # ---- re-evaluate best + references on identical fold draws
    best_val = walk_forward(D, best_cfg, FOLD_YEARS)
    uni_val = uniform_block(D, FOLD_YEARS)
    freq_val = frequency_block(D, FOLD_YEARS)
    control_val = walk_forward(D, best_cfg, FOLD_YEARS, shuffle_labels=True)

    champ_path = OUT / "champion.json"
    champion = c.read_json(champ_path) if champ_path.exists() else {
        "version": "uniform-v0", "kind": "uniform", "config": None, "promoted_utc": started,
        "reason": "starting champion: uniform 1/1,000 forecast (no model)"}
    if champion["kind"] == "uniform":
        champ_val = uni_val
    else:
        champ_val = walk_forward(D, champion["config"], FOLD_YEARS)

    gain_vs_champ, p_vs_champ = paired_p(best_val["per_draw_log_loss"], champ_val["per_draw_log_loss"])
    gain_vs_uni, p_vs_uni = paired_p(best_val["per_draw_log_loss"], uni_val["per_draw_log_loss"])
    promote = (best_cfg != champion.get("config") and gain_vs_champ > MIN_DELTA and p_vs_champ < ALPHA
               and gain_vs_uni > MIN_DELTA and p_vs_uni < ALPHA)
    decision = {
        "challenger_trial": best.number, "challenger_config": best_cfg,
        "gain_vs_champion_nats": gain_vs_champ, "p_vs_champion": p_vs_champ,
        "gain_vs_uniform_nats": gain_vs_uni, "p_vs_uniform": p_vs_uni,
        "rule": f"promote only if gain > {MIN_DELTA} nats/draw and one-sided paired p < {ALPHA} vs champion AND vs uniform",
        "promoted": promote,
    }
    if promote:
        champion = {"version": f"xgb-trial{best.number}", "kind": "positional_xgboost", "config": best_cfg,
                    "promoted_utc": c.utcnow(), "reason": "challenger cleared both practical and statistical thresholds",
                    "data_cutoff": D.s["date"].iloc[-1]}
    c.write_json(champ_path, champion)

    # ---- optional architecture comparison: direct 1,000-class (fixed modest settings, once per run)
    direct = None
    try:
        dcfg = dict(best_cfg, max_depth=min(best_cfg["max_depth"], 3), learning_rate=0.1, max_rounds=40, early_stop=5)
        t0 = time.time()
        dm = walk_forward(D, dcfg, FOLD_YEARS, kind="direct")
        direct = {**clean(dm), "seconds": round(time.time() - t0, 1), "config": dcfg}
    except Exception as e:  # noqa: BLE001 - report, don't fail the run
        direct = {"error": str(e)}

    # ---- locked test (reporting only)
    locked = {"uniform": clean(uniform_block(D, LOCKED_YEARS)),
              "frequency": clean(frequency_block(D, LOCKED_YEARS)),
              "best_candidate": clean(walk_forward(D, best_cfg, LOCKED_YEARS))}
    if champion["kind"] != "uniform":
        locked["champion"] = clean(walk_forward(D, champion["config"], LOCKED_YEARS))

    runs = prev.get("runs", []) + [{"started": started, "finished": c.utcnow(), "trials_this_run": run_trials,
                                    "best_value_after": best.value, "promoted": promote}]
    status = {
        "updated_utc": c.utcnow(), "study": STUDY_NAME, "data_cutoff": D.s["date"].iloc[-1],
        "trials_total": len(study.trials), "trials_complete": len(complete),
        "trials_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "stopped": reason, "patience": PATIENCE, "lifetime_cap": LIFETIME_TRIALS,
        "best_trial": best.number, "best_value": best.value, "best_config": best_cfg,
        "champion": champion, "decision": decision, "runs": runs,
        "protocol": {"fold_years": FOLD_YEARS, "locked_years": LOCKED_YEARS,
                     "year_definition": c.YEAR_DEFINITION, "refit": "yearly within walk-forward; live = every draw",
                     "objective": "mean exact-number log loss (uniform 6.9078)"},
        "audit": c.audit(D.s),
    }
    c.write_json(status_path, status)
    c.write_json(OUT / "report.json", {
        "updated_utc": c.utcnow(),
        "validation": {"best_candidate": clean(best_val), "uniform": clean(uni_val), "frequency": clean(freq_val),
                       "shuffled_label_control": clean(control_val), "champion": clean(champ_val)},
        "direct_1000_class": direct,
        "locked_test": locked,
    })
    print(f"trials this run {run_trials}, total {len(study.trials)}, best {best.value:.5f} "
          f"(uniform {np.log(1000):.5f}); promoted={promote}; stopped={reason}")


if __name__ == "__main__":
    main()
