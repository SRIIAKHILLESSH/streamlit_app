# Daily 3 AutoML experiments

Two separate experiments, built from the two specification PDFs. Both run entirely on GitHub Actions, so nothing depends on a laptop. Results are committed to `artifacts/`, shown on the Streamlit site (pages **AutoML 1** and **AutoML 2**), and available in Telegram via `/predict` and `/status`.

> If Daily 3 draws are fair and independent, no model can beat chance. Both experiments are built to *measure* that honestly, not to manufacture accuracy. A Top-100 hit rate near 10% is a correct result, not a bug.

## AutoML 1: self-improving XGBoost (`automl/model1`)
*Spec: Daily3_Self_Improving_XGBoost_System.pdf*

| Piece | What it does |
|---|---|
| `features.py` | Leakage-safe features for draw *t* from draws < *t* only: lags (5/10/20/30), rolling positional digit frequency (30/100/500; the hundreds digit doubles as the 100-number bucket), time-since-digit (capped at 200), session/weekday/month, and the High-Low counts (900–999 and 000–099 in the last 7/10 draws). `assert_no_leakage` proves that changing future draws never changes past rows. |
| `train.py` | Three 10-class XGBoost models (hundreds/tens/ones) multiplied into 1,000 probabilities, plus a direct 1,000-class `multi:softprob` model for comparison. Early stopping uses a chronological slice. |
| `optimize.py` | Optuna (TPE + median pruner) with persistent SQLite storage (`artifacts/automl1/experiments.db`). Search space: max_depth 2–7, learning_rate 0.01–0.15, subsample 0.6–1, reg_lambda 0.1–20, plus lag/frequency/feature-family choices. Walk-forward folds are dataset-years 33–37 with yearly refits. The objective is exact-number log loss (uniform = 6.9078); Top-100/50/20/10 are secondary. Each run also reports the uniform, frequency and shuffled-label controls. |
| Champion / challenger | The starting champion is the uniform forecast. A challenger is promoted only if it improves log loss by more than 0.0005 nats/draw **and** a one-sided paired t-test gives p < 0.05 against both the champion and uniform, on identical draws. |
| Stopping | 100 trials or 150 minutes per run; patience of 30 trials without a new best; a lifetime cap of 300 trials. |
| Locked test | Dataset-years 38–42, reported for the best candidate and the champion, never used for selection. |
| `predict.py` | Before every draw it saves rankings for the champion (a seeded random ranking while the champion is uniform) and the best candidate (refit on all earlier draws). Each is saved with a timestamp and a SHA-256 hash, append-only, then scored once the result is posted. |

## AutoML 2: strict blind test (`automl/model2`)
*Spec: Daily3_AutoML_Model2_Architecture_AI_Handoff.pdf*

| Contract | Dataset years |
|---|---|
| Train, tune, calibrate, internal CV | 1–38 (to 2022-08-31) |
| Embargo (never used) | 39 |
| Blind evaluation, once | 40–41 |
| Separate one-time confirmation | 42 |

The workflow runs three jobs on three separate machines:
1. **prepare** has the full history and writes `train.csv` (years 1–38) and `forecast_schedule.csv` (years 40–42: date, draw_type and period only).
2. **train** checks out the repo **without `data/`** (asserted) and receives only that bundle. Mode A features are calendar facts only. It establishes baselines first (uniform, seeded random, frequency), then runs an Optuna search over frequency, positional XGBoost, positional LightGBM and direct 1,000-class XGBoost with expanding CV (years 34–38). The search is capped at 60 trials or 60 minutes, with a patience of 25 trials and a 0.0005-nat minimum change. The job then freezes the champion and commits its predictions, all 1,000 probabilities, the rankings and a SHA-256 manifest of data, model, predictions and code.
3. **evaluate** runs `evaluator_private/evaluate_once.py`, which:
   - refuses to run if a report already exists
   - verifies every frozen hash
   - verifies the predictions were committed before it started
   - rebuilds the answer key itself
   - checks for duplicates and alignment
   - reports Top-100/50/20/10 with Wilson 95% intervals, one-sided binomial p-values against the baseline, and log loss, alongside the frozen random and frequency baselines

After the freeze, `prospective.py` pre-registers the frozen model's ranking for every new draw. That becomes the next locked future period.

*Isolation note:* the Daily 3 history is public, so isolation is enforced by process: the trainer machine never receives the files, tests forbid the trainer code from referencing them, and the evaluator is a one-time locked job. *Development note:* while building the evaluator, a 10-trial test build was scored on years 40–42 on a laptop, to check that the code works (8.8% Top-100, no edge). No design choice was changed after it. The official freeze and evaluation are the GitHub Actions run.

## AutoML 3: coverage ladder (`automl/model3/ladder.py`)
Before every draw, a list of K numbers is pre-registered (timestamp + SHA-256, append-only). K starts at 900 in week 1
(Tue 2026-10-13 → Mon 2026-10-19) and drops by 100 each week to 100 (week 9 onward). At the first run of each week the
AutoML back-tests 14 rankers (random, hot/cold windows, overdue, recently drawn, skip-recent, positional digit
frequency) walk-forward on the last 730 draws at that week's K and freezes the best one for the week. Results are
scored against the exact chance rate K/1,000 with Wilson intervals and one-sided binomial p-values. The back-test winner
is the best of 14 on the same draws, so its back-test rate is optimistic; the live weekly rate is the honest number.

## AutoML 4: 2025 blind reconstruction (`automl/model4`)
Owner's benchmark: reconstruct randomly hidden (50%) 2025 midday/evening numbers, exact match, target >= 80%. `run.py` (trainer; sees only
draws <= 2024-12-31 plus the date/session of hidden cases) logs random-search experiments on 2024 validation, freezes one prediction per case with
SHA-256 hashes; `evaluator_private/evaluate_once.py` scores once (refuses re-run), publishes only aggregates, unlocks answers only at >= 80%.
Run once: 0/364 correct (chance predicts 0.36). Do not re-run the evaluator or tune on 2025.

## Workflows (all in `.github/workflows/`)
| Workflow | When | Does |
|---|---|---|
| `automl1-optimize.yml` | Sundays 06:00 UTC, manual, or edit `automl/model1/RUN_REQUEST` | tests → Optuna → champion/challenger → locked test → pre-register |
| `automl-live.yml` | 20:00 and 05:00 UTC daily (between draws) | fetch results → score → pre-register next draw (both models) |
| `automl2-blind.yml` | once, manual or edit `automl/model2/RUN_REQUEST` | prepare → train/freeze → evaluate once (refuses to re-run) |

## Run locally
```
pip install -r automl/requirements.txt
python -m pytest automl/tests -q
AUTOML1_TRIALS=5 python -m automl.model1.optimize
python -m automl.model1.predict
python -m automl.model2.prepare build/bundle
AUTOML2_TRIALS=5 python -m automl.model2.trainer.run build/bundle /tmp/automl2   # never into artifacts/ locally
```
