"""AutoML 2 experiment contract (fixed before any run)."""
TRAIN_YEARS = tuple(range(1, 39))   # 1..38: training, tuning, calibration, internal validation
EMBARGO_YEARS = (39,)               # never used
BLIND_YEARS = (40, 41)              # strict blind evaluation, once
CONFIRM_YEARS = (42,)               # held back: separate one-time confirmation
CV_YEARS = (34, 35, 36, 37, 38)     # expanding-window folds inside the training zone
SEED = 2026
MAX_TRIALS = 60
MAX_MINUTES = 60
PATIENCE = 25
MIN_DELTA = 0.0005                  # nats/draw counted as a meaningful validation change
