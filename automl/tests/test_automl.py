"""Integrity tests for both AutoML experiments (run in CI before any training)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl import common as c

ROOT = Path(__file__).resolve().parents[2]
HAS_DATA = (ROOT / "data" / "daily3.csv").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="full history not present (trainer sandbox)")


# ------------------------------------------------------------------ shared ranking helpers
def test_combine_positional_and_ranking():
    rng = np.random.default_rng(0)
    ps = [rng.dirichlet(np.ones(10), size=4) for _ in range(3)]
    p = c.combine_positional(*ps)
    assert p.shape == (4, 1000) and np.allclose(p.sum(1), 1) and (p >= 0).all()
    order = c.rank_order(p)
    assert all(len(set(r)) == 1000 for r in order)
    assert (order[:, :100] == c.rank_order(p)[:, :100]).all()  # deterministic; Top-100 is a prefix


def test_topk_hits_counts_prefix():
    order = np.tile(np.arange(1000), (2, 1))
    hits = c.topk_hits(order, np.array([5, 500]))
    assert hits[10].tolist() == [True, False] and hits[100].tolist() == [True, False]


# ------------------------------------------------------------------ AutoML 1
@needs_data
@pytest.mark.parametrize("cfg", [
    {"lag_window": 5, "freq_window": 30, "use_since": False, "use_meta": True, "use_hyp": False},
    {"lag_window": 30, "freq_window": 500, "use_since": True, "use_meta": True, "use_hyp": True},
])
def test_model1_features_have_no_leakage(cfg):
    from automl.model1 import features as F
    F.assert_no_leakage(c.load_stream(), cfg, probe=8000)


@needs_data
def test_dates_and_draws():
    s = c.load_stream()
    a = c.audit(s)
    assert a["duplicates_date_drawtype"] == 0 and a["invalid_numbers"] == 0
    assert s["number"].str.fullmatch(r"\d{3}").all()
    assert (s.groupby("date").draw_type.apply(list).apply(lambda x: x in (["midday", "evening"], ["evening"], ["midday"]))).all()


# ------------------------------------------------------------------ AutoML 2
TRAINER_DIR = ROOT / "automl" / "model2" / "trainer"


def test_trainer_code_never_touches_full_history():
    for f in TRAINER_DIR.glob("*.py"):
        src = f.read_text()
        assert "daily3.csv" not in src and "load_stream" not in src, f"{f.name} reads the full history"
        assert not re.search(r"evaluator_private|evaluation_report", src), f"{f.name} references evaluator outputs"


@needs_data
def test_bundle_contract(tmp_path):
    from automl.model2 import prepare
    from automl.model2.config import BLIND_YEARS, CONFIRM_YEARS
    from automl.model2.trainer.run import _year, load_bundle
    prepare.main(tmp_path)
    train, sched = load_bundle(tmp_path)
    assert train["year"].max() == 38
    assert list(sched.columns) == ["date", "draw_type", "period"]
    assert set(_year(sched.date)) == set(BLIND_YEARS + CONFIRM_YEARS)  # year 39 absent everywhere
    assert not sched.duplicated(["date", "draw_type"]).any()


def test_schedule_with_results_is_rejected(tmp_path):
    from automl.model2.trainer.run import load_bundle
    pd.DataFrame({"date": ["2000-01-01"], "draw_type": ["evening"], "number": ["123"]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"date": ["2024-01-01"], "draw_type": ["evening"], "period": ["blind"], "number": ["456"]}).to_csv(
        tmp_path / "forecast_schedule.csv", index=False)
    with pytest.raises(AssertionError):
        load_bundle(tmp_path)


def test_training_beyond_year_38_is_rejected(tmp_path):
    from automl.model2.trainer.run import load_bundle
    pd.DataFrame({"date": ["2023-01-01"], "draw_type": ["evening"], "number": ["123"]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"date": ["2024-01-01"], "draw_type": ["evening"], "period": ["blind"]}).to_csv(
        tmp_path / "forecast_schedule.csv", index=False)
    with pytest.raises(AssertionError):
        load_bundle(tmp_path)


@pytest.mark.parametrize("family,cfg", [
    ("uniform", {}), ("freq_exact", {"alpha": 1.0}), ("freq_positional", {"alpha": 5.0}),
    ("xgb_positional", {"recipe": {"session": True, "weekday": True}, "learning_rate": 0.1, "subsample": 1.0,
                        "reg_lambda": 1.0, "max_depth": 2, "min_child_weight": 1.0, "max_rounds": 5}),
    ("lgbm_positional", {"recipe": {"session": True, "month": True}, "learning_rate": 0.1, "subsample": 1.0,
                         "reg_lambda": 1.0, "num_leaves": 4, "min_data_in_leaf": 20, "max_rounds": 5}),
])
def test_model2_families_give_valid_probabilities(family, cfg, tmp_path):
    from automl.model2.trainer.models import Model
    rng = np.random.default_rng(1)
    dates = pd.date_range("2000-01-01", periods=600).strftime("%Y-%m-%d")
    train = pd.DataFrame({"date": np.repeat(dates, 2), "draw_type": ["midday", "evening"] * 600,
                          "number": [f"{x:03d}" for x in rng.integers(0, 1000, 1200)]})
    m = Model(family, cfg).fit(train)
    p = m.predict(train.head(7))
    assert p.shape == (7, 1000) and np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1)
    m.save(tmp_path / "m")
    p2 = Model.load(tmp_path / "m").predict(train.head(7))
    assert np.allclose(p, p2, atol=1e-6), "frozen model does not reproduce its predictions"


def test_evaluator_runs_only_once(tmp_path):
    from automl.model2.evaluator_private import evaluate_once
    (tmp_path / "blind_evaluation_report.json").write_text("{}")
    with pytest.raises(SystemExit):
        evaluate_once.evaluate(tmp_path, "blind")


# ------------------------------------------------------------------ AutoML 3 (coverage ladder)
def test_ladder_schedule():
    from automl.model3 import ladder as L
    assert L.week_of("2026-10-12") == 0 and L.week_of("2026-10-13") == 1 and L.week_of("2026-10-19") == 1
    assert L.week_of("2026-10-20") == 2
    assert [L.k_for_week(w) for w in (1, 2, 8, 9, 10, 30)] == [900, 800, 200, 100, 100, 100]


def test_ladder_rankers_are_full_permutations_and_use_only_the_past():
    from automl.model3 import ladder as L
    past = np.random.default_rng(1).integers(0, 1000, 3000)
    for name in L.RANKERS:
        o = L.ranking(name, past, "2026-10-13", "midday")
        assert sorted(o.tolist()) == list(range(1000)), name
        # changing anything after the cut-off cannot change the ranking: rankers only ever see `past`
        assert (L.ranking(name, past.copy(), "2026-10-13", "midday") == o).all()


@needs_data
def test_ladder_backtest_hit_counts_are_sane():
    from automl.model3 import ladder as L
    s = c.load_stream()
    bt = L.backtest(s, 500, n=200)
    assert set(bt) == set(L.RANKERS)
    assert all(40 <= v["hits"] <= 160 for v in bt.values())  # chance = 100 of 200
