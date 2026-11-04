"""Truth matching and replicate-level uncertainty for covariance validation."""
import copy
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cedalion")
from pyBrainAnalyzIR.testing.covariance_validation import (  # noqa: E402
    DataSet, MixedEffects, bootstrap_interval, calibration_flags, legacy_floor_for_study, metrics,
    run_study, simulate_subject, summarize,
)

pytestmark = pytest.mark.requires_cedalion


def test_simulation_truth_matches_beta_order_and_legacy_comparison_is_paired():
    records, truth = simulate_subject(np.random.default_rng(42), 4, 12, "correlated_ar")
    legacy, repaired = [rec["stats"] for rec in records]
    np.testing.assert_array_equal(legacy.betas, repaired.betas)
    np.testing.assert_array_equal(legacy.fit_data.data.fitted, repaired.fit_data.data.fitted)
    for stats in (legacy, repaired):
        score = metrics(stats, truth, lambda table: table.Condition == "HRF A")
        shuffled = metrics(stats, truth.sample(frac=1, random_state=5),
                           lambda table: table.Condition == "HRF A")
        assert score == shuffled
        assert score["null_tests"] == 2
        assert score["alternative_tests"] == 6
        assert 0 <= score["auc"] <= 1
    task = np.flatnonzero(legacy.betas.conditions.values == "HRF A")
    legacy.covariance.values[np.ix_(task, task)] = np.ones((8, 8))
    before = copy.deepcopy(legacy.covariance)
    dataset = DataSet([records[0]])
    assert legacy_floor_for_study(dataset) == 1
    assert np.linalg.eigvalsh(legacy.covariance.values[np.ix_(task, task)])[0] > 0
    other = np.flatnonzero(legacy.betas.conditions.values != "HRF A")
    np.testing.assert_array_equal(legacy.covariance.values[np.ix_(other, other)],
                                  before.values[np.ix_(other, other)])


def test_stress_simulation_exercises_projection_not_just_marginal_replacement():
    with pytest.warns(UserWarning, match="substantial regularization"):
        records, truth = simulate_subject(np.random.default_rng(42), 0, 12, "ar_stress")
    legacy, repaired = [rec["stats"] for rec in records]
    np.testing.assert_array_equal(legacy.betas, repaired.betas)
    assert np.linalg.eigvalsh(legacy.covariance)[0] < 0
    assert np.linalg.eigvalsh(repaired.covariance)[0] > 0
    assert repaired.covariance.attrs["relative_block_correction"] > .1
    assert repaired.covariance.attrs["shrinkage"] > 0
    assert repaired.covariance.attrs["final_block_condition"] <= 1000 * (1 + 1e-8)
    assert truth.index.is_unique


def test_summary_retains_pairing_of_independent_replicates():
    rows = []
    for replicate in range(10):
        for variant in ["legacy_floor", "repaired_full"]:
            value = replicate / 10 + (.1 if variant == "repaired_full" else 0)
            rows.append({
                "regime": "iid", "level": "group", "model": "mean", "robust": True,
                "variant": variant, "replicate": replicate,
                **{name: value for name in [
                    "auc", "false_positive_rate", "power", "coverage", "bias", "rmse",
                ]},
            })
    summary, paired = summarize(rows, 42)
    assert len(summary) == 2
    for comparison in paired:
        assert comparison["repaired_minus_legacy"]["mean"] == pytest.approx(.1)
        np.testing.assert_allclose(comparison["repaired_minus_legacy"]["ci95"], [.1, .1])


def test_study_checks_replication_and_balanced_demographics():
    with pytest.raises(ValueError, match="even subject count"):
        run_study(replicates=1, subjects=5)


def test_exploratory_calibration_flags_cannot_mistake_good_auc_for_valid_inference():
    summary = [{
        "regime": "iid", "level": "group", "model": "mean", "robust": True,
        "variant": "repaired_full",
        "metrics": {
            "auc": {"mean": 1., "ci95": [1., 1.], "replicates": 100},
            "false_positive_rate": {"mean": .8, "ci95": [.7, .9], "replicates": 100},
            "coverage": {"mean": .2, "ci95": [.1, .3], "replicates": 100},
        },
    }]
    flags = calibration_flags(summary)
    assert {flag["metric"] for flag in flags} == {"false_positive_rate", "coverage"}


def test_boundary_rates_do_not_have_falsely_zero_monte_carlo_uncertainty():
    rng = np.random.default_rng(42)
    zeros = bootstrap_interval(np.zeros(100), rng, bounded=True)
    ones = bootstrap_interval(np.ones(100), rng, bounded=True)
    assert zeros["ci95"][0] == 0
    assert zeros["ci95"][1] == pytest.approx(1 - .025 ** .01)
    assert ones["ci95"][0] == pytest.approx(.025 ** .01)
    assert ones["ci95"][1] == 1
    assert zeros["ci_method"] == "exact_replicate_boundary_bound"


def test_scoring_fails_instead_of_silently_discarding_unknown_truth():
    records, truth = simulate_subject(np.random.default_rng(42), 0, 12, "iid")
    with pytest.raises(ValueError, match="Nonfinite inference"):
        metrics(records[1]["stats"], pd.Series(dtype=float), lambda table: table.Condition == "HRF A")


def test_robust_conditional_scale_collapse_is_flagged_not_hidden(monkeypatch):
    from tests.test_model_fit_data import make_dataset
    import pyBrainAnalyzIR.math.mixed_effects as mixed_math
    dataset = make_dataset()

    def saturated_fit(x, y, z, robust_flag):
        beta = np.linalg.lstsq(x, y, rcond=None)[0].reshape(-1, 1)
        residual = y.reshape(-1, 1) - x @ beta
        random = np.linalg.lstsq(z, residual, rcond=None)[0]
        covariance = np.eye(len(beta))[:, :, None]
        return beta, random, covariance, np.array([0.]), np.ones((len(y), 1))

    monkeypatch.setattr(mixed_math, "fitlme", saturated_fit)
    job = MixedEffects()
    job.options["RE_formula"] = "subject:cond:channel:chromo"
    with pytest.warns(UserWarning, match="robust covariance scale can collapse"):
        stats = job.run(copy.deepcopy(dataset))["groupstats"]
    assert stats.covariance.attrs["conditional_residual_collapse"]
    assert stats.covariance.attrs["conditional_to_fixed_residual_norm_ratio"] < 1e-4
    job.options["robust"] = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stats = job.run(copy.deepcopy(dataset))["groupstats"]
    assert not any("scale can collapse" in str(item.message) for item in caught)
    assert not stats.covariance.attrs["conditional_residual_collapse"]
