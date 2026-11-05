"""Matched controls for the diagnostic-only first-level calibration study."""
import numpy as np
import pytest

pytest.importorskip("cedalion")
from pyBrainAnalyzIR.testing.helpers.raw_null_calibration import (  # noqa: E402
    compare_matlab, filtered_fit, oracle_fit, robust_sandwich, rows_for_fit,
    run, simulate, summarize, traced_fit,
)

pytestmark = pytest.mark.requires_cedalion


def test_motion_ablations_share_exact_background_noise_and_design():
    design, signals = simulate(np.random.default_rng(42))
    np.testing.assert_allclose(
        signals["both"] - signals["none"],
        signals["impulses"] + signals["shifts"] - 2 * signals["none"], atol=1e-14,
    )
    assert design.shape == (360, 3)
    assert all(value.shape == (360, 8) for value in signals.values())
    assert np.sum(signals["impulses"] != signals["none"]) == 5 * 8
    assert np.sum(signals["shifts"] != signals["none"]) == 8 * 8


def test_tracing_reconstructs_actual_whitened_residual_and_preserves_fit():
    design, signals = simulate(np.random.default_rng(42))
    result, residual, whitening, iterations = traced_fit(signals["both"][:, 0], design, 8)
    np.testing.assert_allclose(result.resid, result.weights * residual, atol=1e-12)
    assert len(whitening) == result.ar_order + 1
    assert 1 <= iterations <= 10
    assert np.isfinite(robust_sandwich(result.model.exog, residual, result.weights, 8)).all()
    repeated = filtered_fit(signals["both"][:, 0], design, whitening)
    np.testing.assert_allclose(result.params, repeated[0].params, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.weights, repeated[0].weights, rtol=0, atol=1e-12)


def test_diagnostic_scoring_separates_variance_and_dof_and_clusters_recordings():
    rows = []
    for replicate in range(2):
        design, signals = simulate(np.random.default_rng(replicate))
        fits = [oracle_fit(signals["none"][:, channel], design) for channel in range(8)]
        results, residuals, filters, iterations = zip(*fits)
        rows.extend(rows_for_fit(results, residuals, filters, iterations,
                                 replicate, "none", "known_ar1"))
    assert len(rows) == 2 * 8 * 10
    summary = summarize(rows, 42)
    assert len(summary) == 10
    assert all(entry["raw_fpr"]["replicates"] == 2 for entry in summary)
    assert all(entry["empirical_beta_variance"] > 0 for entry in summary)
    assert all(entry["raw_sensitivity"] is None for entry in summary)
    with pytest.raises(ValueError, match="two independent"):
        run(1)
    with pytest.raises(ValueError, match="two independent"):
        compare_matlab(None, replicates=1)


def test_sandwich_reduces_to_ols_hc0_for_unit_weights():
    rng = np.random.default_rng(42)
    design = np.column_stack([np.ones(100), rng.normal(size=100)])
    residual = rng.normal(size=100)
    bread = np.linalg.inv(design.T @ design)
    scores = design * residual[:, None]
    expected = bread @ scores.T @ scores @ bread * 100 / 98
    np.testing.assert_allclose(robust_sandwich(design, residual, np.ones(100), 0), expected)


def test_active_experiment_scores_raw_fpr_and_sensitivity_separately():
    result = run(2, orders=(1,), ar_estimator="robust", active_effects=True)
    assert not result["configuration"]["all_null"]
    assert result["configuration"]["ar_estimator"] == "robust"
    assert all(entry["raw_fpr"]["replicates"] == 2 for entry in result["summary"])
    assert all(entry["raw_sensitivity"]["replicates"] == 2 for entry in result["summary"])
    assert {row["expected_beta"] for row in result["raw_metrics"]} == {0., .3}
