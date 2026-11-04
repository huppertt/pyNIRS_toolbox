"""Numerical contracts of the temporary joint covariance approximation."""
import warnings
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy.linalg import block_diag

pytest.importorskip("cedalion")
from pyBrainAnalyzIR.dataclasses.statistics import Statistics, _ar_irls_joint_covariance  # noqa: E402
from pyBrainAnalyzIR.math.glm.covariance import (  # noqa: E402
    MAX_BLOCK_CONDITION, regularize_joint_covariance,
)
from pyBrainAnalyzIR.math.glm.solve import fit  # noqa: E402
from tests.test_model_fit_data import small_model  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


def test_differing_design_counterexample_is_repaired_and_blocks_preserved():
    x = np.arange(1., 6.)[:, None]
    residual = np.arange(-2., 3.)
    results = [
        SimpleNamespace(weighted_design=x, weighted_residual=residual, sigma2=1.),
        SimpleNamespace(weighted_design=x * np.array([1., 2., 1., 2., 1.])[:, None],
                        weighted_residual=residual, sigma2=1.),
    ]
    candidate = _ar_irls_joint_covariance(results)
    assert np.linalg.eigvalsh(candidate)[0] < -1e-4
    blocks = [candidate[i:i + 1, i:i + 1] for i in range(2)]
    repaired, info = regularize_joint_covariance(candidate, blocks)
    assert np.linalg.eigvalsh(repaired)[0] > 0
    np.testing.assert_array_equal(np.diag(repaired), np.diag(candidate))
    assert 0 < info["shrinkage"] < 1
    assert info["final_block_condition"] <= MAX_BLOCK_CONDITION * (1 + 1e-8)


@pytest.mark.parametrize("seed", range(6))
def test_dense_indefinite_candidate_preserves_full_marginal_blocks(seed):
    rng = np.random.default_rng(seed)
    blocks = []
    for scale in [0.01, 1, 100]:
        a = rng.normal(size=(3, 3))
        blocks.append((a @ a.T + np.eye(3)) * scale)
    scale = np.sqrt(np.diag(block_diag(*blocks)))
    raw = rng.normal(size=(9, 9)) * scale[:, None] * scale[None, :]
    raw = (raw + raw.T) / 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        repaired, info = regularize_joint_covariance(raw, blocks)
    assert any("substantial regularization" in str(item.message) for item in caught)
    np.testing.assert_allclose(repaired, repaired.T, atol=1e-12)
    for index, block in enumerate(blocks):
        np.testing.assert_array_equal(repaired[3 * index:3 * index + 3, 3 * index:3 * index + 3], block)
    assert np.linalg.eigvalsh(repaired)[0] > 0
    assert info["final_block_condition"] <= MAX_BLOCK_CONDITION * (1 + 1e-8)
    contrast = rng.normal(size=(5, 9))
    assert np.linalg.eigvalsh(contrast @ repaired @ contrast.T)[0] > 0
    again, second_info = regularize_joint_covariance(repaired, blocks)
    np.testing.assert_allclose(again, repaired, atol=1e-9, rtol=1e-8)
    assert second_info["relative_block_correction"] < 1e-8


def test_already_valid_joint_matrix_is_unchanged():
    blocks = [np.array([[2., .3], [.3, 1.]]), np.array([[3., -.2], [-.2, 2.]])]
    raw = block_diag(*blocks)
    repaired, info = regularize_joint_covariance(raw, blocks)
    np.testing.assert_allclose(repaired, raw, atol=1e-14)
    assert info["shrinkage"] == 0


def test_units_and_signal_permutation_equivariance():
    raw = np.array([[2., 4., -3.], [4., 3., 1.], [-3., 1., 5.]])
    blocks = [raw[i:i + 1, i:i + 1] for i in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected, _ = regularize_joint_covariance(raw, blocks)
        scale = np.array([.01, 3., 100.])
        converted = raw * scale[:, None] * scale[None, :]
        actual, _ = regularize_joint_covariance(
            converted, [converted[i:i + 1, i:i + 1] for i in range(3)],
        )
        order = np.array([2, 0, 1])
        permuted, _ = regularize_joint_covariance(raw[np.ix_(order, order)], [blocks[i] for i in order])
    np.testing.assert_allclose(actual, expected * scale[:, None] * scale[None, :], rtol=1e-8)
    np.testing.assert_allclose(permuted, expected[np.ix_(order, order)], rtol=1e-8)


def test_projection_nonconvergence_is_an_explicit_failure(monkeypatch):
    import pyBrainAnalyzIR.math.glm.covariance as module
    monkeypatch.setattr(module, "MAX_PROJECTION_ITERATIONS", 1)
    with pytest.raises(RuntimeError, match="did not converge"):
        regularize_joint_covariance(np.array([[1., 5.], [5., 1.]]), [np.eye(1)] * 2)


def test_large_cross_covariances_converge_without_changing_marginals():
    rng = np.random.default_rng(14)
    raw = rng.normal(size=(12, 12)) * 100
    raw = (raw + raw.T) / 2
    np.fill_diagonal(raw, 1)
    with pytest.warns(UserWarning, match="substantial regularization"):
        repaired, info = regularize_joint_covariance(raw, [np.eye(1)] * 12)
    assert info["iterations"] > 2000
    np.testing.assert_array_equal(np.diag(repaired), np.ones(12))
    assert np.linalg.eigvalsh(repaired)[0] > 0
    assert info["final_block_condition"] <= MAX_BLOCK_CONDITION * (1 + 1e-8)


def test_unavailable_marginals_and_cross_entries_are_not_silent():
    raw = np.full((5, 5), np.nan)
    blocks = [np.eye(2), np.ones((2, 2)), np.eye(1)]
    with pytest.warns(UserWarning) as caught:
        repaired, info = regularize_joint_covariance(raw, blocks)
    assert len(caught) == 2
    assert info["unavailable_blocks"] == [1]
    assert np.isnan(repaired[2:4]).all()
    assert np.isnan(repaired[:, 2:4]).all()
    np.testing.assert_allclose(repaired[np.ix_([0, 1, 4], [0, 1, 4])], np.eye(3))


@pytest.mark.parametrize("masked", [False, True])
def test_ar_irls_statistics_preserves_legacy_blocks_and_inference(masked):
    ts, design = small_model()
    mask = xr.ones_like(ts.isel(chromo=0, drop=True), dtype=bool).pint.dequantify()
    if masked:
        mask.values[10:20, 0] = False
        mask.values[30:35, 1] = False
    results = fit(ts, design, noise_model="ar_irls", ar_order=4, quality_mask=mask, verbose=False)
    stats = Statistics(results)
    legacy = _ar_irls_joint_covariance(list(results.values.ravel()))
    for index, result in enumerate(results.values.ravel()):
        block = stats.covariance.values[index * 3:index * 3 + 3, index * 3:index * 3 + 3]
        np.testing.assert_array_equal(block, legacy[index * 3:index * 3 + 3, index * 3:index * 3 + 3])
    assert stats.covariance.attrs["marginal_source"] == "legacy_ar_irls_joint_diagonal_blocks"
    baseline = stats.copy()
    baseline.covariance.values[:] = legacy
    np.testing.assert_array_equal(stats.table(), baseline.table())
    np.testing.assert_array_equal(stats.betas, np.concatenate([r.params for r in results.values.ravel()]))
    assert np.linalg.eigvalsh(stats.covariance)[0] > 0
    assert stats.covariance.attrs["method"] == "marginal_block_dykstra_shrinkage_v1"
    restored = pickle.loads(pickle.dumps(stats))
    xr.testing.assert_identical(restored.covariance, stats.covariance)
    assert np.isfinite(stats.table()["Q-values"]).all()
    stats.ttest(["HRF A-HRF B"])
    baseline.ttest(["HRF A-HRF B"])
    np.testing.assert_array_equal(stats.table(), baseline.table())
    assert np.linalg.eigvalsh(stats.covariance)[0] > 0
    assert stats.covariance.attrs == restored.covariance.attrs


def test_missing_channel_retains_nan_uncertainty_and_other_ar_blocks():
    ts, design = small_model()
    mask = xr.ones_like(ts.isel(chromo=0, drop=True), dtype=bool).pint.dequantify()
    mask.loc[dict(channel="S1D1")] = False
    results = fit(ts, design, noise_model="ar_irls", ar_order=4, quality_mask=mask, verbose=False)
    with pytest.warns(UserWarning, match="remain unavailable"):
        stats = Statistics(results)
    assert np.isnan(stats.covariance.values[:6]).all()
    assert np.linalg.eigvalsh(stats.covariance.values[6:, 6:])[0] > 0
    assert stats.table().loc[stats.table().Channel == "S2D2", "Q-values"].notna().all()


def test_no_shared_samples_uses_reported_cross_block_independence():
    ts, design = small_model()
    mask = xr.ones_like(ts.isel(chromo=0, drop=True), dtype=bool).pint.dequantify()
    mask.values[:48, 0] = False
    mask.values[48:, 1] = False
    results = fit(ts, design, noise_model="ar_irls", ar_order=4, quality_mask=mask, verbose=False)
    with pytest.warns((UserWarning, RuntimeWarning)) as caught:
        stats = Statistics(results)
    assert any("unavailable cross-covariances" in str(item.message) for item in caught)
    assert stats.covariance.attrs["missing_cross_entries"] == 72
    np.testing.assert_array_equal(stats.covariance.values[:6, 6:], np.zeros((6, 6)))
    assert np.linalg.eigvalsh(stats.covariance)[0] > 0
