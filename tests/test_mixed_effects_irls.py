"""Verify the experimental outer loop independently of production MixedEffects."""
import copy

import numpy as np
import pytest
from scipy.linalg import block_diag

pytest.importorskip("cedalion")
from pyBrainAnalyzIR.math.mixed_effects import solveLME  # noqa: E402
from pyBrainAnalyzIR.testing.helpers.mixed_effects_irls import covariance_inflation_irls  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


def example(seed=712, contaminated=False):
    rng = np.random.default_rng(seed)
    subjects = 16
    age = np.linspace(-1, 1, subjects)
    x = np.concatenate([
        np.tile(np.eye(4), (subjects, 1)),
        np.repeat(age, 4)[:, None] * np.tile(np.eye(4), (subjects, 1)),
    ], axis=1)
    z = np.repeat(np.eye(subjects), 4, axis=0)
    truth = np.array([1., -.5, 2., .5, .3, -.2, -.1, .2])
    covariances = []
    noise = []
    for _ in range(subjects):
        scales = np.exp(rng.normal(0, .35, 4))
        covariance = np.outer(scales, scales) * (.7 * np.eye(4) + .3)
        covariances.append(covariance)
        noise.extend(rng.multivariate_normal(np.zeros(4), covariance))
    y = x @ truth + z @ rng.normal(0, .3, subjects) + noise
    if contaminated:
        y[0] += 15.
    return y, x, z, covariances


def test_initialization_and_final_state_are_exact_nonrobust_whitened_solves():
    y, x, z, blocks = example()
    before = copy.deepcopy((y, x, z, blocks))
    # A huge Huber cutoff keeps s exactly one and isolates the stacking/solve.
    result = covariance_inflation_irls(y, x, z, blocks, tuning=1e12)
    whitening = block_diag(*[np.linalg.solve(np.linalg.cholesky(c), np.eye(len(c))) for c in blocks])
    assert result.converged
    assert len(result.history) == 2
    np.testing.assert_array_equal(result.inflation, 1.)
    np.testing.assert_array_equal(result.precision_weights, 1.)
    np.testing.assert_allclose(result.whitening, whitening)
    expected = solveLME(whitening @ x, whitening @ y, whitening @ z, result.theta)
    np.testing.assert_allclose(result.beta, expected[1].ravel())
    np.testing.assert_allclose(result.gamma, expected[2].ravel())
    np.testing.assert_allclose(result.covariance, expected[3])
    assert result.residual_variance == pytest.approx(expected[4])
    np.testing.assert_allclose(result.fitted, x @ result.beta + z @ result.gamma)
    np.testing.assert_allclose(result.residuals, y - result.fitted)
    for original, retained in zip(before[:3], (y, x, z)):
        np.testing.assert_array_equal(original, retained)
    for original, retained in zip(before[3], blocks):
        np.testing.assert_array_equal(original, retained)


def test_original_space_update_and_covariance_inflation_match_returned_fit():
    y, x, z, blocks = example(contaminated=True)
    result = covariance_inflation_irls(y, x, z, blocks, max_iterations=200)
    assert result.converged
    assert result.inflation[0] > 1
    assert result.precision_weights[0] < .3
    assert np.unique(np.round(result.inflation[:4], 4)).size > 1
    original_sd = np.sqrt(np.diag(block_diag(*blocks)))
    np.testing.assert_allclose(result.standardized_residuals, result.residuals / original_sd)
    scale = np.median(abs(result.standardized_residuals - np.median(result.standardized_residuals))) / .6744897501960817
    assert result.robust_scale == pytest.approx(scale)
    proposed = np.sqrt(np.maximum(1., abs(result.standardized_residuals / scale) / 1.345))
    assert np.max(abs(proposed - result.inflation) / (1 + result.inflation)) <= 1e-5
    effective = result.inflation[:, None] * block_diag(*blocks) * result.inflation[None, :]
    np.testing.assert_allclose(result.whitening @ effective @ result.whitening.T, np.eye(len(y)), atol=1e-12)
    # Lower-Cholesky convention: W_new = W_original @ diag(1/s), not diag(1/s) @ W_original.
    initial = block_diag(*[np.linalg.inv(np.linalg.cholesky(c)) for c in blocks])
    np.testing.assert_allclose(result.whitening, initial / result.inflation[None, :])
    assert not np.allclose(result.whitening, initial / result.inflation[:, None])
    expected = solveLME(result.whitening @ x, result.whitening @ y, result.whitening @ z, result.theta)
    np.testing.assert_allclose(result.beta, expected[1].ravel())
    np.testing.assert_allclose(result.gamma, expected[2].ravel())


def test_each_iteration_uses_original_covariance_and_unwhitened_residuals(monkeypatch):
    import pyBrainAnalyzIR.testing.helpers.mixed_effects_irls as experiment

    y, x, z, blocks = example(contaminated=True)
    calls = []
    original = experiment._solve

    def capture(xx, yy, zz):
        solved = original(xx, yy, zz)
        calls.append((xx.copy(), yy.copy(), zz.copy(), solved))
        return solved

    monkeypatch.setattr(experiment, "_solve", capture)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = experiment.covariance_inflation_irls(y, x, z, blocks, max_iterations=3)
    inflation = np.ones(len(y))
    for xx, yy, zz, solved in calls:
        beta, gamma = solved[:2]
        effective = inflation[:, None] * block_diag(*blocks) * inflation[None, :]
        whitening = np.linalg.inv(np.linalg.cholesky(effective))
        np.testing.assert_allclose(xx, whitening @ x)
        np.testing.assert_allclose(yy, whitening @ y)
        np.testing.assert_allclose(zz, whitening @ z)
        residual = (y - x @ beta - z @ gamma) / np.sqrt(np.diag(block_diag(*blocks)))
        scale = np.median(abs(residual - np.median(residual))) / .6744897501960817
        inflation = np.sqrt(np.maximum(1., abs(residual / scale) / 1.345))
    np.testing.assert_allclose(result.beta, calls[-1][3][0])
    assert not result.converged
    assert len(result.history) == 3


def test_fixed_effects_only_and_unequal_subject_blocks():
    rng = np.random.default_rng(42)
    blocks = [np.diag([1., 2.]), np.eye(3) * .5, np.eye(4) * 2]
    x = np.column_stack([np.ones(9), np.arange(9)])
    y = x @ [1., .2] + rng.normal(size=9)
    z = np.empty((9, 0))
    result = covariance_inflation_irls(y, x, z, blocks, tuning=1e12)
    assert result.converged
    assert result.theta is None
    assert result.gamma.size == 0
    precision = np.linalg.inv(block_diag(*blocks))
    expected = np.linalg.solve(x.T @ precision @ x, x.T @ precision @ y)
    np.testing.assert_allclose(result.beta, expected)


def test_matching_data_and_covariance_reordering_preserves_original_coefficient_identities():
    y, x, z, blocks = example(contaminated=True)
    first = covariance_inflation_irls(y, x, z, blocks, max_iterations=200)
    order = np.array([2, 3, 0, 1])
    rows = np.arange(len(y)).reshape(-1, 4)[:, order].ravel()
    second = covariance_inflation_irls(
        y[rows], x[rows], z[rows], [c[np.ix_(order, order)] for c in blocks], max_iterations=200,
    )
    assert first.converged and second.converged
    np.testing.assert_allclose(first.beta, second.beta, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(first.gamma, second.gamma, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(first.inflation[rows], second.inflation, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("change, message", [
    ("nonfinite_y", "finite values"), ("nonfinite_x", "finite values"),
    ("rank", "full-column-rank"), ("rows", "matching rows"), ("shape", "two-dimensional"),
    ("block_rows", "cover every"), ("empty", "nonempty square"),
    ("asymmetric", "symmetric"), ("indefinite", "positive definite"),
    ("nonfinite_covariance", "finite and symmetric"),
])
def test_invalid_inputs_are_explicit(change, message):
    y, x, z, blocks = example()
    if change == "nonfinite_y":
        y[0] = np.nan
    elif change == "nonfinite_x":
        x[0, 0] = np.inf
    elif change == "rank":
        x[:, 0] = 0
    elif change == "rows":
        z = z[:-1]
    elif change == "shape":
        x = x.ravel()
    elif change == "block_rows":
        blocks = blocks[:-1]
    elif change == "empty":
        blocks = []
    elif change == "asymmetric":
        blocks[0][0, 1] += 1
    elif change == "indefinite":
        blocks[0][0, 0] = -1
    elif change == "nonfinite_covariance":
        blocks[0][0, 0] = np.nan
    with pytest.raises(ValueError, match=message):
        covariance_inflation_irls(y, x, z, blocks)


@pytest.mark.parametrize("options", [
    {"max_iterations": 0}, {"max_iterations": True}, {"max_iterations": 1.5},
    {"tuning": 0}, {"tuning": np.inf}, {"tolerance": -1}, {"tolerance": np.nan},
])
def test_invalid_options_are_explicit(options):
    with pytest.raises(ValueError):
        covariance_inflation_irls(*example(), **options)


def test_iteration_limit_preserves_last_solved_weights_and_warns():
    y, x, z, blocks = example(contaminated=True)
    with pytest.warns(RuntimeWarning, match="did not converge after 1 iterations"):
        result = covariance_inflation_irls(y, x, z, blocks, max_iterations=1)
    assert not result.converged
    np.testing.assert_array_equal(result.inflation, 1.)
    np.testing.assert_array_equal(result.precision_weights, 1.)


def test_degenerate_residual_scale_is_not_silently_replaced(monkeypatch):
    import pyBrainAnalyzIR.testing.helpers.mixed_effects_irls as experiment

    y, x, z, blocks = example()
    y[:] = 0

    def exact_fit(xx, yy, zz):
        return np.zeros(x.shape[1]), np.zeros(z.shape[1]), np.eye(x.shape[1]), 1., 0.

    monkeypatch.setattr(experiment, "_solve", exact_fit)
    with pytest.raises(ValueError, match="MAD is zero"):
        experiment.covariance_inflation_irls(y, x, z, blocks)
