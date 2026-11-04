"""AR-IRLS GLM fitting with robust AR regression and preserved missing-data grids.

The estimator is a port of MATLAB nirs-toolbox ``nirs.math.ar_irls`` (with
``nirs.math.ar_fit``, ``nirs.math.stepwise`` and ``nirs.math.robustfit``):
forward/backward AR fit with BIC order selection, filter offset correction,
bisquare IRLS with MAD scale, and the same per-channel covariance scale.
The default additionally robustly refits the selected AR regression; ``ols``
AR estimation retains the original MATLAB-compatible estimator.
"""
from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.signal

_TUNE = 4.685
_MAX_OUTER_ITERATIONS = 10
_MAX_ROBUST_AR_ITERATIONS = 500


def _interpolate_missing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Interpolate missing samples for whitening only; excluded samples stay excluded."""
    indices = np.arange(values.size)
    observed = indices[valid]
    samples = values[valid]
    if observed.size == 0:
        raise ValueError("AR-IRLS cannot fit a channel with no valid observations.")
    if observed.size >= 4:
        from scipy.interpolate import CubicSpline

        return CubicSpline(observed, samples, extrapolate=True)(indices)

    filled = np.interp(indices, observed, samples)
    if observed.size > 1:
        left = indices < observed[0]
        right = indices > observed[-1]
        left_slope = (samples[1] - samples[0]) / (observed[1] - observed[0])
        right_slope = (samples[-1] - samples[-2]) / (observed[-1] - observed[-2])
        filled[left] = samples[0] + (indices[left] - observed[0]) * left_slope
        filled[right] = samples[-1] + (indices[right] - observed[-1]) * right_slope
    return filled


class ArIrlsResult:
    """Per-channel AR-IRLS fit exposing the attributes the GLM pipeline uses.

    ``resid`` holds MATLAB's weighted whitened residuals at valid samples.
    ``weighted_design`` and ``weighted_residual`` are full-length, NaN where
    samples were excluded, and feed the MATLAB cross-channel covariance.
    """

    cov_type = "ar_irls"

    def __init__(self, params, covariance, bse, weights, design, residual,
                 weighted_design, weighted_residual, sigma2, order, valid):
        self.params = params
        self._covariance = covariance
        self.bse = bse
        self.weights = weights
        self.model = SimpleNamespace(exog=design, exog_names=list(params.index))
        self.resid = residual
        self.nobs = int(valid.sum())
        self.weighted_design = weighted_design
        self.weighted_residual = weighted_residual
        self.sigma2 = sigma2
        self.ar_order = order
        self.ar_estimator = "ols"

    def cov_params(self):
        return self._covariance.copy()


def _lag_matrix(values, n_lags):
    out = np.zeros((values.size, n_lags))
    for lag in range(1, n_lags + 1):
        out[lag:, lag - 1] = values[: values.size - lag]
    return out


def _stepwise_bic(design, target):
    q, r = np.linalg.qr(design, mode="reduced")
    inverse_r = np.linalg.pinv(r)
    projected = q.T @ target
    n = target.size
    criterion = np.full(design.shape[1], np.nan)
    solutions = []
    for count in range(1, design.shape[1] + 1):
        coefficients = inverse_r[:count, :count] @ projected[:count]
        residual = target - design[:, :count] @ coefficients
        log_likelihood = -n / 2 * np.log(2 * np.pi * np.mean(residual**2)) - n / 2
        criterion[count - 1] = -2 * log_likelihood + count * np.log(n)
        solutions.append(coefficients)
    return solutions[int(np.nanargmin(criterion))]


def _ar_fit(values, pmax, *, robust=False):
    """Forward/backward AR regression, optionally robust after BIC selection.

    BIC retains the legacy OLS order-selection policy. The selected regression
    is refit with leverage-adjusted bisquare IRLS. Rows with extreme lagged
    predictors are excluded using the residual series' median/MAD: residual
    weighting alone cannot protect against motion spikes in the predictors.
    The AR order itself is not robustly selected.
    """
    n = values.size
    pmax = max(min(int(pmax), n - 1), 1)
    forward = _lag_matrix(values, pmax)
    backward = _lag_matrix(values[::-1], pmax)
    design = np.column_stack([np.ones(2 * n), np.vstack([forward, backward])])
    target = np.concatenate([values, values[::-1]])
    keep = ~np.isnan(target) & ~np.isnan(design.sum(axis=1))
    design, target = design[keep], target[keep]
    coefficients = _stepwise_bic(design, target)
    if robust:
        selected = design[:, :len(coefficients)]
        observed = values[np.isfinite(values)]
        center = np.median(observed)
        scale = np.median(np.abs(observed - center)) / 0.6745
        regular = np.ones(len(target), dtype=bool)
        if scale > 0 and len(coefficients) > 1:
            regular = np.all(np.abs(selected[:, 1:] - center) <= _TUNE * scale, axis=1)
        if regular.sum() <= len(coefficients):
            raise ValueError("Too few non-outlying lag rows to estimate robust AR coefficients.")
        coefficients = _robust_fit(
            selected[regular], target[regular], max_iterations=_MAX_ROBUST_AR_ITERATIONS,
        )[0]
        if not np.isfinite(coefficients).all():
            raise ValueError("Robust AR regression produced nonfinite coefficients.")
    return coefficients


def _my_filter(coefficients, values):
    """Filter after removing the first sample, then restore its filtered offset."""
    first = values[0]
    filtered = scipy.signal.lfilter(coefficients, [1.0], values - first, axis=0)
    return filtered + coefficients.sum() * first


def _bisquare(r):
    return (np.abs(r) < 1) * (1 - r**2) ** 2


def _mad_sigma(r, rank):
    ordered = np.sort(np.abs(r))
    return np.median(ordered[max(1, rank) - 1:]) / 0.6745


def _robust_sigma(r, rank, scale, tune, leverage):
    scaled = scale * tune
    n = r.size
    u = r / scaled
    phi = u * _bisquare(u)
    delta = 1e-4
    phi_low = (u - delta) * _bisquare(u - delta)
    phi_high = (u + delta) * _bisquare(u + delta)
    slope = (phi_high - phi_low) / (2 * delta)
    m1 = np.mean(slope)
    m2 = np.sum((1 - leverage) * phi**2) / (n - rank)
    k = 1 + (rank / n) * (1 - m1) / m1
    return k * np.sqrt(m2) * scaled / m1


def _robust_fit(design, target, tune=_TUNE, *, max_iterations=50):
    """MATLAB bisquare regression; robust AR fits allow more iterations.

    The coefficient-wise relative convergence criterion is unchanged. Correlated
    lag predictors and near-zero AR coefficients can require more than the
    legacy 50 iterations even when predictions have already stabilized.
    """
    n, p = design.shape
    if n <= p:
        raise ValueError("AR-IRLS requires more valid samples than regressors.")
    q, r, perm = scipy.linalg.qr(design, mode="economic", pivoting=True)
    tolerance = abs(r[0, 0]) * max(n, p) * np.finfo(float).eps
    rank = int(np.sum(np.abs(np.diag(r)) > tolerance))
    perm = perm[:rank]
    r_rank = r[:rank, :rank]
    b = np.zeros(p)
    b[perm] = np.linalg.solve(r_rank, q[:, :rank].T @ target)

    inverse_r = np.linalg.inv(r_rank)
    basis = design[:, perm] @ inverse_r
    leverage = np.minimum(0.9999, np.sum(basis * basis, axis=1))
    adjustment = 1 / np.sqrt(1 - leverage)
    ols_s = np.linalg.norm(target - design @ b) / np.sqrt(n - rank)
    tiny_s = 1e-6 * np.std(target, ddof=1)
    if tiny_s == 0:
        tiny_s = 1.0

    step = np.sqrt(np.finfo(float).eps)
    previous = np.zeros(p)
    weighted_rank = rank
    weights = np.ones(n)
    iteration = 0
    while iteration == 0 or np.any(
        np.abs(b - previous) > step * np.maximum(np.abs(b), np.abs(previous))
    ):
        iteration += 1
        if iteration > max_iterations:
            warnings.warn(
                f"Bisquare regression reached {max_iterations} iterations before coefficient convergence.",
                RuntimeWarning, stacklevel=2,
            )
            break
        adjusted = (target - design @ b) * adjustment
        scale = _mad_sigma(adjusted, weighted_rank)
        weights = _bisquare(adjusted / (max(scale, tiny_s) * tune))
        previous = b.copy()
        root = np.sqrt(weights)
        solution, _, weighted_rank, _ = np.linalg.lstsq(
            design[:, perm] * root[:, None], target * root, rcond=None
        )
        b[perm] = solution

    residual = target - design @ b
    adjusted = residual * adjustment
    mad_scale = _mad_sigma(adjusted, rank)
    if np.all((weights < step) | (weights > 1 - step)):
        included = weights > 1 - step
        robust_s = np.linalg.norm(residual[included]) / np.sqrt(included.sum() - rank)
    else:
        robust_s = _robust_sigma(
            adjusted, rank, max(mad_scale, tiny_s), tune, leverage
        )
    sigma = max(
        robust_s,
        np.sqrt((ols_s**2 * rank**2 + robust_s**2 * n) / (rank**2 + n)),
    )
    standard_error = np.zeros(p)
    standard_error[perm] = np.sqrt(
        np.maximum(np.finfo(float).eps, np.diag(inverse_r @ inverse_r.T) * sigma**2)
    )
    return b, weights, residual, robust_s, standard_error


def ar_irls_GLM(y, x, pmax=40, valid_mask=None, *, ar_estimator="robust"):
    """Fit AR-IRLS with selectable robust or legacy AR coefficient regression.

    ``ar_estimator="robust"`` refits AR coefficients with bisquare regression;
    ``"ols"`` retains the MATLAB-compatible AR coefficient estimator.
    Missing samples are interpolated only to construct the continuous AR whitening
    filter. The robust regression is fit only at observed, finite samples.
    """
    if ar_estimator not in ("robust", "ols"):
        raise ValueError("ar_estimator must be 'robust' or 'ols'.")
    y_values = np.asarray(y, dtype=float).reshape(-1)
    x_values = np.asarray(x, dtype=float)
    if x_values.ndim != 2 or x_values.shape[0] != y_values.size:
        raise ValueError("AR-IRLS requires a 2D design matrix aligned with the data.")

    valid = np.isfinite(y_values) & np.isfinite(x_values).all(axis=1)
    if valid_mask is not None:
        requested = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if requested.size != y_values.size:
            raise ValueError("The quality mask length must match the time series.")
        valid &= requested
    if not valid.any():
        raise ValueError("AR-IRLS cannot fit a channel with no valid observations.")

    names = getattr(x, "columns", None)
    names = list(names) if names is not None else list(range(x_values.shape[1]))
    design = np.where(np.isfinite(x_values), x_values, 0.0)
    signal = np.where(valid, y_values, np.nan)
    whitening_signal = signal if valid.all() else _interpolate_missing(y_values, valid)

    coefficients = np.linalg.pinv(design[valid]) @ signal[valid]
    previous = np.full(coefficients.shape, 1e6)
    iteration = 0
    while (
        np.linalg.norm(coefficients - previous) / np.linalg.norm(previous) > 1e-2
        and iteration < _MAX_OUTER_ITERATIONS
    ):
        previous = coefficients
        ar_coefficients = _ar_fit(
            signal - design @ coefficients, np.ceil(pmax), robust=ar_estimator == "robust",
        )
        whitening = np.concatenate([[1.0], -ar_coefficients[1:]])
        filtered_design = _my_filter(whitening, design)
        filtered_signal = _my_filter(whitening, whitening_signal)
        coefficients, weights, residual, robust_s, standard_error = _robust_fit(
            filtered_design[valid], filtered_signal[valid]
        )
        iteration += 1

    design_valid = filtered_design[valid]
    weighted_design = np.full(design.shape, np.nan)
    weighted_design[valid] = weights[:, None] * design_valid
    weighted_residual = np.full(y_values.size, np.nan)
    weighted_residual[valid] = residual * weights
    covariance = pd.DataFrame(
        np.linalg.pinv(design_valid.T @ design_valid) * robust_s**2,
        index=names, columns=names,
    )
    result = ArIrlsResult(
        pd.Series(coefficients, index=names),
        covariance,
        pd.Series(standard_error, index=names),
        weights,
        design_valid,
        weighted_residual[valid],
        weighted_design,
        weighted_residual,
        robust_s**2,
        len(ar_coefficients) - 1,
        valid,
    )
    result.ar_estimator = ar_estimator
    result.ar_coefficients = ar_coefficients.copy()
    result.ar_filter = whitening.copy()
    return result
