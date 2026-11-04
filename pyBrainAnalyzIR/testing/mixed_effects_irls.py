"""Experimental original-observation IRLS; does not change MixedEffects.

Covariance blocks and the rows of Y, X and Z must share subject-major order.
The robust step is outside the nonrobust mixed-model solve. Here ``s`` means
uncertainty inflation, not a small precision weight: C_eff = diag(s) C diag(s).
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import warnings

import numpy as np
from scipy.linalg import block_diag, solve_triangular
from scipy.optimize import minimize_scalar

from pyBrainAnalyzIR.math.mixed_effects import calcLogLikelihood, solveLME


@dataclass
class CovarianceIRLSResult:
    """Experimental fit in original beta units.

    ``covariance`` is conditional working-model coefficient covariance, NOT
    validated robust inference. ``inflation`` and ``whitening`` are exactly
    those used for the returned solve, including on iteration-limit exit.
    No Statistics object or GUI inference is created automatically.
    """

    beta: np.ndarray
    gamma: np.ndarray
    covariance: np.ndarray
    residual_variance: float
    theta: float | None
    inflation: np.ndarray
    precision_weights: np.ndarray
    whitening: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    standardized_residuals: np.ndarray
    robust_scale: float
    converged: bool
    history: list[dict[str, float]]


def _solve(x, y, z):
    """Use the existing LME solve with a bounded scalar variance search."""
    theta = None
    if z.shape[1]:
        optimum = minimize_scalar(
            lambda value: calcLogLikelihood(x, y, z, value),
            bounds=(-25., 20.), method="bounded", options={"xatol": 1e-7},
        )
        if not optimum.success or not np.isfinite(optimum.fun):
            raise RuntimeError(f"Experimental LME variance optimization failed: {optimum.message}")
        candidates = [optimum.x, -25., 20.]
        theta = float(min(candidates, key=lambda value: calcLogLikelihood(x, y, z, value)))
        if theta in (-25., 20.):
            warnings.warn(
                f"Experimental LME variance ratio reached search boundary theta={theta:g}.",
                RuntimeWarning, stacklevel=2,
            )
    likelihood, beta, gamma, covariance, variance = solveLME(x, y, z, theta if theta is not None else 0.)
    if (
        not np.isfinite(likelihood) or not np.isfinite(beta).all()
        or not np.isfinite(gamma).all() or not np.isfinite(covariance).all()
        or not np.isfinite(variance) or variance <= 0
    ):
        raise ValueError("Experimental LME returned invalid coefficients or working-model variance.")
    return beta.ravel(), gamma.ravel(), covariance, float(variance), theta


def covariance_inflation_irls(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    covariances: Sequence[np.ndarray],
    *,
    tuning: float = 1.345,
    tolerance: float = 1e-5,
    max_iterations: int = 100,
) -> CovarianceIRLSResult:
    """Test an outer robust loop with subject-wise covariance rewhitening.

    Parameters
    ----------
    y, x, z
        Unwhitened response vector, full fixed-effect design, and random-effect
        design. Rows must be contiguous per subject/recording, in precisely
        the order of its covariance block (channel, condition, chromophore).
        Supply one shared subject-indicator column in Z per subject unless a
        different random design is deliberately being tested. Z may have zero
        columns for a fixed-effects-only check.
    covariances
        Original positive-definite first-level covariance blocks, one per
        subject/recording. Blocks may differ in size and noise level.
    tuning
        Huber cutoff on original conditional residuals divided by their
        ORIGINAL marginal first-level SD and a pooled robust scale. The scale
        is MAD/0.67448975 across all those standardized residuals, not a
        per-subject scale. Precision weight q=min(1, tuning/abs(u)); s=1/sqrt(q).
        Positive Huber weights avoid singular covariance from zero bisquare
        weights. There is no inner robust fit, leverage adjustment, damping,
        or cumulative multiplication into previously inflated covariances.
    tolerance, max_iterations
        Convergence requires the proposed inflation, beta, gamma and log
        variance ratio to stabilize. A cap exit warns and returns
        converged=False, never an apparent successful convergence.

    Notes
    -----
    Every iteration forms W_i = solve(cholesky(S_i C_i S_i.T), I) using LOWER
    Cholesky, stacks W_all = block_diag(W_i), and solves the nonrobust mixed
    model for W_all @ (Y, X, Z). Residuals are Y-X beta-Z gamma in original
    units. All subject/channel/condition rows get their own inflation factor.
    A bounded ML variance-ratio search is used rather than the production
    solver's BFGS search, so this is not an isolated change to weighting order.
    Conditional residuals can mask subject-level outliers; this experiment
    makes no claim to solve that bias or to provide calibrated robust SEs.
    """
    y = np.asarray(y, dtype=float).copy()
    x = np.asarray(x, dtype=float).copy()
    z = np.asarray(z, dtype=float).copy()
    if y.ndim != 1 or x.ndim != 2 or z.ndim != 2:
        raise ValueError("Y must be a vector; X and Z must be two-dimensional.")
    if len(y) != x.shape[0] or len(y) != z.shape[0] or not x.shape[1] or len(y) <= x.shape[1]:
        raise ValueError("Y, X and Z must have matching rows and positive fixed-effect residual degrees of freedom.")
    if not all(np.isfinite(array).all() for array in (y, x, z)):
        raise ValueError("Y, X and Z must contain only finite values.")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Experimental IRLS requires a full-column-rank fixed-effects design.")
    if not np.isfinite(tuning) or tuning <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Huber tuning and convergence tolerance must be finite and positive.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    blocks = [np.asarray(covariance, dtype=float).copy() for covariance in covariances]
    if not blocks or any(c.ndim != 2 or c.shape[0] != c.shape[1] or not c.size for c in blocks):
        raise ValueError("Covariances must be nonempty square subject blocks.")
    if sum(len(c) for c in blocks) != len(y):
        raise ValueError("Covariance blocks must cover every Y/X/Z row exactly once, in matching order.")
    for subject, covariance in enumerate(blocks):
        if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Subject covariance {subject} must be finite and symmetric.")
        values = np.linalg.eigvalsh(covariance)
        if values[0] <= np.finfo(float).eps * len(covariance) * np.max(abs(values)):
            raise ValueError(
                f"Subject covariance {subject} must be numerically positive definite; no flooring applied."
            )
    original_sd = np.concatenate([np.sqrt(np.diag(c)) for c in blocks])
    inflation = np.ones(len(y))
    history = []
    previous = None
    converged = False
    for iteration in range(1, max_iterations + 1):
        whitening_blocks = []
        start = 0
        for covariance in blocks:
            stop = start + len(covariance)
            s = inflation[start:stop]
            effective = s[:, None] * covariance * s[None, :]
            whitening_blocks.append(
                solve_triangular(np.linalg.cholesky(effective), np.eye(len(covariance)), lower=True)
            )
            start = stop
        whitening = block_diag(*whitening_blocks)
        beta, gamma, covariance_beta, variance, theta = _solve(whitening @ x, whitening @ y, whitening @ z)
        fitted = x @ beta + z @ gamma
        residuals = y - fitted
        standardized = residuals / original_sd
        robust_scale = float(np.median(abs(standardized - np.median(standardized))) / .6744897501960817)
        if not np.isfinite(robust_scale) or robust_scale <= 0:
            raise ValueError("Original-space standardized residual MAD is zero or nonfinite; robust weights undefined.")
        proposed = np.sqrt(np.maximum(1., abs(standardized / robust_scale) / tuning))
        if not np.isfinite(proposed).all():
            raise ValueError("Original-space residual update produced nonfinite covariance inflation.")
        weight_change = float(np.max(abs(proposed - inflation) / (1 + abs(inflation))))
        state = np.concatenate([beta, gamma, [] if theta is None else [theta]])
        parameter_change = (
            float(np.max(abs(state - previous) / (1 + abs(previous)))) if previous is not None else np.inf
        )
        history.append(dict(
            iteration=float(iteration), inflation_change=weight_change,
            parameter_change=parameter_change, robust_scale=robust_scale,
            maximum_inflation=float(inflation.max()),
        ))
        if previous is not None and max(weight_change, parameter_change) <= tolerance:
            converged = True
            break
        if iteration < max_iterations:
            inflation = proposed
            previous = state
    if not converged:
        warnings.warn(
            f"Covariance-inflation IRLS did not converge after {max_iterations} iterations "
            f"(inflation change={weight_change:.3g}, parameter change={parameter_change:.3g}).",
            RuntimeWarning, stacklevel=2,
        )
    return CovarianceIRLSResult(
        beta=beta, gamma=gamma, covariance=covariance_beta, residual_variance=variance, theta=theta,
        inflation=inflation, precision_weights=1 / inflation ** 2, whitening=whitening,
        fitted=fitted, residuals=residuals, standardized_residuals=standardized,
        robust_scale=robust_scale, converged=converged, history=history,
    )
