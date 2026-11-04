"""Temporary block-preserving regularization of joint AR-IRLS uncertainty.

This is a numerical approximation, not a new theoretically calibrated estimator.
Projection minimizes Frobenius distance in marginal-whitened coordinates using
Dykstra's alternating projections onto PSD matrices and identity diagonal blocks.
Shrinkage toward block independence bounds the condition number in those same
coordinates to 1000. Marginal blocks and coefficient estimates are not changed.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import block_diag


MAX_BLOCK_CONDITION = 1000.0
PROJECTION_TOLERANCE = 1e-8
MAX_PROJECTION_ITERATIONS = 20000
SUBSTANTIAL_CORRECTION = 0.1


def regularize_joint_covariance(candidate, marginal_blocks):
    """Return a repaired covariance and serializable numerical diagnostics.

    Singular/nonfinite marginal blocks remain unavailable (NaN rows/columns),
    with a warning; no precision is invented for unidentifiable coefficients.
    Nonfinite cross-blocks between otherwise valid signals are explicitly
    approximated by zero before projection and reported.
    """
    candidate = np.asarray(candidate, dtype=float)
    blocks = [np.asarray(block, dtype=float) for block in marginal_blocks]
    sizes = [len(block) for block in blocks]
    if not blocks or any(block.shape != (size, size) or not size for block, size in zip(blocks, sizes)):
        raise ValueError("Marginal covariance blocks must be nonempty square matrices.")
    if candidate.shape != (sum(sizes), sum(sizes)):
        raise ValueError("Joint covariance shape does not match marginal blocks.")
    offsets = np.cumsum([0] + sizes)
    valid, roots, inverse_roots, selected, slices = [], [], [], [], []
    for index, block in enumerate(blocks):
        if not np.isfinite(block).all():
            continue
        symmetric = (block + block.T) / 2
        eigenvalues, vectors = np.linalg.eigh(symmetric)
        tolerance = np.finfo(float).eps * len(block) * max(np.max(np.abs(eigenvalues)), 1e-300)
        if not np.allclose(block, block.T, rtol=1e-10, atol=tolerance) or eigenvalues[0] <= tolerance:
            continue
        start = len(selected)
        selected.extend(range(offsets[index], offsets[index + 1]))
        slices.append(slice(start, len(selected)))
        valid.append(index)
        roots.append((vectors * np.sqrt(eigenvalues)) @ vectors.T)
        inverse_roots.append((vectors * (1 / np.sqrt(eigenvalues))) @ vectors.T)
    diagnostics = {
        "method": "marginal_block_dykstra_shrinkage_v1",
        "diagnostic_scope": "first_level_before_contrasts_and_pooling",
        "max_block_condition": MAX_BLOCK_CONDITION,
        "projection_tolerance": PROJECTION_TOLERANCE,
        "unavailable_blocks": [index for index in range(len(blocks)) if index not in valid],
        "missing_cross_entries": 0,
        "iterations": 0,
        "shrinkage": 0.0,
    }
    if diagnostics["unavailable_blocks"]:
        warnings.warn(
            "AR-IRLS covariance: nonfinite or singular marginal blocks remain unavailable "
            f"(signal indices {diagnostics['unavailable_blocks']}).",
            UserWarning, stacklevel=2,
        )
    output = np.full(candidate.shape, np.nan)
    if not valid:
        return output, diagnostics

    matrix = candidate[np.ix_(selected, selected)].copy()
    marginal_changes = []
    for index, section, inverse_root in zip(valid, slices, inverse_roots):
        original_block = matrix[section, section]
        if np.isfinite(original_block).all():
            standardized = inverse_root @ original_block @ inverse_root
            marginal_changes.append(float(
                np.linalg.norm(standardized - np.eye(len(original_block)), "fro")
                / np.sqrt(len(original_block))
            ))
        matrix[section, section] = blocks[index]
    diagnostics["max_marginal_block_replacement"] = max(marginal_changes, default=0.0)
    missing = ~np.isfinite(matrix) | ~np.isfinite(matrix.T)
    diagnostics["missing_cross_entries"] = int(missing.sum())
    if missing.any():
        warnings.warn(
            "AR-IRLS covariance: unavailable cross-covariances approximated by zero "
            f"({missing.sum()} entries).", UserWarning, stacklevel=2,
        )
        matrix[missing] = 0
    matrix = (matrix + matrix.T) / 2
    root = block_diag(*roots)
    inverse = block_diag(*inverse_roots)
    normalized = inverse @ matrix @ inverse
    normalized = (normalized + normalized.T) / 2
    for section in slices:
        normalized[section, section] = np.eye(section.stop - section.start)
    original = normalized.copy()
    diagnostics["raw_min_block_eigenvalue"] = float(np.linalg.eigvalsh(original)[0])

    correction = np.zeros_like(normalized)
    for iteration in range(1, MAX_PROJECTION_ITERATIONS + 1):
        previous = normalized
        residual = previous - correction
        eigenvalues, vectors = np.linalg.eigh((residual + residual.T) / 2)
        positive = (vectors * np.maximum(eigenvalues, 0)) @ vectors.T
        correction = positive - residual
        normalized = positive.copy()
        for section in slices:
            normalized[section, section] = np.eye(section.stop - section.start)
        normalized = (normalized + normalized.T) / 2
        change = np.linalg.norm(normalized - previous, "fro") / max(1, np.linalg.norm(normalized, "fro"))
        if change <= PROJECTION_TOLERANCE and np.linalg.eigvalsh(normalized)[0] >= -PROJECTION_TOLERANCE:
            break
    else:
        raise RuntimeError("Block-preserving AR-IRLS covariance projection did not converge.")
    diagnostics["iterations"] = iteration
    eigenvalues = np.linalg.eigvalsh(normalized)
    low, high = float(eigenvalues[0]), float(eigenvalues[-1])
    excess = high - MAX_BLOCK_CONDITION * low
    if excess > 0:
        shrinkage = excess / (excess + MAX_BLOCK_CONDITION - 1)
        shrinkage = min(1.0, shrinkage + 10 * np.finfo(float).eps)
        normalized = (1 - shrinkage) * normalized + shrinkage * np.eye(len(normalized))
        diagnostics["shrinkage"] = float(shrinkage)
    eigenvalues = np.linalg.eigvalsh(normalized)
    diagnostics["final_min_block_eigenvalue"] = float(eigenvalues[0])
    diagnostics["final_block_condition"] = float(eigenvalues[-1] / eigenvalues[0])
    relative_change = np.linalg.norm(normalized - original, "fro") / max(np.linalg.norm(original, "fro"), 1e-300)
    diagnostics["relative_block_correction"] = float(relative_change)
    repaired = root @ normalized @ root.T
    repaired = (repaired + repaired.T) / 2
    for index, section in zip(valid, slices):
        repaired[section, section] = blocks[index]
    if np.linalg.eigvalsh(repaired)[0] <= 0:
        raise ValueError("Marginal covariance scaling prevents a numerically positive-definite joint covariance.")
    diagnostics["final_condition"] = float(np.linalg.cond(repaired))
    output[np.ix_(selected, selected)] = repaired
    if max(relative_change, diagnostics["max_marginal_block_replacement"]) > SUBSTANTIAL_CORRECTION:
        warnings.warn(
            f"AR-IRLS joint covariance required substantial regularization "
            f"(relative block correction {relative_change:.1%}, shrinkage "
            f"{diagnostics['shrinkage']:.3g}, maximum marginal block replacement "
            f"{diagnostics['max_marginal_block_replacement']:.1%}). Inspect covariance.attrs; "
            "PSD regularization does not establish statistical calibration.",
            UserWarning, stacklevel=2,
        )
    return output, diagnostics
