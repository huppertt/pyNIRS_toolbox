"""Diagnostic-only matched ablations of first-level AR-IRLS null inference.

No alternative covariance, filter, or degrees of freedom here changes production.
Recordings, not correlated channels, are the independent Monte Carlo units.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.signal import fftconvolve, lfilter
from scipy.stats import gamma, t

from pyBrainAnalyzIR.dataclasses.statistics import _ar_irls_joint_covariance
from pyBrainAnalyzIR.math.glm import ar_irls as estimator
from pyBrainAnalyzIR.math.glm.solve import _fit_ar_irls, _satterthwaite_dof
from pyBrainAnalyzIR.testing.covariance_validation import bootstrap_interval
from pyBrainAnalyzIR.testing.matlab_comparison import find_matlab_executable


MOTION = ("none", "impulses", "shifts", "both")


def simulate(rng, n=360):
    """Same noise/task laws as the previous study, with matched motion ablations."""
    events = np.zeros(n)
    for onset in (12, 38, 67, 94, 122, 151):
        events[2 * onset:2 * onset + 4] = 1
    htime = np.arange(0, 30, .5)
    task = fftconvolve(events, gamma.pdf(htime, 6) - gamma.pdf(htime, 16) / 6)[:n]
    task /= task.max()
    design = np.column_stack([np.ones(n), task, np.roll(task, 24)])
    innovations = rng.normal(size=(n + 100, 8))
    common = rng.normal(size=(n + 100, 1))
    innovations = np.sqrt(.65) * innovations + np.sqrt(.35) * common
    noise = lfilter([1], [1, -.65], innovations, axis=0)[100:]
    noise *= np.sqrt(1 - .65**2) * .25 * np.linspace(.6, 1.4, 8)
    impulses, shifts = np.zeros_like(noise), np.zeros_like(noise)
    for channel in range(8):
        indices = rng.choice(n, 5, replace=False)
        impulses[indices, channel] = rng.normal(0, 2, 5)
        start = int(rng.integers(30, n - 20))
        shifts[start:start + 8, channel] = rng.normal(0, .8)
    return design, {
        "none": 1 + noise, "impulses": 1 + noise + impulses,
        "shifts": 1 + noise + shifts, "both": 1 + noise + impulses + shifts,
    }


def traced_fit(y, design, pmax, *, ar_estimator="ols"):
    """Capture the filter actually used, without modifying result/fit behavior."""
    filters = []
    original = estimator._ar_fit

    def capture(values, order, *, robust=False):
        coefficients = original(values, order, robust=robust)
        filters.append(np.r_[1., -coefficients[1:]])
        return coefficients

    with patch.object(estimator, "_ar_fit", side_effect=capture):
        result = _fit_ar_irls(y, design, pmax, ar_estimator=ar_estimator)
    whitening = filters[-1]
    residual = estimator._my_filter(whitening, y) - result.model.exog @ result.params
    return result, residual, whitening, len(filters)


def robust_sandwich(design, residual, weights, maxlag):
    """Exploratory bisquare M-estimator covariance, conditional on the AR filter.

    Weight derivatives are reconstructed from bisquare weights. This ignores
    estimated scale/filter uncertainty and is a diagnostic, not a validated fix.
    """
    root = np.sqrt(weights)
    derivative = root * (5 * root - 4)
    bread = np.linalg.pinv(design.T @ (derivative[:, None] * design))
    scores = design * (weights * residual)[:, None]
    meat = scores.T @ scores
    for lag in range(1, maxlag + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (maxlag + 1)) * (cross + cross.T)
    return bread @ meat @ bread.T * len(residual) / (len(residual) - design.shape[1])


def acf(values, lag):
    centered = values - np.mean(values)
    return float(centered[lag:] @ centered[:-lag] / (centered @ centered))


def rows_for_fit(results, residuals, filters, iterations, replicate, motion, method, truth=None):
    n, p = results[0].model.exog.shape
    joint = _ar_irls_joint_covariance(results)
    rows = []
    for channel, result in enumerate(results):
        design = result.model.exog
        residual = residuals[channel]
        own_weighted = np.linalg.pinv(result.weighted_design.T @ result.weighted_design) * result.sigma2
        variances = {
            "legacy_marginal": joint[channel * p + 1, channel * p + 1],
            "individual_cov_params": np.asarray(result.cov_params())[1, 1],
            "individual_bse": float(result.bse.iloc[1] ** 2),
            "own_scale_weighted": own_weighted[1, 1],
            "sandwich_hc0": robust_sandwich(design, residual, result.weights, 0)[1, 1],
            "sandwich_hac8": robust_sandwich(design, residual, result.weights, 8)[1, 1],
            "sandwich_hac30": robust_sandwich(design, residual, result.weights, 30)[1, 1],
        }
        dofs = {
            "n_minus_p": n - p,
            "subtract_ar_order": n - p - result.ar_order,
            "satterthwaite": _satterthwaite_dof(design, result.weights),
            "fixed_30": 30.,
        }
        if not np.isfinite(list(variances.values())).all() or min(variances.values()) <= 0:
            raise ValueError(f"Invalid diagnostic variance: {replicate}, {motion}, {method}, {channel}")
        if not np.isfinite(list(dofs.values())).all() or min(dofs.values()) <= 0:
            raise ValueError(f"Invalid diagnostic dof: {replicate}, {motion}, {method}, {channel}")
        for covariance, variance in variances.items():
            for dof_name, dof in dofs.items():
                if dof_name != "n_minus_p" and covariance != "legacy_marginal":
                    continue
                beta = float(result.params.iloc[1])
                statistic = beta / np.sqrt(variance)
                rows.append({
                    "replicate": replicate, "motion": motion, "method": method,
                    "channel": channel, "covariance": covariance, "dof_method": dof_name,
                    "beta": beta, "variance": float(variance), "dof": float(dof),
                    "expected_beta": float(truth[channel]) if truth is not None else 0.,
                    "is_null": bool(truth[channel] == 0) if truth is not None else True,
                    "reject": float(2 * t.sf(abs(statistic), dof) < .05),
                    "t_squared": statistic**2,
                    "selected_ar_order": result.ar_order,
                    "hit_order_limit": float(result.ar_order == int(method.split("_")[-1]))
                    if method.startswith("estimated_") else 0.,
                    "whitened_acf1": acf(residual, 1),
                    "whitened_acf8": acf(residual, 8),
                    "score_acf1": acf(result.weights * residual * design[:, 1], 1),
                    "zero_weight_fraction": float(np.mean(result.weights == 0)),
                    "filter_dc": float(np.sum(filters[channel])),
                    "outer_iterations": iterations[channel],
                })
    return rows


def filtered_fit(y, design, whitening):
    """Diagnostic robust coefficient fit with an externally supplied filter."""
    xw = estimator._my_filter(whitening, design)
    yw = estimator._my_filter(whitening, y)
    beta, weights, residual, scale, bse = estimator._robust_fit(xw, yw)
    names = list(range(design.shape[1]))
    result = estimator.ArIrlsResult(
        pd.Series(beta, index=names),
        pd.DataFrame(np.linalg.pinv(xw.T @ xw) * scale**2),
        pd.Series(bse, index=names), weights, xw, weights * residual,
        weights[:, None] * xw, weights * residual, scale**2, len(whitening) - 1,
        np.ones(len(y), dtype=bool),
    )
    return result, residual, whitening, 1


def oracle_fit(y, design):
    """Known background AR(1) filter; motion itself is not assumed to be AR(1)."""
    return filtered_fit(y, design, np.array([1., -.65]))


def summarize(rows, seed):
    frame = pd.DataFrame(rows)
    keys = ["motion", "method", "covariance", "dof_method"]
    rng = np.random.default_rng(seed)
    summaries = []
    for key, group in frame.groupby(keys):
        rates = group[group.is_null].groupby("replicate").reject.mean()
        sensitivity = group[~group.is_null].groupby("replicate").reject.mean()
        # Center each channel separately so between-channel mean differences
        # cannot masquerade as sampling variance.
        empirical = group.groupby("channel").beta.var(ddof=1).mean()
        summaries.append({
            **dict(zip(keys, key)),
            "raw_fpr": bootstrap_interval(rates.to_numpy(), rng, bounded=True),
            "raw_sensitivity": bootstrap_interval(sensitivity.to_numpy(), rng, bounded=True)
            if len(sensitivity) else None,
            "beta_bias": float((group.beta - group.expected_beta).mean()),
            "empirical_beta_variance": float(empirical),
            "mean_estimated_variance": float(group.variance.mean()),
            "empirical_to_estimated_variance": float(empirical / group.variance.mean()),
            **{name: float(group[name].mean()) for name in (
                "dof", "t_squared", "selected_ar_order", "hit_order_limit",
                "whitened_acf1", "whitened_acf8", "score_acf1",
                "zero_weight_fraction", "filter_dc", "outer_iterations",
            )},
        })
    return summaries


def run(replicates=400, orders=(1, 8, 30), seed=20261032, *, filter_transfer_only=False,
        ar_estimator="ols", active_effects=False):
    if replicates < 2:
        raise ValueError("At least two independent recordings are required.")
    if not orders or any(order < 1 for order in orders):
        raise ValueError("AR order limits must be positive.")
    rows = []
    for replicate in range(replicates):
        design, signals = simulate(np.random.default_rng(np.random.SeedSequence([seed, replicate])))
        clean_filters = [
            traced_fit(signals["none"][:, channel], design, 8)[2] for channel in range(8)
        ] if filter_transfer_only else []
        truth = np.r_[np.full(4, .3), np.zeros(4)] if active_effects else np.zeros(8)
        for motion, values in signals.items():
            values = values + design[:, 1, None] * truth + .5 * design[:, 2, None] * truth
            methods = ["clean_filter8"] if filter_transfer_only else [
                f"estimated_{order}" for order in orders
            ] + ["known_ar1"]
            for method in methods:
                fits = []
                for channel in range(values.shape[1]):
                    if filter_transfer_only:
                        fitted = filtered_fit(values[:, channel], design, clean_filters[channel])
                    elif method.startswith("estimated_"):
                        fitted = traced_fit(values[:, channel], design, int(method.split("_")[-1]),
                                            ar_estimator=ar_estimator)
                    else:
                        fitted = oracle_fit(values[:, channel], design)
                    fits.append(fitted)
                results, residuals, filters, iterations = zip(*fits)
                rows.extend(rows_for_fit(results, residuals, filters, iterations,
                                         replicate, motion, method, truth))
        if (replicate + 1) % 20 == 0:
            print(f"{replicate + 1}/{replicates} independent matched recordings", flush=True)
    return {
        "configuration": {
            "replicates": replicates, "seed": seed, "ar_order_limits": list(orders),
            "background_ar": .65, "spatial_correlation": .35, "n": 360,
            "channels_and_chromophores": 8, "conditions": 2, "all_null": not active_effects,
            "active_task_a_beta": .3 if active_effects else 0.,
            "independent_unit": "recording; channels are correlated",
            "motion": "Matched background noise, impulses and eight-sample shifts across ablations.",
            "caveat": "Sandwich estimates condition on estimated AR filter/scale; diagnostic only.",
            "ar_estimator": ar_estimator,
            "scope": "Diagnostic run; does not modify production configuration.",
            "filter_transfer_only": filter_transfer_only,
            "filter_transfer_caveat": "Clean background is known only in the simulation. "
                                      "This is a causal diagnostic, not an available production estimator.",
        },
        "summary": summarize(rows, seed + 1),
        "raw_metrics": rows,
    }


def compare_matlab(output, replicates=100, seed=20261032):
    """Compare local MATLAB math/module covariance on identical null recordings."""
    if replicates < 2:
        raise ValueError("At least two independent recordings are required.")
    executable = find_matlab_executable()
    driver = Path(__file__).parent / "matlab" / "run_raw_null_calibration.m"
    data = np.empty((360, 8, len(MOTION), replicates))
    python_beta = np.empty((8, len(MOTION), replicates))
    python_variance = np.empty_like(python_beta)
    python_order = np.empty_like(python_beta)
    for replicate in range(replicates):
        design, signals = simulate(np.random.default_rng(np.random.SeedSequence([seed, replicate])))
        for index, motion in enumerate(MOTION):
            data[:, :, index, replicate] = signals[motion]
            results = [traced_fit(signals[motion][:, channel], design, 8)[0] for channel in range(8)]
            python_beta[:, index, replicate] = [result.params.iloc[1] for result in results]
            python_order[:, index, replicate] = [result.ar_order for result in results]
            python_variance[:, index, replicate] = np.diag(_ar_irls_joint_covariance(results))[1::3]
    inputs = output.with_suffix(".inputs.mat")
    outputs = output.with_suffix(".matlab.mat")
    savemat(inputs, {
        "data": data, "design": design, "pmax": 8,
        "python_beta": python_beta, "python_variance": python_variance, "python_order": python_order,
    })
    environment = os.environ.copy()
    environment.update(PYBRAIN_RAW_NULL_INPUT=str(inputs.resolve()),
                       PYBRAIN_RAW_NULL_OUTPUT=str(outputs.resolve()))
    directory = str(driver.parent.resolve()).replace("'", "''")
    completed = subprocess.run(
        [executable, "-singleCompThread", "-batch",
         f"addpath('{directory}'); run_raw_null_calibration()"],
        capture_output=True, text=True, timeout=3600, env=environment, check=False,
    )
    output.with_suffix(".stdout.log").write_text(completed.stdout)
    output.with_suffix(".stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"MATLAB failed ({completed.returncode}):\n{completed.stderr}\n{completed.stdout}")
    matlab = loadmat(outputs)
    rng = np.random.default_rng(seed + 2)
    summary = []
    for index, motion in enumerate(MOTION):
        py_stat = np.abs(python_beta[:, index] / np.sqrt(python_variance[:, index]))
        mat_stat = np.abs(matlab["beta"][:, index] / np.sqrt(matlab["variance"][:, index]))
        module_stat = np.abs(matlab["beta"][:, index] / np.sqrt(matlab["module_variance"][:, index]))
        probabilities = {
            "python_legacy_n_minus_p": 2 * t.sf(py_stat, 357),
            "matlab_math_cov_n_minus_p": 2 * t.sf(mat_stat, 357),
            "matlab_math_cov_satterthwaite": 2 * t.sf(mat_stat, matlab["dfe"][index]),
            "matlab_module_cov_satterthwaite": 2 * t.sf(module_stat, matlab["dfe"][index]),
            "matlab_internal_robustfit_p": matlab["internal_p"][:, index],
        }
        summary.append({
            "motion": motion,
            "raw_fpr": {
                name: bootstrap_interval(np.mean(pvalues < .05, axis=0), rng, bounded=True)
                for name, pvalues in probabilities.items()
            },
            "max_absolute_beta_difference": float(np.max(np.abs(matlab["beta"][:, index] - python_beta[:, index]))),
            "max_relative_variance_difference": float(np.max(
                np.abs(matlab["variance"][:, index] / python_variance[:, index] - 1),
            )),
            "selected_order_mismatches": int(np.count_nonzero(matlab["ar_order"][:, index] != python_order[:, index])),
            "matlab_mean_dof": float(matlab["dfe"][index].mean()),
            "matlab_mean_filter_dc": float(matlab["filter_dc"][:, index].mean()),
            "module_indefinite_candidates": int(matlab["module_indefinite"][index].sum()),
        })
    result = {
        "configuration": {"replicates": replicates, "seed": seed, "pmax": 8, "motion": MOTION,
                          "ar_estimator": "ols",
                          "input_policy": "Identical saved concentration-space design and signals.",
                          "matlab_executable": executable,
                          "ar_irls_path": str(matlab["ar_irls_path"].item()),
                          "ar_fit_path": str(matlab["ar_fit_path"].item()),
                          "scope": "Explicit-design nirs.math.ar_irls; final module SVD reproduced exactly, "
                                   "not a separate end-to-end preprocessing pipeline."},
        "summary": summary,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=400)
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 8, 30])
    parser.add_argument("--ar-estimator", choices=["ols", "robust"], default="ols")
    parser.add_argument("--active-effects", action="store_true",
                        help="Inject task A=.3, task B=.15 in four of eight signals; raw sensitivity.")
    parser.add_argument("--filter-transfer-only", action="store_true")
    parser.add_argument("--matlab-reference", action="store_true",
                        help="Compare local MATLAB on identical saved inputs at pmax=8.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.matlab_reference:
        if args.filter_transfer_only or args.active_effects or args.ar_estimator != "ols":
            parser.error("--matlab-reference requires OLS AR, no active effects, and no filter transfer")
        compare_matlab(args.output, args.replicates)
        print(f"Saved MATLAB comparison {args.output}", flush=True)
        return
    result = run(args.replicates, args.orders, filter_transfer_only=args.filter_transfer_only,
                 ar_estimator=args.ar_estimator, active_effects=args.active_effects)
    pd.DataFrame(result.pop("raw_metrics")).to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
