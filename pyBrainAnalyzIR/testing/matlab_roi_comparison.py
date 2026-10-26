"""Compare MATLAB and Python GLM/ROI processing order on shared simulations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat, savemat

from pyBrainAnalyzIR.testing.matlab_comparison import find_matlab_executable


_MATLAB_DRIVER = Path(__file__).parent / "matlab" / "run_roi_glm_comparison.m"
_MATLAB_AR_IRLS_DRIVER = Path(__file__).parent / "matlab" / "run_roi_ar_irls_comparison.m"
_N_CHANNELS = 6
_N_ROIS = 3
_MAX_CONDITIONS = 2


def run_matlab_python_roi_glm_comparison(
    num_examples: int = 50,
    *,
    seed: int = 13,
    output_dir: str | Path | None = None,
    matlab_executable: str | Path | None = None,
    timeout: float = 1800,
    noise_model: str = "ols",
    ar_order: int = 8,
) -> dict[str, Any]:
    """Compare ROI-before-GLM and GLM-before-ROI in both toolboxes.

    The same 50 or more seeded synthetic time courses, task designs, and
    varying ROI membership/weights are passed to MATLAB and Python. Each
    example has either one or two task conditions. With ``noise_model="ols"``
    (default) the GLM is OLS and ROI pooling uses fixed table weights, so both
    processing orders have the same linear interpretation. With
    ``noise_model="ar_irls"`` MATLAB runs ``nirs.math.ar_irls`` and Python runs
    its AR-IRLS fit (``ar_order`` maximum AR order in both); ROI time courses
    use inverse temporal-variance weights and fitted statistics use
    inverse-coefficient-variance weights (``nirs.util.roiAverage`` with
    ``weighted=true`` in MATLAB). The two processing orders are not expected to
    agree with each other under weighting, only MATLAB with Python for the same
    order. The report includes coefficient and full covariance matrix errors.

    When ``output_dir`` is provided, the shared inputs, MATLAB outputs,
    per-example errors, and aggregate JSON summary are retained there.
    MATLAB is discovered using ``PYBRAIN_MATLAB_EXECUTABLE``, ``PATH``, or the
    standard macOS application location.
    """
    if isinstance(num_examples, bool) or not isinstance(num_examples, int) or num_examples < 50:
        raise ValueError("num_examples must be an integer of at least 50")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if noise_model not in ("ols", "ar_irls"):
        raise ValueError("noise_model must be 'ols' or 'ar_irls'")
    if isinstance(ar_order, bool) or not isinstance(ar_order, int) or ar_order < 1:
        raise ValueError("ar_order must be a positive integer")
    driver = _MATLAB_AR_IRLS_DRIVER if noise_model == "ar_irls" else _MATLAB_DRIVER
    if not driver.is_file():
        raise FileNotFoundError(f"MATLAB ROI comparison driver is missing: {driver}")

    executable = find_matlab_executable(matlab_executable)
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="matlab-roi-glm-") as temporary:
            return _run_comparison(
                Path(temporary), executable, num_examples, seed, timeout,
                noise_model, ar_order,
            )

    work_dir = Path(output_dir).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)
    return _run_comparison(
        work_dir, executable, num_examples, seed, timeout, noise_model, ar_order
    )


def _run_comparison(
    work_dir: Path,
    executable: str,
    num_examples: int,
    seed: int,
    timeout: float,
    noise_model: str,
    ar_order: int,
) -> dict[str, Any]:
    driver = _MATLAB_AR_IRLS_DRIVER if noise_model == "ar_irls" else _MATLAB_DRIVER
    inputs = _simulate_inputs(num_examples, seed)
    inputs["ar_order"] = np.int64(ar_order)
    input_path = work_dir / "roi_glm_inputs.mat"
    output_path = work_dir / "roi_glm_matlab_outputs.mat"
    savemat(input_path, inputs)

    matlab_directory = str(driver.parent.resolve()).replace("'", "''")
    environment = os.environ.copy()
    environment["PYBRAIN_ROI_GLM_INPUT"] = str(input_path.resolve())
    environment["PYBRAIN_ROI_GLM_OUTPUT"] = str(output_path.resolve())
    completed = subprocess.run(
        [
            executable,
            "-batch",
            f"addpath('{matlab_directory}'); {driver.stem}()",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )
    (work_dir / "matlab_stdout.log").write_text(completed.stdout)
    (work_dir / "matlab_stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "No MATLAB output"
        raise RuntimeError(
            f"MATLAB ROI/GLM comparison failed with exit status "
            f"{completed.returncode}:\n{details}"
        )
    if not output_path.is_file():
        raise FileNotFoundError(f"MATLAB did not create the expected output: {output_path}")

    python = _run_python_comparisons(inputs, noise_model)
    matlab = loadmat(output_path)
    per_example = []
    comparisons = {}
    for workflow, prefix in (
        ("roi_timecourse_then_glm", "timecourse_first"),
        ("glm_then_roi_statistics", "glm_first"),
    ):
        beta_matlab = matlab[f"beta_{prefix}"]
        covariance_matlab = matlab[f"covariance_{prefix}"]
        beta_error = _error_metrics(python[f"beta_{prefix}"], beta_matlab)
        covariance_error = _error_metrics(
            python[f"covariance_{prefix}"], covariance_matlab
        )
        comparisons[workflow] = {
            "estimate_error": beta_error,
            "covariance_matrix_difference": covariance_error,
        }
        for iteration in range(beta_matlab.shape[-1]):
            per_example.append({
                "iteration": iteration + 1,
                "workflow": workflow,
                "estimate_max_absolute_error": _error_metrics(
                    python[f"beta_{prefix}"][:, :, iteration],
                    beta_matlab[:, :, iteration],
                )["max_absolute_error"],
                "covariance_max_absolute_error": _error_metrics(
                    python[f"covariance_{prefix}"][:, :, iteration],
                    covariance_matlab[:, :, iteration],
                )["max_absolute_error"],
            })

    summary = {
        "seed": seed,
        "num_examples": num_examples,
        "single_condition_examples": int(np.count_nonzero(inputs["num_conditions"] == 1)),
        "multiple_condition_examples": int(np.count_nonzero(inputs["num_conditions"] == 2)),
        "distinct_roi_definitions": int(
            len({weights.tobytes() for weights in inputs["roi_weights"].transpose(2, 0, 1)})
        ),
        "roi_names": ["ROI_1", "ROI_2", "ROI_3"],
        "pooling": (
            "inverse temporal variance (time courses) / inverse coefficient "
            "variance (statistics)" if noise_model == "ar_irls"
            else "fixed normalized table weights"
        ),
        "noise_model": "AR-IRLS" if noise_model == "ar_irls" else "OLS",
        "ar_order": ar_order if noise_model == "ar_irls" else None,
        "matlab_vs_python": comparisons,
    }
    pd.DataFrame(per_example).to_csv(work_dir / "roi_glm_per_example_errors.csv", index=False)
    (work_dir / "roi_glm_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _simulate_inputs(num_examples: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_time = 160
    design = np.zeros((n_time, 3, num_examples))
    data = np.zeros((n_time, _N_CHANNELS, num_examples))
    roi_weights = np.zeros((_N_ROIS, _N_CHANNELS, num_examples))
    num_conditions = np.empty(num_examples, dtype=np.int64)
    response = np.exp(-np.arange(20) / 5)

    for iteration in range(num_examples):
        condition_count = iteration % _MAX_CONDITIONS + 1
        num_conditions[iteration] = condition_count
        design[:, 0, iteration] = 1
        for condition in range(condition_count):
            events = np.zeros(n_time)
            event_indices = rng.choice(np.arange(5, n_time - 20), size=6, replace=False)
            events[event_indices] = 1
            design[:, condition + 1, iteration] = np.convolve(
                events, response, mode="full"
            )[:n_time]

        condition_design = design[:, :condition_count + 1, iteration]
        coefficients = rng.normal(0, 1, size=(condition_count + 1, _N_CHANNELS))
        shared_noise = _ar_noise(rng, n_time)
        channel_noise = np.column_stack([
            _ar_noise(rng, n_time) for _ in range(_N_CHANNELS)
        ])
        loadings = rng.uniform(0.2, 0.8, size=_N_CHANNELS)
        data[:, :, iteration] = (
            condition_design @ coefficients
            + 0.35 * shared_noise[:, None] * loadings[None, :]
            + 0.8 * channel_noise
        )

        membership = np.zeros((_N_ROIS, _N_CHANNELS))
        for roi in range(_N_ROIS):
            members = rng.choice(_N_CHANNELS, size=rng.integers(2, 5), replace=False)
            membership[roi, members] = rng.uniform(0.25, 2.0, size=len(members))
        roi_weights[:, :, iteration] = membership / membership.sum(axis=1, keepdims=True)

    return {
        "design": design,
        "data": data,
        "roi_weights": roi_weights,
        "num_conditions": num_conditions,
    }


def _ar_noise(rng: np.random.Generator, length: int) -> np.ndarray:
    noise = rng.normal(size=length)
    for index in range(1, length):
        noise[index] += 0.35 * noise[index - 1]
    return noise


def _fit_statistics(
    signal: np.ndarray,
    x: np.ndarray,
    names: list[str],
    labels: np.ndarray,
    source: np.ndarray,
    detector: np.ndarray,
    ar_order: int,
) -> Any:
    """Fit each column with AR-IRLS and assemble Statistics as the GLM module does."""
    from pyBrainAnalyzIR.dataclasses.statistics import Statistics
    from pyBrainAnalyzIR.math.glm.solve import _channel_fit

    design = pd.DataFrame(x, columns=names)
    results = xr.DataArray(
        np.empty((signal.shape[1], 1), dtype=object),
        dims=("channel", "chromo"),
        coords={"channel": labels, "chromo": ["hbo"]},
    )
    for column in range(signal.shape[1]):
        results.values[column, 0] = _channel_fit(
            signal[:, column], design, noise_model="ar_irls", ar_order=ar_order
        )
    stats = Statistics(results)
    stats.betas = stats.betas.assign_coords(
        source=("indices", np.repeat(source, len(names))),
        detector=("indices", np.repeat(detector, len(names))),
    )
    stats.geoROI = None
    return stats


def _run_python_comparisons(
    inputs: dict[str, np.ndarray], noise_model: str = "ols"
) -> dict[str, np.ndarray]:
    from cedalion.dataclasses.geometry import PointType

    from pyBrainAnalyzIR.dataclasses.statistics import (
        Statistics,
        _ols_joint_covariance,
    )
    from pyBrainAnalyzIR.math.glm.solve import _channel_fit
    from pyBrainAnalyzIR.sigproc.roi import roi_average, roi_average_statistics

    data = inputs["data"]
    design = inputs["design"]
    roi_weights = inputs["roi_weights"]
    num_conditions = inputs["num_conditions"]
    n_examples = data.shape[2]
    use_ar_irls = noise_model == "ar_irls"
    ar_order = int(inputs["ar_order"])
    python_outputs = {
        "beta_timecourse_first": np.full((_N_ROIS, _MAX_CONDITIONS, n_examples), np.nan),
        "covariance_timecourse_first": np.full(
            (_N_ROIS * _MAX_CONDITIONS, _N_ROIS * _MAX_CONDITIONS, n_examples),
            np.nan,
        ),
        "beta_glm_first": np.full((_N_ROIS, _MAX_CONDITIONS, n_examples), np.nan),
        "covariance_glm_first": np.full(
            (_N_ROIS * _MAX_CONDITIONS, _N_ROIS * _MAX_CONDITIONS, n_examples),
            np.nan,
        ),
    }
    source = np.asarray([f"S{index + 1}" for index in range(_N_CHANNELS)])
    detector = np.asarray([f"D{index + 1}" for index in range(_N_CHANNELS)])
    channels = np.asarray([f"{s}{d}" for s, d in zip(source, detector)])
    roi_names = [f"ROI_{index + 1}" for index in range(_N_ROIS)]
    optode_positions = np.asarray(
        [[index * 30.0, 0.0, 0.0] for index in range(_N_CHANNELS)]
        + [[index * 30.0 + 15.0, 30.0, 0.0] for index in range(_N_CHANNELS)]
    )
    geo3d = xr.DataArray(
        optode_positions,
        dims=("label", "pos"),
        coords={
            "label": [*source, *detector],
            "type": (
                "label",
                [PointType.SOURCE] * _N_CHANNELS
                + [PointType.DETECTOR] * _N_CHANNELS,
            ),
        },
        attrs={"units": "mm"},
    ).pint.quantify()

    for iteration in range(n_examples):
        condition_count = int(num_conditions[iteration])
        x = design[:, :condition_count + 1, iteration]
        y = data[:, :, iteration]
        weights = roi_weights[:, :, iteration]
        rows = []
        for roi, name in enumerate(roi_names):
            for channel in np.flatnonzero(weights[roi] > 0):
                rows.append({
                    "source": source[channel],
                    "detector": detector[channel],
                    "name": name,
                    "weight": weights[roi, channel],
                })
        table = pd.DataFrame(rows)
        recording = _ArrayRecording(
            xr.DataArray(
                y,
                dims=("time", "channel"),
                coords={
                    "time": np.arange(y.shape[0]),
                    "channel": channels,
                    "source": ("channel", source),
                    "detector": ("channel", detector),
                },
                attrs={"units": "dimensionless"},
            ).pint.quantify(),
            geo3d,
        )
        roi_average(
            recording,
            table,
            input_name="signal",
            output_name="roi_signal",
            roi_name=f"comparison_{iteration}",
            weighted=use_ar_irls,
        )
        roi_data = recording["roi_signal"].pint.dequantify().values
        coefficient_names = ["intercept", *[f"task_{i + 1}" for i in range(condition_count)]]
        if use_ar_irls:
            roi_stats = _fit_statistics(
                roi_data, x, coefficient_names, np.asarray(roi_names),
                np.asarray(roi_names), np.asarray(roi_names), ar_order,
            )
            _store_task_estimates(
                python_outputs,
                "timecourse_first",
                np.asarray(roi_stats.betas),
                np.asarray(roi_stats.covariance),
                condition_count,
                iteration,
            )
            roi_results = []
        else:
            roi_results = [
                _channel_fit(roi_data[:, roi], x, noise_model="ols")
                for roi in range(_N_ROIS)
            ]
            roi_residuals = np.vstack([result.resid for result in roi_results])
            roi_covariance = _ols_joint_covariance(
                roi_results, roi_residuals, condition_count + 1
            )
            _store_task_estimates(
                python_outputs,
                "timecourse_first",
                np.concatenate([np.asarray(result.params) for result in roi_results]),
                roi_covariance,
                condition_count,
                iteration,
            )

        if use_ar_irls:
            stats = _fit_statistics(
                y, x, coefficient_names, channels, source, detector, ar_order
            )
            stats.geo3d = geo3d
            pooled = roi_average_statistics(
                stats, table, roi_name="comparison", weighted=True
            )
            pooled_labels = list(zip(
                pooled.betas.channels.values,
                pooled.betas.conditions.values,
            ))
            pooled_indices = [
                pooled_labels.index((name, label))
                for name in roi_names
                for label in coefficient_names
            ]
            _store_task_estimates(
                python_outputs,
                "glm_first",
                np.asarray(pooled.betas)[pooled_indices],
                np.asarray(pooled.covariance)[np.ix_(pooled_indices, pooled_indices)],
                condition_count,
                iteration,
            )
            continue

        channel_results = [
            _channel_fit(y[:, channel], x, noise_model="ols")
            for channel in range(_N_CHANNELS)
        ]
        channel_residuals = np.vstack([result.resid for result in channel_results])
        channel_covariance = _ols_joint_covariance(
            channel_results, channel_residuals, condition_count + 1
        )
        beta_values = np.concatenate([
            np.asarray(result.params) for result in channel_results
        ])
        beta_coordinates = {
            "indices": np.arange(beta_values.size),
            "channels": ("indices", np.repeat(channels, condition_count + 1)),
            "type": ("indices", np.repeat("hbo", beta_values.size)),
            "conditions": (
                "indices",
                np.tile(["intercept", *[f"task_{i + 1}" for i in range(condition_count)]],
                        _N_CHANNELS),
            ),
            "source": ("indices", np.repeat(source, condition_count + 1)),
            "detector": ("indices", np.repeat(detector, condition_count + 1)),
        }
        stats = Statistics()
        stats.description = "Synthetic OLS GLM"
        stats.geo3d = geo3d
        stats.geoROI = None
        stats.dof = data.shape[0] - condition_count - 1
        stats.betas = xr.DataArray(beta_values, dims="indices", coords=beta_coordinates)
        stats.covariance = xr.DataArray(
            channel_covariance,
            dims=("rows", "cols"),
            coords={"rows": np.arange(beta_values.size), "cols": np.arange(beta_values.size)},
        )
        pooled = roi_average_statistics(
            stats, table, roi_name="comparison", weighted=False
        )
        pooled_labels = list(zip(
            pooled.betas.channels.values,
            pooled.betas.conditions.values,
        ))
        pooled_indices = [
            pooled_labels.index((name, label))
            for name in roi_names
            for label in ["intercept", *[f"task_{i + 1}" for i in range(condition_count)]]
        ]
        pooled_beta = np.asarray(pooled.betas)[pooled_indices]
        pooled_covariance = np.asarray(pooled.covariance)[
            np.ix_(pooled_indices, pooled_indices)
        ]
        _store_task_estimates(
            python_outputs,
            "glm_first",
            pooled_beta,
            pooled_covariance,
            condition_count,
            iteration,
        )

    return python_outputs


class _ArrayRecording(SimpleNamespace):
    def __init__(self, signal: xr.DataArray, geo3d: xr.DataArray) -> None:
        super().__init__(
            timeseries={"signal": signal},
            aux_obj={},
            geo3d=geo3d,
            masks={},
        )

    def __getitem__(self, name: str) -> xr.DataArray:
        return self.timeseries[name]

    def __setitem__(self, name: str, value: xr.DataArray) -> None:
        self.timeseries[name] = value


def _store_task_estimates(
    output: dict[str, np.ndarray],
    suffix: str,
    full_beta: np.ndarray,
    full_covariance: np.ndarray,
    num_conditions: int,
    iteration: int,
) -> None:
    full_coefficients_per_roi = full_beta.size // _N_ROIS
    task_indices = np.concatenate([
        np.arange(roi * full_coefficients_per_roi + 1,
                  roi * full_coefficients_per_roi + num_conditions + 1)
        for roi in range(_N_ROIS)
    ])
    beta = np.asarray(full_beta).reshape(_N_ROIS, full_coefficients_per_roi)[:, 1:]
    covariance = np.asarray(full_covariance)[np.ix_(task_indices, task_indices)]
    beta_key = f"beta_{suffix}"
    covariance_key = f"covariance_{suffix}"
    output[beta_key][:, :num_conditions, iteration] = beta[:, :num_conditions]
    output_indices = np.concatenate([
        np.arange(roi * _MAX_CONDITIONS, roi * _MAX_CONDITIONS + num_conditions)
        for roi in range(_N_ROIS)
    ])
    output[covariance_key][
        np.ix_(output_indices, output_indices, [iteration])
    ] = covariance[:, :, None]


def _error_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError(
            f"MATLAB and Python outputs have different shapes: {left.shape} != {right.shape}"
        )
    if not np.array_equal(np.isfinite(left), np.isfinite(right)):
        raise ValueError("MATLAB and Python outputs have different missing-value locations.")
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        raise ValueError("Cannot report comparison errors without finite paired values.")
    difference = left[valid] - right[valid]
    return {
        "count": int(difference.size),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "root_mean_square_error": float(np.sqrt(np.mean(difference**2))),
    }
