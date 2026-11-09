"""Fast, exact OLS resampling diagnostics for real-data validation."""

from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
import pandas as pd
from cedalion.sigproc.frequency import sampling_rate
from scipy.stats import kurtosis, wasserstein_distance

from pyBrainAnalyzIR.math.glm import design_matrix
from pyBrainAnalyzIR.pipelines.default_pipelines import default_pipelines
from pyBrainAnalyzIR.pipelines.modules.glm import get_basis_function
from pyBrainAnalyzIR.pipelines.modules.nonparametric_statistics import (
    _circular_shift_residual,
    _fit_design,
    _residual_model,
)

__all__ = [
    "OLSResamplingContext",
    "fit_randomized_null_context",
    "make_randomized_stimulus",
    "draw_ols_null_betas",
    "residual_stationarity_metrics",
]


@dataclass
class OLSResamplingContext:
    """Prepared real-data residuals and exact OLS coefficient projections."""

    signal_values: np.ndarray
    baseline: np.ndarray
    residual: np.ndarray
    observed_betas: np.ndarray
    projection_weights: np.ndarray
    original_variance: np.ndarray
    channels: tuple[str, ...]
    signal_types: tuple[str, ...]
    condition: str
    sampling_frequency: float


def make_randomized_stimulus(
    time: np.ndarray,
    seed: int,
    duration_seconds: float = 10.0,
    rest_min_seconds: float = 10.0,
    rest_max_seconds: float = 20.0,
    start_margin_seconds: float = 20.0,
    end_margin_seconds: float = 15.0,
) -> pd.DataFrame:
    """Create a null-design schedule with one type and jittered rests."""
    time = np.asarray(time, dtype=float).reshape(-1)
    if time.size < 2 or not np.isfinite(time).all():
        raise ValueError(
            "A finite time vector with at least two samples is required."
        )
    if not (
        duration_seconds > 0
        and 0 <= rest_min_seconds <= rest_max_seconds
        and start_margin_seconds >= 0
        and end_margin_seconds >= 0
    ):
        raise ValueError("Invalid stimulus timing parameters.")

    rng = np.random.default_rng(seed)
    end = float(time[-1])
    onset = float(time[0]) + start_margin_seconds
    onsets = []
    while onset + duration_seconds <= end - end_margin_seconds:
        onsets.append(onset)
        onset += duration_seconds + float(
            rng.uniform(rest_min_seconds, rest_max_seconds)
        )
    if len(onsets) < 2:
        raise ValueError(
            "Recording is too short for at least two randomized task events."
        )
    return pd.DataFrame(
        {
            "onset": onsets,
            "duration": duration_seconds,
            "value": 1.0,
            "trial_type": "validation_task",
        }
    )


def fit_randomized_null_context(recording, seed: int) -> OLSResamplingContext:
    """Remove recorded task effects, then fit a seeded null schedule.

    The recorded task component is removed using its fitted first-level OLS
    coefficients. The new validation schedule is seeded independently of the
    time series. The resulting task coefficients therefore have no injected
    activation, retaining the non-task baseline and residuals.
    """
    has_recorded_stimulus = (
        hasattr(recording, "stim") and not recording.stim.empty
    )
    if not has_recorded_stimulus:
        recording.stim = make_randomized_stimulus(
            recording["amp"].time.values,
            seed=seed,
        )

    pipeline = default_pipelines.first_level_analysis.pipeline()
    pipeline.set_all_options({"noise_model": "ols", "verbose": False})
    pipeline.run(recording)
    glm = pipeline
    if not has_recorded_stimulus:
        return prepare_ols_resampling_context(
            glm, recording, recording[glm.outputName]
        )

    observed_stats = recording[glm.outputName]
    cleaned_signal = _remove_recorded_task_component(
        recording, glm, observed_stats
    )
    recording[glm.inputName] = cleaned_signal

    recording.stim = make_randomized_stimulus(
        cleaned_signal.time.values,
        seed=seed,
    )
    glm_only = copy.deepcopy(glm)
    glm_only.previous_job = None
    glm_only.options["verbose"] = False
    glm_only.options["max_jobs"] = 1
    glm_only.run(recording)
    return prepare_ols_resampling_context(
        glm_only, recording, recording[glm_only.outputName]
    )


def prepare_ols_resampling_context(
    glm, recording, stats
) -> OLSResamplingContext:
    """Prepare exact linear OLS coefficient projections for bootstrap draws."""
    if str(glm.options["noise_model"]).lower() != "ols":
        raise ValueError(
            "The fast projection is exact only for the OLS noise model."
        )
    (
        signal,
        baseline,
        observed_rows,
        _,
        _,
        residual,
        condition,
    ) = _residual_model(glm, recording, stats)
    design, _ = _fit_design(glm, recording, signal)
    channels = tuple(map(str, signal.channel.values))
    signal_types = tuple(map(str, signal.chromo.values))
    regressor_names = tuple(map(str, design.common.regressor.values))
    if condition not in regressor_names:
        raise ValueError(
            f"Missing task regressor {condition!r} in the OLS design."
        )

    weights = []
    for signal_type in signal_types:
        matrix = (
            design.common.sel(chromo=signal_type, drop=True)
            .transpose("time", "regressor")
            .values
        )
        if not np.isfinite(matrix).all():
            raise ValueError("The OLS design contains non-finite values.")
        coefficient_index = regressor_names.index(condition)
        projection = np.linalg.pinv(matrix)[coefficient_index]
        if not np.isclose(projection @ matrix[:, coefficient_index], 1.0):
            raise ValueError("The OLS task coefficient is not identifiable.")
        weights.append(projection)

    observed_betas = np.empty((len(channels), len(signal_types)), dtype=float)
    for channel_index, channel in enumerate(channels):
        for type_index, signal_type in enumerate(signal_types):
            row = observed_rows.loc[
                (observed_rows.Channel == channel)
                & (observed_rows.Type == signal_type),
                "Beta",
            ]
            if len(row) != 1:
                raise ValueError(
                    f"Missing unique OLS estimate for {channel}/{signal_type}."
                )
            observed_betas[channel_index, type_index] = float(row.iloc[0])

    values = np.asarray(signal.pint.magnitude, dtype=float)
    original_variance = values.reshape(values.shape[0], -1).var(axis=0, ddof=1)
    if (
        not np.isfinite(original_variance).all()
        or (original_variance <= 0).any()
    ):
        raise ValueError(
            "The OLS input must have positive finite signal variance."
        )
    frequency = float(sampling_rate(signal).to("Hz").magnitude)

    return OLSResamplingContext(
        signal_values=values,
        baseline=baseline,
        residual=residual,
        observed_betas=observed_betas,
        projection_weights=np.asarray(weights),
        original_variance=original_variance,
        channels=channels,
        signal_types=signal_types,
        condition=condition,
        sampling_frequency=frequency,
    )


def draw_ols_null_betas(
    context: OLSResamplingContext,
    iterations: int,
    seed: int,
    window_seconds: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact OLS null beta draws and variance ratios."""
    if iterations < 1:
        raise ValueError("At least one bootstrap iteration is required.")
    window_samples = (
        None
        if window_seconds is None
        else int(round(window_seconds * context.sampling_frequency))
    )
    rng = np.random.default_rng(seed)
    n_time = context.residual.shape[0]
    betas = np.empty(
        (iterations, len(context.channels), len(context.signal_types)),
        dtype=float,
    )
    variance_ratios = np.empty(
        (iterations, context.original_variance.size), dtype=float
    )
    for iteration in range(iterations):
        shifted = _circular_shift_residual(
            context.residual, rng, window_samples
        ).reshape(context.signal_values.shape)
        null_values = context.baseline + shifted
        for type_index, weights in enumerate(context.projection_weights):
            betas[iteration, :, type_index] = (
                weights @ null_values[:, :, type_index]
            )
        variance_ratios[iteration] = (
            null_values.reshape(n_time, -1).var(axis=0, ddof=1)
            / context.original_variance
        )
    return betas, variance_ratios


def residual_stationarity_metrics(
    context: OLSResamplingContext,
    window_seconds: list[float | None],
) -> pd.DataFrame:
    """Summarize distribution changes between and within shift windows."""
    rows = []
    n_time = context.residual.shape[0]
    for window in window_seconds:
        window_samples = (
            None
            if window is None
            else int(round(window * context.sampling_frequency))
        )
        segments = _time_segments(n_time, window_samples)
        for signal_index, (channel, signal_type) in enumerate(
            (channel, signal_type)
            for channel in context.channels
            for signal_type in context.signal_types
        ):
            series = context.residual[:, signal_index]
            scale = float(np.std(series, ddof=1))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(
                    f"Invalid residual scale for {channel}/{signal_type}."
                )
            block_variances = []
            block_medians = []
            block_tail_ratios = []
            block_excess_kurtosis = []
            block_acf_1s = []
            block_acf_5s = []
            half_variance_ratios = []
            half_mean_shifts = []
            lag_1s = max(1, int(round(context.sampling_frequency)))
            lag_5s = max(1, int(round(5 * context.sampling_frequency)))
            for segment in segments:
                block = series[segment]
                if len(block) < 2:
                    continue
                block_variance = float(np.var(block, ddof=1))
                block_variances.append(block_variance)
                block_medians.append(float(np.median(block)))
                block_scale = np.sqrt(block_variance)
                block_tail_ratios.append(
                    float(np.quantile(np.abs(block - np.median(block)), 0.95))
                    / block_scale
                )
                if len(block) > 3:
                    block_excess_kurtosis.append(
                        float(kurtosis(block, fisher=True, bias=False))
                    )
                block_acf_1s.append(_sample_autocorrelation(block, lag_1s))
                block_acf_5s.append(_sample_autocorrelation(block, lag_5s))
                midpoint = len(block) // 2
                if midpoint >= 2 and len(block) - midpoint >= 2:
                    first, second = block[:midpoint], block[midpoint:]
                    first_var = max(
                        float(np.var(first, ddof=1)), np.finfo(float).tiny
                    )
                    second_var = max(
                        float(np.var(second, ddof=1)), np.finfo(float).tiny
                    )
                    half_variance_ratios.append(
                        max(first_var, second_var) / min(first_var, second_var)
                    )
                    half_mean_shifts.append(
                        abs(float(np.mean(first) - np.mean(second))) / scale
                    )
            variance_mean = float(np.mean(block_variances))
            block_cv = np.nan
            if len(block_variances) > 1 and variance_mean > 0:
                block_cv = float(
                    np.std(block_variances, ddof=1) / variance_mean
                )
            tail_mean = float(np.mean(block_tail_ratios))
            tail_ratio_cv = np.nan
            if len(block_tail_ratios) > 1 and tail_mean > 0:
                tail_ratio_cv = float(
                    np.std(block_tail_ratios, ddof=1) / tail_mean
                )
            adjacent_wasserstein = [
                wasserstein_distance(series[left], series[right]) / scale
                for left, right in zip(segments, segments[1:])
                if len(left) and len(right)
            ]
            rows.append(
                {
                    "Channel": channel,
                    "Type": signal_type,
                    "WindowSeconds": window,
                    "NumWindows": len(segments),
                    "BetweenWindowVarianceCV": block_cv,
                    "BetweenWindowTailRatioCV": tail_ratio_cv,
                    "MedianWindowExcessKurtosis": (
                        float(np.median(block_excess_kurtosis))
                        if block_excess_kurtosis
                        else np.nan
                    ),
                    "MedianWindowACF1s": (
                        float(np.nanmedian(block_acf_1s))
                        if np.isfinite(block_acf_1s).any()
                        else np.nan
                    ),
                    "BetweenWindowACF1sSD": _finite_sample_sd(block_acf_1s),
                    "MedianWindowACF5s": (
                        float(np.nanmedian(block_acf_5s))
                        if np.isfinite(block_acf_5s).any()
                        else np.nan
                    ),
                    "BetweenWindowACF5sSD": _finite_sample_sd(block_acf_5s),
                    "BetweenWindowMedianRangeSD": (
                        (max(block_medians) - min(block_medians)) / scale
                        if len(block_medians) > 1
                        else np.nan
                    ),
                    "AdjacentWindowWassersteinSD": (
                        float(np.median(adjacent_wasserstein))
                        if adjacent_wasserstein
                        else np.nan
                    ),
                    "WithinWindowHalfVarianceRatioMedian": (
                        float(np.median(half_variance_ratios))
                        if half_variance_ratios
                        else np.nan
                    ),
                    "WithinWindowHalfMeanShiftSDMedian": (
                        float(np.median(half_mean_shifts))
                        if half_mean_shifts
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _remove_recorded_task_component(recording, glm, stats):
    signal = recording[glm.inputName].transpose("time", "channel", "chromo")
    values = np.asarray(signal.pint.magnitude, dtype=float)
    if recording.stim.empty:
        return signal.copy(deep=True)

    task_design = design_matrix.hrf_regressors(
        signal,
        recording.stim,
        get_basis_function(glm.options["basis_function"]),
    )
    estimates = stats.table()
    channels = list(map(str, signal.channel.values))
    signal_types = list(map(str, signal.chromo.values))
    task_component = np.zeros_like(values)
    for condition in map(str, task_design.common.regressor.values):
        waveform = (
            task_design.common.sel(regressor=condition, drop=True)
            .transpose("time", "chromo")
            .values
        )
        for channel_index, channel in enumerate(channels):
            for type_index, signal_type in enumerate(signal_types):
                row = estimates.loc[
                    (estimates.Channel == channel)
                    & (estimates.Type == signal_type)
                    & (estimates.Condition == condition),
                    "Beta",
                ]
                if len(row) != 1:
                    raise ValueError(
                        "Missing unique recorded-task estimate for "
                        f"{channel}/{signal_type}."
                    )
                task_component[:, channel_index, type_index] += waveform[
                    :, type_index
                ] * float(row.iloc[0])
    cleaned = signal.copy(deep=True)
    cleaned.pint.magnitude[:] = values - task_component
    return cleaned


def _time_segments(
    n_time: int, window_samples: int | None
) -> list[np.ndarray]:
    if window_samples is None:
        return [np.arange(n_time)]
    if window_samples < 2 or window_samples >= n_time:
        raise ValueError(
            "Shift window must span at least two and fewer than all samples."
        )
    n_windows = int(np.ceil(n_time / window_samples))
    segments = np.array_split(np.arange(n_time), n_windows)
    if any(len(segment) < 2 for segment in segments):
        raise ValueError("A shift window contains fewer than two samples.")
    return segments


def _sample_autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag + 1:
        return np.nan
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) <= 0 or np.std(right) <= 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _finite_sample_sd(values: list[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan
