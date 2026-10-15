"""Resampling helpers that preserve missing-data locations."""
from __future__ import annotations

import numpy as np
import scipy.signal
import xarray as xr

from cedalion.math.resample import resample as cedalion_resample

from pyBrainAnalyzIR.math.glm.solve import _recording_quality_mask


def _project_bad_samples(good: xr.DataArray, source_time, target_time):
    projected = np.ones((len(target_time), good.sizes["channel"]), dtype=bool)
    source_values = np.asarray(source_time, dtype=float)
    target_values = np.asarray(target_time, dtype=float)
    for channel_index in range(good.sizes["channel"]):
        bad = np.flatnonzero(~good.to_numpy()[:, channel_index])
        if bad.size == 0:
            continue
        target_indices = np.searchsorted(target_values, source_values[bad])
        target_indices = np.clip(target_indices, 0, target_values.size - 1)
        left = np.maximum(target_indices - 1, 0)
        choose_left = (
            np.abs(source_values[bad] - target_values[left])
            <= np.abs(source_values[bad] - target_values[target_indices])
        )
        target_indices[choose_left] = left[choose_left]
        projected[target_indices, channel_index] = False
    return xr.DataArray(
        projected,
        dims=("time", "channel"),
        coords={"time": target_time, "channel": good.channel},
    )


def resample_with_quality_masks(
    data: xr.DataArray,
    sampling_rate: float,
    masks: dict[str, xr.DataArray],
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """Resample data while interpolating missing input points only for filtering.

    Invalid source points are restored as NaN at their nearest output sample, and the
    recording's clean/bad masks are projected to that same output time grid.
    """
    data_plain = data.pint.dequantify()
    time = np.asarray(data_plain.time, dtype=float)
    if time.size < 2:
        raise ValueError("At least two time samples are required for resampling.")
    if sampling_rate <= 0:
        raise ValueError("The target sampling rate must be positive.")

    source_mask = (
        _recording_quality_mask(data_plain, masks)
        if masks
        else xr.DataArray(
            np.ones((data_plain.sizes["time"], data_plain.sizes["channel"]), dtype=bool),
            dims=("time", "channel"),
            coords={
                "time": data_plain.time,
                "channel": data_plain.channel,
            },
        )
    )
    original_dims = data_plain.dims
    other_dims = tuple(dim for dim in original_dims if dim != "time")
    work = data_plain.transpose("time", *other_dims)
    values = np.asarray(work.to_numpy(), dtype=float)
    flat_values = values.reshape((time.size, -1))

    quality_values = source_mask.to_numpy()
    channel_labels = list(data_plain.channel.values)
    invalid_by_channel = ~quality_values
    finite_before = np.isfinite(flat_values)
    flattened_channel_coords = np.asarray(
        work.coords["channel"].broadcast_like(work).transpose(*work.dims).to_numpy()
    ).reshape((time.size, -1))
    for channel_index, label in enumerate(channel_labels):
        columns = flattened_channel_coords[0] == label
        if columns.any():
            flat_values[:, columns] = np.where(
                quality_values[:, channel_index, None],
                flat_values[:, columns],
                np.nan,
            )

    finite_before = np.isfinite(flat_values)
    source_bad = np.zeros((time.size, len(channel_labels)), dtype=bool)
    for channel_index in range(len(channel_labels)):
        source_bad[:, channel_index] = invalid_by_channel[:, channel_index]
        columns = flattened_channel_coords[0] == channel_labels[channel_index]
        if columns.any():
            source_bad[:, channel_index] |= (~finite_before[:, columns]).any(axis=1)

    current_rate = 1 / np.diff(time).mean()
    target_time = time[0] + np.arange(
        int(np.floor((time[-1] - time[0]) * sampling_rate)) + 1
    ) / sampling_rate

    if not source_bad.any() and finite_before.all():
        resampled = cedalion_resample(data, sampling_rate)
        target_time = np.asarray(resampled.time, dtype=float)
        output_masks = {
            name: _project_bad_samples(
                _recording_quality_mask(data_plain, {name: mask}),
                time,
                target_time,
            )
            for name, mask in masks.items()
        }
        return resampled, output_masks

    output = np.full((target_time.size, flat_values.shape[1]), np.nan)
    cutoff = sampling_rate / current_rate
    filter_order = int(np.floor(time.size / 10))
    coefficients = (
        scipy.signal.firwin(filter_order + 1, cutoff)
        if sampling_rate < current_rate and filter_order > 0
        else None
    )

    for column in range(flat_values.shape[1]):
        samples = flat_values[:, column]
        valid = np.isfinite(samples)
        if valid.sum() < 4:
            continue
        filled = np.interp(time, time[valid], samples[valid])
        left = time < time[valid][0]
        right = time > time[valid][-1]
        if valid.sum() > 1:
            left_slope = (samples[valid][1] - samples[valid][0]) / (
                time[valid][1] - time[valid][0]
            )
            right_slope = (samples[valid][-1] - samples[valid][-2]) / (
                time[valid][-1] - time[valid][-2]
            )
            filled[left] = samples[valid][0] + (
                time[left] - time[valid][0]
            ) * left_slope
            filled[right] = samples[valid][-1] + (
                time[right] - time[valid][-1]
            ) * right_slope
        mean = filled.mean()
        centered = filled - mean
        if coefficients is not None:
            padlen = 3 * (coefficients.size - 1)
            if time.size <= padlen:
                raise ValueError(
                    "The recording is too short for the anti-aliasing filter."
                )
            centered = scipy.signal.filtfilt(
                coefficients, [1.0], centered, axis=0
            )
        output[:, column] = np.interp(target_time, time, centered) + mean

    for channel_index, label in enumerate(channel_labels):
        bad_source = np.flatnonzero(source_bad[:, channel_index])
        if not bad_source.size:
            continue
        source_times = time[bad_source]
        target_indices = np.searchsorted(target_time, source_times)
        target_indices = np.clip(target_indices, 0, target_time.size - 1)
        left = np.maximum(target_indices - 1, 0)
        choose_left = (
            np.abs(source_times - target_time[left])
            <= np.abs(source_times - target_time[target_indices])
        )
        target_indices[choose_left] = left[choose_left]
        columns = flattened_channel_coords[0] == label
        output[np.unique(target_indices)[:, None], np.flatnonzero(columns)] = np.nan

    output_shape = (target_time.size, *(values.shape[1:]))
    output = output.reshape(output_shape)
    coords = {
        name: coord
        for name, coord in data_plain.coords.items()
        if "time" not in coord.dims
    }
    coords["time"] = (
        "time",
        target_time,
        dict(data_plain.time.attrs),
    )
    if "samples" in data_plain.coords:
        coords["samples"] = ("time", np.arange(target_time.size))
    resampled_plain = xr.DataArray(
        output,
        dims=("time", *other_dims),
        coords=coords,
        attrs=data_plain.attrs,
        name=data_plain.name,
    ).transpose(*original_dims)
    resampled = resampled_plain.pint.quantify(data.pint.units)

    output_masks = {
        name: _project_bad_samples(
            _recording_quality_mask(data_plain, {name: mask}), time, target_time
        )
        for name, mask in masks.items()
    }
    natural_missing = np.zeros_like(source_bad)
    for channel_index, label in enumerate(channel_labels):
        columns = flattened_channel_coords[0] == label
        natural_missing[:, channel_index] = (
            ~np.isfinite(values.reshape((time.size, -1))[:, columns])
        ).any(axis=1)
    if natural_missing.any():
        natural_good = xr.DataArray(
            ~natural_missing,
            dims=("time", "channel"),
            coords={"time": time, "channel": channel_labels},
        )
        output_masks["missing_data"] = _project_bad_samples(
            natural_good, time, target_time
        )

    return resampled, output_masks
