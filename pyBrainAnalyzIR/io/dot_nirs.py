"""Reader for DotNIRS/HOMER MATLAB ``.nirs`` files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import warnings

import cedalion
import cedalion.dataclasses as cdc
import numpy as np
import pandas as pd
from scipy.io import loadmat

from pyBrainAnalyzIR.dataclasses.dataset import DataSet

units = cedalion.units

__all__ = ["loadDotNIRS"]


def loadDotNIRS(
    filenames: str | Path | Iterable[str | Path],
    force: bool = False,
) -> DataSet:
    """Load one or more DotNIRS ``.nirs`` files into a :class:`DataSet`.

    Files with a sibling ``.wl1`` are skipped unless ``force=True``, matching
    the MATLAB DotNIRS loader's protection against importing NIRx data twice.
    """
    if isinstance(filenames, (str, Path)):
        paths = [Path(filenames)]
    else:
        paths = [Path(path) for path in filenames]

    dataset = DataSet()
    dataset.description = "DotNIRS data"
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".nirs":
            raise ValueError(f"Expected a .nirs file, got: {path}")
        if not force and path.with_suffix(".wl1").exists():
            continue

        contents = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "d" not in contents or "t" not in contents:
            raise ValueError(f"{path} must contain both 'd' and 't' arrays")

        data = np.asarray(contents["d"], dtype=float)
        time = np.asarray(contents["t"], dtype=float).reshape(-1)
        if data.size == 0 or time.size == 0:
            warnings.warn(f"Skipping empty DotNIRS file: {path}", UserWarning)
            continue
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[0] != time.size:
            if data.shape[1] == time.size:
                data = data.T
            else:
                raise ValueError(
                    f"Data dimensions in {path} do not match its time vector"
                )

        data = _fill_missing(data, time)
        sd = contents.get("SD")
        meas_list = _mat_field(sd, "MeasList")
        if meas_list is None:
            meas_list = contents.get("ml")
        if meas_list is None:
            raise ValueError(f"{path} is missing SD.MeasList (or ml)")
        meas_list = np.atleast_2d(np.asarray(meas_list))
        amp, sources, detectors = _amplitude_array(
            data, meas_list, _mat_field(sd, "Lambda"), time
        )

        recording = cdc.Recording()
        recording["amp"] = amp
        recording.description = str(path)
        recording.meta_data.update(_demographics(contents))
        recording.aux_obj["dotnirs"] = {
            "source_file": str(path),
            "extra_fields": {
                key: _mat_to_python(value)
                for key, value in contents.items()
                if not key.startswith("__")
                and key not in {"d", "t", "s", "aux", "aux10", "SD", "ml"}
            },
        }
        _set_geometry(recording, sd, sources, detectors, meas_list)
        recording.stim = _stimuli(contents, time)
        _set_auxiliary(recording, contents, sd, time)
        dataset.import_data(recording)

    return dataset


def _amplitude_array(data, meas_list, lambda_values, time):
    rows = np.asarray(meas_list)
    if rows.ndim != 2 or rows.shape[0] != data.shape[1] or rows.shape[1] < 4:
        raise ValueError(
            "SD.MeasList must have one row per data column and at least four columns"
        )

    lambdas = np.asarray(lambda_values if lambda_values is not None else [690, 830])
    lambdas = lambdas.reshape(-1)
    entries = []
    for index, row in enumerate(rows):
        source, detector, data_type, wavelength_index = (int(value) for value in row[:4])
        if data_type != 1:
            continue
        if source < 1 or detector < 1 or wavelength_index < 1:
            raise ValueError("MeasList indices must be positive, one-based indices")
        if wavelength_index > len(lambdas):
            raise ValueError("MeasList wavelength index exceeds SD.Lambda")
        entries.append((index, source, detector, float(lambdas[wavelength_index - 1])))
    if not entries:
        raise ValueError("No raw-intensity measurements (MeasList data type 1) found")

    source_detector_pairs = list(dict.fromkeys((src, det) for _, src, det, _ in entries))
    wavelengths = list(dict.fromkeys(wavelength for _, _, _, wavelength in entries))
    pair_index = {pair: index for index, pair in enumerate(source_detector_pairs)}
    wavelength_index = {value: index for index, value in enumerate(wavelengths)}
    result = np.full((len(source_detector_pairs), len(wavelengths), data.shape[0]), np.nan)

    for data_index, source, detector, wavelength in entries:
        channel = pair_index[(source, detector)]
        wave = wavelength_index[wavelength]
        if np.isfinite(result[channel, wave]).any():
            raise ValueError(
                f"Duplicate raw-intensity measurement for S{source}D{detector} "
                f"at {wavelength:g} nm"
            )
        result[channel, wave] = data[:, data_index]
    if np.isnan(result).all(axis=2).any():
        raise ValueError("MeasList has incomplete wavelength coverage for a channel")

    channel_names = [f"S{source}D{detector}" for source, detector in source_detector_pairs]
    source_names = [f"S{source}" for source, _ in source_detector_pairs]
    detector_names = [f"D{detector}" for _, detector in source_detector_pairs]
    return (
        cdc.build_timeseries(
            np.transpose(result, (2, 0, 1)),
            dims=["time", "channel", "wavelength"],
            time=time,
            channel=channel_names,
            value_units=units.V,
            time_units=units.s,
            other_coords={"wavelength": wavelengths},
        ).assign_coords(
            source=("channel", source_names),
            detector=("channel", detector_names),
        ).transpose("channel", "wavelength", "time"),
        source_names,
        detector_names,
    )


def _set_geometry(recording, sd, sources, detectors, meas_list):
    source_positions = _position_array(_mat_field(sd, "SrcPos"))
    detector_positions = _position_array(_mat_field(sd, "DetPos"))
    if source_positions is None or detector_positions is None:
        return

    used_rows = np.asarray(meas_list, dtype=int)
    distances = []
    for source, detector in used_rows[:, :2]:
        if 1 <= source <= len(source_positions) and 1 <= detector <= len(detector_positions):
            distances.append(
                np.linalg.norm(source_positions[source - 1] - detector_positions[detector - 1])
            )
    spatial_unit = _string_value(_mat_field(sd, "SpatialUnit"))
    scale = 10.0 if spatial_unit.lower() in {"cm", "centimeter", "centimeters"} else 1.0
    if distances and np.nanmedian(distances) < 20:
        scale = 10.0

    points = []
    labels = []
    kinds = []
    for label in dict.fromkeys(sources):
        index = int(label[1:]) - 1
        if index < len(source_positions):
            points.append(source_positions[index])
            labels.append(label)
            kinds.append(cdc.PointType.SOURCE)
    for label in dict.fromkeys(detectors):
        index = int(label[1:]) - 1
        if index < len(detector_positions):
            points.append(detector_positions[index])
            labels.append(label)
            kinds.append(cdc.PointType.DETECTOR)
    if not points:
        return

    positions = np.asarray(points, dtype=float) * scale
    if positions.shape[1] == 2:
        positions = np.column_stack([positions, np.zeros(len(positions))])
    recording.geo3d = cdc.build_labeled_points(
        positions[:, :3], crs="digitized", units="mm",
        labels=labels, types=kinds,
    )
    if not np.any(np.abs(positions[:, 2]) > 1e-12):
        recording.geo2d = cdc.build_labeled_points(
            positions[:, :2], crs="digitized", units="mm",
            labels=labels, types=kinds,
        )


def _set_auxiliary(recording, contents, sd, time):
    values = contents.get("aux")
    if values is None or np.asarray(values).size == 0:
        values = contents.get("aux10")
    if values is None or np.asarray(values).size == 0:
        return

    auxiliary = np.asarray(values, dtype=float)
    if auxiliary.ndim == 1:
        auxiliary = auxiliary[:, None]
    if auxiliary.shape[0] != len(time) and auxiliary.shape[1] == len(time):
        auxiliary = auxiliary.T
    if auxiliary.shape[0] != len(time):
        raise ValueError("Auxiliary data dimensions do not match the time vector")
    auxiliary = _fill_missing(auxiliary, time)
    names = _string_list(_mat_field(sd, "auxChannels"))
    if len(names) != auxiliary.shape[1]:
        names = [f"aux-{index + 1}" for index in range(auxiliary.shape[1])]
    recording.aux_ts["aux"] = cdc.build_timeseries(
        auxiliary,
        dims=["time", "channel"],
        time=time,
        channel=names,
        value_units=units.dimensionless,
        time_units=units.s,
    ).transpose("channel", "time")


def _stimuli(contents, time):
    design = contents.get("StimDesign")
    if design is not None:
        events = []
        for item in np.asarray(design, dtype=object).reshape(-1):
            name = _string_value(
                _mat_field(item, "name", _mat_field(item, "cond", "stim"))
            )
            onset = _mat_field(item, "onset")
            duration = _mat_field(item, "dur", _mat_field(item, "duration"))
            amplitude = _mat_field(item, "amp", 1.0)
            if onset is None:
                continue
            onsets = np.asarray(onset, dtype=float).reshape(-1)
            durations = _broadcast_values(duration, len(onsets), 0.0)
            amplitudes = _broadcast_values(amplitude, len(onsets), 1.0)
            events.extend(
                (float(start), float(dur), float(value), name)
                for start, dur, value in zip(onsets, durations, amplitudes)
            )
        return _event_frame(events)

    stimulus = contents.get("s")
    if stimulus is None:
        return _empty_events()
    stimulus = np.asarray(stimulus, dtype=float)
    if stimulus.ndim == 1:
        stimulus = stimulus[:, None]
    if stimulus.shape[0] != len(time) and stimulus.shape[1] == len(time):
        stimulus = stimulus.T
    if stimulus.shape[0] != len(time):
        raise ValueError("Stimulus data dimensions do not match the time vector")

    names = _string_list(contents.get("CondNames"))
    events = []
    for index in range(stimulus.shape[1]):
        values = stimulus[:, index]
        maximum = np.nanmax(values) if np.isfinite(values).any() else 0
        if maximum > 0:
            values = values / maximum
        active = np.isfinite(values) & (values > 0.5)
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(~active & np.r_[False, active[:-1]])
        if active[-1]:
            ends = np.r_[ends, len(active)]
        name = names[index] if index < len(names) else f"stim_channel{index + 1}"
        dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.0
        edges = np.r_[time, time[-1] + dt]
        for start, end in zip(starts, ends):
            events.append((float(edges[start]), float(edges[end] - edges[start]), 1.0, name))
    return _event_frame(events)


def _event_frame(events):
    if not events:
        return _empty_events()
    return pd.DataFrame(events, columns=["onset", "duration", "value", "trial_type"])


def _empty_events():
    return pd.DataFrame(columns=["onset", "duration", "value", "trial_type"])


def _demographics(contents):
    result = {}
    for key in ("demographics", "Demographics"):
        value = contents.get(key)
        if value is None:
            continue
        entries = np.asarray(value, dtype=object).reshape(-1)
        if key == "demographics" and getattr(value, "_fieldnames", None):
            result.update({
                field: _mat_to_python(getattr(value, field))
                for field in value._fieldnames
            })
        else:
            for entry in entries:
                name = _mat_field(entry, "name")
                item_value = _mat_field(entry, "value")
                if name is not None:
                    result[_string_value(name)] = _mat_to_python(item_value)
    return result


def _fill_missing(data, time):
    result = np.asarray(data, dtype=float).copy()
    for column in range(result.shape[1]):
        valid = np.isfinite(result[:, column])
        if not valid.any():
            result[:, column] = 1.0
        elif not valid.all():
            result[~valid, column] = np.interp(time[~valid], time[valid], result[valid, column])
    return result


def _position_array(value):
    if value is None:
        return None
    positions = np.asarray(value, dtype=float)
    if positions.ndim == 1 and positions.size in (2, 3):
        positions = positions[None, :]
    if positions.ndim != 2:
        return None
    if positions.shape[1] not in (2, 3) and positions.shape[0] in (2, 3):
        positions = positions.T
    return positions if positions.shape[1] in (2, 3) else None


def _mat_field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _mat_to_python(value):
    if hasattr(value, "_fieldnames"):
        return {
            field: _mat_to_python(getattr(value, field))
            for field in value._fieldnames
        }
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            converted = [_mat_to_python(item) for item in value.flat]
            return converted[0] if value.ndim == 0 and converted else converted
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _broadcast_values(value, size, default):
    if value is None:
        return np.full(size, default, dtype=float)
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == 1:
        return np.full(size, values[0], dtype=float)
    if values.size != size:
        raise ValueError("StimDesign onset, duration, and amplitude lengths differ")
    return values


def _string_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _string_value(value.item())
    return str(value)


def _string_list(value):
    if value is None:
        return []
    values = np.asarray(value, dtype=object).reshape(-1)
    return [_string_value(item) for item in values]
