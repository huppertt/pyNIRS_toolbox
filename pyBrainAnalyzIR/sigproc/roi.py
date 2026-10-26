"""ROI time-course/statistics averages guided by nirs.util.roiAverage and ROIhelper."""
import copy
import warnings

import cedalion.nirs
import numpy as np
import pandas as pd
import xarray as xr

from pyBrainAnalyzIR.dataclasses.roi import GeoROI, get_geo_roi, quality_masks
from pyBrainAnalyzIR.math.glm.solve import _recording_quality_mask


def _roi_membership(roi_table, data):
    """Validate a membership table against physical coordinates for both ROI paths.

    Direct callers: ``roi_average`` and ``roi_average_statistics``.
    Dependencies: pandas/numpy. Tests: ``test_roi.py``, ``test_roi_statistics.py``.
    """
    if not isinstance(roi_table, pd.DataFrame):
        raise TypeError("roi_table must be a pandas DataFrame.")
    table = roi_table.copy()
    if "name" not in table and "Name" in table:
        table = table.rename(columns={"Name": "name"})
    required = {"source", "detector", "name"}
    if not required.issubset(table) or table.empty:
        raise ValueError("ROI table requires source, detector, and name columns.")
    if table[list(required)].isna().any().any():
        raise ValueError("ROI names and optode labels must not be missing.")
    if not table["name"].map(lambda name: isinstance(name, str) and bool(name.strip())).all():
        raise ValueError("ROI names must be nonempty strings.")
    if "weight" not in table:
        table["weight"] = 1.0
    weights = pd.to_numeric(table["weight"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("ROI weights must be finite.")
    table["weight"] = weights
    if table.duplicated(["source", "detector", "name"]).any():
        raise ValueError("Duplicate source-detector pair within an ROI.")
    if not {"source", "detector"}.issubset(data.coords):
        raise ValueError("ROI input needs source and detector coordinates.")
    pairs = list(zip(data.source.values.astype(str), data.detector.values.astype(str)))
    if len(set(pairs)) != len(pairs):
        raise ValueError("Input source-detector pairs must be unique.")
    names = list(pd.unique(table["name"]))
    membership = np.zeros((len(names), len(pairs)))
    for _, row in table.iterrows():
        pair = (str(row["source"]), str(row["detector"]))
        if pair not in pairs:
            raise ValueError(f"ROI source-detector pair {pair!r} is absent from input.")
        membership[names.index(row["name"]), pairs.index(pair)] = row["weight"]
    if not np.any(membership != 0, axis=1).all():
        raise ValueError("Each ROI needs at least one nonzero weight.")
    return table, names, membership


def roi_average(
    rec, roi_table, input_name="amp", output_name="amp_roi",
    roi_name="default", weighted=True, overwrite=False,
):
    """Add an ROI average of time courses or fitted channel statistics.

    ``roi_table`` has source, detector, name (or MATLAB Name), and optional
    finite signed weight columns. Effective weights are table weights divided by
    temporal variance when ``weighted=True`` (the default), normalized over
    the absolute sum of available channels at each sample. Negative weights
    contribute in the opposite direction. Masked samples do not estimate variance
    or contribute to averages. Entirely masked channels are excluded with a
    warning. Constant/undersampled contributors with otherwise valid samples
    are rejected rather than assigned infinite weight.

    Statistics use MATLAB's linear contrast propagation: beta_roi = W beta and
    covariance_roi = W covariance W.T, including cross-condition/ROI covariance.
    With ``weighted=True`` their table weights are divided by coefficient
    variance, separately per condition/type. Use ``weighted=False`` to compare
    processing order with OLS: only fixed identical weights and identical designs
    and valid samples commute. Temporal-variance and coefficient-variance weights
    are generally different, even for OLS.

    Unlike MATLAB's legacy scaling, averages preserve the input signal units
    and amplitude. Data are modified in place and the input object is returned.
    Direct dependencies: GeoROI, Cedalion channel distances, and the local mask
    alignment helper. Example: ``examples/15_ROI_Analysis.ipynb``.
    A direct Statistics input returns a new Statistics object; recording/dataset
    inputs are updated in place. Tests: ``tests/test_roi.py`` and
    ``tests/test_roi_statistics.py``.
    """
    from pyBrainAnalyzIR.dataclasses.statistics import Statistics
    if isinstance(rec, Statistics):
        return roi_average_statistics(rec, roi_table, roi_name, weighted)
    if hasattr(rec, "dataset"):
        if input_name in rec.statistics:
            if not overwrite and output_name in rec.statistics:
                raise ValueError("ROI output already exists; choose a new name.")
            if input_name == output_name:
                raise ValueError("ROI output must not replace its source statistics.")
            rec[output_name] = roi_average_statistics(
                rec[input_name], roi_table, roi_name, weighted
            )
            return rec
        for recording in rec.dataset:
            roi_average(
                recording, roi_table, input_name, output_name,
                roi_name, weighted, overwrite,
            )
        return rec
    if not isinstance(roi_name, str) or not roi_name.strip():
        raise ValueError("roi_name must be a nonempty string.")
    if input_name == output_name:
        raise ValueError("ROI output must not replace its source time series.")
    if not overwrite and (
        output_name in rec.timeseries
        or roi_name in rec.aux_obj.get("roi_definitions", {})
    ):
        raise ValueError("ROI output or definition already exists; choose a new name.")
    if overwrite and any(
        key != output_name and (
            isinstance(value, xr.DataArray) and value.attrs.get("roi_definition") == roi_name
            or isinstance(value, Statistics) and value.geoROI is not None and value.geoROI.name == roi_name
        )
        for key, value in rec.timeseries.items()
    ):
        raise ValueError("Cannot replace an ROI definition used by other time series or statistics.")
    data = rec[input_name]
    if isinstance(data, Statistics):
        pooled = roi_average_statistics(data, roi_table, roi_name, weighted)
        rec.aux_obj.setdefault("roi_definitions", {})[roi_name] = pooled.geoROI
        rec[output_name] = pooled
        return rec
    if get_geo_roi(rec, data) is not None:
        raise ValueError("Define ROIs from physical channel data, not another ROI.")
    table, names, membership = _roi_membership(roi_table, data)

    plain = data.pint.dequantify()
    good = _recording_quality_mask(data, quality_masks(rec, data))
    valid = np.isfinite(plain)
    if good is not None:
        valid = valid & good
    subtype = [dim for dim in data.dims if dim not in ("time", "channel")]
    if subtype:
        valid = valid.all(subtype)
    values = plain.where(valid)
    count = values.count("time")
    mean = values.sum("time") / count.clip(min=1)
    variance = ((values - mean).fillna(0) ** 2).sum("time") / count.clip(min=1)
    used = xr.DataArray(
        membership.any(axis=0), dims="channel", coords={"channel": data.channel}
    )
    if weighted:
        if bool((((variance <= 0) | (count < 2)) & (count > 0) & used).any()):
            raise ValueError(
                "Inverse-variance ROI averaging requires positive finite variance "
                "in every contributing channel/subtype; use weighted=False otherwise."
            )
        if bool(((count == 0) & used).any()):
            warnings.warn(
                "Entirely masked channels are excluded from ROI variance weighting.",
                UserWarning, stacklevel=2,
            )
    base = xr.DataArray(
        membership, dims=("region", "channel"),
        coords={"region": names, "channel": data.channel},
    )
    effective = base / variance.where(variance > 0) if weighted else base
    effective = effective.fillna(0)
    absolute_weight_sum = np.abs(effective).sum("channel")
    effective = effective / absolute_weight_sum.where(absolute_weight_sum > 0)
    effective = effective.fillna(0)
    denominator = np.abs(effective.where(valid, 0)).sum("channel")
    averaged = (values.fillna(0) * effective).sum("channel") / denominator
    averaged = averaged.rename(region="channel").transpose(*data.dims)
    averaged.attrs = data.attrs.copy()
    averaged.attrs["units"] = str(data.pint.units)
    averaged = averaged.pint.quantify()
    averaged.attrs.update(roi_definition=roi_name, roi_mask_key=output_name)
    roi_good = denominator > 0
    mask_subtypes = [dim for dim in subtype if dim in roi_good.dims]
    if mask_subtypes:
        roi_good = roi_good.all(mask_subtypes)
    roi_good = roi_good.rename(region="channel").transpose("time", "channel")
    distances = cedalion.nirs.channel_distances(data, rec.geo3d)
    distance_weights = np.abs(base) / np.abs(base).sum("channel")
    roi_distances = xr.dot(
        distance_weights, distances, dim="channel"
    ).rename(region="channel")
    stored_weights = effective.rename(channel="source_channel", region="channel").assign_coords(
        source=("source_channel", data.source.values),
        detector=("source_channel", data.detector.values),
    )
    geometry = GeoROI(
        roi_name, table.copy(deep=True), roi_distances,
        stored_weights, copy.deepcopy(rec.geo3d),
    )
    rec.aux_obj.setdefault("roi_definitions", {})[roi_name] = geometry
    rec.aux_obj.setdefault("roi_masks", {})[output_name] = {"roi_valid": roi_good}
    rec[output_name] = averaged
    return rec


def roi_average_statistics(stats, roi_table, roi_name="default", weighted=True):
    """Pool channel betas and their full covariance using MATLAB-style contrasts.

    Direct caller: ``roi_average`` (including ``ROIAverage``). Dependencies:
    ``_roi_membership``, ``GeoROI``, Cedalion channel distances, numpy/xarray.
    Example: ``examples/15_ROI_Analysis.ipynb``; tests: ``test_roi_statistics.py``.
    Missing coefficients/variances are excluded and weights renormalized per
    estimate, with a warning. Empty estimates retain NaN betas/covariance.
    Nonfinite covariance between retained estimates is propagated, not imputed.
    The input remains unchanged; its residual degrees of freedom are preserved.
    """
    if not isinstance(roi_name, str) or not roi_name.strip():
        raise ValueError("roi_name must be a nonempty string.")
    if stats.geoROI is not None:
        raise ValueError("Define ROIs from physical channel statistics, not another ROI.")
    beta = stats.betas
    channels = list(pd.unique(beta.channels.values))
    types = list(pd.unique(beta.type.values))
    conditions = list(pd.unique(beta.conditions.values))
    if beta.ndim != 1 or stats.covariance.shape != (beta.size, beta.size):
        raise ValueError("Statistics require a coefficient vector and matching covariance matrix.")
    labels = list(zip(beta.channels.values, beta.type.values, beta.conditions.values))
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate channel/type/condition estimates in statistics.")
    # GLM outputs retain explicit optode coordinates. Legacy stats are resolved by
    # exact source+detector label matches, never by splitting arbitrary labels.
    pairs = []
    for channel in channels:
        indices = np.flatnonzero(beta.channels.values == channel)
        if {"source", "detector"}.issubset(beta.coords):
            matches = set(zip(beta.source.values[indices], beta.detector.values[indices]))
        else:
            from cedalion.dataclasses.geometry import PointType
            sources = stats.geo3d.label.values[stats.geo3d.type.values == PointType.SOURCE]
            detectors = stats.geo3d.label.values[stats.geo3d.type.values == PointType.DETECTOR]
            matches = {
                (source, detector) for source in sources for detector in detectors
                if str(source) + str(detector) == str(channel)
            }
        if len(matches) != 1:
            raise ValueError(f"Cannot resolve a unique source-detector pair for {channel!r}.")
        pairs.append(next(iter(matches)))
    physical = xr.DataArray(
        np.zeros(len(channels)), dims="channel",
        coords={"channel": channels, "source": ("channel", [pair[0] for pair in pairs]),
                "detector": ("channel", [pair[1] for pair in pairs])},
    )
    table, names, membership = _roi_membership(roi_table, physical)
    covariance = np.asarray(stats.covariance, dtype=float)
    values = np.asarray(beta, dtype=float)
    variances = np.diag(covariance)
    valid = np.isfinite(values) & np.isfinite(variances) & (variances >= 0)
    targets = [(name, kind, condition) for name in names for kind in types for condition in conditions]
    projection = np.zeros((len(targets), beta.size))
    effective = np.zeros((len(names), len(channels), len(types), len(conditions)))
    excluded = False
    for row, (name, kind, condition) in enumerate(targets):
        weights = np.array([
            membership[names.index(name), channels.index(channel)]
            if signal_type == kind and cond == condition else 0
            for channel, signal_type, cond in labels
        ])
        excluded |= bool(np.any((weights > 0) & ~valid))
        if weighted and np.any((weights > 0) & valid & (variances == 0)):
            raise ValueError("Inverse-variance statistics pooling requires positive coefficient variance.")
        weights[~valid] = 0
        if weighted:
            weights = np.divide(weights, variances, out=np.zeros_like(weights), where=valid & (variances > 0))
        absolute_weight_sum = np.abs(weights).sum()
        if absolute_weight_sum > 0:
            weights /= absolute_weight_sum
        projection[row] = weights
        for index in np.flatnonzero(weights):
            effective[names.index(name), channels.index(labels[index][0]),
                      types.index(kind), conditions.index(condition)] = weights[index]
    if excluded:
        warnings.warn(
            "Missing channel statistics excluded from ROI pooling; empty ROI estimates remain NaN.",
            UserWarning, stacklevel=2,
        )
    pooled_beta = np.full(len(targets), np.nan)
    pooled_covariance = np.full((len(targets), len(targets)), np.nan)
    contributors = [np.flatnonzero(row) for row in projection]
    for row, indices in enumerate(contributors):
        if not indices.size:
            continue
        pooled_beta[row] = projection[row, indices] @ values[indices]
        for col, other in enumerate(contributors):
            if other.size:
                pooled_covariance[row, col] = (
                    projection[row, indices] @ covariance[np.ix_(indices, other)]
                    @ projection[col, other]
                )
    finite_output = np.isfinite(pooled_beta)
    if not np.isfinite(pooled_covariance[np.ix_(finite_output, finite_output)]).all():
        warnings.warn("Nonfinite covariance propagated into ROI statistics.", UserWarning, stacklevel=2)
    output = stats.copy()
    output.description = "ROI pooled statistics: " + stats.description
    output.betas = xr.DataArray(
        pooled_beta, dims="indices",
        coords={"indices": np.arange(len(targets)),
                "channels": ("indices", [target[0] for target in targets]),
                "type": ("indices", [target[1] for target in targets]),
                "conditions": ("indices", [target[2] for target in targets])},
        attrs=beta.attrs.copy(),
    )
    output.covariance = xr.DataArray(
        pooled_covariance, dims=("rows", "cols"),
        coords={
            axis: np.arange(len(targets)) for axis in ("rows", "cols")
        } | {
            f"{label}_{axis}": (axis, output.betas[label].values)
            for axis in ("rows", "cols") for label in ("channels", "type", "conditions")
        },
        attrs=stats.covariance.attrs.copy(),
    )
    output.roi_pooling_matrix = xr.DataArray(projection, dims=("roi_estimate", "channel_estimate"))
    if hasattr(output, "satterthwaite_dof"):
        del output.satterthwaite_dof
    distances = cedalion.nirs.channel_distances(physical, stats.geo3d)
    base = xr.DataArray(membership, dims=("region", "channel"),
                        coords={"region": names, "channel": channels})
    absolute_weights = np.abs(base)
    roi_distances = xr.dot(
        absolute_weights / absolute_weights.sum("channel"), distances, dim="channel"
    ).rename(region="channel")
    stored_weights = xr.DataArray(
        effective, dims=("channel", "source_channel", "type", "condition"),
        coords={"channel": names, "source_channel": channels, "type": types,
                "condition": conditions, "source": ("source_channel", physical.source.values),
                "detector": ("source_channel", physical.detector.values)},
    )
    output.geoROI = GeoROI(
        roi_name, table, roi_distances, stored_weights, copy.deepcopy(stats.geo3d)
    )
    return output


def roi_od2conc(rec, od, dpf, spectrum):
    """Apply Cedalion's MBLL algebra using ROI fixed distances, not optodes.

    Direct caller: preprocessing.mbll. Tests: tests/test_roi.py.
    """
    from cedalion import units, xrutils
    geometry = get_geo_roi(rec, od)
    extinction = cedalion.nirs.get_extinction_coefficients(spectrum, od.wavelength)
    inverse = xrutils.pinv(extinction)
    if geometry is None:
        raise ValueError("ROI MBLL requires a registered ROI definition.")
    distance = geometry.distances.sel(channel=od.channel).pint.to("mm")
    path = distance if dpf[0] != 1 else 1 * units.mm
    result = xr.dot(inverse, od / (path * dpf), dim=["wavelength"])
    return result.pint.to("micromolar").pint.quantify(
        {"time": od.time.attrs["units"]}
    ).rename("concentration")


def roi_conc2od(rec, conc, dpf, spectrum):
    """Invert ROI MBLL with the same fixed distances; tested in tests/test_roi.py."""
    geometry = get_geo_roi(rec, conc)
    if geometry is None:
        raise ValueError("ROI conversion requires a registered ROI definition.")
    extinction = cedalion.nirs.get_extinction_coefficients(
        spectrum, dpf.wavelength.values.astype(float)
    )
    result = xr.dot(
        extinction, conc.pint.to("molar"), dim=["chromo"]
    ) * (geometry.distances.sel(channel=conc.channel).pint.to("mm") * dpf)
    return result.rename("optical_density").pint.quantify({"time": "s"})
