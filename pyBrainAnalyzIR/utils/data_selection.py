"""Helpers to select conditions, data types and channels across the pyBrainAnalyzIR data
containers (cedalion Recording time series, Statistics and Connectivity objects)."""
import re

import numpy as np
import pandas as pd
import xarray as xr

from cedalion.dataclasses import PointType

from pyBrainAnalyzIR.dataclasses.connectivity import Connectivity
from pyBrainAnalyzIR.dataclasses.statistics import Statistics

TYPE_DIMS = ('chromo', 'wavelength')


# ---------------------------------------------------------------------------
# name matching
# ---------------------------------------------------------------------------

def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _same_name(value, name) -> bool:
    """Case-insensitive name comparison; numeric names (e.g. wavelengths) are compared as numbers."""
    a, b = _as_float(value), _as_float(name)
    if a is not None and b is not None:
        return np.isclose(a, b)
    return str(value).lower() == str(name).lower()


def match_names(values, names, regex: bool = False) -> np.ndarray:
    """Boolean mask over ``values`` that are equal to (or, with ``regex``, match) any of ``names``."""
    values = list(np.asarray(values).ravel())
    mask = np.zeros(len(values), dtype=bool)
    for name in names:
        if regex:
            pattern = re.compile(str(name), re.IGNORECASE)
            mask |= np.array([pattern.search(str(v)) is not None for v in values], dtype=bool)
        else:
            mask |= np.array([_same_name(v, name) for v in values], dtype=bool)
    return mask


def as_list(names):
    if names is None:
        return []
    if isinstance(names, (str, bytes)) or np.isscalar(names):
        return [names]
    return list(names)


# ---------------------------------------------------------------------------
# conditions (stimulus names)
# ---------------------------------------------------------------------------

def list_conditions(obj):
    """Condition / stimulus names held by a Statistics or Connectivity object."""
    if isinstance(obj, Statistics):
        return [str(c) for c in obj.list_conditions()] if hasattr(obj, 'betas') else []
    if isinstance(obj, Connectivity):
        return [str(c) for c in pd.unique(obj.coef['Condition'])] if 'Condition' in obj.coef else []
    return []


_HRF_NAME = re.compile(r'^HRF (.+)$|\[HRF (.+)\]')


def condition_stim_name(name):
    """Stimulus name of a GLM condition ('HRF A' -> 'A', 'Condition[HRF A]' -> 'A'); None if the
    condition is not a stimulus regressor (e.g. drift terms)."""
    m = _HRF_NAME.search(str(name))
    return None if m is None else (m.group(1) or m.group(2))


def filter_conditions(obj, is_kept):
    """Filter the conditions of a Statistics / Connectivity object in place using
    ``is_kept(stim_names) -> bool mask``. GLM stimulus regressors ('HRF <stim>') are matched on their
    stimulus name and nuisance regressors (drift, etc.) are always kept; if no condition is an HRF
    regressor the full condition names are matched."""
    names = list_conditions(obj)
    if not names:
        return obj
    stim = [condition_stim_name(n) for n in names]
    if any(s is not None for s in stim):
        idx = [i for i, s in enumerate(stim) if s is not None]
        keep = np.ones(len(names), dtype=bool)
        keep[idx] = is_kept(np.array([stim[i] for i in idx], dtype=object))
    else:
        keep = np.asarray(is_kept(np.array(names, dtype=object)), dtype=bool)
    return keep_conditions(obj, [n for n, k in zip(names, keep) if k])


def keep_conditions(obj, conditions):
    """Keep only ``conditions`` in a Statistics or Connectivity object (in place)."""
    conditions = list(conditions)
    if isinstance(obj, Statistics) and hasattr(obj, 'betas'):
        obj.betas = obj.betas[obj.betas.conditions.isin(conditions)]
        cov = obj.covariance[obj.covariance.conditions_rows.isin(conditions), :]
        obj.covariance = cov[:, cov.conditions_cols.isin(conditions)]
    elif isinstance(obj, Connectivity) and 'Condition' in obj.coef:
        obj.coef = obj.coef[obj.coef['Condition'].isin(conditions)].reset_index(drop=True)
    return obj


def stats_objects(rec):
    """(container, key, object) for every Statistics/Connectivity object stored in a Recording."""
    out = []
    for container in (rec.timeseries, rec.aux_obj):
        for key, value in container.items():
            if isinstance(value, (Statistics, Connectivity)):
                out.append((container, key, value))
    return out


# ---------------------------------------------------------------------------
# data types (HbO/HbR/wavelengths)
# ---------------------------------------------------------------------------

def type_dim(da: xr.DataArray):
    for dim in TYPE_DIMS:
        if dim in da.dims:
            return dim
    return None


def select_types(obj, types, keep: bool):
    """Keep (``keep=True``) or discard the data ``types`` of a time series, Statistics or Connectivity
    object. Returns the (possibly new) object, or ``None`` if ``obj`` has no data type that matches
    ``types`` (in which case it should be left unchanged)."""
    types = as_list(types)

    if isinstance(obj, xr.DataArray):
        dim = type_dim(obj)
        if dim is None:
            return None
        match = match_names(obj[dim].values, types)
        if not match.any():
            return None
        return obj.isel({dim: np.flatnonzero(match if keep else ~match)})

    if isinstance(obj, Statistics) and hasattr(obj, 'betas'):
        match = match_names(obj.betas.type.values, types)
        if not match.any():
            return None
        sel = match if keep else ~match
        obj.betas = obj.betas[sel]
        obj.covariance = obj.covariance[sel, :][:, sel]
        return obj

    if isinstance(obj, Connectivity) and 'Type_to' in obj.coef:
        match_to = match_names(obj.coef['Type_to'].values, types)
        match_from = match_names(obj.coef['Type_from'].values, types)
        if not (match_to.any() or match_from.any()):
            return None
        sel = (match_to & match_from) if keep else (~match_to & ~match_from)
        obj.coef = obj.coef[sel].reset_index(drop=True)
        return obj

    return None


# ---------------------------------------------------------------------------
# channels
# ---------------------------------------------------------------------------

def is_channel_series(obj) -> bool:
    return (isinstance(obj, xr.DataArray) and 'channel' in obj.dims
            and 'source' in obj.coords and 'detector' in obj.coords)


def remove_channels(rec, channels, prune_geometry: bool = True):
    """Remove ``channels`` (channel labels, e.g. 'S1D1') from every entry of a cedalion Recording
    (time series, masks, aux_ts, and Statistics/Connectivity objects). With ``prune_geometry`` the
    sources/detectors that are no longer used by any remaining channel are also removed from
    ``geo3d`` / ``geo2d``. Modifies ``rec`` in place and returns it."""
    channels = [str(c) for c in channels]
    if not channels:
        return rec

    used = set()
    found_series = False
    for container in (rec.timeseries, rec.masks, rec.aux_ts):
        for key, value in list(container.items()):
            if not is_channel_series(value):
                continue
            found_series = True
            container[key] = value.sel(channel=~value.channel.isin(channels))
            used.update(container[key].source.values.tolist())
            used.update(container[key].detector.values.tolist())

    for _, _, obj in stats_objects(rec):
        if isinstance(obj, Statistics) and hasattr(obj, 'betas'):
            keep = ~np.isin(obj.betas.channels.values.astype(str), channels)
            obj.betas = obj.betas[keep]
            obj.covariance = obj.covariance[keep, :][:, keep]
        elif isinstance(obj, Connectivity) and 'Channel_to' in obj.coef:
            coef = obj.coef
            obj.coef = coef[~coef['Channel_to'].isin(channels) &
                            ~coef['Channel_from'].isin(channels)].reset_index(drop=True)

    if prune_geometry and found_series:
        def prune(geo):
            if geo is None or 'type' not in geo.coords:
                return geo
            is_optode = geo.type.isin([PointType.SOURCE, PointType.DETECTOR]).values
            return geo.sel(label=~is_optode | geo.label.isin(list(used)).values)

        rec.geo3d = prune(rec.geo3d)
        rec.geo2d = prune(rec.geo2d)
    return rec
