"""Short-separation channel helpers (regression filter and channel removal)."""
import numpy as np
import xarray as xr

import cedalion
import cedalion.nirs

units = cedalion.units

DEFAULT_THRESHOLD = 15 * units.mm


def short_channel_mask(ts: xr.DataArray, geo3d: xr.DataArray,
                       distance_threshold=DEFAULT_THRESHOLD) -> xr.DataArray:
    """Boolean mask over ``ts.channel``; True for short-separation channels."""
    dists = cedalion.nirs.channel_distances(ts, geo3d)
    return dists < distance_threshold


def _midpoints(ts: xr.DataArray, geo3d: xr.DataArray) -> np.ndarray:
    geo = geo3d.pint.dequantify()
    return (geo.sel(label=ts.source).values + geo.sel(label=ts.detector).values) / 2


def short_separation_filter(ts: xr.DataArray, geo3d: xr.DataArray,
                            distance_threshold=DEFAULT_THRESHOLD) -> xr.DataArray:
    """Regress the closest short-separation channel out of every channel.

    The regression is done separately for each wavelength / chromophore. The closest
    short channel of a short channel is itself, so after filtering the short channels
    are flat (equal to their mean) and carry no further information.

    Returns a time series with the same shape, coordinates and units as ``ts``. If no
    short-separation channels are present, ``ts`` is returned unchanged.
    """
    is_short = short_channel_mask(ts, geo3d, distance_threshold).values
    if not is_short.any():
        return ts

    other = [d for d in ts.dims if d not in ('time', 'channel')]
    was_quantified = hasattr(ts.data, 'units')
    y = ts.pint.dequantify() if was_quantified else ts
    y = y.transpose('time', 'channel', *other)

    mid = _midpoints(y, geo3d)
    dists = np.linalg.norm(mid[:, None] - mid[None, is_short], axis=2)
    closest = np.flatnonzero(is_short)[np.argmin(dists, axis=1)]

    yv = y.values.astype(float)
    x = yv[:, closest]
    x = x - x.mean(axis=0)
    y0 = yv - yv.mean(axis=0)
    xx = (x * x).sum(axis=0)
    beta = np.divide((x * y0).sum(axis=0), xx, out=np.zeros_like(xx), where=xx > 0)

    out = y.copy(data=yv - beta * x).transpose(*ts.dims)
    return out.pint.quantify() if was_quantified else out


def remove_short_channels(ts: xr.DataArray, geo3d: xr.DataArray,
                          distance_threshold=DEFAULT_THRESHOLD) -> xr.DataArray:
    """Return ``ts`` without its short-separation channels."""
    ts_long, _ = cedalion.nirs.split_long_short_channels(ts, geo3d, distance_threshold)
    return ts_long


def remove_short_channels_recording(rec, distance_threshold=DEFAULT_THRESHOLD):
    """Remove the short-separation channels from every channel-based entry of a
    cedalion ``Recording`` (timeseries, masks, aux_ts, statistics) and drop the optodes that are
    no longer used by any remaining channel from ``geo3d`` / ``geo2d``. Modifies ``rec``
    in place and returns it."""
    from pyBrainAnalyzIR.utils.data_selection import is_channel_series, remove_channels

    short_labels = set()
    for container in (rec.timeseries, rec.masks, rec.aux_ts):
        for value in container.values():
            if is_channel_series(value):
                mask = short_channel_mask(value, rec.geo3d, distance_threshold)
                short_labels.update(value.channel.values[mask.values].tolist())

    return remove_channels(rec, sorted(short_labels))
