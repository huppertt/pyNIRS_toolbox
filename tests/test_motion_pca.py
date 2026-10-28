"""Guarded PCA motion correction matches Cedalion and handles its edge cases."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cedalion")

import xarray as xr  # noqa: E402

import cedalion  # noqa: E402
import cedalion.sigproc.motion as cedalion_motion  # noqa: E402
from pyBrainAnalyzIR.sigproc.motion_pca import (  # noqa: E402
    motion_correct_pca, motion_correct_pca_recurse,
)

pytestmark = pytest.mark.requires_cedalion
units = cedalion.units
TIME = np.arange(0, 60, 0.1)


def _series(offsets=()):
    rng = np.random.default_rng(0)
    values = np.cumsum(rng.normal(size=(4, 2, TIME.size)), axis=-1) * 0.01
    for start, stop, step in offsets:
        values[:, :, start:stop] += step
    return xr.DataArray(
        values, dims=["channel", "wavelength", "time"],
        coords={
            "channel": ["S1D1", "S1D2", "S2D1", "S2D2"], "wavelength": [760.0, 850.0],
            "time": TIME, "samples": ("time", np.arange(TIME.size)),
            "source": ("channel", ["S1", "S1", "S2", "S2"]),
            "detector": ("channel", ["D1", "D2", "D1", "D2"]),
        },
    ).pint.quantify("1")


def _mask(*segments):
    values = np.ones(TIME.size, dtype=bool)
    for start, stop in segments:
        values[start:stop] = False
    return xr.DataArray(values, dims=["time"], coords={"time": TIME})


def _values(ts):
    return ts.pint.dequantify().values


@pytest.mark.parametrize("segments,n_sv", [
    ([(100, 120)], 0.97), ([(0, 20)], 0.97), ([(580, 600)], 2),
    ([(50, 70), (200, 260), (400, 410)], 0.8),
])
def test_pca_matches_cedalion_where_cedalion_succeeds(segments, n_sv):
    ts, mask = _series(), _mask(*segments)
    expected = cedalion_motion.pca(ts, mask, n_sv=n_sv)
    actual = motion_correct_pca(ts, mask, n_sv=n_sv)
    np.testing.assert_array_equal(_values(actual[0]), _values(expected[0]))
    assert actual[1] == expected[1]
    np.testing.assert_allclose(actual[2], expected[2])


@pytest.mark.parametrize("segments", [[(100, 101)], [(0, 1), (100, 110)]])
def test_pca_handles_inputs_that_break_cedalion(segments):
    ts, mask = _series(), _mask(*segments)
    with pytest.raises((ValueError, IndexError)):
        cedalion_motion.pca(ts, mask)
    cleaned, _, _ = motion_correct_pca(ts, mask)
    assert cleaned.dims == ts.dims
    assert np.isfinite(_values(cleaned)).all()


def test_pca_without_motion_returns_input_unchanged():
    ts = _series()
    with pytest.raises(ValueError):
        cedalion_motion.pca(ts, _mask())
    cleaned, n_sv, svs = motion_correct_pca(ts, _mask())
    assert cleaned is ts
    assert n_sv == 0 and svs.size == 0


@pytest.mark.parametrize("offsets", [[], [(300, 600, 1.0), (450, 455, 2.0)]])
def test_recursive_pca_matches_cedalion(offsets):
    ts = _series(offsets)
    options = dict(t_motion=0.5 * units.s, t_mask=1 * units.s, stdev_thresh=7, amp_thresh=0.5)
    expected = cedalion_motion.pca_recurse(ts, **options)
    actual = motion_correct_pca_recurse(ts, **options)
    np.testing.assert_array_equal(_values(actual[0]), _values(expected[0]))
    assert actual[2] == expected[2]
    xr.testing.assert_equal(actual[3], expected[3])
