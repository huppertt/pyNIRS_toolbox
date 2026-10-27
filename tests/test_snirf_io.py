import numpy as np
import pytest

pytest.importorskip("snirf")

import cedalion  # noqa: E402
import cedalion.dataclasses as cdc  # noqa: E402
import xarray as xr  # noqa: E402
from cedalion.dataclasses import PointType  # noqa: E402
from cedalion.io.snirf import read_snirf  # noqa: E402

from pyBrainAnalyzIR.io.snirf import probe_landmarks, write_snirf  # noqa: E402


def _recording(geo3d=True, geo2d_units="cm"):
    sources, detectors = ["S1", "S2"], ["D1"]
    time = np.arange(5) / 10.0
    amp = xr.DataArray(
        np.random.default_rng(0).uniform(1, 2, (2, 2, len(time))),
        dims=("channel", "wavelength", "time"),
        coords={"channel": ["S1D1", "S2D1"], "source": ("channel", sources),
                "detector": ("channel", detectors * 2), "wavelength": [760.0, 850.0],
                "time": time, "samples": ("time", np.arange(len(time)))},
        attrs={"units": "V"},
    ).pint.quantify()
    rec = cdc.Recording()
    rec["amp"] = amp
    labels = sources + detectors + ["Nz", "LPA", "RPA", "H1"]
    types = [PointType.SOURCE] * 2 + [PointType.DETECTOR] + [PointType.LANDMARK] * 4
    if geo3d:
        rec.geo3d = cdc.build_labeled_points(
            np.arange(21, dtype=float).reshape(7, 3) + 50, crs="digitized", units="mm",
            labels=labels, types=types,
        )
    xy = np.array([[0, 1], [2, 1], [1, 0], [1, 3], [-2, 1], [4, 1]], dtype=float)
    rec.geo2d = cdc.build_labeled_points(
        xy, crs="pos2d", units=geo2d_units,
        labels=sources + detectors + ["Nz", "LPA", "Cz"],
        types=[PointType.SOURCE] * 2 + [PointType.DETECTOR] + [PointType.LANDMARK] * 3,
    )
    return rec


def test_write_snirf_keeps_3d_and_2d_landmarks(tmp_path):
    rec = _recording()
    path = tmp_path / "probe.snirf"
    write_snirf(path, rec)
    back = read_snirf(path)[0]

    # All geo3d points survive, in their units, as landmarks.
    geo3d = back.geo3d.pint.dequantify()
    assert set(geo3d.label.values) == {"S1", "S2", "D1", "Nz", "LPA", "RPA", "H1", "Cz"}
    assert back.geo3d.pint.units == cedalion.units.mm
    np.testing.assert_allclose(
        geo3d.sel(label=list(rec.geo3d.label.values)).values,
        rec.geo3d.pint.dequantify().values,
    )
    assert np.isnan(geo3d.sel(label="Cz").values).all()  # only in 2D
    labels, _ = probe_landmarks(back.geo3d)
    assert set(labels) == {"Nz", "LPA", "RPA", "H1", "Cz"}

    # 2D positions are converted to the file's length unit (mm).
    geo2d = back.geo2d.pint.to("mm").pint.dequantify()
    expected = rec.geo2d.pint.to("mm").pint.dequantify()
    np.testing.assert_allclose(
        geo2d.sel(label=list(expected.label.values)).values, expected.values)
    assert np.isnan(geo2d.sel(label=["RPA", "H1"]).values).all()


def test_write_snirf_geo2d_only_and_without_landmarks(tmp_path):
    rec = _recording(geo3d=False, geo2d_units="mm")
    path = tmp_path / "flat.snirf"
    write_snirf(path, rec)
    back = read_snirf(path)[0]
    assert back.geo3d.sizes["label"] == 0
    assert sorted(back.geo2d.label.values) == ["Cz", "D1", "LPA", "Nz", "S1", "S2"]

    rec = _recording()
    rec.geo3d = rec.geo3d.sel(label=["S1", "S2", "D1"])
    rec.geo2d = rec.geo2d.sel(label=["S1", "S2", "D1"])
    write_snirf(tmp_path / "plain.snirf", [rec])
    back = read_snirf(tmp_path / "plain.snirf")[0]
    assert sorted(back.geo3d.label.values) == ["D1", "S1", "S2"]
