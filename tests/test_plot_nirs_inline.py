"""Tests for inline fNIRS probe plotting helpers."""
from __future__ import annotations

import matplotlib
import pytest
import xarray as xr

pytest.importorskip("cedalion")

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from cedalion import units  # noqa: E402
from cedalion.dataclasses.geometry import PointType  # noqa: E402
from pyBrainAnalyzIR.testing import simData  # noqa: E402
from pyBrainAnalyzIR.vis.plot_nirs_inline import draw_probe  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


def _recording_with_scalp_landmarks():
    rec, _ = simData.Data(snr=10)
    landmarks = xr.DataArray(
        [[0.0, 100.0, 0.0], [-100.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        dims=("label", "pos"),
        coords={
            "label": ["Nz", "LPA", "RPA"],
            "type": ("label", [PointType.LANDMARK] * 3),
        },
    ).pint.quantify(units.mm)
    rec.geo3d = xr.concat([rec.geo3d, landmarks], dim="label")
    return rec


def test_draw_probe_uses_scalp_projection_when_landmarks_are_available():
    rec = _recording_with_scalp_landmarks()
    fig, ax = plt.subplots()

    lines = draw_probe(rec, rec["amp"], ax, plot_on_scalp=True)

    assert len(lines) == len(rec["amp"].channel)
    assert any(isinstance(patch, Circle) for patch in ax.patches)
    assert ax.get_xlim() == pytest.approx((-1.15, 1.15))
    plt.close(fig)


def test_draw_probe_legacy_mode_uses_recording_probe_geometry():
    rec, _ = simData.Data(snr=10)
    fig, ax = plt.subplots()

    lines = draw_probe(rec, rec["amp"], ax, plot_on_scalp=False)

    assert len(lines) == len(rec["amp"].channel)
    assert not any(isinstance(patch, Circle) for patch in ax.patches)
    plt.close(fig)


def test_draw_probe_falls_back_when_scalp_landmarks_are_missing():
    import warnings

    rec, _ = simData.Data(snr=10)
    fig, ax = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        lines = draw_probe(rec, rec["amp"], ax, plot_on_scalp=True)

    assert len(lines) == len(rec["amp"].channel)
    # Without landmarks the probe is drawn in its own geometry, not on the unit scalp.
    assert not any(isinstance(patch, Circle) for patch in ax.patches)
    assert ax.get_xlim()[1] > 1.15
    plt.close(fig)
