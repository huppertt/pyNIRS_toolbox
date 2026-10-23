"""Tests for inline fNIRS probe plotting helpers."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("cedalion")

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from cedalion import units  # noqa: E402
from cedalion.dataclasses.geometry import PointType  # noqa: E402
from pyBrainAnalyzIR.testing import simData  # noqa: E402
from pyBrainAnalyzIR.vis.plot_nirs_inline import draw_probe, plot  # noqa: E402
from pyBrainAnalyzIR.sigproc.roi import roi_average  # noqa: E402

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


def _overlapping_roi_table(rec):
    indices = [0, 1, 0, 2]
    return pd.DataFrame({
        "source": rec["amp"].source.values[indices],
        "detector": rec["amp"].detector.values[indices],
        "weight": [2.0, 1.0, 4.0, 4.0],
        "name": ["Region A", "Region A", "Region B", "Region B"],
    })


def _click(fig, axes):
    fig.canvas.draw()
    x, y = axes.transAxes.transform((0.5, 0.5))
    for name in ("button_press_event", "button_release_event"):
        fig.canvas.callbacks.process(name, MouseEvent(name, fig.canvas, x, y, button=1))


@pytest.mark.parametrize("scalp", [False, True])
def test_roi_probe_retains_physical_layout(scalp, monkeypatch):
    rec = _recording_with_scalp_landmarks() if scalp else simData.Data(snr=10)[0]
    roi_average(rec, _overlapping_roi_table(rec))
    monkeypatch.setattr(plt, "show", lambda: None)
    physical_fig, physical_ax = plt.subplots()
    expected = draw_probe(rec, rec["amp"], physical_ax)
    fig = plot(rec, type="amp_roi", show_stim=False)
    probe = fig.axes[0]
    lines = probe.lines[1:] if scalp else probe.lines
    assert len(lines) == len(expected)
    for line, physical in zip(lines, expected):
        np.testing.assert_array_equal(line.get_xydata(), physical.get_xydata())
    assert any(isinstance(patch, Circle) for patch in probe.patches) == scalp
    assert [button.label.get_text() for button in fig._roi_buttons] == ["Region A", "Region B"]
    assert all(button.ax.get_position().y1 < probe.get_position().y0 for button in fig._roi_buttons)
    plt.close(physical_fig)
    plt.close(fig)


def test_roi_buttons_blend_definition_weights_and_toggle(monkeypatch):
    rec, _ = simData.Data(snr=10)
    roi_average(rec, _overlapping_roi_table(rec))
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = plot(rec, type="amp_roi", show_stim=False)
    probe, signals = fig.axes[:2]
    buttons = fig._roi_buttons
    np.testing.assert_allclose(to_rgb(probe.lines[0].get_color()), [1, 0, 0])
    np.testing.assert_allclose(to_rgb(probe.lines[1].get_color()), [1, 0.5, 0.5])
    assert all(to_rgb(line.get_color()) == to_rgb("red") for line in signals.lines)
    assert [line.get_linestyle() for line in signals.lines] == ["-", "--"]

    _click(fig, buttons[1].ax)
    np.testing.assert_allclose(to_rgb(probe.lines[0].get_color()), [0.5, 0, 0.5])
    np.testing.assert_allclose(to_rgb(probe.lines[2].get_color()), [0, 0, 1])
    assert [to_rgb(line.get_color()) for line in signals.lines] == [
        to_rgb("red"), to_rgb("red"), to_rgb("blue"), to_rgb("blue"),
    ]
    _click(fig, buttons[0].ax)
    assert all(to_rgb(line.get_color()) == to_rgb("blue") for line in signals.lines)
    assert to_rgb(probe.lines[1].get_color()) == to_rgb("black")
    # The last region remains selected, and physical channel clicks do not select ROIs.
    _click(fig, buttons[1].ax)
    _click(fig, probe)
    assert len(signals.lines) == 2
    assert all(to_rgb(line.get_color()) == to_rgb("blue") for line in signals.lines)
    assert buttons[1].color == "blue"
    plt.close(fig)


def test_roi_colors_ignore_variance_weights_and_survive_missing_source(monkeypatch):
    rec, _ = simData.Data(snr=10)
    table = _overlapping_roi_table(rec)
    roi_average(rec, table)
    roi_average(rec, table, output_name="amp_equal", roi_name="equal", weighted=False)
    monkeypatch.setattr(plt, "show", lambda: None)
    weighted_fig = plot(rec, type="amp_roi", show_stim=False)
    del rec.timeseries["amp"]
    equal_fig = plot(rec, type="amp_equal", show_stim=False)
    for fig in (weighted_fig, equal_fig):
        _click(fig, fig._roi_buttons[1].ax)
    assert len(equal_fig.axes[0].lines) == 16
    for weighted, equal in zip(weighted_fig.axes[0].lines, equal_fig.axes[0].lines):
        np.testing.assert_allclose(to_rgb(weighted.get_color()), to_rgb(equal.get_color()))
        np.testing.assert_array_equal(weighted.get_xydata(), equal.get_xydata())
    plt.close(weighted_fig)
    plt.close(equal_fig)


def test_roi_overlap_blends_unequal_definition_contributions(monkeypatch):
    rec, _ = simData.Data(snr=10)
    table = _overlapping_roi_table(rec)
    table.loc[0, "weight"] = 0.5
    roi_average(rec, table)
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = plot(rec, type="amp_roi", show_stim=False)
    _click(fig, fig._roi_buttons[1].ax)
    np.testing.assert_allclose(
        to_rgb(fig.axes[0].lines[0].get_color()), [1 / 3, 0, 2 / 3],
    )
    plt.close(fig)


def test_physical_channel_click_selection_is_preserved(monkeypatch):
    rec, _ = simData.Data(snr=10)
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = plot(rec, show_stim=False, plot_on_scalp=False)
    probe, signals = fig.axes
    fig.canvas.draw()
    x, y = probe.transData.transform(probe.lines[3].get_xydata().mean(axis=0))
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    fig.canvas.callbacks.process("button_press_event", event)
    assert signals.get_title() == str(rec["amp"].channel.values[3])
    assert len(signals.lines) == 2
    assert to_rgb(probe.lines[3].get_color()) == to_rgb("red")
    assert not hasattr(fig, "_roi_buttons")
    plt.close(fig)
