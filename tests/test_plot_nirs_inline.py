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


def test_scalp_view_does_not_log_fixed_limit_aspect_warnings(caplog):
    # Matplotlib logs (not warns) "Ignoring fixed x limits to fulfill fixed data aspect".
    rec = _recording_with_scalp_landmarks()
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_axes([0.02, 0.05, 0.22, 0.9])
    draw_probe(rec, rec["amp"], ax, plot_on_scalp=True)
    with caplog.at_level("WARNING", logger="matplotlib"):
        fig.canvas.draw()
    assert "Ignoring fixed" not in caplog.text
    assert ax.get_xlim() == pytest.approx((-1.15, 1.15))
    plt.close(fig)


def test_probe_view_helpers_report_available_views():
    from pyBrainAnalyzIR.vis.plot_nirs_inline import probe_layout, probe_views_available

    plain, _ = simData.Data(snr=10)
    rec = _recording_with_scalp_landmarks()
    assert not probe_views_available(plain)
    assert probe_views_available(rec)
    points, on_scalp = probe_layout(rec, True)
    assert on_scalp and np.abs(np.asarray(points)).max() <= 1.0 + 1e-9
    points, on_scalp = probe_layout(rec, False)
    assert not on_scalp and points.shape == rec.geo2d.shape
    assert probe_layout(plain, True)[1] is False


def _channel_statistics(rec):
    from pyBrainAnalyzIR.dataclasses.statistics import Statistics

    channels = rec["amp"].channel.values[:3].tolist()
    stats = Statistics()
    stats.geo2d, stats.geo3d = rec.geo2d, rec.geo3d
    stats.betas = xr.DataArray(
        [3.0, -0.5, 6.0], dims="indices",
        coords={"channels": ("indices", channels), "type": ("indices", ["HbO"] * 3),
                "conditions": ("indices", ["A"] * 3)},
    )
    stats.covariance = xr.DataArray(np.eye(3), dims=("rows", "cols"))
    stats.dof = 100
    return stats


@pytest.mark.parametrize("plot_on_scalp", [True, False])
def test_channel_statistics_draw_both_probe_views(plot_on_scalp):
    rec = _recording_with_scalp_landmarks()
    _channel_statistics(rec).draw(plot_on_scalp=plot_on_scalp, thresh="p<0.05")
    fig = plt.gcf()
    ax = fig.axes[0]
    assert any(isinstance(patch, Circle) for patch in ax.patches) == plot_on_scalp
    drawn = [line for line in ax.lines if line.get_linewidth() in (1, 3)]
    assert sorted(line.get_linewidth() for line in drawn)[-2:] == [3, 3]
    if not plot_on_scalp:
        optodes = rec.geo2d.pint.dequantify().values
        assert ax.get_xlim()[0] < optodes[:, 0].min() and ax.get_xlim()[1] > optodes[:, 0].max()
    plt.close("all")


def _hyperscan_connectivity(rec, landmarks=True):
    from pyBrainAnalyzIR.dataclasses.connectivity import Connectivity
    from pyBrainAnalyzIR.utils.geo_transforms import geo_rotation

    if not landmarks:
        rec, _ = simData.Data(snr=10)
    channel = rec["amp"].channel.values[0]
    conn = Connectivity()
    for name, theta, offset in (("A", -np.pi / 2, (-100.0, 0.0)), ("B", np.pi / 2, (100.0, 0.0))):
        conn.geo2d[name], conn.geo3d[name] = rec.geo2d, rec.geo3d
        rotator = geo_rotation()
        rotator.set_rotation2D([theta])
        rotator.translation2D = np.array(offset)
        conn.geo2d_rotation_data[name] = rotator
    conn.coef = pd.DataFrame({
        "Condition": ["rest"], "Channel_to": [channel], "Channel_from": [channel],
        "Type_to": ["hbo"], "Type_from": ["hbo"], "geo_to": ["A"], "geo_from": ["B"],
        "Coeff": [0.5], "Z": [0.55], "ZstdErr": [0.1], "Tstat": [5.5], "Pvalue": [0.001], "dfe": [100],
    })
    return conn


def test_hyperscanning_scalp_view_places_facing_heads():
    conn = _hyperscan_connectivity(_recording_with_scalp_landmarks())
    layouts, on_scalp = conn.probe_layouts(True)
    assert on_scalp
    centers = {name: outline[:2, 2] for name, (_, outline) in layouts.items()}
    assert centers["A"] == pytest.approx([-1.25, 0.0], abs=1e-9)
    assert centers["B"] == pytest.approx([1.25, 0.0], abs=1e-9)
    # Each head's nose (+y in the head frame) points at the other participant.
    noses = {name: outline[:2, :2] @ [0, 1] for name, (_, outline) in layouts.items()}
    assert noses["A"] == pytest.approx([1, 0], abs=1e-9)
    assert noses["B"] == pytest.approx([-1, 0], abs=1e-9)
    for name, (points, _) in layouts.items():
        assert np.linalg.norm(np.asarray(points) - centers[name], axis=1).max() <= 1.0 + 1e-9
    layouts_2d, on_scalp = conn.probe_layouts(False)
    assert not on_scalp
    assert layouts_2d["A"][0].pint.dequantify().values == pytest.approx(
        conn.geo_rotated2D("A").pint.dequantify().values)


@pytest.mark.parametrize("plot_on_scalp", [True, False])
def test_connectivity_show_draws_both_probe_views(plot_on_scalp):
    conn = _hyperscan_connectivity(_recording_with_scalp_landmarks())
    fig = conn.show(plot_on_scalp=plot_on_scalp) or plt.gcf()
    ax = fig.axes[0]
    assert sum(isinstance(patch, Circle) for patch in ax.patches) == (2 if plot_on_scalp else 0)
    plt.close("all")


def test_connectivity_scalp_view_falls_back_without_landmarks():
    conn = _hyperscan_connectivity(None, landmarks=False)
    assert conn.probe_layouts(True)[1] is False


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
