"""Tests for the linked data-mask viewer."""
from __future__ import annotations

import copy
import importlib
import inspect

import numpy as np
import pytest
import xarray as xr
from types import SimpleNamespace

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("cedalion")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QToolButton  # noqa: E402

from pyBrainAnalyzIR.dataclasses.dataset import DataSet  # noqa: E402
from pyBrainAnalyzIR.testing import simData  # noqa: E402
from pyBrainAnalyzIR.vis.NIRSviewIR import NIRSviewIR  # noqa: E402
from pyBrainAnalyzIR.vis.data_quality_manager import (  # noqa: E402
    data_quality_manager,
)
from pyBrainAnalyzIR.vis.demographics_manager import (  # noqa: E402
    demographics_manager,
)
from pyBrainAnalyzIR.vis.stimulus_manager import stimulus_manager  # noqa: E402
masking_module = importlib.import_module(
    "pyBrainAnalyzIR.vis.data_masking_manager"
)
from pyBrainAnalyzIR.vis.data_masking_manager import (  # noqa: E402
    DataMaskingManager,
    _segments,
)

pytestmark = [
    pytest.mark.requires_qt,
    pytest.mark.requires_cedalion,
    pytest.mark.usefixtures("qt_app"),
]


def _recording():
    return simData.Data(snr=10)[0]


def test_segments_return_masked_ranges():
    starts, stops = _segments([False, True, True, False, True])

    assert starts.tolist() == [1, 4]
    assert stops.tolist() == [3, 5]


def test_gui_launchers_are_nonblocking_by_default():
    launchers = (
        NIRSviewIR,
        data_quality_manager,
        masking_module.data_masking_manager,
        demographics_manager,
        stimulus_manager,
    )

    for launcher in launchers:
        assert inspect.signature(launcher).parameters["block"].default is False


def test_manager_uses_ipython_qt_loop_without_blocking(monkeypatch):
    real_app = QApplication.instance()

    class AppProxy:
        quit_on_last_window_closed = None

        @staticmethod
        def instance():
            return app_proxy

        def setQuitOnLastWindowClosed(self, value):
            self.quit_on_last_window_closed = value

        @staticmethod
        def exec():
            pytest.fail("The notebook launcher must not start a nested Qt loop.")

    app_proxy = AppProxy()
    shell = SimpleNamespace(active_eventloop="qt")
    monkeypatch.setattr(masking_module, "QApplication", AppProxy)
    monkeypatch.setattr(masking_module, "_active_ipython", lambda: shell)

    window = masking_module.data_masking_manager(_recording())

    assert app_proxy.quit_on_last_window_closed is False
    assert window.isVisible()
    assert window.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    assert window in masking_module._OPEN_WINDOWS

    window.close()
    real_app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    real_app.processEvents()
    assert window not in masking_module._OPEN_WINDOWS


def test_manager_blocks_when_requested_and_exits_when_closed(monkeypatch):
    real_app = QApplication.instance()
    exec_calls = []

    class AppProxy:
        quit_on_last_window_closed = None

        @staticmethod
        def instance():
            return app_proxy

        def setQuitOnLastWindowClosed(self, value):
            self.quit_on_last_window_closed = value

        def exec(self):
            exec_calls.append(True)
            manager = masking_module._OPEN_WINDOWS[-1]
            manager.close()
            real_app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            real_app.processEvents()

    app_proxy = AppProxy()
    monkeypatch.setattr(masking_module, "QApplication", AppProxy)
    monkeypatch.setattr(masking_module, "_active_ipython", lambda: None)

    masking_module.data_masking_manager(_recording(), block=True)

    assert exec_calls == [True]
    assert app_proxy.quit_on_last_window_closed is True
    assert not masking_module._OPEN_WINDOWS


def test_viewer_accepts_a_single_recording_and_plots_time_course():
    recording = _recording()
    window = DataMaskingManager(recording)

    assert window.recording_combo.count() == 1
    assert window.data_combo.count() >= 1
    assert window._current_data() is recording["amp"]
    assert len(window._channel_labels) == recording["amp"].sizes["channel"]
    assert ": " in window._channel_labels[0]
    assert window.waterfall_ax.get_xlabel() == "Time (s)"
    assert window.stimulus_ax.get_title(loc="left") == "Stimulus"
    assert window.probe_canvas is not window.canvas
    assert window.probe_ax.get_title() == ""
    probe_labels = {text.get_text() for text in window.probe_ax.texts}
    assert "S1" in probe_labels
    assert "D1" in probe_labels
    assert not any(label.startswith("Source-") for label in probe_labels)
    assert not any(label.startswith("Detector ") for label in probe_labels)
    for text in window.probe_ax.texts:
        assert text.get_horizontalalignment() == "center"
        assert text.get_verticalalignment() == "center"
    assert len(window.probe_ax.collections) >= 2
    assert not hasattr(window, "selected_label")
    assert all(
        label.get_color() == window._channel_colors[index]
        for index, label in enumerate(window.label_ax.get_yticklabels())
    )
    assert all(color.lower() != "#000000" for color in window._channel_colors)
    assert all(
        line.get_color() == window._channel_colors[index]
        for index, line in enumerate(window._probe_lines)
    )
    window.canvas.draw()
    stimulus_position = window.stimulus_ax.get_position()
    waterfall_position = window.waterfall_ax.get_position()
    label_position = window.label_ax.get_position()
    assert stimulus_position.x0 == pytest.approx(waterfall_position.x0)
    assert stimulus_position.x1 == pytest.approx(waterfall_position.x1)
    assert label_position.x1 < waterfall_position.x0
    assert window.plot_layout.itemAtPosition(0, 1).widget() is window.vertical_scroll
    assert window.plot_layout.itemAtPosition(1, 0).widget() is window.horizontal_scroll


def test_viewer_updates_subtypes_when_data_type_changes():
    recording = _recording()
    amplitude = recording["amp"].mean("wavelength")
    recording["conc"] = amplitude.expand_dims(chromo=["HbO", "HbR"], axis=-1)
    window = DataMaskingManager(recording)
    data_keys = [
        window.data_combo.itemData(index)
        for index in range(window.data_combo.count())
    ]
    assert "conc" in data_keys

    window.data_combo.setCurrentIndex(data_keys.index("conc"))

    subtype_labels = [checkbox.text() for checkbox in window._subtype_checks]
    assert subtype_labels
    assert any("HbO" in label or "HbR" in label for label in subtype_labels)
    assert len(window.waterfall_ax.lines) == (
        len(window._subtype_checks) * window._current_data().sizes["channel"]
    )


def test_viewer_selects_recording_and_renders_existing_mask():
    recording = _recording()
    data = recording["amp"]
    values = np.ones(
        (data.sizes["time"], data.sizes["channel"], data.sizes["wavelength"]),
        dtype=bool,
    )
    values[5:10, 0, 0] = False
    recording.set_mask(
        "quality",
        xr.DataArray(
            values,
            dims=("time", "channel", "wavelength"),
            coords={
                "time": data.time,
                "channel": data.channel,
                "wavelength": data.wavelength,
            },
        ),
    )
    second_recording = copy.deepcopy(recording)
    second_recording.meta_data["subject"] = "sub-2"
    dataset = DataSet([recording, second_recording])
    window = DataMaskingManager(dataset)

    assert window.recording_combo.count() == 2
    assert window.applied_mask_list.count() == 1
    assert window.applied_mask_list.item(0).text() == "Existing mask: quality"
    assert len(window.waterfall_ax.patches) >= 1

    window.recording_combo.setCurrentIndex(1)
    assert window._current_recording() is second_recording
    assert not hasattr(window, "mask_combo")


def test_mask_tool_panel_exposes_quality_options_and_summary():
    window = DataMaskingManager(_recording())

    assert window.mask_tool_combo.count() == 15
    assert window.mask_tool_combo.currentText() == "Peak spectral power"
    assert window.tool_summary.text()
    assert window._option_widgets["window_length"].value() == pytest.approx(10.0)
    assert window.preview_button.text() == "Show preview"
    assert window.add_mask_button.text() == "Add data mask"
    assert window.remove_mask_button.text() == "Remove selected"
    assert window.probe_canvas.parentWidget() is (
        window.applied_mask_list.parentWidget().parentWidget().parentWidget()
    )

    window.mask_tool_combo.setCurrentText("Signal-to-noise ratio")

    assert "snr_thresh" in window._option_widgets
    assert "signal-to-noise ratio" in window.tool_summary.text()
    mask = window._create_mask()
    assert isinstance(mask, xr.DataArray)
    assert "channel" in mask.dims


def test_all_mask_tool_options_call_cedalion_and_return_boolean_masks():
    window = DataMaskingManager(_recording())

    for index in range(window.mask_tool_combo.count() - 3):
        window.mask_tool_combo.setCurrentIndex(index)
        mask = window._create_mask()

        assert isinstance(mask, xr.DataArray), window.mask_tool_combo.currentText()
        assert mask.dtype == bool, window.mask_tool_combo.currentText()


def test_preview_composes_incremental_masks_and_remove_updates_stack():
    recording = _recording()
    data = recording["amp"]
    values = np.ones((data.sizes["time"], data.sizes["channel"]), dtype=bool)
    values[4:8, 0] = False
    first_mask = xr.DataArray(
        values,
        dims=("time", "channel"),
        coords={"time": data.time, "channel": data.channel},
    )
    window = DataMaskingManager(recording)
    window._create_mask = lambda: first_mask

    window._preview_mask()

    assert window.add_mask_button.isEnabled()
    assert any(patch.get_facecolor()[2] > 0.3 for patch in window.waterfall_ax.patches)
    expected_percent = 100 * 4 / data.sizes["time"]
    assert f"{expected_percent:.1f}% removed" in {
        text.get_text() for text in window.summary_ax.texts
    }
    window._add_preview_mask()

    assert window.applied_mask_list.count() == 1
    assert not window.add_mask_button.isEnabled()
    assert all(
        patch.get_facecolor()[0] == pytest.approx(patch.get_facecolor()[2])
        for patch in window.waterfall_ax.patches
    )

    second_values = np.ones(
        (data.sizes["time"], data.sizes["channel"]), dtype=bool
    )
    second_values[6:10, 0] = False
    second_mask = xr.DataArray(
        second_values,
        dims=("time", "channel"),
        coords={"time": data.time, "channel": data.channel},
    )
    window._create_mask = lambda: second_mask
    window._preview_mask()

    purple_patches = [
        patch
        for patch in window.waterfall_ax.patches
        if patch.get_facecolor()[2] > patch.get_facecolor()[0]
    ]
    assert purple_patches
    assert f"{100 * 6 / data.sizes['time']:.1f}% removed" in {
        text.get_text() for text in window.summary_ax.texts
    }

    window._add_preview_mask()
    window.applied_mask_list.setCurrentRow(0)
    window._remove_applied_mask()

    assert window.applied_mask_list.count() == 1
    assert window.remove_mask_button.isEnabled()


def test_preview_is_generated_and_shown_per_recording():
    first = _recording()
    second = copy.deepcopy(first)
    window = DataMaskingManager(DataSet([first, second]))

    def create_mask():
        recording_index = window._current_recording_index()
        amp = window._current_recording()["amp"]
        good = np.ones((amp.sizes["time"], amp.sizes["channel"]), dtype=bool)
        bad_time = 5 if recording_index == 0 else 12
        good[bad_time, 0] = False
        return xr.DataArray(
            good,
            dims=("time", "channel"),
            coords={"time": amp.time, "channel": amp.channel},
        )

    window._create_mask = create_mask

    window._preview_mask()

    assert set(window._preview_masks) == {0, 1}
    assert not window._displayed_good[5, 0]
    assert window._displayed_good[12, 0]

    window.recording_combo.setCurrentIndex(1)
    assert window._displayed_good[5, 0]
    assert not window._displayed_good[12, 0]
    assert window.add_mask_button.isEnabled()

    window._add_preview_mask()
    assert len(window._mask_entries) == 1
    assert set(window._mask_entries[0]["masks"]) == {0, 1}
    assert not window._mask_entries[0]["masks"][0].values[5, 0]
    assert not window._mask_entries[0]["masks"][1].values[12, 0]


def test_mask_summaries_only_show_for_visible_channels():
    window = DataMaskingManager(_recording())
    window._viewport_height = 4.0
    window._draw()

    assert len(window.summary_ax.texts) == 4

    window.vertical_scroll.setValue(2)

    visible_indices = {
        int(round(text.get_position()[1] / window._line_spacing))
        for text in window.summary_ax.texts
    }
    assert len(visible_indices) == 4
    assert min(visible_indices) >= 2
    assert max(visible_indices) <= 5


def test_manual_channel_context_menu_scopes_preserve_other_masks(monkeypatch):
    recording = _recording()
    data = recording["amp"]
    other_good = np.ones((data.sizes["time"], data.sizes["channel"]), dtype=bool)
    other_good[:, 0] = False
    recording.set_mask(
        "automatic",
        xr.DataArray(
            other_good,
            dims=("time", "channel"),
            coords={"time": data.time, "channel": data.channel},
        ),
    )
    window = DataMaskingManager(recording)
    assert window.applied_mask_list.item(0).text() == "Existing mask: automatic"
    window.mask_tool_combo.setCurrentText("Manual removal of channels")

    assert not window.preview_button.isEnabled()
    context_calls = []
    monkeypatch.setattr(
        window,
        "_show_channel_context_menu",
        lambda index, position: context_calls.append(index),
    )
    window._on_plot_click(
        SimpleNamespace(
            inaxes=window.waterfall_ax,
            ydata=2 * window._line_spacing,
            xdata=1.0,
            x=5,
            y=5,
            button=3,
        )
    )
    assert context_calls == [2]

    window._mask_channel_for_scope(0, 2, "current")
    assert not window._mask_matrix(window._manual_mask_draft, data)[:, 2].any()
    window._add_preview_mask()
    assert window.applied_mask_list.count() == 2
    assert not window._base_mask_matrix(data)[:, 0].any()
    assert not window._base_mask_matrix(data)[:, 2].any()


@pytest.mark.parametrize(
    ("tool", "expected_channels"),
    [
        ("Manual removal of time points (all channels)", None),
        ("Manual removal of time points (current channel)", 1),
    ],
)
def test_manual_time_drag_masks_selected_time_range(tool, expected_channels):
    window = DataMaskingManager(_recording())
    data = window._current_data()
    window._select_channel(1)
    window.mask_tool_combo.setCurrentText(tool)
    times = data["time"].values.astype(float)
    first, last = 20, 30

    window._on_plot_click(
        SimpleNamespace(
            inaxes=window.waterfall_ax,
            ydata=0.0,
            xdata=float(times[first]),
            button=3,
        )
    )
    window._on_plot_motion(
        SimpleNamespace(
            inaxes=window.waterfall_ax,
            xdata=float(times[last]),
        )
    )
    window._on_plot_release(
        SimpleNamespace(
            inaxes=window.waterfall_ax,
            xdata=float(times[last]),
        )
    )

    manual = window._mask_matrix(window._manual_mask_draft, data)
    assert not manual[first : last + 1, :].all()
    assert window.add_mask_button.isEnabled()
    assert not window.preview_button.isEnabled()
    if expected_channels is None:
        assert not manual[first : last + 1, :].any()
    else:
        assert not manual[first : last + 1, expected_channels].any()
        assert manual[first : last + 1, 0].all()


def test_probe_and_waterfall_selection_stay_linked():
    recording = _recording()
    window = DataMaskingManager(recording)
    assert window._probe_lines

    window.probe_canvas.draw()
    target_channel = min(2, len(window._channel_labels) - 1)
    line = window._probe_lines[target_channel]
    midpoint = line.get_transform().transform(
        np.asarray(
            [[
                np.mean(line.get_xdata()),
                np.mean(line.get_ydata()),
            ]]
        )
    )[0]
    window._on_probe_click(
        SimpleNamespace(
            inaxes=window.probe_ax,
            x=midpoint[0],
            y=midpoint[1],
        )
    )

    assert window._selected_channel == target_channel
    assert window.vertical_scroll.value() >= 0


def test_manual_channel_mask_scope_uses_demographic_subject_and_channel_identity():
    first = _recording()
    first.meta_data["subject"] = "sub-1"
    second = copy.deepcopy(first)
    second.meta_data["subjID"] = "SUB-1"
    third = copy.deepcopy(first)
    third.meta_data["subject"] = "sub-2"
    window = DataMaskingManager(DataSet([first, second, third]))
    window.mask_tool_combo.setCurrentText("Manual removal of channels")

    window._mask_channel_for_scope(0, 2, "subject")

    assert not window._manual_drafts[(0, "Manual removal of channels")].values[:, 2].any()
    assert not window._manual_drafts[(1, "Manual removal of channels")].values[:, 2].any()
    assert (2, "Manual removal of channels") not in window._manual_drafts


def test_manual_time_drafts_are_per_recording_and_added_together():
    first = _recording()
    second = copy.deepcopy(first)
    third = copy.deepcopy(first)
    window = DataMaskingManager(DataSet([first, second, third]))
    tool = "Manual removal of time points (all channels)"
    window.mask_tool_combo.setCurrentText(tool)
    data = first["amp"]
    times = data.time.values.astype(float)

    window._apply_manual_time_range(float(times[10]), float(times[12]))
    window.recording_combo.setCurrentIndex(2)
    assert window.add_mask_button.isEnabled()
    window.recording_combo.setCurrentIndex(1)
    window._apply_manual_time_range(float(times[20]), float(times[22]))

    assert not window._manual_drafts[(0, tool)].values[10:13].all()
    assert not window._manual_drafts[(1, tool)].values[20:23].all()
    window._add_preview_mask()

    assert len(window._mask_entries) == 1
    assert set(window._mask_entries[0]["masks"]) == {0, 1}
    assert window.applied_mask_list.count() == 1


def test_submit_writes_staged_mask_via_recording_set_mask(monkeypatch):
    recording = _recording()
    window = DataMaskingManager(recording)
    data = recording["amp"]
    good = np.ones((data.sizes["time"], data.sizes["channel"]), dtype=bool)
    good[4:8, 0] = False
    staged_mask = xr.DataArray(
        good,
        dims=("time", "channel"),
        coords={"time": data.time, "channel": data.channel},
    )
    window._mask_entries.append(
        {
            "id": 99,
            "name": "manual",
            "masks": {0: staged_mask},
            "existing_keys": {},
            "existing": False,
        }
    )
    window._staged_dirty = True
    submitted = []
    window.submitted.connect(lambda: submitted.append(True))
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("reviewed", True))

    assert window._submit()

    assert list(recording.masks) == ["reviewed"]
    np.testing.assert_array_equal(recording.masks["reviewed"].values, good)
    assert submitted == [True]


def test_submit_includes_uncommitted_manual_drafts(monkeypatch):
    recording = _recording()
    window = DataMaskingManager(recording)
    window.mask_tool_combo.setCurrentText("Manual removal of channels")
    window._mask_channel_for_scope(0, 2, "current")
    monkeypatch.setattr(
        QInputDialog, "getText", lambda *args, **kwargs: ("manual", True)
    )

    assert window._submit()

    manual = recording.masks["manual"]
    assert not manual.values[:, 2].any()
    assert manual.values[:, 1].all()


@pytest.mark.parametrize("mode", ["combine", "add", "replace"])
def test_submit_modes_preserve_or_replace_existing_masks(mode, monkeypatch):
    recording = _recording()
    data = recording["amp"]
    old_good = np.ones((data.sizes["time"], data.sizes["channel"]), dtype=bool)
    old_good[0:2, 0] = False
    recording.set_mask(
        "existing",
        xr.DataArray(
            old_good,
            dims=("time", "channel"),
            coords={"time": data.time, "channel": data.channel},
        ),
    )
    window = DataMaskingManager(recording)
    new_good = np.ones_like(old_good)
    new_good[2:4, 0] = False
    window._mask_entries.append(
        {
            "id": 99,
            "name": "staged",
            "masks": {
                0: xr.DataArray(
                    new_good,
                    dims=("time", "channel"),
                    coords={"time": data.time, "channel": data.channel},
                )
            },
            "existing_keys": {},
            "existing": False,
        }
    )
    window._staged_dirty = True
    window._choose_submission_mode = lambda: mode
    if mode == "add":
        monkeypatch.setattr(
            QInputDialog, "getText", lambda *args, **kwargs: ("reviewed", True)
        )

    assert window._submit()

    if mode == "combine":
        assert set(recording.masks) == {"existing"}
        expected = np.logical_and(old_good, new_good)
        np.testing.assert_array_equal(recording.masks["existing"].values, expected)
    elif mode == "add":
        assert set(recording.masks) == {"existing", "reviewed"}
        np.testing.assert_array_equal(recording.masks["existing"].values, old_good)
        np.testing.assert_array_equal(recording.masks["reviewed"].values, new_good)
    else:
        assert set(recording.masks) == {"quality_mask"}
        expected = np.logical_and(old_good, new_good)
        np.testing.assert_array_equal(recording.masks["quality_mask"].values, expected)


def test_mask_canonicalization_marks_sample_bad_if_any_subtype_is_bad():
    recording = _recording()
    amp = recording["amp"]
    values = np.ones(
        (amp.sizes["time"], amp.sizes["channel"], amp.sizes["wavelength"]),
        dtype=bool,
    )
    values[5, 0, -1] = False
    mask = xr.DataArray(
        values,
        dims=("time", "channel", "wavelength"),
        coords={
            "time": amp.time,
            "channel": amp.channel,
            "wavelength": amp.wavelength,
        },
    )
    window = DataMaskingManager(recording)

    canonical = window._canonical_mask(mask, amp)

    assert canonical.dtype == bool
    assert not canonical.values[5, 0]
    assert canonical.values[5, 1]


def test_cancel_does_not_mutate_recording_masks():
    recording = _recording()
    window = DataMaskingManager(recording)
    data = recording["amp"]
    window._mask_entries.append(
        {
            "id": 99,
            "name": "staged",
            "masks": {
                0: xr.DataArray(
                    np.zeros((data.sizes["time"], data.sizes["channel"]), dtype=bool),
                    dims=("time", "channel"),
                    coords={"time": data.time, "channel": data.channel},
                )
            },
            "existing_keys": {},
            "existing": False,
        }
    )
    window._staged_dirty = True

    window._cancel()

    assert not recording.masks


def test_close_discard_drops_staged_masks_without_mutation():
    recording = _recording()
    window = DataMaskingManager(recording)
    window._staged_dirty = True
    window._choose_close_action = lambda: "discard"
    window.show()

    window.close()

    assert not window.isVisible()
    assert not recording.masks


def test_close_save_submits_staged_masks(monkeypatch):
    recording = _recording()
    window = DataMaskingManager(recording)
    data = recording["amp"]
    good = np.ones((data.sizes["time"], data.sizes["channel"]), dtype=bool)
    good[3, 0] = False
    window._mask_entries.append(
        {
            "id": 99,
            "name": "staged",
            "masks": {
                0: xr.DataArray(
                    good,
                    dims=("time", "channel"),
                    coords={"time": data.time, "channel": data.channel},
                )
            },
            "existing_keys": {},
            "existing": False,
        }
    )
    window._staged_dirty = True
    window._choose_close_action = lambda: "save"
    monkeypatch.setattr(
        QInputDialog, "getText", lambda *args, **kwargs: ("saved", True)
    )
    window.show()

    window.close()

    assert not window.isVisible()
    assert "saved" in recording.masks
    np.testing.assert_array_equal(recording.masks["saved"].values, good)


def test_clicking_colored_channel_label_selects_channel():
    window = DataMaskingManager(_recording())
    target_channel = min(3, len(window._channel_labels) - 1)

    window._on_label_click(
        SimpleNamespace(
            inaxes=window.label_ax,
            ydata=target_channel * window._line_spacing,
        )
    )

    assert window._selected_channel == target_channel
    assert window.label_ax.get_yticklabels()[target_channel].get_fontweight() == "bold"
    assert window._probe_lines[target_channel].get_linewidth() == 5.0


def test_zoom_controls_keep_stimulus_and_waterfall_x_scales_linked():
    window = DataMaskingManager(_recording())
    original_tick_spacing = np.diff(window.label_ax.get_yticks())[0]
    original_amplitude_zoom = window._amplitude_zoom

    window._change_time_zoom(1)
    window.horizontal_scroll.setValue(5000)
    original_visible_channels = window.vertical_scroll.pageStep()
    original_selected_trace = window.waterfall_ax.lines[0].get_ydata().copy()
    for _ in range(8):
        window._change_spacing(1)

    assert window._time_zoom > 1
    assert np.diff(window.label_ax.get_yticks())[0] > original_tick_spacing
    assert window._line_spacing > 2
    assert window.vertical_scroll.pageStep() < original_visible_channels
    assert window._amplitude_zoom == original_amplitude_zoom
    np.testing.assert_allclose(
        window.waterfall_ax.lines[0].get_ydata(),
        original_selected_trace,
    )
    assert window.stimulus_ax.get_xlim() == window.waterfall_ax.get_xlim()
    assert window.stimulus_ax.get_xlim()[0] > 0


def test_amplitude_zoom_has_a_wide_range_and_buttons_have_light_background():
    window = DataMaskingManager(_recording())
    zoom_buttons = window.findChildren(QToolButton)
    assert len(zoom_buttons) == 7
    assert zoom_buttons[0].accessibleName() == "Reset view"
    assert zoom_buttons[0].toolTip() == "Reset view"
    assert all("#e6e6e6" in button.styleSheet() for button in zoom_buttons)

    for _ in range(10):
        window._change_amplitude_zoom(1)

    assert window._amplitude_zoom > 4.0

    window._change_time_zoom(1)
    window._change_spacing(1)
    window.horizontal_scroll.setValue(5000)
    window.vertical_scroll.setValue(min(2, window.vertical_scroll.maximum()))
    window._select_channel(min(2, len(window._channel_labels) - 1))

    window._reset_view()

    assert window._time_zoom == 1.0
    assert window._amplitude_zoom == window._initial_amplitude_zoom
    assert window._line_spacing == 1.0
    assert window._time_start == 0.0
    assert window.horizontal_scroll.value() == 0
    assert window.vertical_scroll.value() == 0
    assert window._selected_channel == 0
