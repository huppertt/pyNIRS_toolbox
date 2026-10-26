"""Tests for NIRSviewIR session/file menu actions."""
from __future__ import annotations

import importlib
from collections import OrderedDict

import pytest
import numpy as np
import pandas as pd
import xarray as xr

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("cedalion")

from PySide6.QtWidgets import QDialog, QMenuBar, QMessageBox  # noqa: E402

from pyBrainAnalyzIR.dataclasses.dataset import DataSet  # noqa: E402
from pyBrainAnalyzIR.vis.bids_info_manager import BidsInfoManager  # noqa: E402

nirsview = importlib.import_module("pyBrainAnalyzIR.vis.NIRSviewIR")

pytestmark = [pytest.mark.requires_qt, pytest.mark.requires_cedalion]
pytestmark.append(pytest.mark.usefixtures("qt_app"))


class Recording:
    def __init__(self, subject: str = "subj_1"):
        self.meta_data = OrderedDict({"subject": subject})
        self.timeseries = {}
        self.stim = None


def _file_menu(window):
    menu_bar = window.findChild(QMenuBar)
    for action in menu_bar.actions():
        if action.text() == "File":
            return action.menu()
    raise AssertionError("File menu not found")


@pytest.fixture
def roi_dataset(simulated_recording):
    from pyBrainAnalyzIR.sigproc.roi import roi_average
    from pyBrainAnalyzIR.dataclasses.statistics import Statistics

    rec, _ = simulated_recording
    roi_average(rec, pd.DataFrame({
        "source": rec["amp"].source.values[[0, 1, 0, 2]],
        "detector": rec["amp"].detector.values[[0, 1, 0, 2]],
        "name": ["Left", "Left", "Right", "Right"],
        "weight": [2, 1, 4, 4],
    }))
    stats = Statistics()
    stats.geoROI = rec.aux_obj["roi_definitions"]["default"]
    stats.geo2d, stats.geo3d = rec.geo2d, rec.geo3d
    stats.betas = xr.DataArray(
        [4.0, 0.1, -4.0, 0.2], dims="indices",
        coords={"channels": ("indices", ["Left", "Right"] * 2),
                "type": ("indices", ["HbO"] * 4),
                "conditions": ("indices", ["A", "A", "B", "B"])},
    )
    stats.covariance = xr.DataArray(np.eye(4), dims=("rows", "cols"))
    stats.dof = 100
    rec["stats_roi"] = stats
    dset = DataSet([rec])
    dset["groupstats_roi"] = stats.copy()
    return dset


def _canvas_click(canvas, axes):
    from matplotlib.backend_bases import MouseEvent

    canvas.draw()
    x, y = axes.transAxes.transform((0.5, 0.5))
    for name in ("button_press_event", "button_release_event"):
        canvas.callbacks.process(name, MouseEvent(name, canvas, x, y, button=1))


def test_roi_timecourses_embedded_buttons_and_switching(roi_dataset):
    from matplotlib.colors import to_rgb

    window = nirsview.NIRSviewIRWindow(roi_dataset)
    window.type_selector.setCurrentText("amp_roi")
    assert len(window._roi_buttons) == 2
    assert len(window.mllines) == roi_dataset.dataset[0]["amp"].sizes["channel"]
    assert len(window._ts_lines) == 2
    np.testing.assert_allclose(to_rgb(window.mllines[1].get_color()), [1, 0.5, 0.5])
    _canvas_click(window.canvas, window._roi_buttons[1].ax)
    assert window.selected_channels == [0, 1]
    assert len(window._ts_lines) == 4
    np.testing.assert_allclose(to_rgb(window.mllines[0].get_color()), [0.5, 0, 0.5])
    _canvas_click(window.canvas, window.ax_probe)
    assert window.selected_channels == [0, 1]
    _canvas_click(window.canvas, window._roi_buttons[0].ax)
    assert window.selected_channels == [1]
    assert all(to_rgb(line.get_color()) == to_rgb("blue") for line in window._ts_lines)
    old_buttons = window._roi_buttons[:]
    window.type_selector.setCurrentText("amp")
    assert window._roi_buttons == []
    assert len(window.figure.axes) == 2
    assert all(button.ax not in window.figure.axes for button in old_buttons)
    _canvas_click(window.canvas, window.ax_probe)
    assert len(window.selected_channels) == 1
    window.type_selector.setCurrentText("amp_roi")
    assert len(window.figure.axes) == 4
    assert len(window._roi_buttons) == 2
    window.close()


@pytest.mark.parametrize("key", ["stats_roi", "Group: groupstats_roi"])
def test_roi_statistics_embedded_panels_filters_and_fixed_keys(roi_dataset, key):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    external = plt.figure()
    window = nirsview.NIRSviewIRWindow(roi_dataset, plot_on_scalp=False)
    window.type_selector.setCurrentText(key)
    assert window._right_stack.currentIndex() == 1
    assert not window._filter_vartype_combo.isEnabled()
    assert window._filter_vartype_combo.currentText() == "beta"
    fig = window._stats_figure
    assert fig.canvas is window._stats_canvas
    assert len(fig.axes) == 5
    assert not fig.axes[0].images  # Actual embedded artists, not a flattened PNG.
    assert len(fig._roi_buttons) == 2
    assert all(not button.active for button in fig._roi_buttons)
    assert len(fig.axes[1].texts) == 1
    for button in fig._roi_buttons:
        _canvas_click(window._stats_canvas, button.ax)
        assert to_rgb(button.ax.get_facecolor()) == to_rgb(button.color)
    assert len(fig.axes[1].patches) == 2
    np.testing.assert_allclose([bar.get_height() for bar in fig.axes[1].patches], [4, 0.1])
    assert window._stats_model.rowCount() == 4
    assert plt.fignum_exists(external.number)
    window._stats_condition_filter = {"B"}
    window._apply_stats_filter()
    assert len(fig.axes) == 4
    assert fig.axes[1].get_title() == "B: HbO"
    np.testing.assert_allclose([bar.get_height() for bar in fig.axes[1].patches], [-4, 0.2])
    assert window._stats_proxy.rowCount() == 2
    window._filter_q_spin.setValue(0.05)
    assert "q<0.05" in fig._suptitle.get_text()
    window._global_fdr_checkbox.setChecked(False)
    assert all(not button.active for button in fig._roi_buttons)
    window._filter_q_spin.setValue(0)
    assert len(fig.axes[1].texts) == 0
    window.type_selector.setCurrentText("amp_roi")
    assert window._right_stack.currentIndex() == 0
    assert len(window._roi_buttons) == 2
    window.close()
    plt.close(external)


@pytest.mark.parametrize("plot_on_scalp", [False, True])
def test_roi_gui_honors_probe_projection(roi_dataset, plot_on_scalp):
    from cedalion import units
    from cedalion.dataclasses.geometry import PointType
    from matplotlib.patches import Circle

    rec = roi_dataset.dataset[0]
    landmarks = xr.DataArray(
        [[0, 100, 0], [-100, 0, 0], [100, 0, 0]], dims=("label", "pos"),
        coords={"label": ["Nz", "LPA", "RPA"],
                "type": ("label", [PointType.LANDMARK] * 3)},
    ).pint.quantify(units.mm)
    rec.geo3d = xr.concat([rec.geo3d, landmarks], dim="label")
    rec["stats_roi"].geo3d = rec.geo3d
    window = nirsview.NIRSviewIRWindow(roi_dataset, plot_on_scalp=plot_on_scalp)
    window.type_selector.setCurrentText("amp_roi")
    assert any(isinstance(patch, Circle) for patch in window.ax_probe.patches) == plot_on_scalp
    window.type_selector.setCurrentText("stats_roi")
    assert any(
        isinstance(patch, Circle) for patch in window._stats_figure.axes[0].patches
    ) == plot_on_scalp
    window.close()


def test_switching_roi_and_channel_statistics_restores_display_control(roi_dataset):
    import matplotlib.pyplot as plt

    rec = roi_dataset.dataset[0]
    stats = rec["stats_roi"].copy()
    stats.geoROI = None
    stats.betas = stats.betas.assign_coords(
        channels=("indices", rec["amp"].channel.values[:2].tolist() * 2)
    )
    rec["stats_channels"] = stats
    external = plt.figure()
    window = nirsview.NIRSviewIRWindow(roi_dataset)
    window.type_selector.setCurrentText("stats_roi")
    assert not window._filter_vartype_combo.isEnabled()
    window.type_selector.setCurrentText("stats_channels")
    assert window._filter_vartype_combo.isEnabled()
    assert len(window._stats_figure.axes[0].images) == 1
    window._filter_vartype_combo.setCurrentText("tstat")
    assert len(window._stats_figure.axes[0].images) == 1
    assert plt.fignum_exists(external.number)
    window.type_selector.setCurrentText("stats_roi")
    assert not window._filter_vartype_combo.isEnabled()
    assert not window._stats_figure.axes[0].images
    assert len(window._stats_figure._roi_buttons) == 2
    window.close()
    plt.close(external)


def test_roi_gui_with_fitted_glm_and_mixed_effects(simulated_recording):
    import copy
    from pyBrainAnalyzIR.sigproc.roi import roi_average
    from pyBrainAnalyzIR.pipelines.modules import preproccessing as prep
    from pyBrainAnalyzIR.pipelines.modules.glm import GLM
    from pyBrainAnalyzIR.pipelines.modules.mixedeffects import MixedEffects

    rec, _ = simulated_recording
    recordings = []
    for index in range(5):
        recording = copy.deepcopy(rec)
        recording["amp"] = recording["amp"] * (
            1 + np.random.default_rng(index).normal(0, 0.001, recording["amp"].shape)
        )
        recording.meta_data["subject"] = f"subject{index}"
        recordings.append(recording)
    dset = DataSet(recordings)
    roi_average(dset, pd.DataFrame({
        "source": rec["amp"].source.values[:4],
        "detector": rec["amp"].detector.values[:4],
        "name": ["Left", "Left", "Right", "Right"],
    }))
    for module, source, target in [
        (prep.intensity_opticaldensity(), "amp_roi", "od_roi"),
        (prep.mbll(), "od_roi", "conc_roi"),
        (GLM(), "conc_roi", "stats_roi"),
    ]:
        module.inputName, module.outputName = source, target
        if isinstance(module, GLM):
            module.options["noise_model"] = "ols"
        module.run(dset)
    group = MixedEffects()
    group.inputName = "stats_roi"
    group.options["robust"] = False
    group.run(dset)
    window = nirsview.NIRSviewIRWindow(dset)
    for key, stats in [("stats_roi", dset.dataset[0]["stats_roi"]), ("Group: groupstats", dset["groupstats"])]:
        window.type_selector.setCurrentText(key)
        assert window._current_stats_obj is stats
        panels = len(stats.list_conditions()) * len(np.unique(stats.betas.type))
        assert len(window._stats_figure.axes) == 1 + panels + 2
        assert window._stats_model.rowCount() == len(stats.table())
        assert all(not button.active for button in window._stats_figure._roi_buttons)
        window._stats_canvas.draw()
    window.close()


def test_file_menu_actions_are_flat():
    window = nirsview.NIRSviewIRWindow(DataSet())
    file_menu = _file_menu(window)

    action_labels = [action.text() for action in file_menu.actions()]

    assert action_labels == [
        "New Session",
        "Load Session",
        "Save Session",
        "Load File",
        "Exit",
    ]


def test_edit_data_menu_puts_bids_information_first():
    window = nirsview.NIRSviewIRWindow(DataSet([Recording()]))
    menu_bar = window.findChild(QMenuBar)
    edit_data_menu = next(
        action.menu() for action in menu_bar.actions()
        if action.text() == "Edit Data"
    )

    assert [action.text() for action in edit_data_menu.actions()] == [
        "Edit BIDS information",
        "Edit Demographics",
        "Edit Stimulus Timing",
    ]
    review_menu = next(action.menu() for action in menu_bar.actions()
                       if action.text() == "Data Review")
    assert [action.text() for action in review_menu.actions()] == [
        "Data Quality Manager", "Edit Data Quality Mask",
    ]


def test_registration_menu_opens_roi_manager_and_stores_tables(monkeypatch):
    dataset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dataset)
    table = pd.DataFrame({
        "source": ["S1"], "detector": ["D1"], "weight": [1.0], "name": ["ROI 1"],
    })
    calls = []

    def open_roi_manager(data, roi_tables, parent):
        calls.append((data, roi_tables, parent))
        return {"ROI 1": table}

    monkeypatch.setattr(nirsview, "roi_manager", open_roi_manager)
    menu_bar = window.findChild(QMenuBar)
    assert [action.text() for action in menu_bar.actions()][1:4] == [
        "Edit Data", "Registration", "Analysis",
    ]
    registration_menu = menu_bar.actions()[2].menu()
    assert [action.text() for action in registration_menu.actions()] == [
        "Define Region-of-Interest",
    ]
    window._define_rois()
    assert calls == [(dataset, {}, window)]
    pd.testing.assert_frame_equal(window.roi_tables["ROI 1"], table)


def test_pipeline_manager_receives_defined_roi_tables(monkeypatch):
    from pyBrainAnalyzIR.vis.pipeline_manager import PipelineManagerDialog

    dataset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dataset)
    window.roi_tables = {"ROI 1": pd.DataFrame({
        "source": ["S1"], "detector": ["D1"], "weight": [1.0], "name": ["ROI 1"],
    })}
    calls = []

    def accept(dialog):
        calls.append((dialog.data, dialog.roi_tables))
        return QDialog.Accepted

    monkeypatch.setattr(PipelineManagerDialog, "exec", accept)
    window._edit_pipeline()
    assert len(calls) == 1
    assert calls[0][0] is dataset
    assert set(calls[0][1]) == {"ROI 1"}
    pd.testing.assert_frame_equal(calls[0][1]["ROI 1"], window.roi_tables["ROI 1"])


def test_nirsview_runs_combined_rois_with_custom_output_name(simulated_recording):
    from pyBrainAnalyzIR.pipelines.modules.roi import ROIAverage

    rec, _ = simulated_recording
    dataset = DataSet([rec])
    window = nirsview.NIRSviewIRWindow(dataset)
    window.roi_tables = {
        name: pd.DataFrame({
            "source": rec["amp"].source.values[start:start + 2],
            "detector": rec["amp"].detector.values[start:start + 2],
            "weight": [1.0, 1.0],
        })
        for name, start in [("Left", 0), ("Right", 2)]
    }
    module = ROIAverage()
    module.set_roi_selection(window.roi_tables, ["Left", "Right"])
    module.outputName = "my_regions"
    window._pipeline_modules = [module]
    window._run_pipeline()
    assert rec["my_regions"].channel.values.tolist() == ["Left", "Right"]
    assert "amp_roi_Left" not in rec.timeseries
    window.type_selector.setCurrentText("my_regions")
    assert window._right_stack.currentIndex() == 0
    assert window._roi_geometry.region_names == ["Left", "Right"]


@pytest.mark.parametrize("repair_existing", [False, True])
def test_nirsview_pipeline_manager_full_roi_glm_chain(simulated_recording, monkeypatch, repair_existing):
    from pyBrainAnalyzIR.pipelines.modules import preproccessing as prep
    from pyBrainAnalyzIR.pipelines.modules.glm import GLM
    from pyBrainAnalyzIR.pipelines.modules.roi import ROIAverage
    from pyBrainAnalyzIR.vis.pipeline_manager import PipelineManagerDialog

    rec, _ = simulated_recording
    window = nirsview.NIRSviewIRWindow(DataSet([rec]))
    window.roi_tables = {
        name: pd.DataFrame({
            "source": rec["amp"].source.values[start:start + 2],
            "detector": rec["amp"].detector.values[start:start + 2],
            "weight": [1.0, 1.0],
        })
        for name, start in [("Left", 0), ("Right", 2)]
    }

    def configure(dialog):
        for cls in [prep.intensity_opticaldensity, prep.mbll, prep.resample, ROIAverage, GLM]:
            row = next(
                i for i, spec in enumerate(dialog.available_row_specs)
                if spec is not None and spec.cls is cls
            )
            dialog.available_list.setCurrentRow(row)
            dialog.add_button.click()
            module = dialog.pipeline_modules[-1]
            if isinstance(module, prep.resample):
                module.options["Fs"] = 2
            if isinstance(module, ROIAverage):
                assert module.inputName == "last"
                for i in range(dialog.roi_selection_list.count()):
                    dialog.roi_selection_list.item(i).setSelected(True)
                dialog.roi_output_edit.setText("selected_regions")
                dialog.roi_output_edit.editingFinished.emit()
            if isinstance(module, GLM):
                assert module.inputName == "last"
                module.options.update(verbose=False)
        assert dialog._compute_pipeline_validity() == [True] * 5
        return QDialog.Accepted

    monkeypatch.setattr(PipelineManagerDialog, "exec", configure)
    window._edit_pipeline()
    if repair_existing:
        window._pipeline_modules[-1].inputName = "conc"

        def repair(dialog):
            dialog._refresh_pipeline_list(select_row=4)
            assert not dialog.model_input_controls.isHidden()
            assert dialog.model_input_combo.currentText() == "conc"
            choices = {
                dialog.model_input_combo.itemText(i)
                for i in range(dialog.model_input_combo.count())
            }
            assert {"last", "previous", "conc", "selected_regions"} <= choices
            dialog.model_input_combo.setCurrentText("previous")
            assert dialog.pipeline_modules[-1].inputName == "previous"
            assert dialog._compute_pipeline_validity() == [True] * 5
            return QDialog.Accepted

        monkeypatch.setattr(PipelineManagerDialog, "exec", repair)
        window._edit_pipeline()
    window._run_pipeline()
    assert "stats" in rec.timeseries
    assert set(rec["stats"].betas.channels.values) == {"Left", "Right"}
    assert rec["stats"].geoROI.region_names == ["Left", "Right"]
    window.type_selector.setCurrentText("stats")
    assert window._current_stats_obj is rec["stats"]
    assert window._current_stats_obj.geoROI.region_names == ["Left", "Right"]


def test_nirsview_displays_roi_timecourses_and_connectivity(roi_dataset):
    from pyBrainAnalyzIR.pipelines.modules.connectivity import resting_state_connectivity

    rec = roi_dataset.dataset[0]
    module = resting_state_connectivity()
    module.inputName = "amp_roi"
    module.options.update(AR=0, robust=False, short_separation_correction=False)
    module.run(rec)
    window = nirsview.NIRSviewIRWindow(roi_dataset)

    window.type_selector.setCurrentText("amp_roi")
    assert window._right_stack.currentIndex() == 0
    assert window._roi_geometry is not None
    window.type_selector.setCurrentText("stats_roi")
    assert window._right_stack.currentIndex() == 1
    assert window._current_stats_obj is rec["stats_roi"]
    window.type_selector.setCurrentText("conn")
    assert window._right_stack.currentIndex() == 1
    assert window._current_stats_obj is rec["conn"]
    assert window._stats_figure.axes


def test_edit_data_quality_mask_launches_manager(monkeypatch):
    dset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dset)
    manager_module = importlib.import_module(
        "pyBrainAnalyzIR.vis.data_masking_manager"
    )
    calls = []
    managers = []

    class FakeSignal:
        def __init__(self):
            self.callback = None

        def connect(self, callback):
            self.callback = callback

    class FakeManager:
        submitted = FakeSignal()

    def open_manager(dataset, block):
        calls.append((dataset, block))
        manager = FakeManager()
        managers.append(manager)
        return manager

    monkeypatch.setattr(manager_module, "data_masking_manager", open_manager)

    window._open_data_masking_manager()

    assert calls == [(dset, False)]
    assert not window.data_changed
    managers[0].submitted.callback()
    assert window.data_changed


def test_edit_data_quality_mask_without_recordings_shows_message(monkeypatch):
    window = nirsview.NIRSviewIRWindow(DataSet())
    messages = []
    monkeypatch.setattr(
        nirsview.QMessageBox,
        "information",
        lambda *args: messages.append(args),
    )

    window._open_data_masking_manager()

    assert messages
    assert messages[0][1] == "Edit Data Quality Mask"


def test_edit_bids_information_saves_both_templates(monkeypatch):
    dset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dset)
    calls = []

    class FakeBidsInfoManager:
        def __init__(self, dataset, parent):
            assert dataset is dset
            assert parent is window

        def exec(self):
            return QDialog.Accepted

        def readme_values(self):
            return {"Overview": "Edited overview."}

        def readme_include_values(self):
            return {"Overview": False}

        def dataset_description_values(self):
            return {"Name": "Edited dataset."}

    monkeypatch.setattr(
        "pyBrainAnalyzIR.vis.bids_info_manager.BidsInfoManager",
        FakeBidsInfoManager,
    )
    original_add_readme = dset.add_meta_readme
    original_add_readme_include = dset.add_meta_readme_include
    original_add_description = dset.add_meta_dataset_description

    def add_readme(values):
        calls.append(("readme", values))
        original_add_readme(values)

    def add_description(values):
        calls.append(("dataset_description", values))
        original_add_description(values)

    def add_readme_include(values):
        calls.append(("readme_include", values))
        original_add_readme_include(values)

    monkeypatch.setattr(dset, "add_meta_readme", add_readme)
    monkeypatch.setattr(dset, "add_meta_readme_include", add_readme_include)
    monkeypatch.setattr(dset, "add_meta_dataset_description", add_description)

    window._edit_bids_information()

    assert calls == [
        ("readme", {"Overview": "Edited overview."}),
        ("readme_include", {"Overview": False}),
        ("dataset_description", {"Name": "Edited dataset."}),
    ]
    descriptors = dset.dataset[0].meta_data["_bids_descriptors"]
    descriptions = dset.dataset[0].meta_data["_bids_descriptions"]
    assert descriptors["readme"]["Overview"] == "Edited overview."
    assert descriptors["dataset_description"]["Name"] == "Edited dataset."
    assert descriptions["readme_include:Overview"] is False
    assert window.data_changed is True


def test_bids_info_editor_uses_dataset_values_and_preserves_structured_fields():
    dset = DataSet([Recording()])
    dset.add_meta_readme({"Overview": "Stored overview."})
    dset.add_meta_dataset_description({
        "Name": "Stored dataset",
        "Authors": ["A. Researcher", "B. Scientist"],
        "StimulusPresentation": {"SoftwareName": "PsychoPy"},
    })
    dialog = BidsInfoManager(dset)

    assert dialog.tabs.tabText(0) == "ReadMe"
    assert dialog.tabs.tabText(1) == "Dataset Description"
    assert dialog._editors["ReadMe"]["Overview"].toPlainText() == "Stored overview."
    assert dialog._editors["Dataset Description"]["Name"].toPlainText() == (
        "Stored dataset"
    )
    assert dialog._editors["Dataset Description"]["Authors"].toPlainText() == (
        "A. Researcher\nB. Scientist"
    )
    assert dialog._editors["Dataset Description"][
        "StimulusPresentation"
    ].toPlainText() == '{\n  "SoftwareName": "PsychoPy"\n}'
    assert all(dialog.readme_include_values().values())

    dialog._editors["ReadMe"]["Overview"].setPlainText("Updated overview.")
    dialog._readme_checkboxes["Overview"].setChecked(False)
    dialog._editors["Dataset Description"]["Authors"].setPlainText(
        "First Author\nSecond Author"
    )
    dialog._editors["Dataset Description"]["StimulusPresentation"].setPlainText(
        '{"SoftwareName": "Presentation"}'
    )
    dialog._accept_if_valid()

    assert dialog.result() == QDialog.Accepted
    assert dialog.readme_values()["Overview"] == "Updated overview."
    assert dialog.readme_include_values()["Overview"] is False
    assert dialog.dataset_description_values()["Authors"] == [
        "First Author",
        "Second Author",
    ]
    assert dialog.dataset_description_values()["StimulusPresentation"] == {
        "SoftwareName": "Presentation"
    }


def test_demographics_panel_hides_bids_descriptions():
    rec = Recording()
    rec.meta_data["_bids_descriptions"] = {"subject": "Subject identifier"}
    rec.meta_data["age"] = 21
    window = nirsview.NIRSviewIRWindow(DataSet([rec]))

    panel_text = window.demographics_panel.toPlainText()

    assert "subject: subj_1" in panel_text
    assert "age: 21" in panel_text
    assert "_bids_descriptions" not in panel_text
    assert "Subject identifier" not in panel_text


def test_recording_summary_only_shows_populated_demographics_and_stimulus_descriptions():
    rec = Recording()
    rec.meta_data.update(age=0, flag=False, extra="N/A", empty="", nan=float("nan"))
    rec.meta_data["_bids_descriptions"] = {"event:left": "Left-hand tapping", "event:rest": "N/A"}
    rec.stim = pd.DataFrame({"trial_type": ["left", "left", "rest"]})
    window = nirsview.NIRSviewIRWindow(DataSet([rec]))
    text = window.demographics_panel.toPlainText()
    assert "age: 0" in text and "flag: False" in text
    assert all(key not in text for key in ("extra:", "empty:", "nan:", "_bids_descriptions"))
    stimulus_text = window.stimulus_panel.toPlainText()
    assert "left (2): Left-hand tapping" in stimulus_text
    assert "rest (1)" in stimulus_text and "N/A" not in stimulus_text


def test_viewer_passes_dataset_and_uses_executable_pipeline(monkeypatch):
    from pyBrainAnalyzIR.vis.pipeline_manager import PipelineManagerDialog
    from pyBrainAnalyzIR.pipelines.modules.utilities import remove_prior_analysis

    dataset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dataset)
    contexts = []

    def accept(dialog):
        contexts.append(dialog.data)
        dialog.remove_previous_checkbox.setChecked(True)
        return QDialog.Accepted

    monkeypatch.setattr(PipelineManagerDialog, "exec", accept)
    window._edit_pipeline()
    assert contexts == [dataset]
    assert isinstance(window._pipeline_modules[0], remove_prior_analysis)


@pytest.mark.parametrize("roi", [False, True])
@pytest.mark.parametrize("group", [False, True])
def test_connectivity_display_and_filters(simulated_recording, roi, group):
    from pyBrainAnalyzIR.pipelines.modules.connectivity import resting_state_connectivity
    from pyBrainAnalyzIR.sigproc.roi import roi_average

    rec, _ = simulated_recording
    job = resting_state_connectivity()
    if roi:
        roi_average(rec, pd.DataFrame({
            "source": rec["amp"].source.values[[0, 1, 0, 2]],
            "detector": rec["amp"].detector.values[[0, 1, 0, 2]],
            "name": ["Left", "Left", "Right", "Right"],
        }))
        job.inputName = "amp_roi"
    else:
        job.inputName = "amp"
    job.options.update(AR=0, robust=False, short_separation_correction=False)
    job.run(rec)
    connection = rec["conn"]
    original = connection.coef.copy()
    dataset = DataSet([rec])
    if group:
        dataset.statistics["connections"] = connection
    window = nirsview.NIRSviewIRWindow(dataset)
    window.type_selector.setCurrentText("Group: connections" if group else "conn")
    assert window._current_stats_obj is connection
    assert window._right_stack.currentIndex() == 1
    assert {"P-values", "Q-values", "Type"}.issubset(window._current_stats_df.columns)
    assert window._filter_vartype_combo.currentText() == "r"
    assert window._stats_figure.axes[0].images, [
        text.get_text() for ax in window._stats_figure.axes for text in ax.texts
    ]
    window._filter_vartype_combo.setCurrentText("z")
    window._filter_p_spin.setValue(0.01)
    filtered = window._get_filtered_df()
    assert (filtered["P-values"] < 0.01).all()
    window._stats_type_filter = set()
    window._apply_stats_filter()
    assert window._stats_proxy.rowCount() == 0
    assert "No connectivity results" in window._stats_figure.axes[0].texts[0].get_text()
    window._stats_type_filter = None
    window._stats_condition_filter = set()
    window._apply_stats_filter()
    assert "No connectivity results" in window._stats_figure.axes[0].texts[0].get_text()
    pd.testing.assert_frame_equal(connection.coef, original)
    window.type_selector.setCurrentText("amp")
    assert window._right_stack.currentIndex() == 0


def test_hyperscanning_connectivity_is_displayed(simulated_recording):
    import copy
    from pyBrainAnalyzIR.pipelines.modules.connectivity import hyperscanning

    first, _ = simulated_recording
    second = copy.deepcopy(first)
    first.meta_data.update(subject="A", pair="pair")
    second.meta_data.update(subject="B", pair="pair")
    dataset = DataSet([first, second])
    job = hyperscanning()
    job.inputName = "amp"
    job.options.update(AR=0, robust=False, short_separation_correction=False)
    job.run(dataset)
    window = nirsview.NIRSviewIRWindow(dataset)
    window.type_selector.setCurrentText("hyperconn")
    assert window._current_stats_obj is first["hyperconn"]
    assert window._stats_figure.axes[0].images, [
        text.get_text() for ax in window._stats_figure.axes for text in ax.texts
    ]


def test_new_session_clears_loaded_dataset(monkeypatch):
    dset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dset)
    window._pipeline_modules = [object()]
    window.data_changed = True

    def confirm(parent, title, text, buttons, default_button=QMessageBox.No):
        assert parent is window
        assert title == "New Session"
        assert "unsaved changes" in text
        assert "clear all loaded recordings" in text
        assert buttons == QMessageBox.Yes | QMessageBox.No
        assert default_button == QMessageBox.No
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(confirm))

    window._new_session()

    assert window.dataset.dataset == []
    assert window.rec is None
    assert window._pipeline_modules == []
    assert window.data_changed is False
    assert window.file_tree.topLevelItemCount() == 0


def test_new_session_skips_confirmation_when_data_is_saved(monkeypatch):
    dset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dset)

    def fail_if_called():
        raise AssertionError("Clean sessions should not prompt before clearing.")

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fail_if_called))

    window._new_session()

    assert window.dataset.dataset == []


def test_new_session_keeps_unsaved_dataset_when_user_cancels(monkeypatch):
    dset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dset)
    window.data_changed = True

    def cancel_new_session(parent, title, text, buttons, default_button=QMessageBox.No):
        _ = (parent, title, text, buttons, default_button)
        return QMessageBox.No

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(cancel_new_session),
    )

    window._new_session()

    assert window.dataset is dset
    assert window.dataset.dataset
    assert window.data_changed is True


def test_load_session_appends_bids_recordings(monkeypatch):
    window = nirsview.NIRSviewIRWindow(DataSet([Recording("existing")]))
    loaded = DataSet([Recording("loaded")])

    def select_folder(parent, title, directory, options):
        assert parent is window
        assert title == "Load BIDS Session"
        assert directory == ""
        assert options == nirsview.QFileDialog.ShowDirsOnly
        return "/tmp/bids"

    def read_session(folder, include_derivatives=True):
        assert folder == "/tmp/bids"
        assert include_derivatives is True
        return loaded

    monkeypatch.setattr(nirsview.QFileDialog, "getExistingDirectory", staticmethod(select_folder))
    monkeypatch.setattr(nirsview, "read_bids_dataset", read_session)

    window._load_session()

    assert [rec.meta_data["subject"] for rec in window.dataset.dataset] == [
        "existing",
        "loaded",
    ]
    assert window.data_changed is True


def test_load_session_prompts_for_derivative_loading(monkeypatch, tmp_path):
    (tmp_path / "bids_derivatives").mkdir()
    window = nirsview.NIRSviewIRWindow(DataSet())
    loaded = DataSet([Recording("loaded")])
    calls = []

    def select_derivative_folder(parent, title, directory, options):
        _ = (parent, title, directory, options)
        return str(tmp_path)

    monkeypatch.setattr(
        nirsview.QFileDialog,
        "getExistingDirectory",
        staticmethod(select_derivative_folder),
    )

    def choose_raw_only(parent, title, text, buttons, default_button=QMessageBox.Yes):
        assert parent is window
        assert title == "Load BIDS Session"
        assert "bids_derivatives" in text
        assert "raw data" in text
        assert buttons == QMessageBox.Yes | QMessageBox.No
        assert default_button == QMessageBox.Yes
        return QMessageBox.No

    def read_session(folder, include_derivatives=True):
        calls.append((folder, include_derivatives))
        return loaded

    monkeypatch.setattr(QMessageBox, "question", staticmethod(choose_raw_only))
    monkeypatch.setattr(nirsview, "read_bids_dataset", read_session)

    window._load_session()

    assert calls == [(str(tmp_path), False)]


def test_save_session_writes_bids_dataset(monkeypatch):
    dset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dset)
    window.data_changed = True
    calls = []

    def select_folder(parent, title, directory, options):
        assert parent is window
        assert title == "Save BIDS Session"
        assert directory == ""
        assert options == nirsview.QFileDialog.ShowDirsOnly
        return "/tmp/bids"

    def write_session(dataset, folder):
        calls.append((dataset, folder))

    monkeypatch.setattr(nirsview.QFileDialog, "getExistingDirectory", staticmethod(select_folder))
    monkeypatch.setattr(nirsview, "write_bids_dataset", write_session)

    window._save_session()

    assert calls == [(dset, "/tmp/bids")]
    assert window.data_changed is False


def test_load_files_appends_snirf_recordings(monkeypatch):
    window = nirsview.NIRSviewIRWindow(None)

    def select_files(parent, title, directory, file_filter):
        assert parent is window
        assert title == "Load SNIRF File(s)"
        assert directory == ""
        assert "SNIRF Files" in file_filter
        return ["one.snirf", "two.snirf"], ""

    def read_file(path):
        return [Recording(path)]

    monkeypatch.setattr(nirsview.QFileDialog, "getOpenFileNames", staticmethod(select_files))
    monkeypatch.setattr(nirsview, "read_snirf", read_file)

    window._load_files()

    assert [rec.meta_data["subject"] for rec in window.dataset.dataset] == [
        "one.snirf",
        "two.snirf",
    ]
    assert window.data_changed is True


def test_remove_recording_marks_dataset_changed(monkeypatch):
    window = nirsview.NIRSviewIRWindow(DataSet([Recording("one"), Recording("two")]))

    def confirm_remove(parent, title, text, buttons):
        _ = (parent, title, text, buttons)
        return QMessageBox.Yes

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(confirm_remove),
    )

    window._remove_recording(0)

    assert [rec.meta_data["subject"] for rec in window.dataset.dataset] == ["two"]
    assert window.data_changed is True


def test_run_pipeline_marks_data_changed_but_pipeline_clean():
    class PipelineModule:
        def __init__(self):
            self.previous_job = None

        def run(self, dataset):
            dataset.import_data(Recording("processed"))
            return dataset

    window = nirsview.NIRSviewIRWindow(DataSet([Recording("raw")]))
    window._pipeline_modules = [PipelineModule()]
    window._pipeline_dirty = True
    window.data_changed = False
    window._update_run_pipeline_enabled()

    window._run_pipeline()

    assert [rec.meta_data["subject"] for rec in window.dataset.dataset] == [
        "raw",
        "processed",
    ]
    assert window.data_changed is True
    assert window._pipeline_dirty is False


def test_exit_action_closes_window(monkeypatch):
    window = nirsview.NIRSviewIRWindow(DataSet())
    calls = []

    monkeypatch.setattr(window, "close", lambda: calls.append("closed"))

    exit_action = next(
        action for action in _file_menu(window).actions() if action.text() == "Exit"
    )
    exit_action.trigger()

    assert calls == ["closed"]


def test_close_event_prompts_when_data_is_unsaved(monkeypatch):
    window = nirsview.NIRSviewIRWindow(DataSet([Recording()]))
    window.data_changed = True
    event_calls = []

    class Event:
        def accept(self):
            event_calls.append("accepted")

        def ignore(self):
            event_calls.append("ignored")

    def cancel_close(parent, title, text, buttons, default_button=QMessageBox.No):
        assert parent is window
        assert title == "Exit"
        assert "unsaved changes" in text
        assert buttons == QMessageBox.Yes | QMessageBox.No
        assert default_button == QMessageBox.No
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(cancel_close))

    window.closeEvent(Event())

    assert event_calls == ["ignored"]


def test_close_event_accepts_when_data_is_saved(monkeypatch):
    window = nirsview.NIRSviewIRWindow(DataSet([Recording()]))
    event_calls = []

    def fail_if_called():
        raise AssertionError("Clean sessions should close without prompting.")

    class Event:
        def accept(self):
            event_calls.append("accepted")

        def ignore(self):
            event_calls.append("ignored")

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fail_if_called))

    window.closeEvent(Event())

    assert event_calls == ["accepted"]
