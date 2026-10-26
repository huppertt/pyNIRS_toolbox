"""Tests for the pipeline-manager GUI option table.

Mirrors ``examples/Example_GUI_use.ipynb``.  Everything runs against the
``offscreen`` Qt platform plugin (configured in ``conftest.py``) so no windows
are ever shown and the tests are safe on a headless CI runner.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("cedalion")

from PySide6.QtWidgets import (  # noqa: E402
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
    QStyleOptionViewItem,
    QWidget,
)

from pyBrainAnalyzIR.dataclasses.options_variables import (  # noqa: E402
    BooleanOption,
    ChoiceOption,
    NumericOption,
    OptionsDict,
    StringOption,
)
import pyBrainAnalyzIR.pipelines.modules.glm as glm            # noqa: E402

# the package attribute shadows the module, so import it explicitly
pm = importlib.import_module("pyBrainAnalyzIR.vis.pipeline_manager")

pytestmark = [pytest.mark.requires_qt, pytest.mark.requires_cedalion]


@pytest.fixture()
def dialog(qt_app):
    """A pipeline-manager dialog holding a single GLM module."""
    dlg = pm.PipelineManagerDialog()
    dlg.pipeline_modules = [glm.GLM()]
    dlg._refresh_pipeline_list(select_row=0)
    return dlg


def test_pipeline_continues_from_existing_recording_and_can_reset(simulated_recording, qt_app):
    from pyBrainAnalyzIR.pipelines.modules.preproccessing import input_data
    from pyBrainAnalyzIR.pipelines.modules.utilities import remove_prior_analysis

    rec, _ = simulated_recording
    rec["od"] = rec["amp"].copy()
    module = input_data()
    module.inputName = "od"
    module.outputName = "od"
    dlg = pm.PipelineManagerDialog([module], data=rec)
    assert not dlg.remove_previous_checkbox.isHidden()
    assert not dlg.remove_previous_checkbox.isChecked()
    assert dlg._compute_pipeline_validity() == [True]
    dlg.remove_previous_checkbox.setChecked(True)
    assert dlg._compute_pipeline_validity() == [False]
    assert not dlg.done_button.isEnabled()
    dlg.pipeline_modules[0].inputName = "amp"
    dlg.pipeline_modules[0].outputName = "amp"
    dlg._refresh_pipeline_list()
    assert dlg._compute_pipeline_validity() == [True]
    assert dlg.pipeline_list.count() == 1
    assert isinstance(dlg.selected_modules()[0], remove_prior_analysis)
    tail = dlg.build_pipeline()
    assert isinstance(tail.previous_job, remove_prior_analysis)
    assert tail.run(rec) is rec
    assert list(rec.timeseries) == ["amp"]
    reopened = pm.PipelineManagerDialog(dlg.selected_modules(), data=rec)
    assert reopened.remove_previous_checkbox.isChecked()
    assert len(reopened.pipeline_modules) == 1
    assert len(reopened.selected_modules()) == 2


def test_pipeline_validates_every_dataset_recording_and_resolves_last(simulated_recording, qt_app):
    import copy
    from pyBrainAnalyzIR.dataclasses.dataset import DataSet
    from pyBrainAnalyzIR.pipelines.modules.preproccessing import input_data

    rec, _ = simulated_recording
    other = copy.deepcopy(rec)
    rec["od"] = rec["amp"].copy()
    module = input_data()
    module.inputName = "od"
    module.outputName = "od"
    dlg = pm.PipelineManagerDialog([module], data=DataSet([rec, other]))
    assert dlg._compute_pipeline_validity() == [False]
    module.inputName = "last"
    dlg = pm.PipelineManagerDialog([module], data=DataSet([rec, other]))
    assert dlg._compute_pipeline_validity() == [True]
    module.inputName = "amp"
    module.outputName = "processed"
    following = input_data()
    following.inputName = "last"
    dlg = pm.PipelineManagerDialog([module, following], data=other)
    assert dlg._compute_pipeline_validity() == [True, True]


def test_pipeline_missing_context_retains_amp_default_and_reset_only_is_executable(qt_app):
    from pyBrainAnalyzIR.pipelines.modules.utilities import remove_prior_analysis

    dlg = pm.PipelineManagerDialog()
    assert dlg.remove_previous_checkbox.isHidden()
    assert not dlg.done_button.isEnabled()
    reset = pm.PipelineManagerDialog([remove_prior_analysis()])
    assert reset.pipeline_list.count() == 0
    assert reset.done_button.isEnabled()
    assert isinstance(reset.build_pipeline(), remove_prior_analysis)


def test_pipeline_discard_invalidates_later_consumers(simulated_recording, qt_app):
    from pyBrainAnalyzIR.pipelines.modules.utilities import DiscardData
    from pyBrainAnalyzIR.pipelines.modules.preproccessing import input_data

    rec, _ = simulated_recording
    rec["od"] = rec["amp"].copy()
    discard = DiscardData()
    discard.remove_type = "od"
    following = input_data()
    following.inputName = "od"
    dlg = pm.PipelineManagerDialog([discard, following], data=rec)
    assert dlg._compute_pipeline_validity() == [True, False]


def test_advanced_reset_selection_is_hidden_from_step_list(qt_app):
    from pyBrainAnalyzIR.pipelines.modules.utilities import remove_prior_analysis

    dlg = pm.PipelineManagerDialog()
    dlg.show_advanced_checkbox.setChecked(True)
    row = next(index for index, spec in enumerate(dlg.available_row_specs)
               if spec is not None and spec.cls is remove_prior_analysis)
    dlg.available_list.setCurrentRow(row)
    dlg._add_selected_module()
    assert dlg.remove_previous_checkbox.isChecked()
    assert dlg.pipeline_list.count() == 0
    assert isinstance(dlg.build_pipeline(), remove_prior_analysis)


# ---------------------------------------------------------------------------
# table layout
# ---------------------------------------------------------------------------

def test_option_table_has_a_default_column(dialog):
    table = dialog.options_table
    headers = [table.horizontalHeaderItem(i).text()
               for i in range(table.columnCount())]
    assert headers == ["Option", "Value", "Default"]


def test_option_table_lists_every_option(dialog):
    module = dialog.pipeline_modules[0]
    assert set(dialog.option_row_keys) == set(module.options.keys())


def test_default_column_shows_the_default_value(dialog):
    table = dialog.options_table
    module = dialog.pipeline_modules[0]
    row = dialog.option_row_keys.index("drift_order")
    expected = module.options.option("drift_order").default
    assert table.item(row, 2).text() == str(expected)


def test_modified_values_are_marked_bold(dialog):
    table = dialog.options_table
    module = dialog.pipeline_modules[0]
    row = dialog.option_row_keys.index("drift_order")

    assert not table.item(row, 1).font().bold()
    module.options["drift_order"] = 5
    dialog._populate_option_table(0)
    assert table.item(row, 1).font().bold()


def test_default_column_is_not_editable(dialog):
    from PySide6.QtCore import Qt

    table = dialog.options_table
    item = table.item(0, 2)
    assert not item.flags() & Qt.ItemIsEditable


# ---------------------------------------------------------------------------
# type-appropriate editors
# ---------------------------------------------------------------------------

def make_editor(dialog, option):
    """Create the delegate editor widget the table would use for *option*.

    The editor is produced through a real model index so this exercises the
    same code path as a user double-clicking the cell.
    """
    dialog.pipeline_modules[0].options = OptionsDict({"probe": option})
    dialog._populate_option_table(0)

    delegate = pm.OptionEditorDelegate(dialog)
    index = dialog.options_table.model().index(0, 1)
    # the parent must outlive the call: Qt destroys child widgets with it
    parent = QWidget()
    editor = delegate.createEditor(parent, QStyleOptionViewItem(), index)
    editor._keepalive_parent = parent
    return editor


@pytest.mark.parametrize("option,expected", [
    (ChoiceOption(["ols", "ar_irls"], "ols"), QComboBox),
    (StringOption("ols", allowed=["ols", "ar_irls"]), QComboBox),
    (NumericOption(3, integer_only=True), QSpinBox),
    (NumericOption(0.5, minimum=0.0, maximum=1.0), QDoubleSpinBox),
    (StringOption("free text"), QLineEdit),
    (glm.BasisFunctionOption("AFNIGamma"), QComboBox),
])
def test_editor_type_matches_option_type(dialog, option, expected):
    editor = make_editor(dialog, option)
    assert isinstance(editor, expected)


def test_integer_spinbox_respects_bounds(dialog):
    editor = make_editor(dialog, NumericOption(4, minimum=1, maximum=32,
                                               integer_only=True))
    assert isinstance(editor, QSpinBox)
    assert editor.minimum() == 1
    assert editor.maximum() == 32


def test_float_spinbox_respects_bounds(dialog):
    editor = make_editor(dialog, NumericOption(0.5, minimum=0.0, maximum=1.0))
    assert isinstance(editor, QDoubleSpinBox)
    assert editor.minimum() == pytest.approx(0.0)
    assert editor.maximum() == pytest.approx(1.0)


def test_combo_lists_all_choices(dialog):
    editor = make_editor(dialog, ChoiceOption(["ols", "ar_irls", "wls"], "ols"))
    assert isinstance(editor, QComboBox)
    labels = [editor.itemText(i) for i in range(editor.count())]
    for choice in ("ols", "ar_irls", "wls"):
        assert choice in labels


def test_basis_combo_selects_the_current_preset(dialog):
    editor = make_editor(dialog, glm.BasisFunctionOption("AFNIGamma"))
    delegate = pm.OptionEditorDelegate(dialog)
    index = dialog.options_table.model().index(0, 1)
    delegate.setEditorData(editor, index)
    assert editor.currentText() == "AFNIGamma"
    editor.setCurrentIndex(editor.findText("boxcar"))
    delegate.setModelData(editor, dialog.options_table.model(), index)
    assert dialog.pipeline_modules[0].options.option("probe").preset_name == "boxcar"


def test_boolean_option_uses_a_check_box(dialog):
    """Booleans are rendered as a checkable item rather than a text editor."""
    from PySide6.QtCore import Qt

    module = dialog.pipeline_modules[0]
    module.options = OptionsDict({"flag": BooleanOption(True,
                                                        description="d",
                                                        help="h")})
    dialog._populate_option_table(0)

    item = dialog.options_table.item(0, 1)
    assert item.flags() & Qt.ItemIsUserCheckable
    assert not item.flags() & Qt.ItemIsEditable
    assert item.checkState() == Qt.Checked


def test_toggling_the_check_box_updates_the_option(dialog):
    from PySide6.QtCore import Qt

    module = dialog.pipeline_modules[0]
    module.options = OptionsDict({"flag": BooleanOption(True,
                                                        description="d",
                                                        help="h")})
    dialog._populate_option_table(0)

    item = dialog.options_table.item(0, 1)
    item.setCheckState(Qt.Unchecked)
    assert module.options["flag"] is False


# ---------------------------------------------------------------------------
# editing and validation
# ---------------------------------------------------------------------------

def test_apply_option_value_updates_the_module(dialog):
    module = dialog.pipeline_modules[0]
    row = dialog.option_row_keys.index("drift_order")
    dialog.apply_option_value(row, 4)
    assert module.options["drift_order"] == 4


def test_apply_option_value_rejects_invalid_input(dialog, monkeypatch):
    """An invalid edit is refused and reported, leaving the value untouched."""
    warnings = []
    monkeypatch.setattr(pm.QMessageBox, "warning",
                        lambda *args, **kwargs: warnings.append(args))

    module = dialog.pipeline_modules[0]
    before = module.options["noise_model"]
    row = dialog.option_row_keys.index("noise_model")

    accepted = dialog.apply_option_value(row, "not_a_model")

    assert accepted is False
    assert warnings, "the user was not warned about the invalid value"
    assert module.options["noise_model"] == before


# ---------------------------------------------------------------------------
# resetting to defaults
# ---------------------------------------------------------------------------

def test_reset_single_option_row(dialog):
    module = dialog.pipeline_modules[0]
    module.options["drift_order"] = 5
    module.options["noise_model"] = "ar_irls"
    dialog._populate_option_table(0)

    row = dialog.option_row_keys.index("drift_order")
    dialog._reset_option_row(row)

    assert module.options.option("drift_order").is_default
    # other options must be left alone
    assert module.options["noise_model"] == "ar_irls"


def test_reset_all_options(dialog):
    module = dialog.pipeline_modules[0]
    module.options["drift_order"] = 5
    module.options["noise_model"] = "ar_irls"
    dialog._populate_option_table(0)

    dialog._reset_all_options()

    for opt in module.options.options().values():
        assert opt.is_default


def test_reset_clears_the_bold_marker(dialog):
    table = dialog.options_table
    module = dialog.pipeline_modules[0]
    module.options["drift_order"] = 5
    dialog._populate_option_table(0)

    dialog._reset_all_options()

    bold = [table.item(i, 1).font().bold()
            for i in range(table.rowCount()) if table.item(i, 1)]
    assert not any(bold)


# ---------------------------------------------------------------------------
# pipeline round-trip
# ---------------------------------------------------------------------------

def test_pipeline_survives_a_json_round_trip(dialog):
    """Saving and reloading a pipeline must preserve the option values."""
    module = dialog.pipeline_modules[0]
    module.options["noise_model"] = "ar_irls"
    module.options["drift_order"] = 2

    payload = pm.pipeline_to_json(dialog.pipeline_modules)
    reloaded = pm.pipeline_from_json(payload)[0]
    assert reloaded.options["noise_model"] == "ar_irls"
    assert reloaded.options["drift_order"] == 2


def test_dialog_lists_available_modules(dialog):
    """The palette of modules the user can add must not be empty."""
    assert len(dialog.available_specs) > 0
    names = [spec.cls.__name__ for spec in dialog.available_specs]
    assert "GLM" in names
    assert "resample" in names


BASIC_MODULES = {
    "hyperscanning", "resting_state_connectivity", "bandpass_filter", "pca_filter", "GLM",
    "MixedEffects", "TDDR", "resample", "TrimBaseline", "mbll", "intensity_opticaldensity",
    "ROIAverage",
}


def _listed_classes(dialog):
    return {spec.cls.__name__ for spec in dialog.available_row_specs if spec is not None}


def test_advanced_modules_hidden_by_default(dialog):
    assert not dialog.show_advanced_checkbox.isChecked()
    assert _listed_classes(dialog) == BASIC_MODULES
    assert {spec.cls.__name__ for spec in dialog.available_specs if not spec.advanced} \
        == BASIC_MODULES


def test_show_advanced_modules_lists_everything(dialog):
    dialog.show_advanced_checkbox.setChecked(True)
    assert _listed_classes(dialog) == {spec.cls.__name__ for spec in dialog.available_specs}
    assert "ShortSeparationFilter" in _listed_classes(dialog)
    dialog.show_advanced_checkbox.setChecked(False)
    assert _listed_classes(dialog) == BASIC_MODULES


def test_toggling_advanced_keeps_the_current_selection(dialog):
    row = next(i for i, s in enumerate(dialog.available_row_specs)
               if s is not None and s.cls.__name__ == "GLM")
    dialog.available_list.setCurrentRow(row)
    dialog.show_advanced_checkbox.setChecked(True)
    selected = dialog.available_row_specs[dialog.available_list.currentRow()]
    assert selected.cls.__name__ == "GLM"
