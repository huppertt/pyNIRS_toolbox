"""Tests for BIDS event descriptions in the stimulus manager."""
from __future__ import annotations

from collections import OrderedDict

import pandas as pd
import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QDialog, QMenuBar, QMessageBox  # noqa: E402

from pyBrainAnalyzIR.dataclasses.dataset import DataSet  # noqa: E402
from pyBrainAnalyzIR.vis.stimulus_manager import (  # noqa: E402
    EventDescriptionsDialog,
    StimulusManagerWindow,
)

pytestmark = [pytest.mark.requires_qt, pytest.mark.usefixtures("qt_app")]


class Recording:
    def __init__(self, subject: str, event_types: list[str]):
        self.meta_data = OrderedDict({"subject": subject})
        self.timeseries = {}
        self.stim = pd.DataFrame({"trial_type": event_types})


def _dataset() -> DataSet:
    dataset = DataSet([
        Recording("sub-A", ["left", "left", "right"]),
        Recording("sub-B", ["right", "rest"]),
    ])
    dataset.add_meta_events_description("left", "Existing left-hand description.")
    return dataset


def test_bids_info_menu_is_between_file_and_validate():
    window = StimulusManagerWindow(_dataset())
    menu_bar = window.findChild(QMenuBar)

    assert [action.text() for action in menu_bar.actions()] == [
        "File",
        "Bids Info",
        "Validate",
    ]
    assert [action.text() for action in menu_bar.actions()[1].menu().actions()] == [
        "Add Descriptions",
    ]


def test_event_descriptions_dialog_lists_unique_types_and_existing_descriptions():
    dialog = EventDescriptionsDialog(_dataset())

    assert dialog.table.rowCount() == 3
    assert [
        dialog.table.item(row, 0).text()
        for row in range(dialog.table.rowCount())
    ] == ["left", "right", "rest"]
    assert dialog.table.item(0, 1).text() == "Existing left-hand description."
    assert dialog.table.item(1, 1).text() == ""
    assert not dialog.table.item(0, 0).flags() & Qt.ItemIsEditable


def test_add_descriptions_menu_action_updates_dataset(monkeypatch):
    dataset = _dataset()
    original = [dict(rec.meta_data["_bids_descriptions"]) for rec in dataset.dataset]
    window = StimulusManagerWindow(dataset)

    def accept_with_edits(dialog):
        dialog.table.item(0, 1).setText("Updated left-hand description.")
        dialog.table.item(1, 1).setText("Right-hand tapping.")
        dialog.table.item(2, 1).setText("Resting condition.")
        return QDialog.Accepted

    monkeypatch.setattr(EventDescriptionsDialog, "exec", accept_with_edits)
    window._add_event_descriptions()

    for rec in dataset.dataset:
        descriptions = rec.meta_data["_bids_descriptions"]
        assert descriptions["event:left"] == "Updated left-hand description."
        assert descriptions["event:right"] == "Right-hand tapping."
        assert descriptions["event:rest"] == "Resting condition."
    assert window._dirty
    window._restore_original_stims()
    for rec, descriptions in zip(dataset.dataset, original):
        assert rec.meta_data["_bids_descriptions"] == descriptions
    assert not window._dirty


def test_summary_hides_missing_demographics_and_data_selector():
    dataset = _dataset()
    dataset.dataset[0].meta_data.update(age=0, flag=False, extra="N/A", empty="", nan=float("nan"))
    window = StimulusManagerWindow(dataset)
    text = window.summary_panel.toPlainText()
    assert "age: 0" in text and "flag: False" in text
    assert all(key not in text for key in ("extra:", "empty:", "nan:", "_bids_descriptions"))
    assert not hasattr(window, "type_selector")


def test_stimulus_close_without_edits_does_not_prompt(monkeypatch):
    window = StimulusManagerWindow(_dataset())
    prompts = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args: prompts.append(args))
    window.stim_model.sort(1)
    window.stim_model.set_type_filter({"left"})
    window._on_file_changed(1)
    assert not window._dirty
    window._cancel_and_close()
    assert not prompts
    assert window._finish_state is False


@pytest.mark.parametrize("operation", ["edit", "add", "remove", "rename", "duplicate", "import"])
def test_stimulus_changes_prompt_and_cancel_restores_all_recordings(monkeypatch, operation):
    dataset = _dataset()
    original = [rec.stim.copy() for rec in dataset.dataset]
    window = StimulusManagerWindow(dataset)
    window._set_edit_rows([0, 1], follow_current=False)
    model = window.stim_model
    if operation == "edit":
        model.setData(model.index(0, 1), "changed")
    elif operation == "add":
        model.add_row()
    elif operation == "remove":
        model.remove_rows([0])
    elif operation == "rename":
        model.rename_type("left", "changed")
    elif operation == "duplicate":
        model.duplicate_type("left", "changed")
    else:
        model.import_frames({0: pd.DataFrame({
            "trial_type": ["imported"], "onset": [1.0], "duration": [2.0], "value": [1.0],
        })}, "replace")
    assert window._dirty
    prompts = []

    def decline(*args):
        prompts.append(args)
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", decline)
    window._cancel_and_close()
    assert len(prompts) == 1 and window._finish_state is None
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    window._cancel_and_close()
    for rec, stim in zip(dataset.dataset, original):
        pd.testing.assert_frame_equal(rec.stim, stim)


def test_stimulus_cancel_restores_originally_absent_stimuli_and_descriptions():
    dataset = _dataset()
    rec = dataset.dataset[0]
    rec.stim = None
    rec.meta_data.pop("_bids_descriptions")
    window = StimulusManagerWindow(dataset)
    window.stim_model.add_row()
    dataset.add_meta_events_description("new_event", "New")
    window._update_dirty()
    assert window._dirty
    window._restore_original_stims()
    assert rec.stim is None
    assert "_bids_descriptions" not in rec.meta_data


def test_window_close_checks_dirty_flag_and_can_be_declined(monkeypatch):
    window = StimulusManagerWindow(_dataset())
    window.stim_model.rename_type("left", "changed")
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.No)
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert window._finish_state is None
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    assert window._finish_state is False


def test_update_keeps_edits_without_discard_prompt(monkeypatch):
    dataset = _dataset()
    window = StimulusManagerWindow(dataset)
    window.stim_model.rename_type("left", "changed")
    prompts = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args: prompts.append(args))
    window._apply_and_close()
    assert window._finish_state is True
    assert "changed" in dataset.dataset[0].stim.trial_type.values
    assert not prompts
