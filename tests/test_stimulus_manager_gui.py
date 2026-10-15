"""Tests for BIDS event descriptions in the stimulus manager."""
from __future__ import annotations

from collections import OrderedDict

import pandas as pd
import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QDialog, QMenuBar  # noqa: E402

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
