"""Tests for demographics manager column actions and descriptions."""
from __future__ import annotations

from collections import OrderedDict
from unittest.mock import Mock

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QInputDialog, QMessageBox  # noqa: E402

from pyBrainAnalyzIR.dataclasses.dataset import DataSet  # noqa: E402
from pyBrainAnalyzIR.vis.demographics_manager import DemographicsManager  # noqa: E402

pytestmark = [pytest.mark.requires_qt, pytest.mark.usefixtures("qt_app")]


class Recording:
    def __init__(self, age: int, gender: str):
        self.meta_data = OrderedDict({
            "subject": "sub-A",
            "age": age,
            "gender": gender,
            "_bids_descriptions": {
                "age": "Participant age in years.",
            },
        })


def test_demographics_header_find_replace_is_scoped_to_selected_column():
    manager = DemographicsManager(DataSet([
        Recording(21, "M"),
        Recording(21, "F"),
        Recording(34, "M"),
    ]))
    age_column = manager._columns.index("age")
    gender_column = manager._columns.index("gender")

    replaced = manager._replace_column_values(age_column, "21", "30")

    assert replaced == 2
    assert [
        manager.table.item(row, age_column).text()
        for row in range(manager.table.rowCount())
    ] == ["30", "30", "34"]
    assert [
        manager.table.item(row, gender_column).text()
        for row in range(manager.table.rowCount())
    ] == ["M", "F", "M"]


def test_demographics_header_find_replace_prompts_for_both_values(monkeypatch):
    manager = DemographicsManager(DataSet([
        Recording(21, "M"),
        Recording(21, "F"),
    ]))
    age_column = manager._columns.index("age")
    get_text = Mock(side_effect=[("21", True), ("35", True)])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(get_text))
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(Mock(return_value=QMessageBox.Ok)),
    )

    manager._find_replace_column(age_column)

    assert get_text.call_count == 2
    assert [
        manager.table.item(row, age_column).text()
        for row in range(manager.table.rowCount())
    ] == ["35", "35"]


def test_demographics_header_tooltip_shows_saved_description():
    manager = DemographicsManager(DataSet([Recording(21, "M")]))
    age_column = manager._columns.index("age")
    gender_column = manager._columns.index("gender")

    assert manager.table.horizontalHeaderItem(age_column).toolTip() == (
        "Participant age in years."
    )
    assert manager.table.horizontalHeaderItem(gender_column).toolTip() == ""
