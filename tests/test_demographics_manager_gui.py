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


def test_hide_extra_demographics_preserves_values_and_mixed_columns():
    recordings = [Recording(21, "M"), Recording(34, "F")]
    for rec in recordings:
        rec.meta_data.update(empty="", missing=float("nan"), placeholder="N/A", zero=0, flag=False)
    recordings[0].meta_data["mixed"] = "N/A"
    recordings[1].meta_data["mixed"] = "Present"
    manager = DemographicsManager(DataSet(recordings))
    original = manager._current_data()
    assert not manager.hide_extra_checkbox.isChecked()
    assert not any(manager.table.isColumnHidden(i) for i in range(len(manager._columns)))

    manager.hide_extra_checkbox.setChecked(True)
    for name in ("empty", "missing", "placeholder"):
        assert manager.table.isColumnHidden(manager._columns.index(name))
    for name in ("zero", "flag", "mixed", "age"):
        assert not manager.table.isColumnHidden(manager._columns.index(name))
    assert manager._current_data() == original

    column = manager._columns.index("placeholder")
    manager.table.item(0, column).setText("Now populated")
    assert not manager.table.isColumnHidden(column)
    manager.hide_extra_checkbox.setChecked(False)
    assert not any(manager.table.isColumnHidden(i) for i in range(len(manager._columns)))


@pytest.mark.parametrize("value", [None, float("nan"), "", " ", "N/A", "n/a", "not-a-number"])
def test_meaningless_demographic_values(value):
    from pyBrainAnalyzIR.utils.metadata import has_meaningful_value
    assert not has_meaningful_value(value)


@pytest.mark.parametrize("value", [0, False, "0", "False", "Healthy", 12])
def test_meaningful_demographic_values(value):
    from pyBrainAnalyzIR.utils.metadata import has_meaningful_value
    assert has_meaningful_value(value)
