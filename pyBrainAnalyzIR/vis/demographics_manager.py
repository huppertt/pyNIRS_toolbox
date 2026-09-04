"""PySide6 GUI to edit dataset demographics and generic dictionary tables.

Usage
-----
    from pyBrainAnalyzIR.vis.demographics_manager import demographics_manager

    result = demographics_manager(dataset)  # dataset is returned after closing

In a notebook, the returned window keeps the Qt event loop responsive and
updates `dataset` when it is closed. In a script, `result` is the edited
dataset.
"""
from __future__ import annotations

import argparse
import math
import numbers
import pickle
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

SUBJECT_ID_KEY = "subjectID"

NO_UNITS_LABEL = "No Units"

#: Item role holding the input-order index of a row, so re-sorting the view for
#: display never changes which recording a row is written back to.
ORIGINAL_ROW_ROLE = Qt.UserRole + 1

_OPEN_WINDOWS: List["DemographicsManager"] = []


def _load_common_units() -> "OrderedDict[str, List[Any]]":
    """Return the `common_units` groups, or an empty mapping if unavailable.

    Imported lazily because `cedalion` is an optional dependency; the rest of
    the editor must stay usable without it.
    """
    try:
        from pyBrainAnalyzIR.utils.cedalion_helpers import common_units
    except Exception:  # noqa: BLE001 - cedalion missing or unit lookup failed
        return OrderedDict()
    return OrderedDict(common_units)


def _is_numeric(value: Any) -> bool:
    """True for real numbers that pint can attach a unit to.

    `bool` is a `numbers.Number` but multiplying it by a unit raises, and
    strings/empty cells raise as well, so they are all excluded here.
    """
    return isinstance(value, numbers.Number) and not isinstance(value, bool)


def _known_unit_strings() -> set:
    """All unit spellings offered by `common_units`, used to parse cell text."""
    names = set()
    for unit_list in _load_common_units().values():
        for unit in unit_list:
            names.add(str(unit))
    return names


def _split_magnitude_units(value: Any):
    """Split `value` into (magnitude, unit), stripping any existing pint unit."""
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        return value.magnitude, value.units
    if isinstance(value, str) and " " in value.strip():
        # Cells re-read from the table look like "24 centimeter"; recover the
        # numeric part so the old unit can be dropped and a new one assigned.
        head, tail = value.strip().split(" ", 1)
        magnitude = _coerce_value(head)
        if _is_numeric(magnitude) and tail in _known_unit_strings():
            return magnitude, tail
    return value, None


class UnitsDialog(QDialog):
    """Assign a physical unit to each column of the demographics table."""

    def __init__(
        self,
        columns: List[str],
        current: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Add Units")
        self.resize(480, 400)

        self._columns = list(columns)
        self._groups = _load_common_units()
        current = current or {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select the units associated with each variable:"))

        self.table = QTableWidget(len(self._columns), 2, self)
        self.table.setHorizontalHeaderLabels(["Variable", "Units"])
        self.table.verticalHeader().setVisible(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.horizontalHeader().setStretchLastSection(True)

        self._combos: Dict[str, QComboBox] = {}
        for row, name in enumerate(self._columns):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)

            combo = self._build_combo()
            self._select_current(combo, current.get(name))
            self.table.setCellWidget(row, 1, combo)
            self._combos[name] = combo
        layout.addWidget(self.table)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Confirm")
        buttons.button(QDialogButtonBox.Cancel).setText("Cancel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_combo(self) -> QComboBox:
        """One combo per row: "No Units" first, then non-selectable group headers."""
        combo = QComboBox(self)
        combo.addItem(NO_UNITS_LABEL, None)
        for group, unit_list in self._groups.items():
            combo.insertSeparator(combo.count())
            header_index = combo.count()
            combo.addItem(str(group), None)
            # Group names organize the list but are not valid choices.
            item = combo.model().item(header_index)
            item.setEnabled(False)
            for unit in unit_list:
                combo.addItem(f"   {unit}", unit)
        return combo

    @staticmethod
    def _select_current(combo: QComboBox, unit: Any) -> None:
        if unit is None:
            combo.setCurrentIndex(0)
            return
        for index in range(combo.count()):
            data = combo.itemData(index)
            if data is not None and str(data) == str(unit):
                combo.setCurrentIndex(index)
                return
        combo.setCurrentIndex(0)

    def selected_units(self) -> Dict[str, Any]:
        """Column name -> pint unit, omitting columns left at "No Units"."""
        chosen: Dict[str, Any] = {}
        for name, combo in self._combos.items():
            unit = combo.currentData()
            if unit is not None:
                chosen[name] = unit
        return chosen


def _to_display_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _coerce_value(text: str):
    """Try to turn a cell's text back into an int/float, otherwise keep str."""
    text = text.strip()
    if text == "":
        return None
    try:
        as_int = int(text)
        return as_int
    except ValueError:
        pass
    try:
        as_float = float(text)
        return as_float
    except ValueError:
        pass
    return text


class MatchColumnsDialog(QDialog):
    """Lets the user pick one or more shared columns to match rows on."""

    def __init__(self, shared_columns: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Select Matching Column(s)")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select column(s) shared between the table and the "
                                "Excel file to use as the match key:"))

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        for col in shared_columns:
            item = QListWidgetItem(col)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_columns(self) -> List[str]:
        return [item.text() for item in self.list_widget.selectedItems()]


class StatisticsDialog(QDialog):
    """Summary statistics for one column, expandable to a histogram."""

    def __init__(self, name: str, values: List[float], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Statistics - {name}")
        self._name = name
        self._values = values
        self._canvas = None

        layout = QVBoxLayout(self)

        count = len(values)
        avg = sum(values) / count
        vmin, vmax = min(values), max(values)
        if count > 1:
            var = sum((v - avg) ** 2 for v in values) / (count - 1)
            sd_text = f"\nStd Dev: {math.sqrt(var):.4g}"
        else:
            sd_text = ""
        layout.addWidget(QLabel(
            f"Column: {name}\n"
            f"Count: {count}\n"
            f"Average: {avg:.4g}\n"
            f"Min: {vmin:.4g}\n"
            f"Max: {vmax:.4g}\n"
            f"Range: {vmax - vmin:.4g}" + sd_text
        ))

        self._plot_container = QVBoxLayout()
        layout.addLayout(self._plot_container)

        buttons = QHBoxLayout()
        self.more_info_button = QPushButton("Show More Info")
        self.more_info_button.clicked.connect(self.show_histogram)
        buttons.addWidget(self.more_info_button)
        buttons.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def show_histogram(self) -> None:
        """Expand the dialog to show the distribution of the column values."""
        if self._canvas is not None:
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        figure = Figure(figsize=(5, 3), tight_layout=True)
        axes = figure.add_subplot(111)
        bins = max(1, min(20, len(set(self._values))))
        axes.hist(self._values, bins=bins, color="steelblue", edgecolor="black")
        axes.set_xlabel(self._name)
        axes.set_ylabel("Count")
        self._canvas = FigureCanvas(figure)
        self._plot_container.addWidget(self._canvas)
        self.more_info_button.setEnabled(False)
        self.resize(600, 500)


class DictTableEditor(QDialog):
    """Composite table editor for a list of dictionaries."""

    def __init__(self, data: List[Dict[str, Any]], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Dictionary Table Editor")
        self.resize(900, 600)

        self._columns: List[str] = self._build_columns(data)
        self._n_rows = len(data)
        self.result: List[Dict[str, Any]] = []
        # column name -> pint unit chosen through the Add Units dialog
        self.column_units: Dict[str, Any] = {}
        self.show_outliers = False
        self._sort_column: Optional[int] = None
        self._sort_ascending = True

        self._init_ui()
        self._populate_table(data)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    @staticmethod
    def _build_columns(data: List[Dict[str, Any]]) -> List[str]:
        columns: List[str] = []
        if any(SUBJECT_ID_KEY in d for d in data):
            columns.append(SUBJECT_ID_KEY)
        for d in data:
            for key in d.keys():
                if key not in columns:
                    columns.append(key)
        return columns

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # `setNativeMenuBar(False)` keeps the menus inside the window instead of
        # the macOS global menu bar, matching stimulus_manager/NIRSviewIR.
        menu_bar = QMenuBar(self)
        menu_bar.setNativeMenuBar(False)

        self.file_menu = menu_bar.addMenu("File")
        self.file_menu.addAction("Load from Excel").triggered.connect(self.import_excel)
        self.file_menu.addAction("Save to Excel").triggered.connect(self.export_excel)

        self.edit_menu = menu_bar.addMenu("Edit")
        self.edit_menu.addAction("Add Additional BIDS Variables").triggered.connect(
            self.add_bids_variable
        )
        self.edit_menu.addAction("Add Units").triggered.connect(self.add_units)
        self.show_outliers_action = self.edit_menu.addAction("Show Outliers")
        self.show_outliers_action.setCheckable(True)
        self.show_outliers_action.toggled.connect(self._toggle_outliers)
        main_layout.addWidget(menu_bar)

        toolbar = QHBoxLayout()
        add_col_btn = QPushButton("Add Column")
        add_col_btn.clicked.connect(self.add_column)
        toolbar.addWidget(add_col_btn)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        self.table = QTableWidget()
        self.table.setSortingEnabled(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.horizontalHeader().customContextMenuRequested.connect(
            self._show_header_menu
        )
        # Clicking a header sorts every column by that one.
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().sectionClicked.connect(self._sort_by_column)
        main_layout.addWidget(self.table)

        done_btn = QPushButton("Done")
        done_btn.clicked.connect(self.accept)
        main_layout.addWidget(done_btn, alignment=Qt.AlignRight)

    def _populate_table(self, data: List[Dict[str, Any]]) -> None:
        self.table.setColumnCount(len(self._columns))
        self.table.setHorizontalHeaderLabels(self._columns)
        self.table.setRowCount(len(data))
        for row, d in enumerate(data):
            for col, key in enumerate(self._columns):
                item = QTableWidgetItem(_to_display_str(d.get(key)))
                if col == 0:
                    # Remember which input row (and therefore which recording)
                    # this line came from, so sorting the view for display never
                    # changes where the values are written back.
                    item.setData(ORIGINAL_ROW_ROLE, row)
                self.table.setItem(row, col, item)

    def _original_row(self, row: int) -> int:
        """The input-order index of view row `row`."""
        item = self.table.item(row, 0)
        if item is not None:
            stored = item.data(ORIGINAL_ROW_ROLE)
            if stored is not None:
                return int(stored)
        return row

    # ------------------------------------------------------------------
    # header context menu
    # ------------------------------------------------------------------
    def _show_header_menu(self, pos) -> None:
        col = self.table.horizontalHeader().logicalIndexAt(pos)
        if col < 0:
            return
        menu = QMenu(self)
        rename_action = menu.addAction("Rename Column")
        remove_action = menu.addAction("Remove Column")
        menu.addSeparator()
        assign_action = menu.addAction("Assign All Values...")
        fill_action = menu.addAction("Fill Missing Values...")
        stats_action = menu.addAction("Show Statistics")
        action = menu.exec_(self.table.horizontalHeader().mapToGlobal(pos))
        if action == rename_action:
            self._rename_column(col)
        elif action == remove_action:
            self._remove_column(col)
        elif action == assign_action:
            self._assign_all_values(col)
        elif action == fill_action:
            self._fill_missing(col)
        elif action == stats_action:
            self._show_statistics(col)

    def _assign_all_values(self, col: int) -> None:
        """Set every row of `col` to a single user-supplied value."""
        name = self._columns[col]
        value, ok = QInputDialog.getText(
            self, "Assign All Values", f"Value to assign to every row in '{name}':"
        )
        if not ok:
            return
        confirm = QMessageBox.question(
            self, "Assign All Values",
            f"Assign '{value}' to all {self.table.rowCount()} row(s) of '{name}'?\n\n"
            "This will overwrite any existing values in this column.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        for row in range(self.table.rowCount()):
            item = self.table.item(row, col)
            if item is None:
                item = QTableWidgetItem()
                self.table.setItem(row, col, item)
            item.setText(value)
        self._refresh_outlier_highlighting()

    def _sort_by_column(self, col: int) -> None:
        """Sort all rows by `col`, toggling between ascending and descending."""
        if col < 0 or col >= len(self._columns):
            return
        if self._sort_column == col:
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_column = col
            self._sort_ascending = True

        # Snapshot whole rows (text + original-row identity) so the values keep
        # travelling together when the view is reordered.
        snapshot = []
        for row in range(self.table.rowCount()):
            texts = [
                self.table.item(row, c).text() if self.table.item(row, c) else ""
                for c in range(len(self._columns))
            ]
            snapshot.append((self._original_row(row), texts))

        def sort_key(entry):
            text = entry[1][col]
            magnitude, _ = _split_magnitude_units(_coerce_value(text))
            if _is_numeric(magnitude):
                # Numbers sort before text; blanks sink to the bottom.
                return (0, float(magnitude), "")
            if text.strip() == "":
                return (2, 0.0, "")
            return (1, 0.0, text.lower())

        snapshot.sort(key=sort_key, reverse=not self._sort_ascending)

        for row, (original, texts) in enumerate(snapshot):
            for c, text in enumerate(texts):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setData(ORIGINAL_ROW_ROLE, original)
                self.table.setItem(row, c, item)

        self.table.horizontalHeader().setSortIndicatorShown(True)
        self.table.horizontalHeader().setSortIndicator(
            col, Qt.AscendingOrder if self._sort_ascending else Qt.DescendingOrder
        )
        self._refresh_outlier_highlighting()

    def _toggle_outliers(self, enabled: bool) -> None:
        self.show_outliers = bool(enabled)
        self._refresh_outlier_highlighting()

    def _refresh_outlier_highlighting(self) -> None:
        """Colour studentized outliers (|t| > 2) red in every numeric column."""
        default_brush = QBrush()
        red_brush = QBrush(QColor("red"))
        for col in range(len(self._columns)):
            scores = self._studentized_scores(col) if self.show_outliers else {}
            for row in range(self.table.rowCount()):
                item = self.table.item(row, col)
                if item is None:
                    continue
                score = scores.get(row)
                if score is not None and abs(score) > 2.0:
                    item.setForeground(red_brush)
                else:
                    item.setForeground(default_brush)

    def _studentized_scores(self, col: int) -> Dict[int, float]:
        """Row -> studentized norm for the numeric values of one column."""
        values: Dict[int, float] = {}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, col)
            if item is None:
                continue
            magnitude, _ = _split_magnitude_units(_coerce_value(item.text()))
            if _is_numeric(magnitude):
                values[row] = float(magnitude)

        if len(values) < 3:
            # Too few points for a meaningful deleted-residual estimate.
            return {}

        scores: Dict[int, float] = {}
        for row, value in values.items():
            # Studentized: compare each point against the mean/SD of the others,
            # so a single extreme value cannot mask itself.
            others = [v for r, v in values.items() if r != row]
            mean = sum(others) / len(others)
            var = sum((v - mean) ** 2 for v in others) / (len(others) - 1)
            sd = math.sqrt(var)
            if sd == 0:
                continue
            scores[row] = (value - mean) / sd
        return scores

    def _rename_column(self, col: int) -> None:
        old_name = self._columns[col]
        new_name, ok = QInputDialog.getText(
            self, "Rename Column", "New column name:", text=old_name
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name in self._columns and new_name != old_name:
            QMessageBox.warning(self, "Rename Column", "A column with that name already exists.")
            return
        self._columns[col] = new_name
        self.table.setHorizontalHeaderItem(col, QTableWidgetItem(new_name))

    def _remove_column(self, col: int) -> None:
        name = self._columns[col]
        confirm = QMessageBox.question(
            self, "Remove Column", f"Remove column '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        self.table.removeColumn(col)
        del self._columns[col]

    def add_column(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Column", "New column name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self._columns:
            QMessageBox.warning(self, "Add Column", "A column with that name already exists.")
            return
        col = self.table.columnCount()
        self.table.insertColumn(col)
        self._columns.append(name)
        self.table.setHorizontalHeaderItem(col, QTableWidgetItem(name))
        for row in range(self.table.rowCount()):
            self.table.setItem(row, col, QTableWidgetItem(""))

    def add_bids_variable(self) -> None:
        """Add BIDS variables.

        The base editor has no dataset to derive BIDS metadata from, so it just
        asks for a column name.  `DemographicsManager` overrides this to call
        `pyBrainAnalyzIR.io.bids.add_missing_bids_to_metadata`.
        """
        name, ok = QInputDialog.getText(
            self, "Add Additional BIDS Variables", "BIDS variable name:"
        )
        if ok and name.strip():
            self._add_column_named(name.strip(), "Add Additional BIDS Variables")

    def add_units(self) -> None:
        """Attach a pint unit to the values of one or more columns."""
        if not self._columns:
            QMessageBox.information(self, "Add Units", "The table has no columns.")
            return

        dialog = UnitsDialog(self._columns, self.column_units, self)
        if not dialog._groups:
            QMessageBox.warning(
                self, "Add Units",
                "The common units list is unavailable (cedalion could not be "
                "imported), so units cannot be assigned.",
            )
            return
        if dialog.exec_() != QDialog.Accepted:
            return

        self.column_units = dialog.selected_units()
        self._refresh_unit_display()

    def _refresh_unit_display(self) -> None:
        """Re-render cells so the chosen unit is shown alongside each value."""
        for col, name in enumerate(self._columns):
            for row in range(self.table.rowCount()):
                item = self.table.item(row, col)
                if item is None:
                    continue
                value = self._apply_unit(name, _coerce_value(item.text()))
                item.setText(_to_display_str(value))

    def _add_column_named(self, name: str, title: str) -> None:
        if name in self._columns:
            QMessageBox.information(self, title, f"The column '{name}' already exists.")
            return
        col = self.table.columnCount()
        self.table.insertColumn(col)
        self._columns.append(name)
        self.table.setHorizontalHeaderItem(col, QTableWidgetItem(name))
        for row in range(self.table.rowCount()):
            self.table.setItem(row, col, QTableWidgetItem(""))

    def _fill_missing(self, col: int) -> None:
        value, ok = QInputDialog.getText(
            self, "Fill Missing Values", f"Value to use for empty cells in '{self._columns[col]}':"
        )
        if not ok:
            return
        for row in range(self.table.rowCount()):
            item = self.table.item(row, col)
            if item is None or item.text().strip() == "":
                self.table.setItem(row, col, QTableWidgetItem(value))

    def _show_statistics(self, col: int) -> None:
        values = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, col)
            if item is None:
                continue
            magnitude, _ = _split_magnitude_units(_coerce_value(item.text()))
            if _is_numeric(magnitude):
                values.append(float(magnitude))

        name = self._columns[col]
        if not values:
            QMessageBox.information(self, "Statistics", f"'{name}' has no numeric values.")
            return

        StatisticsDialog(name, values, self).exec_()

    # ------------------------------------------------------------------
    # Excel import / export
    # ------------------------------------------------------------------
    def _current_data(self) -> List[Dict[str, Any]]:
        """Rows in the *original* input order, regardless of the view's sorting."""
        rows: List[Dict[str, Any]] = [{} for _ in range(self.table.rowCount())]
        for row in range(self.table.rowCount()):
            d: Dict[str, Any] = {}
            for col, key in enumerate(self._columns):
                item = self.table.item(row, col)
                text = item.text() if item is not None else ""
                value = _coerce_value(text)
                if value is not None:
                    d[key] = self._apply_unit(key, value)
            target = self._original_row(row)
            if 0 <= target < len(rows):
                rows[target] = d
            else:
                rows.append(d)
        return rows

    def _apply_unit(self, key: str, value: Any) -> Any:
        """Attach the column's chosen unit, replacing any existing pint unit."""
        unit = self.column_units.get(key)
        magnitude, existing = _split_magnitude_units(value)
        if unit is None:
            # "No Units": drop any unit that was previously attached.
            return magnitude if existing is not None else value
        if not _is_numeric(magnitude):
            # Text values (e.g. subject IDs) and empty cells cannot carry units.
            return value
        return magnitude * unit

    def import_excel(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load from Excel", "", "Excel Files (*.xlsx *.xls)"
        )
        if not path:
            return
        try:
            excel_df = pd.read_excel(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load from Excel", f"Failed to read file:\n{exc}")
            return

        shared_columns = [c for c in self._columns if c in excel_df.columns]
        if shared_columns:
            match_cols = self._resolve_match_columns(shared_columns, excel_df)
            if match_cols is None:
                return
            self._merge_excel(excel_df, match_cols)
            return

        self._merge_excel_by_index(excel_df)

    def _resolve_match_columns(
        self, shared_columns: List[str], excel_df: pd.DataFrame
    ) -> Optional[List[str]]:
        """Pick the column(s) to match rows on, asking the user when ambiguous."""
        unique_cols = [
            col for col in shared_columns
            if self._is_unique_match_column(col, excel_df)
        ]
        if unique_cols:
            return unique_cols

        dialog = MatchColumnsDialog(shared_columns, self)
        dialog.setWindowTitle("Select Matching Column")
        dialog.list_widget.setSelectionMode(QListWidget.SingleSelection)
        if dialog.exec_() != QDialog.Accepted:
            return None
        selected = dialog.selected_columns()
        if not selected:
            QMessageBox.warning(self, "Load from Excel", "No matching column selected.")
            return None
        return selected[:1]

    def _is_unique_match_column(self, col: str, excel_df: pd.DataFrame) -> bool:
        """True when `col` uniquely identifies every row on both sides."""
        table_values = [
            _to_display_str(row.get(col)) for row in self._current_data()
        ]
        if not table_values or any(value == "" for value in table_values):
            return False
        if len(set(table_values)) != len(table_values):
            return False

        excel_values = [_to_display_str(value) for value in excel_df[col].tolist()]
        if any(value == "" for value in excel_values):
            return False
        if len(set(excel_values)) != len(excel_values):
            return False

        # Every table row must be resolvable in the Excel file.
        return set(table_values) <= set(excel_values)

    def _merge_excel_by_index(self, excel_df: pd.DataFrame) -> None:
        """Fallback path used when the Excel file shares no column names."""
        if len(excel_df) != self.table.rowCount():
            QMessageBox.warning(
                self, "Load from Excel",
                "No matching column names were found between the table and the "
                f"Excel file, and the Excel file has {len(excel_df)} row(s) while "
                f"the table has {self.table.rowCount()} row(s), so the rows cannot "
                "be matched.",
            )
            return

        confirm = QMessageBox.question(
            self, "Load from Excel",
            "No matching column names were found between the table and the Excel "
            "file, but both have the same number of rows.\n\n"
            "Do you want to match the new data using the row index (row order)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        if not self._merge_rows({row: row for row in range(self.table.rowCount())}, excel_df, []):
            return

        QMessageBox.warning(
            self, "Load from Excel",
            "The data was matched by row index only. Please verify the "
            "demographics table carefully before running any further analysis.",
        )

    def _merge_excel(self, excel_df: pd.DataFrame, match_cols: List[str]) -> None:
        match_col_idx = [self._columns.index(c) for c in match_cols]
        excel_lookup: Dict[tuple, int] = {}
        for excel_pos, (_, excel_row) in enumerate(excel_df.iterrows()):
            key = tuple(_to_display_str(excel_row.get(c)) for c in match_cols)
            excel_lookup.setdefault(key, excel_pos)

        row_map: Dict[int, int] = {}
        for row in range(self.table.rowCount()):
            key = tuple(
                self.table.item(row, idx).text() if self.table.item(row, idx) else ""
                for idx in match_col_idx
            )
            excel_pos = excel_lookup.get(key)
            if excel_pos is not None:
                row_map[row] = excel_pos

        if not row_map:
            QMessageBox.warning(
                self, "Load from Excel",
                "No rows in the Excel file matched the table on "
                f"{', '.join(match_cols)}.",
            )
            return

        if self._merge_rows(row_map, excel_df, match_cols):
            QMessageBox.information(
                self, "Load from Excel",
                f"Matched and updated {len(row_map)} row(s).",
            )

    def _merge_rows(
        self, row_map: Dict[int, int], excel_df: pd.DataFrame, match_cols: List[str]
    ) -> bool:
        """Apply `excel_df` onto the table rows in `row_map`.

        Returns False when the user declines to overwrite existing values.
        """
        incoming_cols = [c for c in excel_df.columns if c not in match_cols]

        conflicts: List[str] = []
        for row, excel_pos in row_map.items():
            for name in incoming_cols:
                if name not in self._columns:
                    continue
                col = self._columns.index(name)
                item = self.table.item(row, col)
                old_text = item.text() if item is not None else ""
                new_text = _to_display_str(excel_df.iloc[excel_pos][name])
                if old_text.strip() == "" or old_text == new_text:
                    continue
                conflicts.append(
                    f"row {row + 1}, '{name}': '{old_text}' -> '{new_text}'"
                )

        if conflicts:
            preview = "\n".join(conflicts[:15])
            if len(conflicts) > 15:
                preview += f"\n... and {len(conflicts) - 15} more"
            confirm = QMessageBox.question(
                self, "Load from Excel",
                f"{len(conflicts)} existing value(s) would be overwritten:\n\n"
                f"{preview}\n\nDo you want to continue and overwrite them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return False

        for name in incoming_cols:
            if name in self._columns:
                continue
            col = self.table.columnCount()
            self.table.insertColumn(col)
            self._columns.append(name)
            self.table.setHorizontalHeaderItem(col, QTableWidgetItem(name))
            for row in range(self.table.rowCount()):
                self.table.setItem(row, col, QTableWidgetItem(""))

        for row, excel_pos in row_map.items():
            for name in incoming_cols:
                col = self._columns.index(name)
                value = excel_df.iloc[excel_pos][name]
                self.table.setItem(row, col, QTableWidgetItem(_to_display_str(value)))
        return True

    def export_excel(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export to Excel", "", "Excel Files (*.xlsx)"
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            df = pd.DataFrame(self._current_data(), columns=self._columns)
            df.to_excel(path, index=False)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export to Excel", f"Failed to write file:\n{exc}")
            return
        QMessageBox.information(self, "Export to Excel", f"Saved to:\n{path}")

    # ------------------------------------------------------------------
    def accept(self) -> None:
        self.result = self._current_data()
        super().accept()

    def reject(self) -> None:
        self.result = self._current_data()
        super().reject()


def edit_dict_table(data: Union[List[Dict[str, Any]], pd.DataFrame]) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """Show the table editor for `data` and return the edited list of dicts.

    Blocks until the user closes the window. The returned list preserves the
    row order of the input `data`, reflecting any edits, renamed/added/removed
    columns, and filled-in values.
    """
    is_frame = isinstance(data, pd.DataFrame)
    rows = data.to_dict(orient="records") if is_frame else data
    owns_app = QApplication.instance() is None
    app = QApplication.instance() or QApplication(sys.argv)
    editor = DictTableEditor(rows)
    editor.exec_()
    result = editor.result
    if owns_app:
        app.quit()
    return pd.DataFrame(result) if is_frame else result


class DemographicsManager(DictTableEditor):
    """Dialog that applies edited rows to every recording in a DataSet."""

    def __init__(self, dataset: Any, parent: Optional[QWidget] = None):
        self.dataset = dataset
        rows = self._rows_from_dataset(dataset)
        super().__init__(rows, parent)
        self.setWindowTitle("Demographics Manager")
        self.column_units = self._units_from_rows(rows)
        self._refresh_unit_display()

    @staticmethod
    def _units_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pick up units already carried by pint quantities in the meta data."""
        units: Dict[str, Any] = {}
        for row in rows:
            for key, value in row.items():
                if key in units:
                    continue
                _, unit = _split_magnitude_units(value)
                if unit is not None and not isinstance(unit, str):
                    units[key] = unit
        return units

    @staticmethod
    def _rows_from_dataset(dataset: Any) -> List[Dict[str, Any]]:
        recordings = list(getattr(dataset, "dataset", []))
        rows = [
            {key: value for key, value in row.items() if _to_display_str(value) != ""}
            for row in dataset.get_demographics().to_dict(orient="records")
        ]
        while len(rows) < len(recordings):
            rows.append({})
        return rows

    def _apply_to_dataset(self) -> None:
        recordings = list(getattr(self.dataset, "dataset", []))
        if len(self.result) != len(recordings):
            QMessageBox.warning(
                self,
                "Demographics Manager",
                f"Edited table has {len(self.result)} row(s), but the dataset has "
                f"{len(recordings)} recording(s).",
            )
            return
        for rec, row in zip(recordings, self.result):
            old = getattr(rec, "meta_data", None)
            new = OrderedDict()
            for key, value in row.items():
                if old is not None and key in old and _to_display_str(old[key]) == _to_display_str(value):
                    new[key] = old[key]
                else:
                    new[key] = value
            if hasattr(old, "clear") and hasattr(old, "update"):
                old.clear()
                old.update(new)
            else:
                rec.meta_data = new

    def accept(self) -> None:
        self.result = self._current_data()
        self._apply_to_dataset()
        QDialog.accept(self)

    def add_bids_variable(self) -> None:
        """Populate missing BIDS metadata fields on every recording."""
        # Keep the edits currently shown in the table before the BIDS defaults
        # are merged into `meta_data`, otherwise they would be discarded by the
        # reload below.
        self.result = self._current_data()
        self._apply_to_dataset()

        from pyBrainAnalyzIR.io.bids import add_missing_bids_to_metadata

        try:
            add_missing_bids_to_metadata(self.dataset)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Add Additional BIDS Variables",
                f"Failed to add BIDS metadata:\n{exc}",
            )
            return

        self._reload_from_dataset()
        QMessageBox.information(
            self, "Add Additional BIDS Variables",
            "Missing BIDS metadata fields were added to the demographics table.",
        )

    def _reload_from_dataset(self) -> None:
        rows = self._rows_from_dataset(self.dataset)
        self._columns = self._build_columns(rows)
        self.table.clear()
        self._populate_table(rows)
        self.column_units.update(self._units_from_rows(rows))
        self._refresh_unit_display()

    def reject(self) -> None:
        self.result = self._current_data()
        self._apply_to_dataset()
        QDialog.reject(self)


def _active_ipython():
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def _release_window(window: DemographicsManager) -> None:
    if window in _OPEN_WINDOWS:
        _OPEN_WINDOWS.remove(window)


def demographics_manager(dataset: Any, block: Optional[bool] = None) -> Any:
    """Launch the demographics manager for ``dataset``.

    In a notebook, the window is returned immediately and edits are applied to
    ``dataset`` when it is closed. In a script, this blocks and returns the
    edited dataset.
    """
    if not hasattr(dataset, "dataset") or not hasattr(dataset, "get_demographics"):
        raise TypeError("dataset must provide a dataset collection and get_demographics()")
    app = QApplication.instance() or QApplication(sys.argv or ["demographics_manager"])
    shell = _active_ipython()
    if block is None:
        block = shell is None
    if not block:
        app.setQuitOnLastWindowClosed(False)
    window = DemographicsManager(dataset)
    window.setAttribute(Qt.WA_DeleteOnClose, True)
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(lambda *_: _release_window(window))
    window.show()
    window.raise_()
    window.activateWindow()
    if not block and shell is not None and shell.active_eventloop != "qt":
        try:
            shell.enable_gui("qt")
        except Exception:
            block = True
    if block:
        app.exec()
        return dataset
    return window


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Edit pyBrainAnalyzIR DataSet demographics.")
    parser.add_argument("dataset", help="path to a pickled DataSet")
    args = parser.parse_args(argv)
    with open(args.dataset, "rb") as fid:
        dataset = pickle.load(fid)
    if not hasattr(dataset, "dataset"):
        parser.error(f"{args.dataset} does not contain a DataSet object")
    demographics_manager(dataset, block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
