"""PySide6 GUI to view/edit a list of dictionaries as a composite table.

Usage
-----
    from pyBrainAnalyzIR.vis.dict_table_editor import edit_dict_table

    data = [
        {"subjectID": "S01", "age": 24, "group": "control"},
        {"subjectID": "S02", "age": 31, "group": "patient"},
    ]
    result = edit_dict_table(data)  # blocks until the window is closed

`result` is a list of dictionaries (same row order as the input `data`)
reflecting any edits, renamed/added/removed columns and filled-in values.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

SUBJECT_ID_KEY = "subjectID"


def _to_display_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
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


class DictTableEditor(QDialog):
    """Composite table editor for a list of dictionaries."""

    def __init__(self, data: List[Dict[str, Any]], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Dictionary Table Editor")
        self.resize(900, 600)

        self._columns: List[str] = self._build_columns(data)
        self._n_rows = len(data)
        self.result: List[Dict[str, Any]] = []

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

        toolbar = QHBoxLayout()
        add_col_btn = QPushButton("Add Column")
        add_col_btn.clicked.connect(self.add_column)
        import_btn = QPushButton("Import from Excel...")
        import_btn.clicked.connect(self.import_excel)
        export_btn = QPushButton("Export to Excel...")
        export_btn.clicked.connect(self.export_excel)
        toolbar.addWidget(add_col_btn)
        toolbar.addWidget(import_btn)
        toolbar.addWidget(export_btn)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        self.table = QTableWidget()
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.horizontalHeader().customContextMenuRequested.connect(
            self._show_header_menu
        )
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
                self.table.setItem(row, col, item)

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
        fill_action = menu.addAction("Fill Missing Values...")
        stats_action = menu.addAction("Show Statistics")
        action = menu.exec_(self.table.horizontalHeader().mapToGlobal(pos))
        if action == rename_action:
            self._rename_column(col)
        elif action == remove_action:
            self._remove_column(col)
        elif action == fill_action:
            self._fill_missing(col)
        elif action == stats_action:
            self._show_statistics(col)

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
            coerced = _coerce_value(item.text())
            if isinstance(coerced, (int, float)):
                values.append(float(coerced))

        name = self._columns[col]
        if not values:
            QMessageBox.information(self, "Statistics", f"'{name}' has no numeric values.")
            return

        avg = sum(values) / len(values)
        vmin, vmax = min(values), max(values)
        msg = (
            f"Column: {name}\n"
            f"Count: {len(values)}\n"
            f"Average: {avg:.4g}\n"
            f"Min: {vmin:.4g}\n"
            f"Max: {vmax:.4g}\n"
            f"Range: {vmax - vmin:.4g}"
        )
        QMessageBox.information(self, "Statistics", msg)

    # ------------------------------------------------------------------
    # Excel import / export
    # ------------------------------------------------------------------
    def _current_data(self) -> List[Dict[str, Any]]:
        rows = []
        for row in range(self.table.rowCount()):
            d: Dict[str, Any] = {}
            for col, key in enumerate(self._columns):
                item = self.table.item(row, col)
                text = item.text() if item is not None else ""
                value = _coerce_value(text)
                if value is not None:
                    d[key] = value
            rows.append(d)
        return rows

    def import_excel(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import from Excel", "", "Excel Files (*.xlsx *.xls)"
        )
        if not path:
            return
        try:
            excel_df = pd.read_excel(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Import from Excel", f"Failed to read file:\n{exc}")
            return

        shared_columns = [c for c in self._columns if c in excel_df.columns]
        if not shared_columns:
            QMessageBox.warning(
                self, "Import from Excel",
                "No matching column names found between the table and the Excel file.",
            )
            return

        dialog = MatchColumnsDialog(shared_columns, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        match_cols = dialog.selected_columns()
        if not match_cols:
            QMessageBox.warning(self, "Import from Excel", "No matching column(s) selected.")
            return

        self._merge_excel(excel_df, match_cols)

    def _merge_excel(self, excel_df: pd.DataFrame, match_cols: List[str]) -> None:
        # Build a lookup: match-key tuple -> excel row (as dict)
        excel_lookup: Dict[tuple, Dict[str, Any]] = {}
        for _, excel_row in excel_df.iterrows():
            key = tuple(_to_display_str(excel_row.get(c)) for c in match_cols)
            excel_lookup[key] = excel_row.to_dict()

        # New columns coming from the Excel file (excluding match columns).
        new_cols = [c for c in excel_df.columns if c not in self._columns]
        for name in new_cols:
            col = self.table.columnCount()
            self.table.insertColumn(col)
            self._columns.append(name)
            self.table.setHorizontalHeaderItem(col, QTableWidgetItem(name))
            for row in range(self.table.rowCount()):
                self.table.setItem(row, col, QTableWidgetItem(""))

        match_col_idx = [self._columns.index(c) for c in match_cols]
        n_matched = 0
        for row in range(self.table.rowCount()):
            key = tuple(
                self.table.item(row, idx).text() if self.table.item(row, idx) else ""
                for idx in match_col_idx
            )
            excel_row = excel_lookup.get(key)
            if excel_row is None:
                continue
            n_matched += 1
            for name, value in excel_row.items():
                if name in match_cols:
                    continue
                col = self._columns.index(name)
                self.table.setItem(row, col, QTableWidgetItem(_to_display_str(value)))

        QMessageBox.information(
            self, "Import from Excel", f"Matched and updated {n_matched} row(s)."
        )

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


def edit_dict_table(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Show the table editor for `data` and return the edited list of dicts.

    Blocks until the user closes the window. The returned list preserves the
    row order of the input `data`, reflecting any edits, renamed/added/removed
    columns, and filled-in values.
    """
    owns_app = QApplication.instance() is None
    app = QApplication.instance() or QApplication(sys.argv)

    editor = DictTableEditor(data)
    editor.exec_()
    result = editor.result

    if owns_app:
        app.quit()

    return result


if __name__ == "__main__":
    sample_data = [
        {"subjectID": "S01", "age": 24, "group": "control"},
        {"subjectID": "S02", "age": 31, "group": "patient"},
        {"subjectID": "S03", "group": "control"},
    ]
    out = edit_dict_table(sample_data)
    print(out)
