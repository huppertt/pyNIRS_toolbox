"""Interactive manual region-of-interest definition for fNIRS recordings."""
from __future__ import annotations

import re
import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import to_rgb
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from pyBrainAnalyzIR.vis.plot_nirs_inline import draw_probe


class _WeightDelegate(QStyledItemDelegate):
    def createEditor(self, parent, _option, _index):
        editor = QDoubleSpinBox(parent)
        editor.setDecimals(6)
        editor.setRange(-1e12, 1e12)
        editor.setSingleStep(0.1)
        return editor

    def setEditorData(self, editor, index):
        editor.setValue(float(index.data(Qt.EditRole) or 0))

    def setModelData(self, editor, model, index):
        editor.interpretText()
        model.setData(index, editor.value(), Qt.EditRole)


class ROIDefinitionDialog(QDialog):
    """Collect named, weighted source-detector membership tables."""

    def __init__(self, data: Any, roi_tables: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Regions-of-Interest")
        self.resize(1050, 690)
        self.recording, self.probe_data = _recording_and_probe_data(data)
        self.is_3d_probe = _has_3d_probe(self.recording)
        self.rois: Dict[str, pd.DataFrame] = {
            str(name): _normalise_roi_table(table, str(name))
            for name, table in (roi_tables or {}).items()
        }
        self._editing_name: Optional[str] = None
        self._lines = []
        self._channel_pairs = [
            (str(self.probe_data.source.values[i]), str(self.probe_data.detector.values[i]))
            for i in range(self.probe_data.sizes["channel"])
        ]
        self._next_roi_number = 1 + max(
            (int(match.group(1)) for name in self.rois
             if (match := re.fullmatch(r"ROI (\d+)", name))),
            default=0,
        )

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(QLabel("Probe channels (right-click a channel to add or remove it)"))
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        left.addWidget(self.canvas, stretch=1)
        root.addLayout(left, stretch=1)

        right = QVBoxLayout()
        right.addWidget(QLabel("ROI construction"))
        if self.is_3d_probe:
            self.definition_mode = QComboBox(self)
            self.definition_mode.setObjectName("roiDefinitionMode")
            self.definition_mode.addItem("Manual channel selection")
            self.definition_mode.addItem("Anatomical location (coming soon)")
            mode_model = self.definition_mode.model()
            if isinstance(mode_model, QStandardItemModel):
                mode_model.item(1).setEnabled(False)
            right.addWidget(self.definition_mode)
        else:
            self.definition_mode = None

        right.addWidget(QLabel("Current ROI channels"))
        self.channel_table = QTableWidget(0, 3, self)
        self.channel_table.setHorizontalHeaderLabels(["Source", "Detector", "Weight"])
        self.channel_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.channel_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.channel_table.setItemDelegateForColumn(2, _WeightDelegate(self.channel_table))
        self.channel_table.horizontalHeader().setStretchLastSection(True)
        self.channel_table.customContextMenuRequested.connect(self._table_context_menu)
        self.channel_table.itemChanged.connect(self._redraw_probe)
        right.addWidget(self.channel_table, stretch=2)

        edit_buttons = QHBoxLayout()
        self.save_roi_button = QPushButton("Add ROI to list", self)
        self.cancel_edits_button = QPushButton("Cancel Edits", self)
        self.cancel_edits_button.hide()
        edit_buttons.addWidget(self.save_roi_button)
        edit_buttons.addWidget(self.cancel_edits_button)
        right.addLayout(edit_buttons)
        self.save_roi_button.clicked.connect(self._save_current_roi)
        self.cancel_edits_button.clicked.connect(self._cancel_edits)

        right.addWidget(QLabel("Defined ROIs"))
        self.roi_list = QListWidget(self)
        self.roi_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roi_list.customContextMenuRequested.connect(self._roi_context_menu)
        right.addWidget(self.roi_list, stretch=1)
        self._populate_roi_list()

        dialog_buttons = QHBoxLayout()
        dialog_buttons.addStretch()
        self.save_button = QPushButton("Save ROIs", self)
        self.cancel_button = QPushButton("Cancel", self)
        dialog_buttons.addWidget(self.save_button)
        dialog_buttons.addWidget(self.cancel_button)
        right.addLayout(dialog_buttons)
        self.save_button.clicked.connect(self._save_and_close)
        self.cancel_button.clicked.connect(self.reject)
        root.addLayout(right, stretch=1)
        self._draw_probe()

    def _populate_roi_list(self, select_name: Optional[str] = None) -> None:
        self.roi_list.clear()
        for name in self.rois:
            self.roi_list.addItem(name)
        if select_name is not None:
            matches = self.roi_list.findItems(select_name, Qt.MatchExactly)
            if matches:
                self.roi_list.setCurrentItem(matches[0])

    def _current_table(self) -> pd.DataFrame:
        rows = []
        for row in range(self.channel_table.rowCount()):
            source = self.channel_table.item(row, 0)
            detector = self.channel_table.item(row, 1)
            weight = self.channel_table.item(row, 2)
            if source is None or detector is None or weight is None:
                continue
            rows.append({
                "source": source.text(),
                "detector": detector.text(),
                "weight": float(weight.text()),
            })
        return pd.DataFrame(rows, columns=["source", "detector", "weight"])

    def _set_current_table(self, table: Optional[pd.DataFrame] = None) -> None:
        table = table if table is not None else pd.DataFrame(
            columns=["source", "detector", "weight"]
        )
        self.channel_table.blockSignals(True)
        self.channel_table.setRowCount(len(table))
        for row, values in enumerate(table.itertuples(index=False)):
            source, detector, weight = values[:3]
            for column, value in enumerate((source, detector, weight)):
                item = QTableWidgetItem(str(value))
                if column < 2:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.channel_table.setItem(row, column, item)
        self.channel_table.blockSignals(False)
        self._redraw_probe()

    def _draw_probe(self) -> None:
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self._lines = draw_probe(
            self.recording, self.probe_data, self.axes,
            plot_on_scalp=self.is_3d_probe,
        )
        self.figure.tight_layout()
        self._redraw_probe()
        self.canvas.mpl_connect("button_press_event", self._probe_context_menu)
        self.canvas.draw_idle()

    def _redraw_probe(self, *_args) -> None:
        if not self._lines:
            self.canvas.draw_idle()
            return
        table = self._current_table()
        weight_by_pair = {
            (row.source, row.detector): row.weight
            for row in table.itertuples(index=False)
        }
        max_weight = max((abs(value) for value in weight_by_pair.values()), default=1.0) or 1.0
        gray = np.asarray(to_rgb("#9a9a9a"))
        for line, pair in zip(self._lines, self._channel_pairs):
            weight = weight_by_pair.get(pair, 0.0)
            if weight == 0:
                line.set_color(gray)
                line.set_linewidth(1.6)
                continue
            color = np.asarray(to_rgb("#d62728" if weight > 0 else "#2166ac"))
            strength = min(abs(weight) / max_weight, 1.0)
            line.set_color(gray * (1 - strength) + color * strength)
            line.set_linewidth(1.6 + 2.2 * strength)
        self.canvas.draw_idle()

    def _nearest_channel(self, event) -> Optional[int]:
        if event.x is None or event.y is None or event.inaxes is not self.axes:
            return None
        distances = []
        for line in self._lines:
            points = self.axes.transData.transform(
                np.column_stack(line.get_data())
            )
            start, end = points[0], points[-1]
            segment = end - start
            length_squared = float(segment @ segment)
            fraction = (float((np.array([event.x, event.y]) - start) @ segment)
                        / length_squared if length_squared else 0.0)
            closest = start + np.clip(fraction, 0.0, 1.0) * segment
            distances.append(float(np.linalg.norm(np.array([event.x, event.y]) - closest)))
        index = int(np.argmin(distances)) if distances else -1
        return index if index >= 0 and distances[index] <= 12 else None

    def _probe_context_menu(self, event) -> None:
        if event.button != 3:
            return
        index = self._nearest_channel(event)
        if index is None:
            return
        pair = self._channel_pairs[index]
        included = pair in set(zip(
            self._current_table().get("source", []),
            self._current_table().get("detector", []),
        ))
        menu = QMenu(self)
        action = menu.addAction(
            "Remove channel from current ROI" if included else "Add channel to current ROI"
        )
        if menu.exec(QCursor.pos()) is not action:
            return
        if included:
            self._remove_channel(pair)
        else:
            self._add_channel(pair)

    def _add_channel(self, pair) -> None:
        table = self._current_table()
        if pair in set(zip(table.get("source", []), table.get("detector", []))):
            return
        table.loc[len(table)] = [pair[0], pair[1], 1.0]
        self._set_current_table(table)

    def _remove_channel(self, pair) -> None:
        table = self._current_table()
        if table.empty:
            return
        keep = ~((table.source == pair[0]) & (table.detector == pair[1]))
        self._set_current_table(table.loc[keep].reset_index(drop=True))

    def _table_context_menu(self, pos) -> None:
        row = self.channel_table.rowAt(pos.y())
        if row < 0:
            return
        self.channel_table.selectRow(row)
        menu = QMenu(self)
        remove_action = menu.addAction("Remove channel from ROI")
        if menu.exec(self.channel_table.viewport().mapToGlobal(pos)) is remove_action:
            self._remove_channel((
                self.channel_table.item(row, 0).text(),
                self.channel_table.item(row, 1).text(),
            ))

    def _save_current_roi(self) -> None:
        table = self._current_table()
        if table.empty:
            QMessageBox.information(self, "Define ROI", "Add at least one channel first.")
            return
        if not np.isfinite(table["weight"].to_numpy(dtype=float)).all():
            QMessageBox.warning(self, "Define ROI", "ROI weights must be finite numbers.")
            return
        if not np.any(table["weight"].to_numpy(dtype=float) != 0):
            QMessageBox.warning(self, "Define ROI", "At least one ROI weight must be nonzero.")
            return
        name = self._editing_name or f"ROI {self._next_roi_number}"
        self.rois[name] = table.assign(name=name)
        if self._editing_name is None:
            self._next_roi_number += 1
        self._editing_name = None
        self._populate_roi_list(name)
        self._set_current_table()
        self._set_editing_state(False)

    def _set_editing_state(self, editing: bool) -> None:
        self.save_roi_button.setText("Save Changes to ROI" if editing else "Add ROI to list")
        self.cancel_edits_button.setVisible(editing)

    def _edit_roi(self, name: str) -> None:
        self._editing_name = name
        self._set_current_table(self.rois[name])
        self._set_editing_state(True)

    def _cancel_edits(self) -> None:
        self._editing_name = None
        self._set_current_table()
        self._set_editing_state(False)

    def _roi_context_menu(self, pos) -> None:
        item = self.roi_list.itemAt(pos)
        if item is None:
            return
        name = item.text()
        menu = QMenu(self)
        rename_action = menu.addAction("Rename ROI")
        remove_action = menu.addAction("Remove ROI")
        edit_action = menu.addAction("Edit ROI")
        action = menu.exec(self.roi_list.viewport().mapToGlobal(pos))
        if action is rename_action:
            new_name, accepted = QInputDialog.getText(
                self, "Rename ROI", "ROI name:", text=name
            )
            if accepted:
                self._rename_roi(name, new_name)
        elif action is remove_action:
            self._remove_roi(name)
        elif action is edit_action:
            self._edit_roi(name)

    def _rename_roi(self, old_name: str, new_name: str) -> bool:
        new_name = new_name.strip()
        if not new_name:
            QMessageBox.warning(self, "Rename ROI", "ROI name cannot be blank.")
            return False
        if new_name != old_name and new_name in self.rois:
            QMessageBox.warning(self, "Rename ROI", f"ROI '{new_name}' already exists.")
            return False
        self.rois[new_name] = self.rois.pop(old_name).assign(name=new_name)
        if self._editing_name == old_name:
            self._editing_name = new_name
        self._populate_roi_list(new_name)
        return True

    def _remove_roi(self, name: str) -> None:
        self.rois.pop(name, None)
        if self._editing_name == name:
            self._cancel_edits()
        self._populate_roi_list()

    def _save_and_close(self) -> None:
        if self._editing_name is not None:
            QMessageBox.information(
                self, "Save ROIs", "Save or cancel the current ROI edits first."
            )
            return
        self.accept()


def _recording_and_probe_data(data):
    recordings = list(data.dataset) if hasattr(data, "dataset") else [data]
    if not recordings:
        raise ValueError("ROI definition requires at least one recording.")
    recording = recordings[0]
    series = getattr(recording, "timeseries", {})
    if "amp" in series:
        probe = series["amp"]
    else:
        probe = next(
            (item for item in series.values()
             if isinstance(item, xr.DataArray) and "channel" in item.dims
             and {"source", "detector"}.issubset(item.coords)),
            None,
        )
    if probe is None:
        raise ValueError("ROI definition requires channel data with source and detector coordinates.")
    return recording, xr.DataArray(
        np.zeros(probe.sizes["channel"]),
        dims="channel",
        coords={
            "channel": probe.channel.values,
            "source": ("channel", probe.source.values.astype(str)),
            "detector": ("channel", probe.detector.values.astype(str)),
        },
    )


def _has_3d_probe(recording) -> bool:
    geometry = getattr(recording, "geo3d", None)
    if geometry is None or getattr(geometry, "ndim", 0) < 2:
        return False
    try:
        values = np.asarray(geometry.pint.dequantify().values, dtype=float)
    except (AttributeError, TypeError, ValueError):
        values = np.asarray(geometry.values, dtype=float)
    extents = np.ptp(values, axis=0)
    return values.shape[-1] >= 3 and bool(extents[-1] > 1e-9)


def _normalise_roi_table(table, name):
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"ROI {name!r} must be a pandas DataFrame.")
    result = table.copy(deep=True)
    result = result.rename(columns={
        "Source": "source", "Detector": "detector", "Weight": "weight", "Name": "name",
    })
    required = {"source", "detector", "weight"}
    if not required.issubset(result.columns):
        raise ValueError(f"ROI {name!r} must contain source, detector, and weight columns.")
    if result.empty or result[["source", "detector"]].isna().any().any():
        raise ValueError(f"ROI {name!r} must contain source-detector pairs.")
    result["source"] = result["source"].astype(str)
    result["detector"] = result["detector"].astype(str)
    result["weight"] = pd.to_numeric(result["weight"], errors="raise")
    if not np.isfinite(result["weight"].to_numpy(dtype=float)).all():
        raise ValueError(f"ROI {name!r} weights must be finite.")
    if not np.any(result["weight"].to_numpy(dtype=float) != 0):
        raise ValueError(f"ROI {name!r} needs at least one nonzero weight.")
    if result.duplicated(["source", "detector"]).any():
        raise ValueError(f"ROI {name!r} contains duplicate source-detector pairs.")
    result["name"] = name
    return result[["source", "detector", "weight", "name"]]


def roi_manager(data: Any, roi_tables: Optional[dict] = None, parent=None) -> Optional[dict]:
    """Open the ROI editor and return ``{name: membership_dataframe}`` on Save.

    ``data`` may be a single Cedalion recording or a ``DataSet``. In notebooks
    this call opens the dialog modally; Cancel returns ``None``.
    """
    if data is None:
        raise TypeError("data must be a recording or DataSet.")
    app = QApplication.instance() or QApplication(sys.argv or ["roi_manager"])
    dialog = ROIDefinitionDialog(data, roi_tables=roi_tables, parent=parent)
    if dialog.exec() != QDialog.Accepted:
        return None
    return {name: table.copy(deep=True) for name, table in dialog.rois.items()}
