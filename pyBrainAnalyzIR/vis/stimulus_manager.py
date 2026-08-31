"""PySide6 GUI to browse and edit the stimulus events stored in a `DataSet`.

Usage
-----
    from pyBrainAnalyzIR.vis.stimulus_manager import stimulus_manager

    stimulus_manager(dataset)

In a script this blocks until the window is closed.  Inside a Jupyter/IPython
notebook the Qt event loop is integrated with the kernel instead, so the call
returns immediately and the notebook stays responsive.

From the command line, with a pickled `DataSet`::

    python -m pyBrainAnalyzIR.vis.stimulus_manager mydataset.pkl
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.patches as patches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import (
    QAbstractTableModel,
    QEventLoop,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    Qt,
    Signal,
)
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStatusBar,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pyBrainAnalyzIR.vis.dict_table_editor import _to_display_str
from pyBrainAnalyzIR.vis.NIRSviewIR import _build_labels, _pick_values_dialog

# (display header, column name in the table, is-numeric)
FILE_COL = "__file__"
REC_COL = "__recording__"
STIM_COLUMNS = (
    ("File", FILE_COL, False),
    ("Name", "trial_type", False),
    ("Onset", "onset", True),
    ("Duration", "duration", True),
    ("Amplitude", "value", True),
)
COL_FILE, COL_NAME, COL_ONSET, COL_DURATION, COL_AMPLITUDE = range(5)

IMAGE_TIME_STEP = 0.5   # seconds per column of the stimulus design image


class StimTableModel(QAbstractTableModel):
    """Editable table model over the stimulus events of one or more recordings.

    Rows may be filtered by trial type; `_rows` maps view rows onto positions in
    the underlying (unfiltered) DataFrame.  `REC_COL` records which recording
    each event belongs to so edits can be written back to the right file.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._df = pd.DataFrame(columns=[c[1] for c in STIM_COLUMNS] + [REC_COL])
        self._recs: List[Any] = []
        self._labels: List[str] = []
        self._rows: List[int] = []
        self._type_filter: Optional[set] = None   # None = show all types
        self._highlight_outliers = False
        self._duration_outliers: set = set()
        self._gap_outliers: set = set()

    # -------------------------------------------------------------- data load

    def set_recordings(self, recs: List[Any], labels: List[str]) -> None:
        self.beginResetModel()
        self._recs = list(recs)
        self._labels = list(labels)

        frames = []
        for i, rec in enumerate(self._recs):
            stim = getattr(rec, "stim", None)
            df = stim.copy() if isinstance(stim, pd.DataFrame) else pd.DataFrame()
            for _, col, numeric in STIM_COLUMNS:
                if col in (FILE_COL,):
                    continue
                if col not in df.columns:
                    df[col] = pd.Series([float("nan")] * len(df) if numeric else [""] * len(df))
            df[FILE_COL] = self._labels[i] if i < len(self._labels) else str(i)
            df[REC_COL] = i
            frames.append(df)

        if frames:
            self._df = pd.concat(frames, ignore_index=True)
        else:
            self._df = pd.DataFrame(columns=[c[1] for c in STIM_COLUMNS] + [REC_COL])
        self._type_filter = None
        self._rebuild_rows()
        self.endResetModel()
        self.refresh_outliers()

    def set_recording(self, rec: Any, label: str = "") -> None:
        self.set_recordings([] if rec is None else [rec], [] if rec is None else [label])

    # ------------------------------------------------------------- filtering

    def trial_types(self) -> List[str]:
        if "trial_type" not in self._df.columns:
            return []
        seen: List[str] = []
        for name in self._df["trial_type"].tolist():
            name = str(name)
            if name not in seen:
                seen.append(name)
        return seen

    @property
    def type_filter(self) -> Optional[set]:
        return self._type_filter

    def set_type_filter(self, types: Optional[set]) -> None:
        self.beginResetModel()
        self._type_filter = None if types is None else set(types)
        self._rebuild_rows()
        self.endResetModel()

    def _rebuild_rows(self) -> None:
        if self._type_filter is None:
            self._rows = list(range(len(self._df)))
        else:
            self._rows = [
                i for i, name in enumerate(self._df.get("trial_type", []))
                if str(name) in self._type_filter
            ]

    def dataframe(self) -> pd.DataFrame:
        """The full (unfiltered) stimulus table."""
        return self._df

    def rows_dataframe(self, view_rows: Optional[List[int]] = None) -> pd.DataFrame:
        """The visible rows (or a subset of them) as a DataFrame."""
        rows = self._rows if view_rows is None else self.source_rows(sorted(view_rows))
        return self._df.iloc[rows] if rows else self._df.iloc[0:0]

    def view_row_for_source(self, source_pos: int) -> Optional[int]:
        """View row currently showing the given DataFrame position, if visible."""
        try:
            return self._rows.index(int(source_pos))
        except ValueError:
            return None

    def source_rows(self, view_rows: List[int]) -> List[int]:
        return [self._rows[r] for r in view_rows if 0 <= r < len(self._rows)]

    # ----------------------------------------------------- QAbstractTableModel

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(STIM_COLUMNS)

    def headerData(self, section: int, orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return STIM_COLUMNS[section][0]
        return section + 1

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.NoItemFlags
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == COL_FILE:
            return base
        return base | Qt.ItemIsEditable

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        _, col, numeric = STIM_COLUMNS[index.column()]
        value = self._df.iloc[self._rows[index.row()]][col]
        if role in (Qt.DisplayRole, Qt.EditRole):
            if numeric:
                try:
                    return f"{float(value):g}"
                except (TypeError, ValueError):
                    return ""
            return _to_display_str(value)
        if role == Qt.TextAlignmentRole and numeric:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.ForegroundRole and self._highlight_outliers:
            position = self._rows[index.row()]
            bad_duration = position in self._duration_outliers
            bad_gap = position in self._gap_outliers
            if bad_duration and bad_gap:
                return QBrush(QColor("purple"))
            if bad_duration:
                return QBrush(QColor("red"))
            if bad_gap:
                return QBrush(QColor("blue"))
        return None

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:
        if not index.isValid() or role != Qt.EditRole or index.column() == COL_FILE:
            return False
        _, col, numeric = STIM_COLUMNS[index.column()]
        if numeric:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return False
        else:
            value = str(value)
        self._df.at[self._df.index[self._rows[index.row()]], col] = value
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        self._write_back()
        if col == "trial_type" and self._type_filter is not None:
            self.set_type_filter(self._type_filter)
        return True

    def sort(self, column: int, order=Qt.AscendingOrder) -> None:
        if len(self._df) == 0:
            return
        _, col, _ = STIM_COLUMNS[column]

        self.layoutAboutToBeChanged.emit()
        old_persistent = self.persistentIndexList()
        old_rows = list(self._rows)

        sorted_df = self._df.sort_values(
            by=col, ascending=(order == Qt.AscendingOrder), kind="mergesort"
        )
        # sorted_df.index holds the previous positions, in their new order
        new_position = {old: new for new, old in enumerate(sorted_df.index)}
        self._df = sorted_df.reset_index(drop=True)
        self._rebuild_rows()

        # Selections follow the events themselves, not the row numbers.
        view_of_source = {src: view for view, src in enumerate(self._rows)}
        new_persistent = []
        for idx in old_persistent:
            view = view_of_source.get(new_position.get(old_rows[idx.row()])) \
                if idx.row() < len(old_rows) else None
            new_persistent.append(
                self.index(view, idx.column()) if view is not None else QModelIndex()
            )
        self.changePersistentIndexList(old_persistent, new_persistent)

        self.layoutChanged.emit()
        self._write_back()

    # ------------------------------------------------------------- row editing

    def add_row(self) -> None:
        """Append one blank event to every recording being edited."""
        if not self._recs:
            return
        self.beginResetModel()
        for i in range(len(self._recs)):
            new = {col: (0.0 if numeric else "new_event") for _, col, numeric in STIM_COLUMNS}
            new[FILE_COL] = self._labels[i] if i < len(self._labels) else str(i)
            new[REC_COL] = i
            for col in self._df.columns:
                new.setdefault(col, None)
            self._df.loc[len(self._df)] = new
        self._df = self._df.reset_index(drop=True)
        if self._type_filter is not None:
            self._type_filter.add("new_event")
        self._rebuild_rows()
        self.endResetModel()
        self._write_back()

    def remove_rows(self, rows: List[int]) -> None:
        """Remove the given *view* rows."""
        if not rows or not self._recs:
            return
        source = self.source_rows(sorted(rows))
        self.beginResetModel()
        self._df = self._df.drop(index=[self._df.index[r] for r in source]).reset_index(drop=True)
        self._rebuild_rows()
        self.endResetModel()
        self._write_back()

    def refresh(self) -> None:
        """Re-read the recordings after their stimulus tables changed elsewhere."""
        keep = self._type_filter
        self.set_recordings(self._recs, self._labels)
        if keep is not None:
            self.set_type_filter(keep & set(self.trial_types()))

    # ----------------------------------------------------------- type editing

    def _type_mask(self, name: str):
        return self._df["trial_type"].astype(str) == str(name)

    def rename_type(self, old: str, new: str) -> int:
        """Rename every event of type `old` to `new`; returns the row count changed."""
        mask = self._type_mask(old)
        n = int(mask.sum())
        if n == 0:
            return 0
        self.beginResetModel()
        self._df.loc[mask, "trial_type"] = str(new)
        if self._type_filter is not None and old in self._type_filter:
            self._type_filter.discard(old)
            self._type_filter.add(str(new))
        self._rebuild_rows()
        self.endResetModel()
        self._write_back()
        return n

    def remove_type(self, name: str) -> int:
        mask = self._type_mask(name)
        n = int(mask.sum())
        if n == 0:
            return 0
        self.beginResetModel()
        self._df = self._df.loc[~mask].reset_index(drop=True)
        if self._type_filter is not None:
            self._type_filter.discard(str(name))
        self._rebuild_rows()
        self.endResetModel()
        self._write_back()
        return n

    def duplicate_type(self, source: str, new: str) -> int:
        mask = self._type_mask(source)
        n = int(mask.sum())
        if n == 0:
            return 0
        copies = self._df.loc[mask].copy()
        copies["trial_type"] = str(new)
        self.beginResetModel()
        self._df = pd.concat([self._df, copies], ignore_index=True)
        if self._type_filter is not None:
            self._type_filter.add(str(new))
        self._rebuild_rows()
        self.endResetModel()
        self._write_back()
        return n

    def add_type(self, name: str) -> None:
        """Append an empty event carrying the new type name to every edited recording."""
        if not self._recs:
            return
        self.beginResetModel()
        for i in range(len(self._recs)):
            new = {col: (float("nan") if numeric else "") for _, col, numeric in STIM_COLUMNS}
            new["trial_type"] = str(name)
            new[FILE_COL] = self._labels[i] if i < len(self._labels) else str(i)
            new[REC_COL] = i
            for col in self._df.columns:
                new.setdefault(col, None)
            self._df.loc[len(self._df)] = new
        self._df = self._df.reset_index(drop=True)
        if self._type_filter is not None:
            self._type_filter.add(str(name))
        self._rebuild_rows()
        self.endResetModel()
        self._write_back()

    # ------------------------------------------------------- outlier checking

    @property
    def highlight_outliers(self) -> bool:
        return self._highlight_outliers

    def set_highlight_outliers(self, enabled: bool) -> None:
        self._highlight_outliers = bool(enabled)
        self.refresh_outliers()

    def outlier_flags(self, position: int) -> tuple:
        """Return (bad_duration, bad_gap) for a position in the unfiltered DataFrame."""
        if not self._highlight_outliers:
            return False, False
        return position in self._duration_outliers, position in self._gap_outliers

    def refresh_outliers(self) -> None:
        """Flag durations >1.5 SD from their type mean and gaps >3 SD from the mean gap."""
        duration_outliers: set = set()
        gap_outliers: set = set()

        if self._highlight_outliers and len(self._df):
            onset = pd.to_numeric(self._df["onset"], errors="coerce")
            duration = pd.to_numeric(self._df["duration"], errors="coerce")

            for _, group in self._df.groupby(self._df["trial_type"].astype(str), sort=False):
                values = duration.loc[group.index]
                std = values.std(ddof=1)
                if not np.isfinite(std) or std == 0:
                    continue
                z = (values - values.mean()) / std
                duration_outliers.update(int(i) for i in z.index[z.abs() > 1.5])

            # Gaps between consecutive events of any type, measured within a file.
            gaps, pairs = [], []
            for _, group in self._df.groupby(REC_COL, sort=False):
                order = list(onset.loc[group.index].sort_values(kind="mergesort").index)
                for first, second in zip(order, order[1:]):
                    gaps.append(float(onset.at[second] - (onset.at[first] + duration.at[first])))
                    pairs.append((int(first), int(second)))
            if len(gaps) > 1:
                gaps = np.asarray(gaps, dtype=float)
                std = np.nanstd(gaps, ddof=1)
                if np.isfinite(std) and std > 0:
                    z = (gaps - np.nanmean(gaps)) / std
                    for k in np.flatnonzero(np.abs(z) > 3.0):
                        gap_outliers.update(pairs[int(k)])

        changed = (duration_outliers != self._duration_outliers
                   or gap_outliers != self._gap_outliers)
        self._duration_outliers = duration_outliers
        self._gap_outliers = gap_outliers
        if changed and self.rowCount():
            top = self.index(0, 0)
            bottom = self.index(self.rowCount() - 1, self.columnCount() - 1)
            self.dataChanged.emit(top, bottom, [Qt.ForegroundRole])

    # ------------------------------------------------------------ import/export

    def export_frames(self) -> "OrderedDict[str, pd.DataFrame]":
        """One event table per edited recording, keyed by its file label."""
        frames = OrderedDict()
        lead = ["trial_type", "onset", "duration", "value"]
        for i, label in enumerate(self._labels):
            subset = self._df.loc[self._df[REC_COL] == i]
            subset = subset.drop(columns=[FILE_COL, REC_COL]).reset_index(drop=True)
            extra = [c for c in subset.columns if c not in lead]
            frames[label] = subset[[c for c in lead if c in subset.columns] + extra]
        return frames

    def import_frames(self, frames: Dict[int, pd.DataFrame], mode: str) -> int:
        """Append or replace events for the given recording indices."""
        if not self._recs:
            return 0
        self.beginResetModel()
        kept = self._df if mode == "append" else \
            self._df.loc[~self._df[REC_COL].isin(frames.keys())]
        added = []
        for i, frame in frames.items():
            block = frame.copy()
            block[FILE_COL] = self._labels[i] if i < len(self._labels) else str(i)
            block[REC_COL] = i
            added.append(block)
        self._df = pd.concat([kept] + added, ignore_index=True) if added else kept.reset_index(drop=True)
        self._type_filter = None
        self._rebuild_rows()
        self.endResetModel()
        self._write_back()
        return sum(len(f) for f in frames.values())

    # ------------------------------------------------------------ timing edits

    def apply_onset_to_offset(self, start: str, end: str, remove_offsets: bool) -> tuple:
        """Stretch each `start` event up to the next `end` event of the same file."""
        if len(self._df) == 0:
            return 0, 0
        onset = pd.to_numeric(self._df["onset"], errors="coerce")
        types = self._df["trial_type"].astype(str)
        updated, offsets_used = 0, []

        for _, group in self._df.groupby(REC_COL, sort=False):
            idx = group.index
            order = onset.loc[idx].sort_values(kind="mergesort").index
            if start == end:
                paired = [i for i in order if types.at[i] == start]
                for first, second in zip(paired[0::2], paired[1::2]):
                    self._df.at[first, "duration"] = float(onset.at[second] - onset.at[first])
                    offsets_used.append(second)
                    updated += 1
            else:
                starts = [i for i in order if types.at[i] == start]
                ends = [i for i in order if types.at[i] == end]
                for first in starts:
                    later = [i for i in ends if onset.at[i] > onset.at[first]]
                    if not later:
                        continue
                    second = later[0]
                    self._df.at[first, "duration"] = float(onset.at[second] - onset.at[first])
                    offsets_used.append(second)
                    updated += 1

        removed = 0
        if remove_offsets and offsets_used:
            removed = len(set(offsets_used))
            self.beginResetModel()
            self._df = self._df.drop(index=list(set(offsets_used))).reset_index(drop=True)
            self._rebuild_rows()
            self.endResetModel()
        elif updated:
            self.beginResetModel()
            self._rebuild_rows()
            self.endResetModel()

        self._write_back()
        return updated, removed

    def alternating_rows(self, view_rows: List[int], which: str) -> List[int]:
        """DataFrame positions of the odd- or even-numbered events per file and type."""
        source = self.source_rows(sorted(view_rows))
        if not source:
            return []
        subset = self._df.iloc[source]
        onset = pd.to_numeric(subset["onset"], errors="coerce")
        offset = 0 if which == "odd" else 1
        targets: List[int] = []
        for _, group in subset.groupby([REC_COL, subset["trial_type"].astype(str)], sort=False):
            order = onset.loc[group.index].sort_values(kind="mergesort").index
            targets.extend(list(order)[offset::2])
        return sorted(targets)

    def remove_source_rows(self, positions: List[int]) -> int:
        if not positions:
            return 0
        self.beginResetModel()
        self._df = self._df.drop(index=list(positions)).reset_index(drop=True)
        self._rebuild_rows()
        self.endResetModel()
        self._write_back()
        return len(positions)

    # --------------------------------------------------------- bulk numeric edits

    def apply_numeric(self, view_rows: List[int], column: int, value: float,
                      relative: bool = False) -> int:
        """Set (or offset, when `relative`) a numeric column for the given view rows."""
        source = self.source_rows(sorted(view_rows))
        if not source:
            return 0
        _, col, numeric = STIM_COLUMNS[column]
        if not numeric:
            return 0
        labels = [self._df.index[r] for r in source]
        if relative:
            self._df.loc[labels, col] = pd.to_numeric(
                self._df.loc[labels, col], errors="coerce") + float(value)
        else:
            self._df.loc[labels, col] = float(value)
        top = self.index(min(view_rows), column)
        bottom = self.index(max(view_rows), column)
        self.dataChanged.emit(top, bottom, [Qt.DisplayRole, Qt.EditRole])
        self._write_back()
        return len(source)

    def rows_of_type(self, name: str) -> List[int]:
        """View rows whose trial type matches `name`."""
        return [
            view for view, src in enumerate(self._rows)
            if str(self._df.iloc[src]["trial_type"]) == str(name)
        ]

    def type_at_row(self, view_row: int) -> Optional[str]:
        if 0 <= view_row < len(self._rows):
            return str(self._df.iloc[self._rows[view_row]]["trial_type"])
        return None

    # --------------------------------------------------------------- write out

    def _write_back(self) -> None:
        if not self._recs or REC_COL not in self._df.columns:
            return
        for i, rec in enumerate(self._recs):
            subset = self._df.loc[self._df[REC_COL] == i]
            rec.stim = subset.drop(columns=[FILE_COL, REC_COL]).reset_index(drop=True)
        self.refresh_outliers()

    def write_back(self) -> None:
        """Push the edited events back onto the recordings."""
        self._write_back()


def _sheet_name(label: str, used: set) -> str:
    """Excel-safe worksheet name (<=31 chars), disambiguated when already used."""
    name = re.sub(r"[\[\]:*?/\\]", "_", str(label)).strip() or "sheet"
    name = name[:31]
    base = name
    suffix = 2
    while name in used:
        name = f"{base[:28]}_{suffix}"
        suffix += 1
    return name


def _normalise_import(frame: pd.DataFrame) -> pd.DataFrame:
    """Map imported headers onto the stimulus columns; extra columns are kept."""
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    aliases = {
        "trial_type": ("type", "trial_type", "name"),
        "onset": ("onset",),
        "duration": ("duration",),
        "value": ("amplitude", "value"),
    }
    found = {}
    for target, names in aliases.items():
        for name in names:
            if name in lookup:
                found[target] = lookup[name]
                break

    missing = [t for t in ("trial_type", "onset", "duration") if t not in found]
    if missing:
        raise ValueError(
            "The file must contain 'type', 'onset' and 'duration' columns; missing: "
            + ", ".join("type" if m == "trial_type" else m for m in missing)
        )

    out = pd.DataFrame()
    out["onset"] = pd.to_numeric(frame[found["onset"]], errors="coerce")
    out["trial_type"] = frame[found["trial_type"]].astype(str)
    out["value"] = (pd.to_numeric(frame[found["value"]], errors="coerce")
                    if "value" in found else 1.0)
    out["duration"] = pd.to_numeric(frame[found["duration"]], errors="coerce")
    for column in frame.columns:
        if column not in found.values():
            out[str(column)] = frame[column].values
    return out


def _unique_type_name(base: str, existing) -> str:
    """Return `base`, or `base_2`, `base_3`, ... if `base` is already taken."""
    existing = set(existing)
    if base not in existing:
        return base
    suffix = 2
    while f"{base}_{suffix}" in existing:
        suffix += 1
    return f"{base}_{suffix}"


class _TypeDialog(QDialog):
    """Base dialog with an optional type selector, optional name field and OK/Cancel."""

    def __init__(self, parent: QWidget, title: str, types: List[str],
                 with_selector: bool = True, with_name: bool = True,
                 name_label: str = "New name:"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        row = QWidget(self)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        self.selector: Optional[QComboBox] = None
        if with_selector:
            row_layout.addWidget(QLabel("Type:", row))
            self.selector = QComboBox(row)
            self.selector.addItems(types)
            row_layout.addWidget(self.selector, stretch=1)

        self.name_edit: Optional[QLineEdit] = None
        if with_name:
            row_layout.addWidget(QLabel(name_label, row))
            self.name_edit = QLineEdit(row)
            row_layout.addWidget(self.name_edit, stretch=1)

        layout.addWidget(row)

        self.message = QLabel("", self)
        self.message.setStyleSheet("color: #b00;")
        layout.addWidget(self.message)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def selected_type(self) -> str:
        return self.selector.currentText() if self.selector is not None else ""

    def entered_name(self) -> str:
        return self.name_edit.text().strip() if self.name_edit is not None else ""


class RenameTypeDialog(_TypeDialog):
    def __init__(self, parent: QWidget, types: List[str]):
        super().__init__(parent, "Rename Stimulus Type", types)
        if types:
            self.name_edit.setText(types[0])
            self.selector.currentTextChanged.connect(self.name_edit.setText)


class RemoveTypeDialog(_TypeDialog):
    def __init__(self, parent: QWidget, types: List[str]):
        super().__init__(parent, "Remove Stimulus Type", types, with_name=False)


class DuplicateTypeDialog(_TypeDialog):
    def __init__(self, parent: QWidget, types: List[str]):
        super().__init__(parent, "Duplicate Stimulus Type", types)
        if types:
            self.name_edit.setText(_unique_type_name(types[0], types))
            self.selector.currentTextChanged.connect(
                lambda name: self.name_edit.setText(_unique_type_name(name, types))
            )


class AddTypeDialog(_TypeDialog):
    def __init__(self, parent: QWidget, types: List[str]):
        super().__init__(parent, "Add Stimulus Type", types, with_selector=False,
                         name_label="Type name:")
        self._types = set(types)
        self.name_edit.textChanged.connect(self._validate)
        self._validate("")

    def _validate(self, text: str) -> None:
        name = text.strip()
        if not name:
            self.message.setText("Enter a name for the new stimulus type.")
            valid = False
        elif name in self._types:
            self.message.setText(f"'{name}' already exists in the table.")
            valid = False
        else:
            self.message.setText("")
            valid = True
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(valid)


class OnsetOffsetDialog(QDialog):
    """Pick the onset and offset marker types used to derive event durations."""

    def __init__(self, parent: QWidget, types: List[str]):
        super().__init__(parent)
        self.setWindowTitle("Onset to Offset marks")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        row = QWidget(self)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(QLabel("Start:", row))
        self.start_combo = QComboBox(row)
        self.start_combo.addItems(types)
        row_layout.addWidget(self.start_combo, stretch=1)
        row_layout.addWidget(QLabel("End:", row))
        self.end_combo = QComboBox(row)
        self.end_combo.addItems(types)
        if len(types) > 1:
            self.end_combo.setCurrentIndex(1)
        row_layout.addWidget(self.end_combo, stretch=1)
        layout.addWidget(row)

        self.remove_checkbox = QCheckBox("remove offset marks when finished", self)
        layout.addWidget(self.remove_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def start_type(self) -> str:
        return self.start_combo.currentText()

    def end_type(self) -> str:
        return self.end_combo.currentText()

    def remove_offsets(self) -> bool:
        return self.remove_checkbox.isChecked()


class FileSelectionDialog(QDialog):
    """Choose which recordings the stimulus table edits."""

    def __init__(self, parent: QWidget, labels: List[str], current_row: int,
                 selected: List[int]):
        super().__init__(parent)
        self.setWindowTitle("Select Files to Edit")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        self._current_row = current_row
        self.current_radio = QRadioButton("Only current data file", self)
        self.all_radio = QRadioButton("All data files", self)
        self.custom_radio = QRadioButton("Selected data files:", self)
        layout.addWidget(self.current_radio)
        layout.addWidget(self.all_radio)
        layout.addWidget(self.custom_radio)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget()
        cb_layout = QVBoxLayout(container)
        cb_layout.setSpacing(2)
        self.checkboxes: List[QCheckBox] = []
        for i, label in enumerate(labels):
            cb = QCheckBox(label, container)
            cb.setChecked(i in selected)
            cb_layout.addWidget(cb)
            self.checkboxes.append(cb)
        scroll.setWidget(container)
        scroll.setMinimumHeight(160)
        layout.addWidget(scroll)

        if len(selected) == len(labels) and labels:
            self.all_radio.setChecked(True)
        elif selected == [current_row]:
            self.current_radio.setChecked(True)
        else:
            self.custom_radio.setChecked(True)

        for radio in (self.current_radio, self.all_radio, self.custom_radio):
            radio.toggled.connect(self._sync_enabled)
        self._sync_enabled()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _sync_enabled(self) -> None:
        custom = self.custom_radio.isChecked()
        for cb in self.checkboxes:
            cb.setEnabled(custom)

    def selected_rows(self) -> List[int]:
        if self.all_radio.isChecked():
            return list(range(len(self.checkboxes)))
        if self.current_radio.isChecked():
            return [self._current_row] if self._current_row >= 0 else []
        return [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]


class StimulusManagerWindow(QMainWindow):
    # True when the user pressed Update, False when the edits were discarded.
    finished = Signal(bool)

    def __init__(self, dataset: Any, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Stimulus Manager")
        self.resize(1100, 650)

        self.dataset = dataset
        self.rec = None
        self._edit_rows: List[int] = []   # recordings currently loaded in the table
        self._follow_current_file = True
        self._image_selection_only = False
        self._image_sort_by = "file"
        self._image_keys: List[tuple] = []
        self._image_row_of: Dict[tuple, int] = {}
        self._image_df: Optional[pd.DataFrame] = None
        self._finish_state: Optional[bool] = None
        self._original_stims = [
            rec.stim.copy() if isinstance(getattr(rec, "stim", None), pd.DataFrame) else None
            for rec in getattr(dataset, "dataset", [])
        ]

        root = QWidget(self)
        outer_layout = QVBoxLayout(root)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.setCentralWidget(root)

        # Embedded (non-native) menu bar so it stays inside the window on macOS.
        menu_bar = QMenuBar(root)
        menu_bar.setNativeMenuBar(False)
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("Export to Excel").triggered.connect(self._export_to_excel)
        file_menu.addAction("Import from Excel").triggered.connect(self._import_from_excel)
        validate_menu = menu_bar.addMenu("Validate")
        self._highlight_action = validate_menu.addAction("Highlight outliers")
        self._highlight_action.setCheckable(True)
        self._highlight_action.toggled.connect(self._toggle_outlier_highlighting)
        outer_layout.addWidget(menu_bar)

        central = QWidget(root)
        outer_layout.addWidget(central, stretch=1)
        layout = QHBoxLayout(central)

        # ------------------------------------------------------------ left side
        left = QWidget(central)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Recordings"))
        self.file_list = QTableWidget(left)
        self.file_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.file_list.setAlternatingRowColors(True)
        self.file_list.verticalHeader().setVisible(True)
        self.file_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.file_list.horizontalHeader().setStretchLastSection(True)
        self._populate_file_table()
        left_layout.addWidget(self.file_list)
        left_layout.addWidget(QLabel("Selected recording summary"))
        self.summary_panel = QTextEdit(left)
        self.summary_panel.setReadOnly(True)
        self.summary_panel.setMinimumHeight(170)
        left_layout.addWidget(self.summary_panel)
        left_layout.addWidget(QLabel("Data type"))
        self.type_selector = QComboBox(left)
        left_layout.addWidget(self.type_selector)
        left.setMaximumWidth(280)
        layout.addWidget(left)

        # ----------------------------------------------------------- right side
        right = QWidget(central)
        right_layout = QVBoxLayout(right)

        file_bar = QWidget(right)
        file_bar_layout = QHBoxLayout(file_bar)
        file_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.select_files_button = QPushButton("Select Files to Edit…", file_bar)
        self.select_files_button.clicked.connect(self._select_files_to_edit)
        file_bar_layout.addWidget(self.select_files_button)
        self.editing_label = QLabel("Editing: <none>", file_bar)
        file_bar_layout.addWidget(self.editing_label, stretch=1)
        right_layout.addWidget(file_bar)

        right_layout.addWidget(QLabel(
            "Stimulus events (click a header to sort, right-click the Name header to filter,"
            " double-click a cell to edit)"
        ))

        self.stim_model = StimTableModel(self)
        self.stim_table = QTableView(right)
        self.stim_table.setModel(self.stim_model)
        self.stim_table.setSortingEnabled(True)
        self.stim_table.setAlternatingRowColors(True)
        self.stim_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.stim_table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
            | QAbstractItemView.EditKeyPressed
        )
        self.stim_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stim_table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.stim_table.horizontalHeader().customContextMenuRequested.connect(
            self._show_header_context_menu
        )
        self.stim_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.stim_table.customContextMenuRequested.connect(self._show_stim_context_menu)
        right_layout.addWidget(self.stim_table, stretch=1)

        right_layout.addWidget(QLabel("Stimulus design (right-click for display options)"))
        self._image_figure = Figure(figsize=(6, 2.4))
        self._image_canvas = FigureCanvas(self._image_figure)
        self._image_ax = self._image_figure.add_subplot(1, 1, 1)
        self._image_canvas.setMinimumHeight(180)
        self._image_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self._image_canvas.customContextMenuRequested.connect(self._show_image_context_menu)
        right_layout.addWidget(self._image_canvas, stretch=1)

        button_row = QWidget(right)
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        add_button = QPushButton("Add Event", button_row)
        add_button.clicked.connect(self._add_event)
        button_layout.addWidget(add_button)
        remove_button = QPushButton("Remove Selected", button_row)
        remove_button.clicked.connect(self._remove_events)
        button_layout.addWidget(remove_button)
        button_layout.addStretch()
        self.update_button = QPushButton("Update", button_row)
        self.update_button.setDefault(True)
        self.update_button.clicked.connect(self._apply_and_close)
        button_layout.addWidget(self.update_button)
        self.cancel_button = QPushButton("Cancel", button_row)
        self.cancel_button.clicked.connect(self._cancel_and_close)
        button_layout.addWidget(self.cancel_button)
        right_layout.addWidget(button_row)

        layout.addWidget(right, stretch=1)

        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready.")

        self.file_list.itemSelectionChanged.connect(self._on_file_selection_changed)
        self.type_selector.currentTextChanged.connect(self._on_type_changed)

        for signal in (self.stim_model.dataChanged, self.stim_model.modelReset,
                       self.stim_model.layoutChanged, self.stim_model.rowsInserted,
                       self.stim_model.rowsRemoved):
            signal.connect(self._update_stim_image)
        self.stim_table.selectionModel().selectionChanged.connect(
            self._on_table_selection_changed
        )

        if len(getattr(dataset, "dataset", [])) > 0:
            self.file_list.setCurrentCell(0, 0)
            self._on_file_changed(0)
        else:
            self._update_summary(-1)
        self._update_stim_image()

    # ------------------------------------------------------------ left helpers

    def _populate_file_table(self) -> None:
        recordings = list(getattr(self.dataset, "dataset", []))
        labels = _build_labels(self.dataset) if recordings else []
        demo = self.dataset.get_demographics() if recordings else None

        columns: List[str] = []
        rows: List[Dict[str, Any]] = []
        if demo is not None:
            columns = [str(col) for col in demo.columns]
            rows = demo.to_dict(orient="records")

        while len(rows) < len(recordings):
            rows.append({})
        rows = rows[: len(recordings)]

        headers = ["Recording"] + columns
        self.file_list.setColumnCount(len(headers))
        self.file_list.setHorizontalHeaderLabels(headers)
        self.file_list.setRowCount(len(recordings))
        self.file_list.setVerticalHeaderLabels([str(i + 1) for i in range(len(recordings))])

        for row_idx, label in enumerate(labels):
            self.file_list.setItem(row_idx, 0, QTableWidgetItem(label))
            row_data = rows[row_idx] if row_idx < len(rows) else {}
            for col_idx, key in enumerate(columns, start=1):
                self.file_list.setItem(
                    row_idx, col_idx, QTableWidgetItem(_to_display_str(row_data.get(key)))
                )

    def _recording_metrics(self, rec: Any, key: Optional[str]) -> tuple[Optional[float], Optional[int]]:
        series = getattr(rec, "timeseries", None)
        if series is None:
            return None, None
        keys = list(series.keys())
        if not keys:
            return None, None
        data = series[key if key in keys else keys[0]]

        duration = None
        time = getattr(data, "time", None)
        if time is not None and len(time) > 0:
            duration = float(time.max()) - float(time.min())

        channels = getattr(data, "channel", None)
        n_channels = int(len(channels)) if channels is not None else None
        return duration, n_channels

    def _update_summary(self, row: int, key: Optional[str] = None) -> None:
        recordings = getattr(self.dataset, "dataset", [])
        if row < 0 or row >= len(recordings):
            self.summary_panel.setPlainText("No recording selected.")
            return

        rec = recordings[row]
        meta = getattr(rec, "meta_data", None) or {}
        if meta:
            demographic_lines = [f"  - {k}: {_to_display_str(v)}" for k, v in meta.items()]
        else:
            demographic_lines = ["  - <none>"]

        stim_names: List[str] = []
        stim = getattr(rec, "stim", None)
        if stim is not None and "trial_type" in stim:
            for name in stim["trial_type"].dropna().to_list():
                name_str = str(name)
                if name_str not in stim_names:
                    stim_names.append(name_str)

        duration, n_channels = self._recording_metrics(rec, key)
        duration_text = f"{duration:.2f}" if duration is not None else "N/A"
        channels_text = str(n_channels) if n_channels is not None else "N/A"
        stim_text = ", ".join(stim_names) if stim_names else "<none>"

        summary = (
            "Demographics:\n"
            + "\n".join(demographic_lines)
            + "\n\nStimulus events:\n"
            + f"  - {stim_text}\n\n"
            + f"Recording length (s): {duration_text}\n"
            + f"Number of channels: {channels_text}"
        )
        self.summary_panel.setPlainText(summary)

    # ------------------------------------------------------------------ slots

    def _on_file_selection_changed(self) -> None:
        self._on_file_changed(self.file_list.currentRow())

    def _on_file_changed(self, row: int) -> None:
        recordings = getattr(self.dataset, "dataset", [])
        if row < 0 or row >= len(recordings):
            self.rec = None
            self._set_edit_rows([], follow_current=True)
            self._update_summary(-1)
            return

        self.rec = recordings[row]
        keys = list(getattr(self.rec, "timeseries", {}).keys())

        self.type_selector.blockSignals(True)
        self.type_selector.clear()
        self.type_selector.addItems(keys)
        if keys:
            self.type_selector.setCurrentIndex(0)
        self.type_selector.blockSignals(False)

        if self._follow_current_file:
            self._set_edit_rows([row], follow_current=True)
        self._update_summary(row, keys[0] if keys else None)
        self._status_bar.showMessage(
            f"Recording {row + 1}: {self.stim_model.rowCount()} stimulus event(s) shown."
        )

    # ---------------------------------------------------------- file selection

    def _recording_labels(self) -> List[str]:
        recordings = getattr(self.dataset, "dataset", [])
        return _build_labels(self.dataset) if recordings else []

    def _set_edit_rows(self, rows: List[int], follow_current: bool) -> None:
        recordings = getattr(self.dataset, "dataset", [])
        labels = self._recording_labels()
        rows = [r for r in rows if 0 <= r < len(recordings)]
        self._edit_rows = rows
        self._follow_current_file = follow_current
        self.stim_model.set_recordings([recordings[r] for r in rows],
                                       [labels[r] for r in rows])
        self._update_editing_label()

    def _update_editing_label(self) -> None:
        recordings = getattr(self.dataset, "dataset", [])
        labels = self._recording_labels()
        n_total, n_sel = len(recordings), len(self._edit_rows)
        if n_sel == 0:
            text = "Editing: <none>"
        elif n_sel == 1:
            text = f"Editing file {labels[self._edit_rows[0]]}"
        elif n_sel == n_total:
            text = f"Editing All ({n_total} files)"
        else:
            text = f"Editing {n_sel} of {n_total} files"
        self.editing_label.setText(text)

    def _select_files_to_edit(self) -> None:
        labels = self._recording_labels()
        if not labels:
            QMessageBox.information(self, "Select Files", "No recordings loaded.")
            return
        current = self.file_list.currentRow()
        dialog = FileSelectionDialog(self, labels, current, self._edit_rows)
        if dialog.exec() != QDialog.Accepted:
            return
        rows = dialog.selected_rows()
        if not rows:
            QMessageBox.information(self, "Select Files", "Select at least one file.")
            return
        self._set_edit_rows(rows, follow_current=dialog.current_radio.isChecked())
        self._status_bar.showMessage(
            f"{self.stim_model.rowCount()} stimulus event(s) across {len(rows)} file(s)."
        )

    def _on_type_changed(self, key: str) -> None:
        if self.rec is not None and key:
            self._update_summary(self.file_list.currentRow(), key)

    # ------------------------------------------------------- stimulus filter

    def _show_header_context_menu(self, pos) -> None:
        header = self.stim_table.horizontalHeader()
        column = header.logicalIndexAt(pos)
        menu = QMenu(header)

        if column == COL_NAME:
            menu.addAction("Filter by Type…").triggered.connect(self._pick_type_filter)
            show_all = menu.addAction("Show All Types")
            show_all.setEnabled(self.stim_model.type_filter is not None)
            show_all.triggered.connect(self._clear_type_filter)
        elif column == COL_ONSET:
            menu.addAction("Shift All Onsets").triggered.connect(self._shift_onsets)
        elif column == COL_DURATION:
            menu.addAction("Change All Durations").triggered.connect(self._change_durations)
        elif column == COL_AMPLITUDE:
            menu.addAction("Change All Amplitudes").triggered.connect(self._change_amplitudes)
        else:
            return

        menu.exec(header.mapToGlobal(pos))

    # ------------------------------------------------------- bulk value edits

    def _selected_view_rows(self) -> List[int]:
        return sorted({idx.row() for idx in self.stim_table.selectionModel().selectedRows()})

    def _prompt_numeric(self, title: str, label: str, default: float = 0.0) -> Optional[float]:
        value, ok = QInputDialog.getDouble(self, title, label, default, -1e9, 1e9, 4)
        return value if ok else None

    def _bulk_edit(self, title: str, label: str, column: int, relative: bool) -> None:
        rows = self._selected_view_rows()
        if not rows:
            QMessageBox.information(self, title, "No rows are selected.")
            return
        value = self._prompt_numeric(title, f"{label} ({len(rows)} selected row(s)):")
        if value is None:
            return
        n = self.stim_model.apply_numeric(rows, column, value, relative=relative)
        self._status_bar.showMessage(f"{title}: updated {n} event(s).")

    def _shift_onsets(self) -> None:
        self._bulk_edit("Shift All Onsets", "Shift onsets by (s)", COL_ONSET, relative=True)

    def _change_durations(self) -> None:
        self._bulk_edit("Change All Durations", "New duration (s)", COL_DURATION, relative=False)

    def _change_amplitudes(self) -> None:
        self._bulk_edit("Change All Amplitudes", "New amplitude", COL_AMPLITUDE, relative=False)

    # ------------------------------------------------------------- selection

    def _on_table_selection_changed(self, *_args) -> None:
        if self._image_selection_only:
            self._update_stim_image()

    def _select_all_rows(self) -> None:
        self.stim_table.selectAll()
        self._status_bar.showMessage(f"Selected {self.stim_model.rowCount()} event(s).")

    def _select_rows_of_type(self, name: Optional[str]) -> None:
        if not name:
            return
        rows = self.stim_model.rows_of_type(name)
        selection = QItemSelection()
        last_col = self.stim_model.columnCount() - 1
        for row in rows:
            selection.select(self.stim_model.index(row, 0),
                             self.stim_model.index(row, last_col))
        self.stim_table.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect
        )
        self._status_bar.showMessage(f"Selected {len(rows)} event(s) of type '{name}'.")

    def _filter_description(self) -> str:
        selected = self.stim_model.type_filter
        if selected is None:
            return "all types"
        return ", ".join(sorted(selected)) or "no types"

    def _update_filter_status(self) -> None:
        self._status_bar.showMessage(
            f"Showing {self.stim_model.rowCount()} of "
            f"{len(self.stim_model.dataframe())} event(s) [{self._filter_description()}]."
        )

    def _pick_type_filter(self) -> None:
        types = self.stim_model.trial_types()
        if not types:
            QMessageBox.information(self, "Filter", "No stimulus types to filter.")
            return
        chosen = _pick_values_dialog(self, "Filter by trial type", types,
                                     self.stim_model.type_filter)
        if chosen is None:
            return
        self.stim_model.set_type_filter(None if len(chosen) == len(types) else chosen)
        self._update_filter_status()

    def _clear_type_filter(self) -> None:
        self.stim_model.set_type_filter(None)
        self._update_filter_status()

    # --------------------------------------------------------- context menu

    def _show_stim_context_menu(self, pos) -> None:
        menu = QMenu(self.stim_table)

        clicked_row = self.stim_table.indexAt(pos).row()
        clicked_type = self.stim_model.type_at_row(clicked_row)

        select_menu = menu.addMenu("Select Events")
        select_menu.addAction("Select All Rows").triggered.connect(self._select_all_rows)
        same_type = select_menu.addAction("Select All Rows of Same Type")
        same_type.setEnabled(clicked_type is not None)
        same_type.triggered.connect(lambda: self._select_rows_of_type(clicked_type))

        types_menu = menu.addMenu("Edit Types")
        types_menu.addAction("Rename Type").triggered.connect(self._rename_type)
        types_menu.addAction("Add Type").triggered.connect(self._add_type)
        types_menu.addAction("Remove Type").triggered.connect(self._remove_type)
        types_menu.addAction("Duplicate Types").triggered.connect(self._duplicate_types)

        timing_menu = menu.addMenu("Edit Timing")
        timing_menu.addAction("Match to template").triggered.connect(self._match_to_template)
        timing_menu.addAction("Shift all times").triggered.connect(self._shift_all_times)
        prune_menu = timing_menu.addMenu("Prune by Pattern")
        prune_menu.addAction("Onset to Offset marks").triggered.connect(
            self._onset_to_offset_marks)
        prune_menu.addAction("Remove all odd events").triggered.connect(
            lambda: self._remove_alternating("odd"))
        prune_menu.addAction("Remove all even events").triggered.connect(
            lambda: self._remove_alternating("even"))

        meta_menu = menu.addMenu("Edit Metadata")
        meta_menu.addAction("Add info from file").triggered.connect(self._add_info_from_file)
        meta_menu.addAction("Add metadata column").triggered.connect(self._add_metadata_column)

        menu.exec(self.stim_table.viewport().mapToGlobal(pos))

    def _not_implemented(self, action: str) -> None:
        QMessageBox.information(self, action, f"'{action}' is not implemented yet.")
        self._status_bar.showMessage(f"{action}: not implemented yet.")

    def _require_types(self, action: str) -> Optional[List[str]]:
        types = self.stim_model.trial_types()
        if not types:
            QMessageBox.information(self, action, "No stimulus types in this recording.")
            return None
        return types

    def _rename_type(self) -> None:
        types = self._require_types("Rename Type")
        if types is None:
            return
        dialog = RenameTypeDialog(self, types)
        if dialog.exec() != QDialog.Accepted:
            return

        old = dialog.selected_type()
        new = dialog.entered_name()
        if not new:
            QMessageBox.warning(self, "Rename Type", "Please enter a new name.")
            return
        if new == old:
            return

        if new in types:
            box = QMessageBox(self)
            box.setWindowTitle("Rename Type")
            box.setText(f"A stimulus type named '{new}' already exists.")
            box.setInformativeText("Merge the renamed events into it, or create a new type?")
            merge_btn = box.addButton("Merge", QMessageBox.AcceptRole)
            create_btn = box.addButton("Create New", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is create_btn:
                new = _unique_type_name(new, types)
            elif clicked is not merge_btn:
                return

        n = self.stim_model.rename_type(old, new)
        self._after_type_change(f"Renamed {n} event(s) from '{old}' to '{new}'.")

    def _add_type(self) -> None:
        dialog = AddTypeDialog(self, self.stim_model.trial_types())
        if dialog.exec() != QDialog.Accepted:
            return
        name = dialog.entered_name()
        self.stim_model.add_type(name)
        self._after_type_change(f"Added stimulus type '{name}'.")

    def _remove_type(self) -> None:
        types = self._require_types("Remove Type")
        if types is None:
            return
        dialog = RemoveTypeDialog(self, types)
        if dialog.exec() != QDialog.Accepted:
            return
        name = dialog.selected_type()
        n = self.stim_model.remove_type(name)
        self._after_type_change(f"Removed {n} event(s) of type '{name}'.")

    def _duplicate_types(self) -> None:
        types = self._require_types("Duplicate Types")
        if types is None:
            return
        dialog = DuplicateTypeDialog(self, types)
        if dialog.exec() != QDialog.Accepted:
            return
        source = dialog.selected_type()
        new = dialog.entered_name()
        if not new:
            QMessageBox.warning(self, "Duplicate Types", "Please enter a name for the copy.")
            return
        n = self.stim_model.duplicate_type(source, new)
        self._after_type_change(f"Duplicated {n} event(s) of '{source}' as '{new}'.")

    def _after_type_change(self, message: str) -> None:
        self._update_summary(self.file_list.currentRow(), self.type_selector.currentText())
        self._status_bar.showMessage(message)

    # Placeholders: the actual editing logic is wired up separately.
    def _match_to_template(self) -> None:
        self._not_implemented("Match to template")

    def _shift_all_times(self) -> None:
        self._bulk_edit("Shift all times", "Shift onsets by (s)", COL_ONSET, relative=True)

    def _onset_to_offset_marks(self) -> None:
        types = self._require_types("Onset to Offset marks")
        if types is None:
            return
        dialog = OnsetOffsetDialog(self, types)
        if dialog.exec() != QDialog.Accepted:
            return
        updated, removed = self.stim_model.apply_onset_to_offset(
            dialog.start_type(), dialog.end_type(), dialog.remove_offsets()
        )
        self._after_type_change(
            f"Updated {updated} event duration(s); removed {removed} offset mark(s)."
        )

    def _remove_alternating(self, which: str) -> None:
        title = f"Remove all {which} events"
        rows = self._selected_view_rows()
        if not rows:
            QMessageBox.information(self, title, "No rows are selected.")
            return
        targets = self.stim_model.alternating_rows(rows, which)
        if not targets:
            QMessageBox.information(self, title, "No matching events found.")
            return
        confirm = QMessageBox.question(
            self, title,
            f"Remove {len(targets)} {which}-numbered event(s) from the selection?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        n = self.stim_model.remove_source_rows(targets)
        self._after_type_change(f"{title}: removed {n} event(s).")

    def _add_info_from_file(self) -> None:
        self._not_implemented("Add info from file")

    def _add_metadata_column(self) -> None:
        self._not_implemented("Add metadata column")

    def _add_event(self) -> None:
        if not self._edit_rows:
            QMessageBox.information(self, "Add Event", "No recording selected.")
            return
        self.stim_model.add_row()
        self._update_summary(self.file_list.currentRow(), self.type_selector.currentText())

    def _remove_events(self) -> None:
        rows = sorted({idx.row() for idx in self.stim_table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.information(self, "Remove Events", "No rows selected.")
            return
        self.stim_model.remove_rows(rows)
        self._update_summary(self.file_list.currentRow(), self.type_selector.currentText())

    # ---------------------------------------------------------- stimulus image

    def _toggle_outlier_highlighting(self, checked: bool) -> None:
        self.stim_model.set_highlight_outliers(checked)
        self._status_bar.showMessage(
            "Outlier highlighting on: durations >1.5 SD from their type mean are red."
            if checked else "Outlier highlighting off."
        )

    # ------------------------------------------------------------ import/export

    def _export_to_excel(self) -> None:
        frames = self.stim_model.export_frames()
        if not frames:
            QMessageBox.information(self, "Export to Excel", "No files are being edited.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Stimulus Events", "", "Excel Workbook (*.xlsx);;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            with pd.ExcelWriter(path) as writer:
                used = set()
                for label, frame in frames.items():
                    sheet = _sheet_name(label, used)
                    used.add(sheet)
                    frame.to_excel(writer, sheet_name=sheet, index=False)
            self._status_bar.showMessage(f"Exported {len(frames)} file(s) to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export to Excel", f"Export failed:\n{exc}")

    def _import_from_excel(self) -> None:
        if not self._edit_rows:
            QMessageBox.information(self, "Import from Excel", "No files are being edited.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Stimulus Events", "",
            "Excel or CSV (*.xlsx *.xls *.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            if path.lower().endswith(".csv"):
                sheets = {"": pd.read_csv(path)}
            else:
                sheets = pd.read_excel(path, sheet_name=None)
        except Exception as exc:
            QMessageBox.critical(self, "Import from Excel", f"Could not read the file:\n{exc}")
            return

        try:
            normalised = {name: _normalise_import(frame) for name, frame in sheets.items()}
        except ValueError as exc:
            QMessageBox.warning(self, "Import from Excel", str(exc))
            return

        labels = self.stim_model._labels
        frames: Dict[int, pd.DataFrame] = {}
        if len(normalised) == 1 and len(labels) == 1:
            frames[0] = next(iter(normalised.values()))
        else:
            by_sheet = {str(name): frame for name, frame in normalised.items()}
            for i, label in enumerate(labels):
                frame = by_sheet.get(_sheet_name(label, set()))
                if frame is None:
                    frame = by_sheet.get(str(label))
                if frame is not None:
                    frames[i] = frame
        if not frames:
            QMessageBox.warning(
                self, "Import from Excel",
                "No sheet names matched the files being edited."
            )
            return

        box = QMessageBox(self)
        box.setWindowTitle("Import from Excel")
        box.setText(f"Import {sum(len(f) for f in frames.values())} event(s) "
                    f"into {len(frames)} file(s).")
        box.setInformativeText("Append to the existing events, or replace them?")
        append_btn = box.addButton("Append", QMessageBox.AcceptRole)
        replace_btn = box.addButton("Replace", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked not in (append_btn, replace_btn):
            return

        n = self.stim_model.import_frames(
            frames, "append" if clicked is append_btn else "replace"
        )
        self._after_type_change(f"Imported {n} event(s) from {path}.")

    # ---------------------------------------------------------- stimulus image

    def _show_image_context_menu(self, pos) -> None:
        menu = QMenu(self._image_canvas)

        event = self._event_at_image_pos(pos)
        if event is not None:
            label, row = event
            menu.addAction("Select Event").triggered.connect(
                lambda: self._select_event_row(label)
            )
            menu.addSeparator()
            for text in (
                f"File: {row[FILE_COL]}",
                f"Type: {row['trial_type']}",
                f"Onset: {row['onset']:g} s",
                f"Duration: {row['duration']:g} s",
                f"Amplitude: {row['value']:g}",
            ):
                menu.addAction(text).setEnabled(False)
            menu.addSeparator()

        highlighted = menu.addAction("Show Only Highlighted Rows")
        highlighted.setCheckable(True)
        highlighted.setChecked(self._image_selection_only)
        highlighted.triggered.connect(lambda: self._set_image_selection_only(True))
        show_all = menu.addAction("Show All Rows")
        show_all.setCheckable(True)
        show_all.setChecked(not self._image_selection_only)
        show_all.triggered.connect(lambda: self._set_image_selection_only(False))

        menu.addSeparator()
        by_file = menu.addAction("Sort by File")
        by_file.setCheckable(True)
        by_file.setChecked(self._image_sort_by == "file")
        by_file.triggered.connect(lambda: self._set_image_sort("file"))
        by_type = menu.addAction("Sort by Type")
        by_type.setCheckable(True)
        by_type.setChecked(self._image_sort_by == "type")
        by_type.triggered.connect(lambda: self._set_image_sort("type"))

        menu.exec(self._image_canvas.mapToGlobal(pos))

    def _set_image_selection_only(self, only: bool) -> None:
        self._image_selection_only = only
        self._update_stim_image()

    def _set_image_sort(self, key: str) -> None:
        self._image_sort_by = key
        self._update_stim_image()

    def _event_at_image_pos(self, pos):
        """Return (DataFrame label, row) of the event under a canvas position."""
        if self._image_df is None or len(self._image_df) == 0:
            return None
        ratio = self._image_canvas.devicePixelRatioF()
        x_px = pos.x() * ratio
        y_px = self._image_canvas.figure.bbox.height - pos.y() * ratio
        if not self._image_ax.bbox.contains(x_px, y_px):
            return None

        time, row_pos = self._image_ax.transData.inverted().transform((x_px, y_px))
        row_index = int(np.floor(row_pos))
        if row_index < 0 or row_index >= len(self._image_keys):
            return None
        file_label, trial_type = self._image_keys[row_index]

        df = self._image_df
        onset = pd.to_numeric(df["onset"], errors="coerce").fillna(0.0)
        duration = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
        match = df[
            (df[FILE_COL].astype(str) == file_label)
            & (df["trial_type"].astype(str) == trial_type)
            & (onset <= time)
            & (time < onset + duration.clip(lower=IMAGE_TIME_STEP))
        ]
        if match.empty:
            return None
        label = match.index[0]
        row = {
            FILE_COL: str(df.at[label, FILE_COL]),
            "trial_type": str(df.at[label, "trial_type"]),
            "onset": float(onset.at[label]),
            "duration": float(duration.at[label]),
            "value": float(pd.to_numeric(df.at[label, "value"], errors="coerce") or 0.0),
        }
        return label, row

    def _select_event_row(self, label) -> None:
        view_row = self.stim_model.view_row_for_source(label)
        if view_row is None:
            QMessageBox.information(self, "Select Event",
                                    "That event is not visible in the table.")
            return
        index = self.stim_model.index(view_row, COL_NAME)
        self.stim_table.setCurrentIndex(index)
        self.stim_table.selectRow(view_row)
        self.stim_table.scrollTo(index, QAbstractItemView.PositionAtCenter)
        self.stim_table.setFocus()
        self._status_bar.showMessage(f"Selected event on table row {view_row + 1}.")

    def _stim_matrix(self, df: pd.DataFrame, dt: float = IMAGE_TIME_STEP):
        """Build a (file, type) x time amplitude matrix; overlapping events add up."""
        onset = pd.to_numeric(df.get("onset"), errors="coerce").fillna(0.0).to_numpy(float)
        duration = pd.to_numeric(df.get("duration"), errors="coerce").fillna(0.0).to_numpy(float)
        amplitude = pd.to_numeric(df.get("value"), errors="coerce").fillna(0.0).to_numpy(float)
        files = df[FILE_COL].astype(str).to_numpy()
        types = df["trial_type"].astype(str).to_numpy()

        keys = sorted(
            set(zip(files, types)),
            key=(lambda k: k) if self._image_sort_by == "file" else (lambda k: (k[1], k[0])),
        )
        row_of = {key: i for i, key in enumerate(keys)}
        self._image_keys = keys
        self._image_row_of = row_of

        t_end = float(np.max(onset + np.maximum(duration, dt))) if len(onset) else dt
        n_cols = min(int(np.ceil(t_end / dt)) + 1, 200000)
        matrix = np.zeros((len(keys), n_cols), dtype=float)

        for i in range(len(onset)):
            start = max(int(round(onset[i] / dt)), 0)
            stop = max(int(round((onset[i] + duration[i]) / dt)), start + 1)
            matrix[row_of[(files[i], types[i])], start:min(stop, n_cols)] += amplitude[i]

        labels = [f"{f} : {t}" for f, t in keys]
        return matrix, labels, n_cols * dt

    def _update_stim_image(self, *_args) -> None:
        rows = self._selected_view_rows() if self._image_selection_only else None
        df = self.stim_model.rows_dataframe(rows)
        self._image_df = df
        self._image_keys = []
        self._image_row_of = {}

        self._image_ax.clear()
        if len(df) == 0:
            self._image_ax.text(0.5, 0.5, "No stimulus events to display",
                                ha="center", va="center", transform=self._image_ax.transAxes)
            self._image_ax.set_xticks([])
            self._image_ax.set_yticks([])
        else:
            matrix, labels, t_max = self._stim_matrix(df)
            self._image_ax.imshow(matrix, cmap="gray", aspect="auto", origin="upper",
                                  interpolation="nearest", extent=[0, t_max, len(labels), 0])
            self._image_ax.set_yticks([i + 0.5 for i in range(len(labels))])
            self._image_ax.set_yticklabels(labels, fontsize=7)
            self._image_ax.set_xlabel("time (s)")
            self._draw_outlier_boxes(df)
        self._image_figure.tight_layout()
        self._image_canvas.draw_idle()

    def _draw_outlier_boxes(self, df: pd.DataFrame) -> None:
        """Outline outlier events in the design image with the same colors as the table."""
        if not self.stim_model.highlight_outliers:
            return
        onset = pd.to_numeric(df["onset"], errors="coerce").fillna(0.0)
        duration = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
        for position, row in df.iterrows():
            bad_duration, bad_gap = self.stim_model.outlier_flags(int(position))
            if not (bad_duration or bad_gap):
                continue
            color = "purple" if bad_duration and bad_gap else ("red" if bad_duration else "blue")
            key = (str(row[FILE_COL]), str(row["trial_type"]))
            row_i = self._image_row_of.get(key)
            if row_i is None:
                continue
            width = max(float(duration.at[position]), IMAGE_TIME_STEP)
            rect = patches.Rectangle(
                (float(onset.at[position]), row_i), width, 1,
                fill=False, edgecolor=color, linewidth=2, zorder=3,
            )
            self._image_ax.add_patch(rect)

    # ------------------------------------------------------------ finish/close

    def _restore_original_stims(self) -> None:
        for rec, stim in zip(getattr(self.dataset, "dataset", []), self._original_stims):
            if stim is not None:
                rec.stim = stim.copy()

    def _confirm_discard(self) -> bool:
        answer = QMessageBox.question(
            self,
            "Cancel",
            "Changes will not be saved.  Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _apply_and_close(self) -> None:
        self.stim_model.write_back()
        self._finish_state = True
        self.close()

    def _cancel_and_close(self) -> None:
        if not self._confirm_discard():
            return
        self._restore_original_stims()
        self._finish_state = False
        self.close()

    def closeEvent(self, event) -> None:
        if self._finish_state is None:
            if not self._confirm_discard():
                event.ignore()
                return
            self._restore_original_stims()
            self._finish_state = False
        self.finished.emit(bool(self._finish_state))
        event.accept()


def edit_stimulus_events(dataset: Any, parent: Optional[QWidget] = None) -> bool:
    """Show the Stimulus Manager modally; returns True when the edits were kept."""
    window = StimulusManagerWindow(dataset, parent)
    window.setWindowModality(Qt.ApplicationModal)

    outcome = {"updated": False}
    loop = QEventLoop()

    def _done(updated: bool) -> None:
        outcome["updated"] = updated
        loop.quit()

    window.finished.connect(_done)
    window.show()
    window.raise_()
    window.activateWindow()
    loop.exec()
    return outcome["updated"]


# Windows opened from a notebook must be kept alive; Python would otherwise
# garbage-collect them as soon as the call returns.
_OPEN_WINDOWS: List["StimulusManagerWindow"] = []


def _release_window(window) -> None:
    """Drop the reference kept for notebook windows once Qt has destroyed it."""
    if window in _OPEN_WINDOWS:
        _OPEN_WINDOWS.remove(window)


def _active_ipython():
    """Return the running IPython shell, or None outside IPython."""
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def stimulus_manager(dataset: Any, block: Optional[bool] = None) -> Any:
    """Launch the Stimulus Manager for `dataset` (a `DataSet` instance).

    `block` defaults to True in plain scripts and False under IPython/Jupyter,
    where the Qt event loop is instead hooked into the kernel.  When blocking,
    the (possibly edited) dataset is returned once the window closes; otherwise
    the window itself is returned so it can be inspected programmatically.
    """
    app = QApplication.instance() or QApplication(sys.argv or ["stimulus_manager"])

    shell = _active_ipython()
    if block is None:
        block = shell is None

    window = StimulusManagerWindow(dataset)
    # Without WA_DeleteOnClose the window (and the Qt event-loop hook installed
    # below) would outlive the notebook cell and block kernel shutdown.
    window.setAttribute(Qt.WA_DeleteOnClose, True)
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(lambda *_: _release_window(window))
    window.show()
    window.raise_()
    window.activateWindow()

    if not block and shell is not None:
        try:
            shell.enable_gui("qt")
        except Exception:
            block = True   # kernel cannot host the Qt loop; fall back to blocking

    if block:
        app.exec()
        return dataset

    return window


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Browse and edit stimulus events of a pyBrainAnalyzIR DataSet."
    )
    parser.add_argument("dataset", help="path to a pickled pyBrainAnalyzIR DataSet (.pkl)")
    args = parser.parse_args(argv)

    # Unpickling executes code from the file; only load files you trust.
    with open(args.dataset, "rb") as fid:
        dataset = pickle.load(fid)

    if not hasattr(dataset, "dataset"):
        parser.error(f"{args.dataset} does not contain a DataSet object")

    stimulus_manager(dataset, block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
