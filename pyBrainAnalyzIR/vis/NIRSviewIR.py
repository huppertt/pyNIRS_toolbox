"""PySide6 GUI to browse the recordings stored in a `DataSet`.

Usage
-----
    from pyBrainAnalyzIR.vis.NIRSviewIR import NIRSviewIR

    NIRSviewIR(dataset)

In a script this blocks until the window is closed.  Inside a Jupyter/IPython
notebook the Qt event loop is integrated with the kernel instead, so the call
returns immediately and the notebook stays responsive.

From the command line, with a pickled `DataSet`::

    python -m pyBrainAnalyzIR.vis.NIRSviewIR mydataset.pkl
"""

from __future__ import annotations

import argparse
import io
import sys
import copy
import pickle
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QSortFilterProxyModel, QModelIndex
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QStatusBar,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from pyBrainAnalyzIR.vis.demographics_manager import DemographicsManager, _to_display_str
from pyBrainAnalyzIR.vis.plot_nirs_inline import linecolors, to_string
from pyBrainAnalyzIR.vis.pipeline_manager import (
    PipelineManagerDialog,
    pipeline_to_json,
    pipeline_from_json,
)
from pyBrainAnalyzIR.vis._about import get_about_text
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module
from pyBrainAnalyzIR.dataclasses.statistics import Statistics
from statsmodels.stats.multitest import multipletests

#: Item role storing a leaf's index into `dataset.dataset`.
RECORDING_INDEX_ROLE = Qt.UserRole + 1

SUBJECT_KEYS = ("subject", "subj", "subjID", "subjectID", "ID", "name")

#: Alternative meta_data names for each tree level, in order of preference.
#: Matching is case-insensitive.
GROUP_KEYS = ("group",)
SESSION_KEYS = ("session", "sess", "visit")


def _meta_lookup(rec: Any, keys: tuple) -> Optional[str]:
    """First non-empty meta_data value among `keys`, matched case-insensitively."""
    meta = getattr(rec, "meta_data", None) or {}
    lowered = {}
    for actual_key, value in meta.items():
        lowered.setdefault(str(actual_key).strip().lower(), value)
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None and _to_display_str(value).strip() != "":
            return _to_display_str(value).strip()
    return None


def _subject_label(rec: Any) -> Optional[str]:
    return _meta_lookup(rec, SUBJECT_KEYS)


def _group_label(rec: Any) -> Optional[str]:
    return _meta_lookup(rec, GROUP_KEYS)


def _session_label(rec: Any) -> Optional[str]:
    return _meta_lookup(rec, SESSION_KEYS)


def _build_labels(dataset: Any) -> List[str]:
    """Label each recording by subject, disambiguating repeats with a file index."""
    names = [_subject_label(rec) for rec in dataset.dataset]
    counts = {}
    for name in names:
        if name is not None:
            counts[name] = counts.get(name, 0) + 1

    seen = {}
    labels = []
    for i, name in enumerate(names):
        if name is None:
            labels.append(f"[{i}] <no subject>")
        elif counts[name] == 1:
            labels.append(f"[{i}] {name}")
        else:
            seen[name] = seen.get(name, 0) + 1
            labels.append(f"[{i}] {name} (file {seen[name]} of {counts[name]})")
    return labels


# ---------------------------------------------------------------------------
# Stats table helpers
# ---------------------------------------------------------------------------

class _StatsFilterProxy(QSortFilterProxyModel):
    """Proxy that filters the stats table by category selections and numeric thresholds.

    Sorting uses Qt.UserRole data so numeric columns sort as numbers, not strings.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition_col = -1
        self._type_col = -1
        self._p_col = -1
        self._q_col = -1
        self._condition_filter: Optional[set] = None  # None = all
        self._type_filter: Optional[set] = None
        self._p_threshold: float = 1.0
        self._q_threshold: float = 1.0
        self.setSortRole(Qt.UserRole)

    def set_column_indices(self, condition_col: int, type_col: int,
                           p_col: int, q_col: int) -> None:
        self._condition_col = condition_col
        self._type_col = type_col
        self._p_col = p_col
        self._q_col = q_col

    def set_filters(self, condition_filter, type_filter,
                    p_threshold: float, q_threshold: float) -> None:
        self._condition_filter = condition_filter
        self._type_filter = type_filter
        self._p_threshold = p_threshold
        self._q_threshold = q_threshold

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()

        def cell_text(col):
            if col < 0:
                return None
            item = model.item(source_row, col)
            return item.text() if item else None

        def cell_float(col):
            if col < 0:
                return None
            item = model.item(source_row, col)
            if item is None:
                return None
            try:
                return float(item.data(Qt.UserRole))
            except (TypeError, ValueError):
                return None

        # Condition filter
        if self._condition_filter is not None:
            val = cell_text(self._condition_col)
            if val is not None and val not in self._condition_filter:
                return False

        # Type filter
        if self._type_filter is not None:
            val = cell_text(self._type_col)
            if val is not None and val not in self._type_filter:
                return False

        # P-value threshold (strict less-than)
        p_val = cell_float(self._p_col)
        if p_val is not None and p_val >= self._p_threshold:
            return False

        # Q-value threshold (strict less-than)
        q_val = cell_float(self._q_col)
        if q_val is not None and q_val >= self._q_threshold:
            return False

        return True


def _pick_values_dialog(parent: QWidget, title: str, all_values: List[str],
                        current: Optional[set] = None) -> Optional[set]:
    """Show a dialog with checkboxes for each value; return selected set or None if cancelled."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setMinimumWidth(260)
    layout = QVBoxLayout(dlg)

    scroll = QScrollArea(dlg)
    scroll.setWidgetResizable(True)
    container = QWidget()
    cb_layout = QVBoxLayout(container)
    cb_layout.setSpacing(2)

    checkboxes: List[QCheckBox] = []
    for val in all_values:
        cb = QCheckBox(val, container)
        cb.setChecked(current is None or val in current)
        cb_layout.addWidget(cb)
        checkboxes.append(cb)

    scroll.setWidget(container)
    layout.addWidget(scroll)

    # Select all / none helper buttons
    btn_row = QWidget(dlg)
    btn_row_layout = QHBoxLayout(btn_row)
    btn_row_layout.setContentsMargins(0, 0, 0, 0)
    sel_all = QPushButton("Select All", btn_row)
    sel_none = QPushButton("Select None", btn_row)
    sel_all.clicked.connect(lambda: [cb.setChecked(True) for cb in checkboxes])
    sel_none.clicked.connect(lambda: [cb.setChecked(False) for cb in checkboxes])
    btn_row_layout.addWidget(sel_all)
    btn_row_layout.addWidget(sel_none)
    layout.addWidget(btn_row)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec() != QDialog.Accepted:
        return None  # cancelled
    return {cb.text() for cb in checkboxes if cb.isChecked()}


class NIRSviewIRWindow(QMainWindow):
    def __init__(self, dataset: Any, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("NIRSviewIR")
        self.resize(1200, 700)

        self.dataset = dataset
        self.rec = None
        self.data = None
        self.mllines = []
        self.line0 = None
        self.line1 = None
        self.selected_channels = [0]
        self._ts_lines = []
        self._stim_handles = []
        self._ts_legends = {"channels": None, "stim": None}
        self._pipeline_modules: List[cedalion_module] = []  # current stored pipeline
        self._pipeline_dirty = True   # False once the pipeline ran on the current data
        self._current_stats_df = None       # last drawn statistics DataFrame (for export)
        self._current_stats_obj = None      # the Statistics object currently displayed
        self._stats_condition_filter: Optional[set] = None  # None = all
        self._stats_type_filter: Optional[set] = None        # None = all

        # ---------------------------------------------------------------- menus
        # Use setNativeMenuBar(False) so the menu bar is embedded inside the
        # window widget rather than being absorbed into the macOS system bar.
        menu_bar = QMenuBar(self)
        menu_bar.setNativeMenuBar(False)

        # File menu
        file_menu = menu_bar.addMenu("File")
        sessions_menu = file_menu.addMenu("Sessions")
        sessions_menu.addAction("New Analysis Session")
        sessions_menu.addAction("Load Analysis Session")
        sessions_menu.addAction("Save Analysis Session")
        files_menu = file_menu.addMenu("Files")
        files_menu.addAction("Load File")

        # Edit Data menu
        edit_data_menu = menu_bar.addMenu("Edit Data")
        edit_data_menu.addAction("Edit Demographics").triggered.connect(
            self.edit_demographics
        )
        edit_data_menu.addAction("Edit Stimulus Timing").triggered.connect(
            self._edit_stimulus_timing
        )

        # Analysis menu
        analysis_menu = menu_bar.addMenu("Analysis")
        self._run_pipeline_action = analysis_menu.addAction("Run Pipeline")
        self._run_pipeline_action.setEnabled(False)
        analysis_menu.addAction("Edit Pipeline").triggered.connect(self._edit_pipeline)
        analysis_menu.addAction("Save Pipeline").triggered.connect(self._save_pipeline)
        analysis_menu.addAction("Load Pipeline").triggered.connect(self._load_pipeline)

        # Help menu
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("About").triggered.connect(self._show_about)

        # ---------------------------------------------------------------- central
        central = QWidget(self)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.addWidget(menu_bar)

        content_widget = QWidget(central)
        layout = QHBoxLayout(content_widget)
        outer_layout.addWidget(content_widget, stretch=1)

        left = QWidget(content_widget)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Recordings"))
        self.file_tree = QTreeWidget(left)
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._populate_file_tree()
        left_layout.addWidget(self.file_tree)
        left_layout.addWidget(QLabel("Selected recording summary"))
        self.summary_tabs = QTabWidget(left)
        self.summary_tabs.setMinimumHeight(170)
        self.demographics_panel = QTextEdit(left)
        self.data_quality_panel = QTextEdit(left)
        self.stimulus_panel = QTextEdit(left)
        for panel, title in (
            (self.demographics_panel, "Demographics"),
            (self.data_quality_panel, "Data Quality"),
            (self.stimulus_panel, "Stimulus Information"),
        ):
            panel.setReadOnly(True)
            self.summary_tabs.addTab(panel, title)
        left_layout.addWidget(self.summary_tabs)
        left_layout.addWidget(QLabel("Data type"))
        self.type_selector = QComboBox(left)
        left_layout.addWidget(self.type_selector)
        left.setMaximumWidth(280)
        layout.addWidget(left)

        # Right-clicking anywhere in the recordings panel offers the
        # demographics editor.
        for widget in (left, self.file_tree):
            widget.setContextMenuPolicy(Qt.CustomContextMenu)
            widget.customContextMenuRequested.connect(self._show_file_list_menu)

        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax_probe = self.figure.add_subplot(1, 2, 1)
        self.ax_ts = self.figure.add_subplot(1, 2, 2)
        self.ax_probe.set_position([0.02, 0.05, 0.22, 0.9])
        self.ax_ts.set_position([0.32, 0.12, 0.64, 0.8])

        right = QWidget(content_widget)
        right_layout = QVBoxLayout(right)

        run_bar = QWidget(right)
        run_bar_layout = QHBoxLayout(run_bar)
        run_bar_layout.setContentsMargins(0, 0, 0, 0)
        run_bar_layout.addStretch()
        self._run_pipeline_button = QPushButton("Run Pipeline", run_bar)
        self._run_pipeline_button.clicked.connect(self._run_pipeline)
        self._run_pipeline_button.setVisible(False)
        run_bar_layout.addWidget(self._run_pipeline_button)
        right_layout.addWidget(run_bar)

        # The button mirrors the Analysis > Run Pipeline action's availability.
        self._run_pipeline_action.changed.connect(
            lambda: self._run_pipeline_button.setVisible(
                self._run_pipeline_action.isEnabled()
            )
        )

        # QStackedWidget: page 0 = timeseries view, page 1 = statistics view
        self._right_stack = QStackedWidget(right)

        # --- Page 0: timeseries (existing canvas) ---
        ts_page = QWidget()
        ts_layout = QVBoxLayout(ts_page)
        ts_layout.setContentsMargins(0, 0, 0, 0)
        ts_layout.addWidget(self.canvas)
        self._right_stack.addWidget(ts_page)   # index 0

        # --- Page 1: statistics view ---
        stats_page = QWidget()
        stats_layout = QHBoxLayout(stats_page)
        stats_layout.setContentsMargins(0, 0, 0, 0)

        # Left half: stats figure canvas
        self._stats_figure = Figure(figsize=(7, 6))
        self._stats_canvas = FigureCanvas(self._stats_figure)
        stats_layout.addWidget(self._stats_canvas, stretch=1)

        # Right half: filter controls + table + export button
        stats_right = QWidget(stats_page)
        stats_right_layout = QVBoxLayout(stats_right)
        stats_right_layout.setContentsMargins(0, 0, 0, 0)
        stats_right_layout.setSpacing(4)

        # --- Filter bar ---
        filter_bar = QWidget(stats_right)
        filter_bar_layout = QHBoxLayout(filter_bar)
        filter_bar_layout.setContentsMargins(2, 2, 2, 2)
        filter_bar_layout.setSpacing(6)

        filter_bar_layout.addWidget(QLabel("Condition:", filter_bar))
        self._filter_condition_btn = QPushButton("All", filter_bar)
        self._filter_condition_btn.setMaximumWidth(110)
        self._filter_condition_btn.clicked.connect(self._pick_condition_filter)
        filter_bar_layout.addWidget(self._filter_condition_btn)

        filter_bar_layout.addWidget(QLabel("Type:", filter_bar))
        self._filter_type_btn = QPushButton("All", filter_bar)
        self._filter_type_btn.setMaximumWidth(110)
        self._filter_type_btn.clicked.connect(self._pick_type_filter)
        filter_bar_layout.addWidget(self._filter_type_btn)

        filter_bar_layout.addWidget(QLabel("P<", filter_bar))
        self._filter_p_spin = QDoubleSpinBox(filter_bar)
        self._filter_p_spin.setRange(0.0, 1.0)
        self._filter_p_spin.setSingleStep(0.01)
        self._filter_p_spin.setDecimals(4)
        self._filter_p_spin.setValue(1.0)
        self._filter_p_spin.setMaximumWidth(80)
        self._filter_p_spin.valueChanged.connect(self._apply_stats_filter)
        filter_bar_layout.addWidget(self._filter_p_spin)

        filter_bar_layout.addWidget(QLabel("Q<", filter_bar))
        self._filter_q_spin = QDoubleSpinBox(filter_bar)
        self._filter_q_spin.setRange(0.0, 1.0)
        self._filter_q_spin.setSingleStep(0.01)
        self._filter_q_spin.setDecimals(4)
        self._filter_q_spin.setValue(1.0)
        self._filter_q_spin.setMaximumWidth(80)
        self._filter_q_spin.valueChanged.connect(self._apply_stats_filter)
        filter_bar_layout.addWidget(self._filter_q_spin)

        filter_bar_layout.addWidget(QLabel("Display:", filter_bar))
        self._filter_vartype_combo = QComboBox(filter_bar)
        self._filter_vartype_combo.addItems(["tstat", "beta"])
        self._filter_vartype_combo.setMaximumWidth(80)
        self._filter_vartype_combo.currentTextChanged.connect(self._redraw_stats_figure)
        filter_bar_layout.addWidget(self._filter_vartype_combo)

        filter_bar_layout.addStretch()
        stats_right_layout.addWidget(filter_bar)

        # --- Table view backed by QStandardItemModel + proxy for sorting/filtering ---
        self._stats_model = QStandardItemModel(0, 0, self)
        self._stats_proxy = _StatsFilterProxy(self)
        self._stats_proxy.setSourceModel(self._stats_model)

        self._stats_table_view = QTableView(stats_right)
        self._stats_table_view.setModel(self._stats_proxy)
        self._stats_table_view.setSortingEnabled(True)
        self._stats_table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._stats_table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._stats_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._stats_table_view.horizontalHeader().setStretchLastSection(True)
        self._stats_table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self._stats_table_view.customContextMenuRequested.connect(self._stats_table_context_menu)
        stats_right_layout.addWidget(self._stats_table_view, stretch=1)

        # --- Global FDR checkbox ---
        self._global_fdr_checkbox = QCheckBox("Global FDR correction", stats_right)
        self._global_fdr_checkbox.setChecked(True)
        self._global_fdr_checkbox.stateChanged.connect(self._on_global_fdr_changed)
        stats_right_layout.addWidget(self._global_fdr_checkbox)

        # --- Export buttons ---
        export_row = QWidget(stats_right)
        export_row_layout = QHBoxLayout(export_row)
        export_row_layout.setContentsMargins(0, 0, 0, 0)
        self._export_stats_button = QPushButton("Export Table to Excel…", export_row)
        self._export_stats_button.clicked.connect(lambda: self._export_stats_table(filtered_only=False))
        export_row_layout.addWidget(self._export_stats_button)
        self._export_stats_filtered_button = QPushButton("Export Filtered Rows…", export_row)
        self._export_stats_filtered_button.clicked.connect(lambda: self._export_stats_table(filtered_only=True))
        export_row_layout.addWidget(self._export_stats_filtered_button)
        stats_right_layout.addWidget(export_row)

        stats_layout.addWidget(stats_right, stretch=1)

        self._right_stack.addWidget(stats_page)   # index 1

        right_layout.addWidget(self._right_stack)
        layout.addWidget(right, stretch=1)

        self.setCentralWidget(central)

        # ---------------------------------------------------------------- status bar
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready.")

        self.file_tree.itemSelectionChanged.connect(self._on_file_selection_changed)
        self.type_selector.currentTextChanged.connect(self._on_type_changed)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._run_pipeline_action.triggered.connect(self._run_pipeline)

        if len(dataset.dataset) > 0:
            self._select_row(0)
            self._on_file_changed(0)
        else:
            self._update_summary(-1)

    # ------------------------------------------------------------------ slots

    # --------------------------------------------------------- pipeline helpers

    def _set_status(self, msg: str) -> None:
        self._status_bar.showMessage(msg)
        QApplication.processEvents()

    def _mark_dataset_changed(self) -> None:
        """Re-enable Run Pipeline after the data or the pipeline changed."""
        self._pipeline_dirty = True
        self._update_run_pipeline_enabled()

    def _update_run_pipeline_enabled(self) -> None:
        self._run_pipeline_action.setEnabled(
            bool(self._pipeline_modules) and self._pipeline_dirty
        )

    def _show_about(self) -> None:
        """Display the About dialog with author/version/build info."""
        dlg = QDialog(self)
        dlg.setWindowTitle("About pyBrainAnalyzIR")
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)
        text = QTextEdit(dlg)
        text.setReadOnly(True)
        text.setPlainText(get_about_text())
        text.setMinimumHeight(200)
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    def _edit_stimulus_timing(self) -> None:
        """Open the Stimulus Manager and refresh the view with its result."""
        from pyBrainAnalyzIR.vis.stimulus_manager import edit_stimulus_events

        if not getattr(self.dataset, "dataset", []):
            QMessageBox.information(self, "Edit Stimulus Timing", "No recordings loaded.")
            return

        self._set_status("Editing stimulus timing…")
        updated = edit_stimulus_events(self.dataset, self)
        self._refresh_labels(preferred_row=self._current_row())
        if updated:
            self._mark_dataset_changed()
        self._set_status(
            "Stimulus timing updated." if updated else "Stimulus editing cancelled."
        )

    def _edit_pipeline(self) -> None:
        self._set_status("Opening Pipeline Manager…")
        dialog = PipelineManagerDialog(
            initial_modules=self._pipeline_modules if self._pipeline_modules else None,
            parent=self,
        )
        if dialog.exec():
            self._pipeline_modules = list(dialog.pipeline_modules)
            n = len(self._pipeline_modules)
            self._mark_dataset_changed()
            self._set_status(
                f"Pipeline updated: {n} module(s)." if n else "Pipeline cleared."
            )
        else:
            self._set_status("Pipeline edit cancelled.")

    def _save_pipeline(self) -> None:
        if not self._pipeline_modules:
            QMessageBox.information(self, "Save Pipeline", "No pipeline to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", "", "Pipeline JSON (*.json);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            json_str = pipeline_to_json(self._pipeline_modules)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            self._set_status(f"Pipeline saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Pipeline", f"Failed to save pipeline:\n{exc}")
            self._set_status("Pipeline save failed.")

    def _load_pipeline(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pipeline", "", "Pipeline JSON (*.json);;All Files (*)"
        )
        if not path:
            return
        self._set_status(f"Loading pipeline from {path}…")
        try:
            with open(path, "r", encoding="utf-8") as f:
                json_str = f.read()
            modules = pipeline_from_json(json_str)
            self._pipeline_modules = modules
            self._mark_dataset_changed()
            self._set_status(f"Pipeline loaded: {len(modules)} module(s) from {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Pipeline", f"Failed to load pipeline:\n{exc}")
            self._set_status("Pipeline load failed.")

    def _run_pipeline(self) -> None:
        if not self._pipeline_modules:
            QMessageBox.information(self, "Run Pipeline", "No pipeline has been configured.")
            return
        recordings = getattr(self.dataset, "dataset", [])
        if not recordings:
            QMessageBox.information(self, "Run Pipeline", "No recordings loaded in dataset.")
            return

        # Build the linked pipeline tail
        modules_copy = [copy.deepcopy(m) for m in self._pipeline_modules]
        previous = None
        for m in modules_copy:
            m.previous_job = previous
            previous = m
        job = previous

        self._set_status("Running pipeline…")
        try:
            result = job.run(self.dataset)
            if result is not None:
                self.dataset = result
            self._set_status("Pipeline complete. Refreshing display…")
            self._refresh_labels(preferred_row=self._current_row())
            self._pipeline_dirty = False
            self._update_run_pipeline_enabled()
            self._set_status("Pipeline complete.")
        except Exception as exc:
            QMessageBox.critical(self, "Run Pipeline", f"Pipeline failed:\n{exc}")
            self._set_status("Pipeline run failed.")

    # --------------------------------------------------------- file list menu

    def _show_file_list_menu(self, pos) -> None:
        widget = self.sender() or self.file_tree
        menu = QMenu(self)
        edit_action = menu.addAction("Edit Demographics...")
        remove_action = None
        target_row = -1
        if widget is self.file_tree:
            clicked = self.file_tree.itemAt(pos)
            if clicked is not None:
                self.file_tree.setCurrentItem(clicked)
                # Only leaves map to a recording; grouping rows have no index.
                target_row = self._current_row()
                if target_row >= 0:
                    remove_action = menu.addAction("Remove Recording...")

        chosen = menu.exec(widget.mapToGlobal(pos))
        if chosen == edit_action:
            self.edit_demographics()
        elif remove_action is not None and chosen == remove_action:
            self._remove_recording(target_row)

    def edit_demographics(self) -> None:
        """Edit the dataset demographics table and write the result back."""
        recordings = getattr(self.dataset, "dataset", [])
        if not recordings:
            QMessageBox.information(
                self, "Edit Demographics", "The dataset contains no recordings."
            )
            return

        # Pass the dataset itself (not a plain list of rows) so dataset-aware
        # actions such as "Add Additional BIDS Variables" behave exactly as they
        # do when the manager is launched from a notebook.
        editor = DemographicsManager(self.dataset, self)
        editor.setWindowTitle("Edit Demographics")
        editor.exec()

        # `DemographicsManager` writes back to `self.dataset` on both Done and
        # window close, so the edits are applied either way.
        self._refresh_labels(preferred_row=self._current_row())
        self._mark_dataset_changed()

    def _apply_demographics(self, rows: List[Dict[str, Any]]) -> None:
        """Write edited demographics rows back onto the recordings.

        `edit_demographics` now delegates this to `DemographicsManager`; this is
        kept for callers that already hold a plain list of rows.
        """
        recordings = self.dataset.dataset
        if len(rows) != len(recordings):
            QMessageBox.warning(
                self, "Edit Demographics",
                f"Edited table has {len(rows)} row(s) but the dataset has "
                f"{len(recordings)} recording(s); demographics were not applied.",
            )
            return

        for rec, row in zip(recordings, rows):
            old = getattr(rec, "meta_data", None)
            new = OrderedDict()
            for key, value in row.items():
                # Keep the original object when the displayed text is unchanged
                # so non-string meta data isn't flattened by a round trip.
                if old is not None and key in old and \
                        _to_display_str(old[key]) == _to_display_str(value):
                    new[key] = old[key]
                else:
                    new[key] = value
            if hasattr(old, "clear") and hasattr(old, "update"):
                old.clear()
                old.update(new)
            else:
                rec.meta_data = new

        self._refresh_labels(preferred_row=self._current_row())

    def _populate_file_tree(self) -> None:
        """Rebuild the recordings tree grouped by group / subject / session.

        Levels that no recording provides are skipped entirely, and when no
        subject naming is found at all the tree degrades to one flat branch per
        file. Each leaf carries its recording index in `RECORDING_INDEX_ROLE`.
        """
        recordings = list(getattr(self.dataset, "dataset", []))
        self.file_tree.clear()
        if not recordings:
            return

        labels = _build_labels(self.dataset)

        levels = []
        if any(_group_label(rec) is not None for rec in recordings):
            levels.append(_group_label)
        if any(_subject_label(rec) is not None for rec in recordings):
            levels.append(_subject_label)
            if any(_session_label(rec) is not None for rec in recordings):
                levels.append(_session_label)

        for index, rec in enumerate(recordings):
            parent = self.file_tree.invisibleRootItem()
            for level in levels:
                parent = self._child_named(parent, level(rec) or "<unknown>")
            label = labels[index] if index < len(labels) else f"recording {index}"
            leaf = QTreeWidgetItem(parent, [label])
            leaf.setData(0, RECORDING_INDEX_ROLE, index)

        self.file_tree.expandAll()

    @staticmethod
    def _child_named(parent: QTreeWidgetItem, name: str) -> QTreeWidgetItem:
        """Find (case-insensitively) or create a grouping child under `parent`."""
        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.data(0, RECORDING_INDEX_ROLE) is None and \
                    child.text(0).strip().lower() == name.strip().lower():
                return child
        return QTreeWidgetItem(parent, [name])

    def _item_for_row(self, row: int) -> Optional[QTreeWidgetItem]:
        """The leaf item representing recording index `row`."""
        iterator = QTreeWidgetItemIterator(self.file_tree)
        while iterator.value():
            item = iterator.value()
            if item.data(0, RECORDING_INDEX_ROLE) == row:
                return item
            iterator += 1
        return None

    def _current_row(self) -> int:
        """Recording index of the selected leaf, or -1 when none is selected."""
        item = self.file_tree.currentItem()
        if item is None:
            return -1
        value = item.data(0, RECORDING_INDEX_ROLE)
        return -1 if value is None else int(value)

    def _select_row(self, row: int) -> None:
        item = self._item_for_row(row)
        if item is not None:
            self.file_tree.setCurrentItem(item)

    def _remove_recording(self, row: int) -> None:
        recordings = getattr(self.dataset, "dataset", None)
        if recordings is None or row < 0 or row >= len(recordings):
            return

        label_item = self._item_for_row(row)
        label = label_item.text(0) if label_item is not None else f"recording {row}"
        confirm = QMessageBox.question(
            self,
            "Remove Recording",
            f"Remove '{label}' from the dataset?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        del recordings[row]
        self._mark_dataset_changed()
        self._refresh_labels(preferred_row=min(row, len(recordings) - 1))

    def _recording_metrics(self, rec: Any, key: Optional[str]) -> tuple[Optional[float], Optional[int]]:
        series = getattr(rec, "timeseries", None)
        if series is None:
            return None, None
        keys = list(series.keys())
        if not keys:
            return None, None
        selected_key = key if key in keys else keys[0]
        data = series[selected_key]

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
            for panel in (self.demographics_panel, self.data_quality_panel,
                          self.stimulus_panel):
                panel.setPlainText("No recording selected.")
            return

        rec = recordings[row]
        meta = getattr(rec, "meta_data", None) or {}
        if meta:
            demographic_lines = [f"  - {k}: {_to_display_str(v)}" for k, v in meta.items()]
        else:
            demographic_lines = ["  - <none>"]

        stim_names: List[str] = []
        stim_counts: Dict[str, int] = {}
        stim = getattr(rec, "stim", None)
        if stim is not None and "trial_type" in stim:
            for name in stim["trial_type"].dropna().to_list():
                name_str = str(name)
                if name_str not in stim_names:
                    stim_names.append(name_str)
                stim_counts[name_str] = stim_counts.get(name_str, 0) + 1

        duration, n_channels = self._recording_metrics(rec, key)
        duration_text = f"{duration:.2f}" if duration is not None else "N/A"
        channels_text = str(n_channels) if n_channels is not None else "N/A"
        stim_text = (
            "\n".join(f"  - {name} ({stim_counts[name]})" for name in stim_names)
            if stim_names else "  - <none>"
        )

        self.demographics_panel.setPlainText(
            "Demographics:\n" + "\n".join(demographic_lines)
        )
        self.data_quality_panel.setPlainText(
            f"Recording length (s): {duration_text}\n"
            f"Number of channels: {channels_text}"
        )
        self.stimulus_panel.setPlainText("Stimulus events:\n" + stim_text)

    def _refresh_labels(self, preferred_row: int = -1) -> None:
        recordings = getattr(self.dataset, "dataset", [])
        self.file_tree.blockSignals(True)
        self._populate_file_tree()
        self.file_tree.blockSignals(False)
        if recordings:
            row = max(preferred_row, 0)
            row = min(row, len(recordings) - 1)
            self._select_row(row)
            self._on_file_changed(row)
        else:
            self.rec = None
            self.data = None
            self.type_selector.blockSignals(True)
            self.type_selector.clear()
            self.type_selector.blockSignals(False)
            self.figure.clear()
            self.ax_probe = self.figure.add_subplot(1, 2, 1)
            self.ax_ts = self.figure.add_subplot(1, 2, 2)
            self.ax_probe.set_position([0.02, 0.05, 0.22, 0.9])
            self.ax_ts.set_position([0.32, 0.12, 0.64, 0.8])
            self.canvas.draw_idle()
            self._update_summary(-1)

    def _on_file_selection_changed(self) -> None:
        row = self._current_row()
        # Clicking a group/subject/session row is just navigation, so keep the
        # current plot instead of clearing it.
        if row < 0:
            return
        self._on_file_changed(row)

    def _on_file_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.dataset.dataset):
            self._update_summary(-1)
            return
        self.rec = self.dataset.dataset[row]
        keys = list(self.rec.timeseries.keys())

        self.type_selector.blockSignals(True)
        self.type_selector.clear()
        self.type_selector.addItems(keys)
        self.type_selector.blockSignals(False)

        if keys:
            self.type_selector.setCurrentIndex(0)
            self._draw(keys[0])
            self._update_summary(row, keys[0])
        else:
            self.figure.clear()
            self.canvas.draw_idle()
            self._update_summary(row, None)

    def _on_type_changed(self, key: str) -> None:
        if self.rec is not None and key:
            self._draw(key)
            self._update_summary(self._current_row(), key)

    def _on_click(self, event) -> None:
        if self._right_stack.currentIndex() != 0:
            return
        if event.inaxes is not self.ax_probe or not self.mllines:
            return
        distances = []
        for line in self.mllines:
            xdata, ydata = line.get_data()
            linelength = ((xdata[0] - xdata[-1]) ** 2 + (ydata[0] - ydata[-1]) ** 2) ** 0.5
            lineseg1 = ((xdata[0] - event.xdata) ** 2 + (ydata[0] - event.ydata) ** 2) ** 0.5
            lineseg2 = ((xdata[-1] - event.xdata) ** 2 + (ydata[-1] - event.ydata) ** 2) ** 0.5
            distances.append(abs(linelength - lineseg1 - lineseg2))

        clicked = int(np.argmin(distances))
        if event.button == 3:   # right click adds to (or removes from) the selection
            if clicked in self.selected_channels:
                if len(self.selected_channels) > 1:
                    self.selected_channels.remove(clicked)
            else:
                self.selected_channels.append(clicked)
        else:
            self.selected_channels = [clicked]

        self._update_timecourse()
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- drawing

    def _channel_components(self, selchann):
        data = self.data
        if hasattr(data, "wavelength"):
            wl = to_string(data.wavelength)
            return [(data.sel(channel=selchann, wavelength=wl[-1]), "-", f"{wl[-1]}nm"),
                    (data.sel(channel=selchann, wavelength=wl[0]), "--", f"{wl[0]}nm")]
        return [(data.sel(channel=selchann, chromo="HbO"), "-", "HbO"),
                (data.sel(channel=selchann, chromo="HbR"), "--", "HbR")]

    def _update_timecourse(self) -> None:
        data = self.data
        if data is None or not self.mllines:
            return

        for line in self._ts_lines:
            line.remove()
        self._ts_lines = []

        multi = len(self.selected_channels) > 1
        handles = []
        names = []
        for k, index in enumerate(self.selected_channels):
            selchann = to_string(data.channel[index])
            names.append(selchann)
            chancolor = linecolors[(k + 1) % len(linecolors)] if multi else None
            for j, (series, style, comp) in enumerate(self._channel_components(selchann)):
                color = chancolor if multi else ("r" if j == 0 else "b")
                label = f"{selchann} {comp}" if multi else comp
                line, = self.ax_ts.plot(data.time, series, linestyle=style,
                                        color=color, label=label)
                self._ts_lines.append(line)
                handles.append(line)

        for line in self.mllines:
            line.set_color("k")
        for k, index in enumerate(self.selected_channels):
            self.mllines[index].set_color(
                linecolors[(k + 1) % len(linecolors)] if multi else "r"
            )

        for legend in self._ts_legends.values():
            if legend is not None:
                legend.remove()
        self._ts_legends["channels"] = self.ax_ts.legend(
            handles=handles, loc="upper right", fontsize=8
        )
        if self._stim_handles:
            self._ts_legends["stim"] = self.ax_ts.legend(
                handles=self._stim_handles, loc="lower right"
            )
            self.ax_ts.add_artist(self._ts_legends["channels"])

        if multi:
            self.ax_ts.set_title(f"{len(names)} channels: " + ", ".join(names), fontsize=9)
        else:
            self.ax_ts.set_title(names[0] if names else "")

    def _draw(self, key: str, show_stim: bool = True) -> None:
        rec = self.rec
        self.data = rec[key]
        data = self.data

        # Dispatch to the statistics viewer when the data object is a Statistics instance.
        if isinstance(data, Statistics):
            self._draw_stats(data)
            return

        # Switch to timeseries page
        self._right_stack.setCurrentIndex(0)

        geo2d = rec.geo3d if len(rec.geo2d) == 0 else rec.geo2d

        self.ax_probe.clear()
        self.ax_ts.clear()
        self.mllines = []

        for chan in data.channel:
            sdstr = to_string(chan)
            source = sdstr[: sdstr.find("D")]
            detector = sdstr[sdstr.find("D"):]
            srcpos = geo2d[geo2d.label == source].to_numpy()
            detpos = geo2d[geo2d.label == detector].to_numpy()
            ll, = self.ax_probe.plot(
                [srcpos[0, 0], detpos[0, 0]], [srcpos[0, 1], detpos[0, 1]], "k"
            )
            self.ax_probe.text(srcpos[0, 0], srcpos[0, 1], source, fontsize=10,
                               ha="center", va="center")
            self.ax_probe.text(detpos[0, 0], detpos[0, 1], detector, fontsize=10,
                               ha="center", va="center")
            self.mllines.append(ll)

        if not self.mllines:
            self.canvas.draw_idle()
            return

        self.selected_channels = [0]
        self._ts_lines = []
        self._stim_handles = []
        self._ts_legends = {"channels": None, "stim": None}

        optodes = geo2d.to_numpy()
        s = (optodes.max() - optodes.min()) / 10
        self.ax_probe.set_ylim(optodes[:, 1].min() - s, optodes[:, 1].max() + s)
        self.ax_probe.set_xlim(optodes[:, 0].min() - s, optodes[:, 0].max() + s)
        self.ax_probe.set_axis_off()

        vmin = float(data.to_numpy().min())
        vmax = float(data.to_numpy().max())

        stim = getattr(rec, "stim", None)
        if show_stim and stim is not None and len(stim) > 0:
            cond_names = np.unique(stim["trial_type"].to_numpy())
            for _, row in stim.iterrows():
                thiscolor = linecolors[np.argwhere(row["trial_type"] == cond_names)[0]][0]
                rectangle = patches.Rectangle(
                    (row["onset"], vmin), row["duration"], vmax - vmin,
                    facecolor=thiscolor, edgecolor=thiscolor, linewidth=2, alpha=0.1)
                self.ax_ts.add_patch(rectangle)

            for index in range(cond_names.shape[0]):
                l, = self.ax_ts.plot([data.time[0], data.time[0]], [vmin, vmin],
                                     color=linecolors[index], label=cond_names[index])
                self._stim_handles.append(l)

        self._update_timecourse()

        self.ax_ts.set_xlabel("time / s")
        self.ax_ts.set_ylabel(f"{key} / a.u.")
        self.ax_ts.set_ylim(vmin, vmax)
        self.ax_ts.set_xlim(float(data.time.min()), float(data.time.max()))

        self.canvas.draw_idle()

    # ------------------------------------------------------------ statistics view

    def _draw_stats(self, stats: Statistics) -> None:
        """Render the Statistics view: channel map figure + sortable/filterable results table."""
        self._right_stack.setCurrentIndex(1)
        self._set_status("Rendering statistics…")

        # Store stats object for re-draw on filter change
        self._current_stats_obj = stats

        # Reset filters when loading new stats data
        self._stats_condition_filter = None
        self._stats_type_filter = None
        self._filter_condition_btn.setText("All")
        self._filter_type_btn.setText("All")
        self._filter_p_spin.blockSignals(True)
        self._filter_q_spin.blockSignals(True)
        self._filter_p_spin.setValue(1.0)
        self._filter_q_spin.setValue(1.0)
        self._filter_p_spin.blockSignals(False)
        self._filter_q_spin.blockSignals(False)

        # --- table (load full unfiltered data first) --------------------------
        try:
            df = stats.table()
            self._current_stats_df = df
            self._populate_stats_model(df)
            self._export_stats_button.setEnabled(True)
            self._export_stats_filtered_button.setEnabled(True)
        except Exception as exc:
            self._current_stats_df = None
            self._stats_model.clear()
            self._stats_model.setHorizontalHeaderLabels(["Error"])
            self._stats_model.appendRow([QStandardItem(str(exc))])
            self._export_stats_button.setEnabled(False)
            self._export_stats_filtered_button.setEnabled(False)

        # --- figure (no filter active yet, draw all) -------------------------
        self._redraw_stats_figure()
        self._set_status("Statistics rendered.")

    def _redraw_stats_figure(self) -> None:
        """Re-render the stats figure using the current filter state."""
        stats = self._current_stats_obj
        if stats is None:
            return

        # Determine condnames from Condition filter
        if self._stats_condition_filter is not None:
            condnames = sorted(list(self._stats_condition_filter))
        else:
            condnames = None  # stats.draw() will use all

        # Determine types from Type filter
        if self._stats_type_filter is not None:
            types_arg = sorted(list(self._stats_type_filter))
        else:
            types_arg = None

        # Determine threshold string: prefer Q if Q is filtered, else P, else none
        p_thresh = self._filter_p_spin.value()
        q_thresh = self._filter_q_spin.value()
        p_filtered = p_thresh < 1.0
        q_filtered = q_thresh < 1.0

        if q_filtered and not p_filtered:
            thresh_str = f"q<{q_thresh:.4g}"
        elif p_filtered and not q_filtered:
            thresh_str = f"p<{p_thresh:.4g}"
        elif p_filtered and q_filtered:
            # Both filtered: derive a p threshold from the max p-value in filtered rows
            filtered_df = self._get_filtered_df()
            if filtered_df is not None and "P-values" in filtered_df.columns and len(filtered_df) > 0:
                max_p = float(filtered_df["P-values"].max())
                thresh_str = f"p<{max_p:.4g}"
            else:
                thresh_str = f"p<{p_thresh:.4g}"
        else:
            thresh_str = "p<1.0"  # effectively no threshold

        vartype = self._filter_vartype_combo.currentText()
        fdr_full = self._global_fdr_checkbox.isChecked()

        try:
            plt.close("all")
            stats.draw(
                vartype=vartype,
                thresh=thresh_str,
                condnames=condnames,
                types=types_arg,
                fdr_correct_full=fdr_full,
            )
            src_fig = plt.gcf()
            self._stats_figure.clear()
            src_fig.canvas.draw()
            buf = io.BytesIO()
            src_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            plt.close(src_fig)
            img = mpimg.imread(buf)
            ax = self._stats_figure.add_subplot(1, 1, 1)
            ax.imshow(img)
            ax.axis("off")
            self._stats_canvas.draw_idle()
        except Exception as exc:
            self._stats_figure.clear()
            ax = self._stats_figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error rendering stats figure:\n{exc}",
                    ha="center", va="center", transform=ax.transAxes, wrap=True)
            ax.axis("off")
            self._stats_canvas.draw_idle()

    def _populate_stats_model(self, df) -> None:
        """Load DataFrame into QStandardItemModel; numeric columns use float user-data for sorting."""
        self._stats_model.clear()
        self._stats_model.setHorizontalHeaderLabels(list(df.columns))
        cols = list(df.columns)
        for _r_idx, row in df.iterrows():
            items = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    item = QStandardItem()
                    item.setData(f"{val:.4g}", Qt.DisplayRole)
                    item.setData(float(val), Qt.UserRole)
                else:
                    item = QStandardItem(str(val))
                    item.setData(str(val), Qt.UserRole)
                item.setEditable(False)
                items.append(item)
            self._stats_model.appendRow(items)

        col_map = {c.lower(): i for i, c in enumerate(cols)}
        self._stats_proxy.set_column_indices(
            condition_col=col_map.get("condition", -1),
            type_col=col_map.get("type", -1),
            p_col=col_map.get("p-values", col_map.get("p-value", -1)),
            q_col=col_map.get("q-values", col_map.get("q-value", -1)),
        )
        self._stats_table_view.sortByColumn(0, Qt.AscendingOrder)

    def _recompute_q_for_filtered(self) -> None:
        """When Global FDR is OFF (local) or Condition/Type filters are active with global FDR ON,
        recompute Q-values on the reduced set of rows and update the model."""
        if self._current_stats_df is None:
            return
        df = self._current_stats_df.copy()
        if "P-values" not in df.columns or "Q-values" not in df.columns:
            return

        # Determine which rows are in the filtered set (Condition + Type mask only,
        # not the p/q threshold — that comes after)
        cond_mask = pd.Series([True] * len(df), index=df.index)
        type_mask = pd.Series([True] * len(df), index=df.index)
        if self._stats_condition_filter is not None and "Condition" in df.columns:
            cond_mask = df["Condition"].astype(str).isin(self._stats_condition_filter)
        if self._stats_type_filter is not None and "Type" in df.columns:
            type_mask = df["Type"].astype(str).isin(self._stats_type_filter)
        subset_mask = cond_mask & type_mask

        q_new = df["Q-values"].values.copy()  # default: keep original q

        global_fdr = self._global_fdr_checkbox.isChecked()
        if not global_fdr or subset_mask.sum() < len(df):
            # Recompute Q only on the subset
            p_sub = df.loc[subset_mask, "P-values"].values
            if len(p_sub) > 1:
                _, q_sub, _, _ = multipletests(p_sub, alpha=0.05, method='fdr_bh')
            elif len(p_sub) == 1:
                q_sub = p_sub.copy()
            else:
                return
            q_new[subset_mask.values] = q_sub

        # Update the Q-values column in the model
        cols = list(self._current_stats_df.columns)
        if "Q-values" not in cols:
            return
        q_col_idx = cols.index("Q-values")
        for row_idx in range(self._stats_model.rowCount()):
            item = self._stats_model.item(row_idx, q_col_idx)
            if item is None:
                continue
            q_val = q_new[row_idx]
            item.setData(f"{q_val:.4g}", Qt.DisplayRole)
            item.setData(float(q_val), Qt.UserRole)

    def _apply_stats_filter(self) -> None:
        """Push current filter state into the proxy, recompute Q if needed, refresh figure."""
        # Recompute Q-values based on current Condition/Type filter
        self._recompute_q_for_filtered()

        self._stats_proxy.set_filters(
            condition_filter=self._stats_condition_filter,
            type_filter=self._stats_type_filter,
            p_threshold=self._filter_p_spin.value(),
            q_threshold=self._filter_q_spin.value(),
        )
        self._stats_proxy.invalidateFilter()
        self._redraw_stats_figure()

    def _on_global_fdr_changed(self) -> None:
        """Recompute Q-values and redraw when global FDR checkbox changes."""
        self._apply_stats_filter()

    def _pick_condition_filter(self) -> None:
        """Open a checkbox dialog to select visible Condition values."""
        if self._current_stats_df is None:
            return
        col = "Condition"
        if col not in self._current_stats_df.columns:
            return
        unique_vals = sorted(self._current_stats_df[col].astype(str).unique().tolist())
        selected = _pick_values_dialog(self, "Filter by Condition", unique_vals,
                                       current=self._stats_condition_filter)
        if selected is None:
            return  # cancelled
        self._stats_condition_filter = None if selected == set(unique_vals) else selected
        label = "All" if self._stats_condition_filter is None else f"{len(self._stats_condition_filter)} sel."
        self._filter_condition_btn.setText(label)
        self._apply_stats_filter()

    def _pick_type_filter(self) -> None:
        """Open a checkbox dialog to select visible Type values."""
        if self._current_stats_df is None:
            return
        col = "Type"
        if col not in self._current_stats_df.columns:
            return
        unique_vals = sorted(self._current_stats_df[col].astype(str).unique().tolist())
        selected = _pick_values_dialog(self, "Filter by Type", unique_vals,
                                       current=self._stats_type_filter)
        if selected is None:
            return  # cancelled
        self._stats_type_filter = None if selected == set(unique_vals) else selected
        label = "All" if self._stats_type_filter is None else f"{len(self._stats_type_filter)} sel."
        self._filter_type_btn.setText(label)
        self._apply_stats_filter()

    def _reset_stats_filters(self) -> None:
        """Reset all filters to their defaults."""
        self._stats_condition_filter = None
        self._stats_type_filter = None
        self._filter_condition_btn.setText("All")
        self._filter_type_btn.setText("All")
        self._filter_p_spin.blockSignals(True)
        self._filter_q_spin.blockSignals(True)
        self._filter_p_spin.setValue(1.0)
        self._filter_q_spin.setValue(1.0)
        self._filter_p_spin.blockSignals(False)
        self._filter_q_spin.blockSignals(False)
        self._apply_stats_filter()

    def _stats_table_context_menu(self, pos) -> None:
        """Right-click context menu on the stats table."""
        menu = QMenu(self)
        menu.addAction("Reset All Filters", self._reset_stats_filters)
        menu.exec(self._stats_table_view.viewport().mapToGlobal(pos))

    def _get_filtered_df(self):
        """Return a DataFrame containing only the currently visible (filtered) rows."""
        if self._current_stats_df is None:
            return None
        proxy = self._stats_proxy
        source = self._stats_model
        cols = [source.horizontalHeaderItem(c).text()
                for c in range(source.columnCount())]
        rows = []
        for proxy_row in range(proxy.rowCount()):
            source_row = proxy.mapToSource(proxy.index(proxy_row, 0)).row()
            row_data = {}
            for c_idx, col in enumerate(cols):
                item = source.item(source_row, c_idx)
                row_data[col] = item.data(Qt.UserRole) if item else None
            rows.append(row_data)
        return pd.DataFrame(rows, columns=cols)

    def _export_stats_table(self, filtered_only: bool = False) -> None:
        """Prompt for a file path and write the stats table to Excel or CSV."""
        if self._current_stats_df is None:
            QMessageBox.information(self, "Export Table", "No statistics table to export.")
            return
        df = self._get_filtered_df() if filtered_only else self._current_stats_df
        if df is None or len(df) == 0:
            QMessageBox.information(self, "Export Table", "No rows to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Statistics Table", "",
            "Excel Workbook (*.xlsx);;CSV (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                df.to_csv(path, index=False)
            else:
                if not path.lower().endswith(".xlsx"):
                    path += ".xlsx"
                df.to_excel(path, index=False)
            self._set_status(f"Statistics table exported to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Table", f"Export failed:\n{exc}")
            self._set_status("Statistics table export failed.")


# Windows opened from a notebook must be kept alive; Python would otherwise
# garbage-collect them as soon as the call returns.
_OPEN_WINDOWS: List["NIRSviewIRWindow"] = []


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


def NIRSviewIR(dataset: Any, block: Optional[bool] = None) -> "NIRSviewIRWindow":
    """Launch the NIRSviewIR browser for `dataset` (a `DataSet` instance).

    `block` defaults to True in plain scripts and False under IPython/Jupyter,
    where the Qt event loop is instead hooked into the kernel.  The window is
    returned so it can be inspected or closed programmatically.
    """
    app = QApplication.instance() or QApplication(sys.argv or ["NIRSviewIR"])
    # Closing the last window must not quit the shared QApplication, otherwise
    # the kernel's qt event-loop integration is torn down and a second call to
    # NIRSviewIR() in the same session hangs/crashes the kernel.
    app.setQuitOnLastWindowClosed(False)

    shell = _active_ipython()
    if block is None:
        block = shell is None

    window = NIRSviewIRWindow(dataset)
    # Without WA_DeleteOnClose the window (and the Qt event-loop hook installed
    # below) would outlive the notebook cell and block kernel shutdown.
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
            block = True   # kernel cannot host the Qt loop; fall back to blocking

    if block:
        app.exec()

    return window


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Browse the recordings of a pyBrainAnalyzIR DataSet."
    )
    parser.add_argument("dataset", help="path to a pickled pyBrainAnalyzIR DataSet (.pkl)")
    args = parser.parse_args(argv)

    # Unpickling executes code from the file; only load files you trust.
    with open(args.dataset, "rb") as fid:
        dataset = pickle.load(fid)

    if not hasattr(dataset, "dataset"):
        parser.error(f"{args.dataset} does not contain a DataSet object")

    NIRSviewIR(dataset, block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
