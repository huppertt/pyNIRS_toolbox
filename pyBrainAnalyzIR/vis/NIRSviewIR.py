"""PySide6 GUI to browse the recordings stored in a `DataSet`.

Usage
-----
    from pyBrainAnalyzIR.vis.NIRSviewIR import NIRSviewIR

    NIRSviewIR(dataset)   # blocks until the window is closed
"""

from __future__ import annotations

import io
import sys
import copy
import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pyBrainAnalyzIR.vis.dict_table_editor import DictTableEditor, _to_display_str
from pyBrainAnalyzIR.vis.plot_nirs_inline import linecolors, to_string
from pyBrainAnalyzIR.vis.pipeline_manager import (
    PipelineManagerDialog,
    pipeline_to_json,
    pipeline_from_json,
)
from pyBrainAnalyzIR.vis._about import get_about_text
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module
from pyBrainAnalyzIR.dataclasses.statistics import Statistics

SUBJECT_KEYS = ("subject", "subjectID", "ID")


def _subject_label(rec: Any) -> Optional[str]:
    meta = getattr(rec, "meta_data", None) or {}
    for key in SUBJECT_KEYS:
        if key in meta and meta[key] is not None and str(meta[key]).strip() != "":
            return str(meta[key])
    return None


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
        self.selected_index = 0
        self._pipeline_modules: List[cedalion_module] = []  # current stored pipeline
        self._current_stats_df = None  # last drawn statistics DataFrame (for export)

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

        # Demographics menu
        demographics_menu = menu_bar.addMenu("Demographics")
        demographics_menu.addAction("Edit Demographics").triggered.connect(
            self.edit_demographics
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
        self.file_list = QTableWidget(left)
        self.file_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.file_list.setAlternatingRowColors(True)
        self.file_list.verticalHeader().setVisible(False)
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

        # Right-clicking anywhere in the recordings panel offers the
        # demographics editor.
        for widget in (left, self.file_list):
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

        # Right half: table + export button
        stats_right = QWidget(stats_page)
        stats_right_layout = QVBoxLayout(stats_right)
        stats_right_layout.setContentsMargins(0, 0, 0, 0)
        self._stats_table = QTableWidget(stats_right)
        self._stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        stats_right_layout.addWidget(self._stats_table, stretch=1)
        self._export_stats_button = QPushButton("Export Table to Excel…", stats_right)
        self._export_stats_button.clicked.connect(self._export_stats_table)
        stats_right_layout.addWidget(self._export_stats_button)
        stats_layout.addWidget(stats_right, stretch=1)

        self._right_stack.addWidget(stats_page)   # index 1

        right_layout.addWidget(self._right_stack)
        layout.addWidget(right, stretch=1)

        self.setCentralWidget(central)

        # ---------------------------------------------------------------- status bar
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready.")

        self.file_list.itemSelectionChanged.connect(self._on_file_selection_changed)
        self.type_selector.currentTextChanged.connect(self._on_type_changed)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._run_pipeline_action.triggered.connect(self._run_pipeline)

        if len(dataset.dataset) > 0:
            self.file_list.setCurrentCell(0, 0)
            self._on_file_changed(0)
        else:
            self._update_summary(-1)

    # ------------------------------------------------------------------ slots

    # --------------------------------------------------------- pipeline helpers

    def _set_status(self, msg: str) -> None:
        self._status_bar.showMessage(msg)
        QApplication.processEvents()

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

    def _edit_pipeline(self) -> None:
        self._set_status("Opening Pipeline Manager…")
        dialog = PipelineManagerDialog(
            initial_modules=self._pipeline_modules if self._pipeline_modules else None,
            parent=self,
        )
        if dialog.exec():
            self._pipeline_modules = list(dialog.pipeline_modules)
            n = len(self._pipeline_modules)
            self._run_pipeline_action.setEnabled(n > 0)
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
            self._run_pipeline_action.setEnabled(len(modules) > 0)
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
            self._refresh_labels(preferred_row=self.file_list.currentRow())
            self._set_status("Pipeline complete.")
        except Exception as exc:
            QMessageBox.critical(self, "Run Pipeline", f"Pipeline failed:\n{exc}")
            self._set_status("Pipeline run failed.")

    # --------------------------------------------------------- file list menu

    def _show_file_list_menu(self, pos) -> None:
        widget = self.sender() or self.file_list
        menu = QMenu(self)
        edit_action = menu.addAction("Edit Demographics...")
        remove_action = None
        target_row = -1
        if widget is self.file_list:
            target_row = self.file_list.rowAt(pos.y())
            if target_row >= 0:
                self.file_list.setCurrentCell(target_row, 0)
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

        table = self.dataset.get_demographics()
        rows: List[Dict[str, Any]] = [
            {k: v for k, v in row.items() if _to_display_str(v) != ""}
            for row in table.to_dict(orient="records")
        ]
        while len(rows) < len(recordings):
            rows.append({})

        editor = DictTableEditor(rows, self)
        editor.setWindowTitle("Edit Demographics")
        editor.exec()

        # `DictTableEditor` fills in `result` on both Done and window close, so
        # the edits are applied either way (same contract as `edit_dict_table`).
        self._apply_demographics(editor.result)

    def _apply_demographics(self, rows: List[Dict[str, Any]]) -> None:
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

        self._refresh_labels(preferred_row=self.file_list.currentRow())

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
        if len(rows) > len(recordings):
            rows = rows[: len(recordings)]

        headers = ["Recording"] + columns
        self.file_list.setColumnCount(len(headers))
        self.file_list.setHorizontalHeaderLabels(headers)
        self.file_list.setRowCount(len(recordings))

        for row_idx, label in enumerate(labels):
            self.file_list.setItem(row_idx, 0, QTableWidgetItem(label))
            row_data = rows[row_idx] if row_idx < len(rows) else {}
            for col_idx, key in enumerate(columns, start=1):
                self.file_list.setItem(
                    row_idx, col_idx, QTableWidgetItem(_to_display_str(row_data.get(key)))
                )

    def _remove_recording(self, row: int) -> None:
        recordings = getattr(self.dataset, "dataset", None)
        if recordings is None or row < 0 or row >= len(recordings):
            return

        label_item = self.file_list.item(row, 0)
        label = label_item.text() if label_item is not None else f"recording {row}"
        confirm = QMessageBox.question(
            self,
            "Remove Recording",
            f"Remove '{label}' from the dataset?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        del recordings[row]
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

    def _refresh_labels(self, preferred_row: int = -1) -> None:
        recordings = getattr(self.dataset, "dataset", [])
        self.file_list.blockSignals(True)
        self._populate_file_table()
        self.file_list.blockSignals(False)
        if recordings:
            row = preferred_row
            if row < 0:
                row = 0
            row = min(row, len(recordings) - 1)
            self.file_list.setCurrentCell(row, 0)
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
        self._on_file_changed(self.file_list.currentRow())

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
            self._update_summary(self.file_list.currentRow(), key)

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
            line.set_color("k")

        self.selected_index = int(np.argmin(distances))
        self.mllines[self.selected_index].set_color("r")
        self._update_timecourse()
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- drawing

    def _update_timecourse(self) -> None:
        data = self.data
        selchann = to_string(data.channel[self.selected_index])
        if hasattr(data, "wavelength"):
            wl = to_string(data.wavelength)
            self.line0.set_ydata(data.sel(channel=selchann, wavelength=wl[-1]))
            self.line1.set_ydata(data.sel(channel=selchann, wavelength=wl[0]))
        else:
            self.line0.set_ydata(data.sel(channel=selchann, chromo="HbO"))
            self.line1.set_ydata(data.sel(channel=selchann, chromo="HbR"))
        self.ax_ts.set_title(selchann)

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
            detector = sdstr[sdstr.find("D") :]
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

        self.selected_index = 0
        self.mllines[0].set_color("r")

        optodes = geo2d.to_numpy()
        s = (optodes.max() - optodes.min()) / 10
        self.ax_probe.set_ylim(optodes[:, 1].min() - s, optodes[:, 1].max() + s)
        self.ax_probe.set_xlim(optodes[:, 0].min() - s, optodes[:, 0].max() + s)
        self.ax_probe.set_axis_off()

        selchann = to_string(data.channel[0])
        if hasattr(data, "wavelength"):
            wl = to_string(data.wavelength)
            self.line0, = self.ax_ts.plot(
                data.time, data.sel(channel=selchann, wavelength=wl[-1]), "r-",
                label=f"{wl[-1]}nm")
            self.line1, = self.ax_ts.plot(
                data.time, data.sel(channel=selchann, wavelength=wl[0]), "b-",
                label=f"{wl[0]}nm")
        else:
            self.line0, = self.ax_ts.plot(
                data.time, data.sel(channel=selchann, chromo="HbO"), "r-", label="HbO")
            self.line1, = self.ax_ts.plot(
                data.time, data.sel(channel=selchann, chromo="HbR"), "b-", label="HbR")

        legend1 = self.ax_ts.legend(handles=[self.line0, self.line1], loc="upper right")

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

            lines = []
            for index in range(cond_names.shape[0]):
                l, = self.ax_ts.plot([data.time[0], data.time[0]], [vmin, vmin],
                                     color=linecolors[index], label=cond_names[index])
                lines.append(l)
            self.ax_ts.legend(handles=lines, loc="lower right")
            self.ax_ts.add_artist(legend1)

        self.ax_ts.set_title(selchann)
        self.ax_ts.set_xlabel("time / s")
        self.ax_ts.set_ylabel(f"{key} / a.u.")
        self.ax_ts.set_ylim(vmin, vmax)
        self.ax_ts.set_xlim(float(data.time.min()), float(data.time.max()))

        self.canvas.draw_idle()

    # ------------------------------------------------------------ statistics view

    def _draw_stats(self, stats: Statistics) -> None:
        """Render the Statistics view: channel map figure + results table."""
        self._right_stack.setCurrentIndex(1)
        self._set_status("Rendering statistics…")

        # --- figure: call stats.draw() which creates its own plt figure --------
        try:
            plt.close("all")
            stats.draw()
            src_fig = plt.gcf()

            # Transfer the matplotlib figure into the embedded canvas via PNG round-trip
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

        # --- table: stats.table() returns a DataFrame -------------------------
        try:
            df = stats.table()
            self._current_stats_df = df
            self._stats_table.setRowCount(len(df))
            self._stats_table.setColumnCount(len(df.columns))
            self._stats_table.setHorizontalHeaderLabels(list(df.columns))
            for r_idx, row in df.iterrows():
                for c_idx, val in enumerate(row):
                    cell_text = f"{val:.4g}" if isinstance(val, float) else str(val)
                    self._stats_table.setItem(r_idx, c_idx, QTableWidgetItem(cell_text))
            self._export_stats_button.setEnabled(True)
        except Exception as exc:
            self._stats_table.setRowCount(0)
            self._stats_table.setColumnCount(1)
            self._stats_table.setHorizontalHeaderLabels(["Error"])
            self._stats_table.setItem(0, 0, QTableWidgetItem(str(exc)))
            self._current_stats_df = None
            self._export_stats_button.setEnabled(False)

        self._set_status("Statistics rendered.")

    def _export_stats_table(self) -> None:
        """Prompt for a file path and write the current stats table to Excel."""
        if self._current_stats_df is None:
            QMessageBox.information(self, "Export Table", "No statistics table to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Statistics Table", "",
            "Excel Workbook (*.xlsx);;CSV (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                self._current_stats_df.to_csv(path, index=False)
            else:
                if not path.lower().endswith(".xlsx"):
                    path += ".xlsx"
                self._current_stats_df.to_excel(path, index=False)
            self._set_status(f"Statistics table exported to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Table", f"Export failed:\n{exc}")
            self._set_status("Statistics table export failed.")


def NIRSviewIR(dataset: Any) -> None:
    """Launch the NIRSviewIR browser for `dataset` (a `DataSet` instance)."""
    owns_app = QApplication.instance() is None
    app = QApplication.instance() or QApplication(sys.argv)

    window = NIRSviewIRWindow(dataset)
    window.show()

    if owns_app:
        app.exec()
