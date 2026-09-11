"""PySide6 GUI for reviewing fNIRS data-quality metrics."""
from __future__ import annotations

import math
import sys
import base64
import io
from collections import OrderedDict
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from cedalion import units
import cedalion.sigproc.quality as quality

import pyBrainAnalyzIR.dataclasses.dataset as dataset_module
from pyBrainAnalyzIR.vis.demographics_manager import _to_display_str
from pyBrainAnalyzIR.vis.plot_nirs_inline import draw_probe


RECORDING_INDEX_ROLE = Qt.UserRole + 1
QUALITY_METRICS: OrderedDict[str, str] = OrderedDict(
    [
        ("snr", "Signal-to-Noise Ratio"),
        ("sci", "Scalp Coupling Index"),
        ("psp", "Peak Spectral Power"),
        ("gvtd", "Global Variance of the Temporal Derivative"),
        ("clean_percent", "Percent of Clean time"),
    ]
)
DEFAULT_WINDOW_LENGTH = 5 * units.s
SNR_THRESHOLD = 16.0
SCI_THRESHOLD = 0.75
PSP_THRESHOLD = 0.1
GVTD_THRESHOLD = 0.1

#: Metrics where values *above* the threshold are considered clean.
_CLEAN_ABOVE_THRESHOLD = ("snr", "sci", "psp", "clean_percent")

DEFAULT_THRESHOLDS: OrderedDict[str, float] = OrderedDict(
    [
        ("snr", SNR_THRESHOLD),
        ("sci", SCI_THRESHOLD),
        ("psp", PSP_THRESHOLD),
        ("gvtd", GVTD_THRESHOLD),
        ("clean_percent", 0.5),
    ]
)
DISPLAY_MODES: OrderedDict[str, str] = OrderedDict(
    [("value", "Time-course"), ("mask", "Mask")]
)
#: Compact labels used for the checkbox row and threshold spin boxes.
SHORT_METRIC_LABELS: Dict[str, str] = {
    "snr": "SNR",
    "sci": "SCI",
    "psp": "PSP",
    "gvtd": "GVTD",
    "clean_percent": "Percent of Clean time",
}


def _short_metric_label(metric: str) -> str:
    return SHORT_METRIC_LABELS.get(metric, QUALITY_METRICS.get(metric, metric))


def _threshold_for(metric: str, thresholds: Optional[Dict[str, float]]) -> float:
    if thresholds and metric in thresholds:
        return float(thresholds[metric])
    return float(DEFAULT_THRESHOLDS[metric])


def metric_mask(
    values: xr.DataArray,
    metric: str,
    thresholds: Optional[Dict[str, float]] = None,
) -> xr.DataArray:
    """Convert metric values to a clean (1.0) / tainted (0.0) mask."""
    threshold = _threshold_for(metric, thresholds)
    if metric in _CLEAN_ABOVE_THRESHOLD:
        mask = values >= threshold
    else:
        mask = values < threshold
    return mask.astype(float).where(np.isfinite(values), 0.0)


def _as_dataset(data: Any) -> dataset_module.DataSet:
    if data is None:
        return dataset_module.DataSet()
    if hasattr(data, "dataset"):
        return data
    return dataset_module.DataSet([data])


def _meta_lookup(rec: Any, keys: Iterable[str]) -> Optional[str]:
    meta = getattr(rec, "meta_data", None) or {}
    lowered = {str(k).strip().lower(): v for k, v in meta.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None and _to_display_str(value).strip():
            return _to_display_str(value).strip()
    return None


def _recording_labels(dset: Any) -> List[str]:
    labels = []
    for index, rec in enumerate(getattr(dset, "dataset", [])):
        subject = _meta_lookup(rec, ("subject", "subj", "subjID", "subjectID", "ID", "name"))
        labels.append(f"[{index}] {subject}" if subject else f"[{index}] <no subject>")
    return labels


def _subject_name(rec: Any) -> str:
    return _meta_lookup(rec, ("subject", "subj", "subjID", "subjectID", "ID", "name")) or ""


def _scan_number(rec: Any) -> str:
    return _meta_lookup(rec, ("scan", "scan_number", "scanNumber", "run", "Run")) or ""


def _collapse_to_channel(metric: xr.DataArray) -> xr.DataArray:
    metric = metric.pint.dequantify() if hasattr(metric, "pint") else metric
    for dim in list(metric.dims):
        if dim not in ("channel",):
            metric = metric.mean(dim, skipna=True)
    return metric


def _collapse_to_channel_time(metric: xr.DataArray) -> xr.DataArray:
    metric = metric.pint.dequantify() if hasattr(metric, "pint") else metric
    metric = metric.where(~np.isinf(metric), np.nan)
    for dim in list(metric.dims):
        if dim not in ("channel", "time"):
            metric = metric.median(dim, skipna=True)
    if metric.dims != ("time", "channel"):
        metric = metric.transpose("time", "channel")
    return metric


def _sampling_rate_hz(amp) -> float:
    dt = amp.time.diff("time").mean()
    try:
        seconds = dt.pint.to(units.s).pint.magnitude.item()
    except Exception:
        seconds = float(dt)
    return 1.0 / float(seconds)


def _numpy_amp(amp) -> tuple[np.ndarray, np.ndarray, list[str]]:
    arr = amp.pint.dequantify().transpose("time", "channel", ...).to_numpy()
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 2:
        arr = np.nanmean(arr, axis=tuple(range(2, arr.ndim)))
    times = np.asarray(amp.time.pint.dequantify(), dtype=float)
    channels = [str(ch) for ch in amp.channel.values]
    return arr, times, channels


def _rolling_window_metric(amp, func, window_length=DEFAULT_WINDOW_LENGTH) -> xr.DataArray:
    arr, times, channels = _numpy_amp(amp)
    fs = _sampling_rate_hz(amp)
    nsamples = max(2, int(math.ceil((window_length * fs).to_base_units().magnitude)))
    starts = np.arange(0, max(arr.shape[0] - nsamples + 1, 1), nsamples)
    values = []
    value_times = []
    for start in starts:
        stop = min(start + nsamples, arr.shape[0])
        window = arr[start:stop]
        if window.shape[0] < 2:
            continue
        values.append(func(window, fs))
        value_times.append(times[start])
    if not values:
        values = [func(arr, fs)]
        value_times = [times[0] if len(times) else 0.0]
    return xr.DataArray(
        np.vstack(values),
        dims=("time", "channel"),
        coords={"time": value_times, "channel": channels},
    )


def _fallback_snr_time(amp, window_length=DEFAULT_WINDOW_LENGTH) -> xr.DataArray:
    def snr_window(window, fs):
        _ = fs
        return np.abs(np.nanmean(window, axis=0)) / (np.nanstd(window, axis=0) + 1e-16)

    return _rolling_window_metric(amp, snr_window, window_length)


def _fallback_psp_time(amp, window_length=DEFAULT_WINDOW_LENGTH) -> xr.DataArray:
    def psp_window(window, fs):
        window = window - np.nanmean(window, axis=0, keepdims=True)
        power = np.abs(np.fft.rfft(window, axis=0)) ** 2
        freqs = np.fft.rfftfreq(window.shape[0], d=1.0 / fs)
        cardiac = (freqs >= 0.5) & (freqs <= 2.5)
        if not cardiac.any():
            cardiac = freqs > 0
        return np.nanmax(power[cardiac, :], axis=0)

    return _rolling_window_metric(amp, psp_window, window_length)


def _gvtd_per_channel(amp, window_length=DEFAULT_WINDOW_LENGTH) -> xr.DataArray:
    """GVTD computed per channel so it can be shown on the probe/heatmap."""
    _ = window_length
    channels = [str(ch) for ch in amp.channel.values]
    order = [dim for dim in ("channel", "wavelength", "time") if dim in amp.dims]
    columns = []
    times = None
    for index in range(amp.sizes["channel"]):
        sub = amp.isel(channel=[index]).transpose(*order)
        gvtd, _mask = quality.gvtd(sub)
        gvtd = gvtd.pint.dequantify() if hasattr(gvtd, "pint") else gvtd
        values = np.asarray(gvtd.to_numpy(), dtype=float).reshape(-1)
        if times is None:
            time_coord = gvtd.time
            time_coord = (
                time_coord.pint.dequantify()
                if hasattr(time_coord, "pint")
                else time_coord
            )
            times = np.asarray(time_coord, dtype=float)
        columns.append(values)
    if times is None:
        times = np.asarray([], dtype=float)
    return xr.DataArray(
        np.column_stack(columns) if columns else np.zeros((len(times), 0)),
        dims=("time", "channel"),
        coords={"time": times, "channel": channels},
    )


def compute_quality_metric(
    rec: Any,
    metric: str,
    thresholds: Optional[Dict[str, float]] = None,
    window_length=DEFAULT_WINDOW_LENGTH,
) -> tuple[xr.DataArray, xr.DataArray]:
    amp = rec["amp"]
    if metric == "snr":
        scalar, _ = quality.snr(amp, _threshold_for("snr", thresholds))
        timeseries = _fallback_snr_time(amp, window_length)
        return _collapse_to_channel(scalar), timeseries
    if metric == "sci":
        sci, _ = quality.sci(amp, window_length, _threshold_for("sci", thresholds))
        sci = _collapse_to_channel_time(sci)
        return _collapse_to_channel(sci), sci
    if metric == "psp":
        try:
            psp, _ = quality.psp(
                amp, window_length, _threshold_for("psp", thresholds)
            )
            psp = _collapse_to_channel_time(psp)
        except Exception as exc:
            warnings.warn(
                f"Cedalion PSP calculation failed ({exc}); using local PSP fallback.",
                UserWarning,
                stacklevel=2,
            )
            psp = _fallback_psp_time(amp, window_length)
        return _collapse_to_channel(psp), psp
    if metric == "gvtd":
        gvtd = _gvtd_per_channel(amp, window_length)
        return _collapse_to_channel(gvtd), gvtd
    if metric == "clean_percent":
        sci = compute_quality_metric(rec, "sci", thresholds, window_length)[1]
        psp = compute_quality_metric(rec, "psp", thresholds, window_length)[1]
        common_times = np.intersect1d(sci.time.values, psp.time.values)
        if common_times.size:
            sci = sci.sel(time=common_times)
            psp = psp.sel(time=common_times)
        clean = (sci > _threshold_for("sci", thresholds)) & (
            psp > _threshold_for("psp", thresholds)
        )
        return _collapse_to_channel(clean.astype(float)), clean.astype(float)
    raise ValueError(f"Unsupported quality metric: {metric}")


def _metric_limits(metric: str, display_mode: str = "value") -> tuple[float | None, float | None, str]:
    if display_mode == "mask":
        return 0.0, 1.0, "RdYlGn"
    if metric == "sci":
        return 0.0, 1.0, "viridis"
    if metric == "clean_percent":
        return 0.0, 1.0, "viridis"
    if metric == "snr":
        return 0.0, None, "jet"
    if metric == "gvtd":
        return 0.0, None, "plasma"
    return 0.0, None, "magma"


def _studentize(series) -> np.ndarray:
    values = np.asarray(series.pint.dequantify() if hasattr(series, "pint") else series)
    values = values.astype(float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return values - mean
    return (values - mean) / std


def _safe_sheet_name(name: str, used: set[str]) -> str:
    invalid = set("[]:*?/\\")
    cleaned = "".join("_" if char in invalid else char for char in name)[:31] or "Sheet"
    candidate = cleaned
    counter = 1
    while candidate in used:
        suffix = f"_{counter}"
        candidate = f"{cleaned[:31 - len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def _distance_to_line_segment(line, x: float | None, y: float | None) -> float:
    if x is None or y is None:
        return float("inf")
    xdata, ydata = line.get_data()
    x0, y0 = float(xdata[0]), float(ydata[0])
    x1, y1 = float(xdata[-1]), float(ydata[-1])
    dx = x1 - x0
    dy = y1 - y0
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return math.hypot(x - x0, y - y0)
    t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / length2))
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    return math.hypot(x - proj_x, y - proj_y)


class DataQualityManager(QMainWindow):
    def __init__(self, data: Any, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Data Quality Manager")
        self.resize(1200, 760)
        self.dataset = _as_dataset(data)
        self._metric_cache: Dict[tuple[int, str], tuple[xr.DataArray, xr.DataArray]] = {}
        self._channel_line_map: Dict[Any, tuple[int, str, int]] = {}
        self._channel_detail_selection: tuple[int, str, int] | None = None
        self.thresholds: Dict[str, float] = dict(DEFAULT_THRESHOLDS)
        self.window_length_s: float = float(
            DEFAULT_WINDOW_LENGTH.to(units.s).magnitude
        )

        root = QWidget(self)
        outer_layout = QVBoxLayout(root)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        menu_bar = QMenuBar(root)
        menu_bar.setNativeMenuBar(False)
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction("Refresh").triggered.connect(self._refresh_current_tab)
        export_menu = menu_bar.addMenu("Export Results")
        export_menu.addAction("Export to Excel").triggered.connect(self._export_to_excel)
        export_menu.addAction("Export to PDF").triggered.connect(self._export_to_pdf)
        export_menu.addAction("Export to HTML").triggered.connect(self._export_to_html)
        outer_layout.addWidget(menu_bar)

        central = QWidget(root)
        outer_layout.addWidget(central, stretch=1)
        layout = QHBoxLayout(central)

        left = QWidget(central)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Recordings"))
        self.file_tree = QTreeWidget(left)
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._populate_file_tree()
        left_layout.addWidget(self.file_tree)
        left_layout.addWidget(QLabel("Selected recording summary"))
        self.summary_panel = QTextEdit(left)
        self.summary_panel.setReadOnly(True)
        self.summary_panel.setMinimumHeight(170)
        left_layout.addWidget(self.summary_panel)
        left.setMaximumWidth(280)
        layout.addWidget(left)

        self.tabs = QTabWidget(central)
        self.channel_tab = QWidget(self.tabs)
        self.artifact_tab = QWidget(self.tabs)
        self.tabs.addTab(self.channel_tab, "Channel Quality")
        self.tabs.addTab(self.artifact_tab, "Time-series artifacts")
        layout.addWidget(self.tabs, stretch=1)

        self._setup_channel_tab()
        self._setup_artifact_tab()
        self.setCentralWidget(root)

        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready.")

        self.file_tree.itemSelectionChanged.connect(self._on_file_selection_changed)
        self.tabs.currentChanged.connect(lambda *_: self._refresh_current_tab())
        if getattr(self.dataset, "dataset", []):
            self._select_row(0)
            self._update_summary(0)
        else:
            self._update_summary(-1)
        self._refresh_current_tab()

    def _setup_channel_tab(self) -> None:
        layout = QVBoxLayout(self.channel_tab)
        layout.addWidget(self._metric_controls("channel"))
        layout.addWidget(self._settings_controls("channel"))
        self.channel_figure = Figure(figsize=(9, 6))
        self.channel_canvas = FigureCanvas(self.channel_figure)
        self.channel_canvas.mpl_connect("button_press_event", self._on_channel_plot_click)
        layout.addWidget(self.channel_canvas, stretch=1)
        self.channel_detail_label = QLabel("Selected channel detail")
        layout.addWidget(self.channel_detail_label)
        self.channel_detail_figure = Figure(figsize=(9, 3))
        self.channel_detail_canvas = FigureCanvas(self.channel_detail_figure)
        self.channel_detail_canvas.setMinimumHeight(210)
        layout.addWidget(self.channel_detail_canvas, stretch=1)
        self.channel_detail_label.setVisible(False)
        self.channel_detail_canvas.setVisible(False)

    def _setup_artifact_tab(self) -> None:
        layout = QVBoxLayout(self.artifact_tab)
        layout.addWidget(self._metric_controls("artifact"))
        layout.addWidget(self._settings_controls("artifact"))
        self.artifact_figure = Figure(figsize=(9, 6))
        self.artifact_canvas = FigureCanvas(self.artifact_figure)
        layout.addWidget(self.artifact_canvas, stretch=1)

    def _settings_controls(self, prefix: str) -> QWidget:
        """Display-mode selector plus user-editable thresholds and window length."""
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Display:", row))
        display_mode = QComboBox(row)
        for key, label in DISPLAY_MODES.items():
            display_mode.addItem(label, key)
        display_mode.currentIndexChanged.connect(
            lambda *_: self._on_display_mode_changed(prefix)
        )
        layout.addWidget(display_mode)
        setattr(self, f"_{prefix}_display_mode", display_mode)

        layout.addWidget(QLabel("Window (s):", row))
        window_spin = QDoubleSpinBox(row)
        window_spin.setRange(0.1, 600.0)
        window_spin.setDecimals(2)
        window_spin.setSingleStep(0.5)
        window_spin.setValue(self.window_length_s)
        window_spin.valueChanged.connect(self._on_window_length_changed)
        layout.addWidget(window_spin)
        setattr(self, f"_{prefix}_window_spin", window_spin)

        threshold_spins = {}
        for metric, label in QUALITY_METRICS.items():
            layout.addWidget(QLabel(f"{_short_metric_label(metric)}:", row))
            spin = QDoubleSpinBox(row)
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.setValue(self.thresholds[metric])
            spin.setToolTip(f"{label} threshold")
            spin.valueChanged.connect(self._threshold_handler(metric))
            layout.addWidget(spin)
            threshold_spins[metric] = spin
        setattr(self, f"_{prefix}_threshold_spins", threshold_spins)

        layout.addStretch()
        return row

    def _threshold_handler(self, metric: str):
        def handler(value: float) -> None:
            self._set_threshold(metric, float(value))

        return handler

    def _set_threshold(self, metric: str, value: float) -> None:
        if math.isclose(self.thresholds.get(metric, float("nan")), value):
            return
        self.thresholds[metric] = value
        for prefix in ("channel", "artifact"):
            spins = getattr(self, f"_{prefix}_threshold_spins", None)
            if not spins or metric not in spins:
                continue
            spin = spins[metric]
            if not math.isclose(spin.value(), value):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        # SNR / SCI / PSP thresholds feed the cedalion calls and clean-percent.
        self._metric_cache.clear()
        self._refresh_current_tab()

    def _on_window_length_changed(self, value: float) -> None:
        value = float(value)
        if math.isclose(self.window_length_s, value):
            return
        self.window_length_s = value
        for prefix in ("channel", "artifact"):
            spin = getattr(self, f"_{prefix}_window_spin", None)
            if spin is not None and not math.isclose(spin.value(), value):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self._metric_cache.clear()
        self._refresh_current_tab()

    def _on_display_mode_changed(self, prefix: str) -> None:
        _ = prefix
        self._refresh_current_tab()

    def _display_mode(self, prefix: str) -> str:
        combo = getattr(self, f"_{prefix}_display_mode", None)
        if combo is None:
            return "value"
        return combo.currentData() or "value"

    @property
    def window_length(self):
        return self.window_length_s * units.s

    def _metric_controls(self, prefix: str) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        all_files = QCheckBox("Show all data files", row)
        layout.addWidget(all_files)
        setattr(self, f"_{prefix}_all_files", all_files)
        metric_checks = {}
        for index, (metric, label) in enumerate(QUALITY_METRICS.items()):
            checkbox = QCheckBox(_short_metric_label(metric), row)
            checkbox.setToolTip(label)
            checkbox.setChecked(index == 0)
            checkbox.stateChanged.connect(self._metric_selection_handler(prefix))
            layout.addWidget(checkbox)
            metric_checks[metric] = checkbox
        all_files.stateChanged.connect(self._all_files_handler(prefix))
        setattr(self, f"_{prefix}_metric_checks", metric_checks)
        layout.addStretch()
        refresh = QPushButton("Refresh", row)
        refresh.clicked.connect(self._refresh_current_tab)
        layout.addWidget(refresh)
        return row

    def _metric_selection_handler(self, prefix: str):
        def handler(state):
            _ = state
            self._on_metric_selection_changed(prefix)

        return handler

    def _all_files_handler(self, prefix: str):
        def handler(state):
            _ = state
            self._on_all_files_changed(prefix)

        return handler

    def _populate_file_tree(self) -> None:
        self.file_tree.clear()
        labels = _recording_labels(self.dataset)
        for index, label in enumerate(labels):
            item = QTreeWidgetItem(self.file_tree, [label])
            item.setData(0, RECORDING_INDEX_ROLE, index)
        self.file_tree.expandAll()

    def _item_for_row(self, row: int) -> Optional[QTreeWidgetItem]:
        iterator = QTreeWidgetItemIterator(self.file_tree)
        while iterator.value():
            item = iterator.value()
            if item.data(0, RECORDING_INDEX_ROLE) == row:
                return item
            iterator += 1
        return None

    def _select_row(self, row: int) -> None:
        item = self._item_for_row(row)
        if item is not None:
            self.file_tree.setCurrentItem(item)

    def _current_row(self) -> int:
        item = self.file_tree.currentItem()
        if item is None:
            return -1
        value = item.data(0, RECORDING_INDEX_ROLE)
        return -1 if value is None else int(value)

    def _on_file_selection_changed(self) -> None:
        self._channel_detail_selection = None
        self._update_summary(self._current_row())
        self._refresh_current_tab()

    def _update_summary(self, row: int) -> None:
        recordings = getattr(self.dataset, "dataset", [])
        if row < 0 or row >= len(recordings):
            self.summary_panel.setPlainText("No recording selected.")
            return
        rec = recordings[row]
        meta = getattr(rec, "meta_data", None) or {}
        visible_meta = {
            k: v for k, v in meta.items()
            if str(k).strip().lower() != "_bids_descriptions"
        }
        lines = [f"  - {k}: {_to_display_str(v)}" for k, v in visible_meta.items()]
        if not lines:
            lines = ["  - <none>"]
        self.summary_panel.setPlainText("Demographics:\n" + "\n".join(lines))

    def _selected_metrics(self, prefix: str) -> list[str]:
        checks = getattr(self, f"_{prefix}_metric_checks")
        selected = [metric for metric, checkbox in checks.items() if checkbox.isChecked()]
        if selected:
            return selected
        first = next(iter(checks))
        checks[first].setChecked(True)
        return [first]

    def _on_all_files_changed(self, prefix: str) -> None:
        checks = getattr(self, f"_{prefix}_metric_checks")
        all_files = getattr(self, f"_{prefix}_all_files").isChecked()
        selected = self._selected_metrics(prefix)[0]
        for metric, checkbox in checks.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(metric == selected)
            checkbox.setEnabled(not all_files or metric == selected)
            checkbox.blockSignals(False)
        self._refresh_current_tab()

    def _on_metric_selection_changed(self, prefix: str) -> None:
        all_files = getattr(self, f"_{prefix}_all_files").isChecked()
        if all_files:
            self._channel_detail_selection = None
            self._on_all_files_changed(prefix)
            return
        if prefix == "channel":
            self._channel_detail_selection = None
        self._refresh_current_tab()

    def _refresh_current_tab(self) -> None:
        if self.tabs.currentWidget() is self.channel_tab:
            self._draw_channel_quality()
        else:
            self._draw_artifact_heatmaps()

    def _metric_for_recording(self, row: int, metric: str) -> tuple[xr.DataArray, xr.DataArray]:
        key = (row, metric)
        if key not in self._metric_cache:
            rec = self.dataset.dataset[row]
            self._metric_cache[key] = compute_quality_metric(
                rec, metric, self.thresholds, self.window_length
            )
        return self._metric_cache[key]

    def _metric_display_values(
        self, row: int, metric: str, display_mode: str
    ) -> xr.DataArray:
        """Metric time-course, converted to a clean/tainted mask when requested."""
        values = self._metric_for_recording(row, metric)[1]
        if display_mode == "mask":
            return metric_mask(values, metric, self.thresholds)
        return values

    def _metric_display_scalar(
        self, row: int, metric: str, display_mode: str
    ) -> xr.DataArray:
        if display_mode == "mask":
            return _collapse_to_channel(
                self._metric_display_values(row, metric, display_mode)
            )
        return self._metric_for_recording(row, metric)[0]

    def _selected_rows_for_prefix(self, prefix: str) -> list[int]:
        recordings = getattr(self.dataset, "dataset", [])
        if getattr(self, f"_{prefix}_all_files").isChecked():
            return list(range(len(recordings)))
        row = self._current_row()
        return [row] if 0 <= row < len(recordings) else []

    def _subplot_grid(self, count: int) -> tuple[int, int]:
        cols = max(1, math.ceil(math.sqrt(count)))
        rows = max(1, math.ceil(count / cols))
        return rows, cols

    def _draw_channel_quality(self) -> None:
        self.channel_figure.clear()
        self._channel_line_map = {}
        rows = self._selected_rows_for_prefix("channel")
        metrics = self._selected_metrics("channel")
        plots = [(row, metric) for row in rows for metric in metrics]
        if not plots:
            ax = self.channel_figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No recording selected.", ha="center", va="center")
            ax.set_axis_off()
            self.channel_canvas.draw_idle()
            self._hide_channel_detail()
            return

        single_detail_enabled = len(plots) == 1
        display_mode = self._display_mode("channel")
        grid_rows, grid_cols = self._subplot_grid(len(plots))
        labels = _recording_labels(self.dataset)
        for index, (row, metric) in enumerate(plots, start=1):
            ax = self.channel_figure.add_subplot(grid_rows, grid_cols, index)
            rec = self.dataset.dataset[row]
            scalar = self._metric_display_scalar(row, metric, display_mode)
            vmin, vmax, cmap = _metric_limits(metric, display_mode)
            channel_lines = self._draw_metric_probe(rec, scalar, ax, vmin, vmax, cmap)
            if single_detail_enabled:
                for channel_index, line in enumerate(channel_lines):
                    self._channel_line_map[line] = (row, metric, channel_index)
            title = _short_metric_label(metric)
            if display_mode == "mask":
                title = f"{title} (fraction clean)"
            if len(rows) > 1:
                title = f"{labels[row]}\n{title}"
            ax.set_title(title)
        self.channel_canvas.draw_idle()
        if not single_detail_enabled:
            self._hide_channel_detail()
        elif self._channel_detail_selection is not None:
            self._draw_channel_detail(*self._channel_detail_selection)
        else:
            self._hide_channel_detail()
        self._status_bar.showMessage("Channel quality updated.")

    def _draw_metric_probe(
        self,
        rec: Any,
        values: xr.DataArray,
        ax,
        vmin: float | None,
        vmax: float | None,
        cmap_name: str,
    ) -> list[Any]:
        data = rec["amp"]
        channel_lines = draw_probe(rec, data, ax, plot_on_scalp=True)
        values = values.sel(channel=data.channel.values)
        arr = np.asarray(values.to_numpy(), dtype=float)
        if vmin is None:
            vmin = float(np.nanmin(arr))
        if vmax is None:
            vmax = float(np.nanmax(arr))
        if vmin == vmax:
            vmax = vmin + 1.0
        cmap = plt.get_cmap(cmap_name)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for line, value in zip(channel_lines, arr):
            line.set_color(cmap(norm(value)))
            line.set_linewidth(3.0)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        self.channel_figure.colorbar(sm, ax=ax, shrink=0.7)
        return channel_lines

    def _on_channel_plot_click(self, event) -> None:
        if event.button != 3 or event.inaxes is None or not self._channel_line_map:
            return
        closest = self._closest_channel_line(event)
        if closest is None:
            return
        self._channel_detail_selection = self._channel_line_map[closest]
        self._draw_channel_detail(*self._channel_detail_selection)
        self._highlight_selected_channel(closest)
        self.channel_canvas.draw_idle()

    def _closest_channel_line(self, event):
        candidates = [
            line for line in self._channel_line_map
            if line.axes is event.inaxes
        ]
        if not candidates:
            return None
        distances = [
            _distance_to_line_segment(line, event.xdata, event.ydata)
            for line in candidates
        ]
        return candidates[int(np.nanargmin(distances))]

    def _highlight_selected_channel(self, selected_line) -> None:
        for line in self._channel_line_map:
            line.set_linewidth(3.0)
        selected_line.set_linewidth(5.0)

    def _hide_channel_detail(self) -> None:
        self.channel_detail_label.setVisible(False)
        self.channel_detail_canvas.setVisible(False)
        self.channel_detail_figure.clear()
        self.channel_detail_canvas.draw_idle()

    def _show_channel_detail(self) -> None:
        self.channel_detail_label.setVisible(True)
        self.channel_detail_canvas.setVisible(True)

    def _clear_channel_detail(self, message: str) -> None:
        self._show_channel_detail()
        self.channel_detail_figure.clear()
        ax = self.channel_detail_figure.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_axis_off()
        self.channel_detail_canvas.draw_idle()

    def _draw_channel_detail(self, row: int, metric: str, channel_index: int) -> None:
        self._show_channel_detail()
        rec = self.dataset.dataset[row]
        amp = rec["amp"]
        channel = amp.channel.values[channel_index]
        display_mode = self._display_mode("channel")
        metric_ts = self._metric_display_values(row, metric, display_mode).sel(
            channel=channel
        )
        metric_label = _short_metric_label(metric)
        if display_mode == "mask":
            metric_label = f"{metric_label} mask"

        self.channel_detail_figure.clear()
        ax = self.channel_detail_figure.add_subplot(1, 1, 1)

        self._plot_channel_timeseries(ax, amp, channel)

        metric_ax = ax.twinx()
        metric_values = np.asarray(metric_ts, dtype=float)
        drawstyle = "steps-post" if display_mode == "mask" else "default"
        metric_line, = metric_ax.plot(
            metric_ts.time,
            metric_values,
            color="magenta",
            linestyle="-",
            linewidth=0.9,
            drawstyle=drawstyle,
            label=metric_label,
        )
        if display_mode == "mask":
            metric_ax.set_ylim(-0.05, 1.05)
            metric_ax.set_yticks([0.0, 1.0])
            metric_ax.set_yticklabels(["tainted", "clean"])
        else:
            self._scale_metric_axis(metric_ax, metric_ts)
        metric_ax.set_ylabel(metric_label, color="magenta")
        metric_ax.tick_params(axis="y", colors="magenta")

        ax.set_title(f"{channel}: normalized amplitude and {metric_label}")
        ax.set_xlabel("time / s")
        ax.set_ylabel("normalized amplitude")
        ax.grid(True, alpha=0.25)
        ax.set_zorder(metric_ax.get_zorder() + 1)
        ax.patch.set_visible(False)
        handles = list(ax.lines) + [metric_line]
        ax.legend(
            handles,
            [handle.get_label() for handle in handles],
            loc="upper right",
            fontsize=8,
        )

        self.channel_detail_canvas.draw_idle()
        self._status_bar.showMessage(
            f"Showing {metric_label} detail for channel {channel}."
        )

    def _scale_metric_axis(self, metric_ax, metric_ts) -> None:
        """Use the full y-range of the metric axis for the plotted values."""
        values = np.asarray(metric_ts, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        low = float(np.min(finite))
        high = float(np.max(finite))
        if math.isclose(low, high):
            pad = abs(low) * 0.05 or 0.5
        else:
            pad = (high - low) * 0.05
        metric_ax.set_ylim(low - pad, high + pad)

    def _plot_channel_timeseries(self, ax, amp, channel) -> None:
        if "wavelength" in amp.dims:
            wavelengths = [str(wl) for wl in amp.wavelength.values]
            colors = ["r", "b", "g", "k"]
            for index, wavelength in enumerate(amp.wavelength.values):
                series = amp.sel(channel=channel, wavelength=wavelength)
                label = f"{wavelengths[index]} nm"
                ax.plot(
                    series.time,
                    _studentize(series),
                    color=colors[index % len(colors)],
                    linewidth=0.8,
                    label=f"{label} (normalized)",
                )
            return

        extra_dims = [dim for dim in amp.dims if dim not in ("time", "channel")]
        if extra_dims:
            for dim in extra_dims:
                for index, value in enumerate(amp[dim].values):
                    series = amp.sel(channel=channel, **{dim: value})
                    ax.plot(
                        series.time,
                        _studentize(series),
                        linewidth=0.8,
                        label=f"{dim}={value} (normalized)",
                    )
                    if index >= 2:
                        break
                break
        else:
            series = amp.sel(channel=channel)
            ax.plot(series.time, _studentize(series), "k-", linewidth=0.8,
                    label="signal (normalized)")

    def _prompt_save_path(self, title: str, file_filter: str, suffix: str) -> str:
        """Prompt for an output path using a non-native dialog.

        The native macOS save panel can lose keyboard focus when the GUI is
        launched from a notebook event loop, so the Qt dialog is used instead.
        """
        dialog = QFileDialog(self, title)
        dialog.setOptions(QFileDialog.DontUseNativeDialog)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter(file_filter)
        dialog.setDefaultSuffix(suffix)
        dialog.setModal(True)
        if dialog.exec() != QFileDialog.Accepted:
            return ""
        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def _export_to_excel(self) -> None:
        path = self._prompt_save_path(
            "Export Data Quality to Excel", "Excel Workbook (*.xlsx)", "xlsx"
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            self.export_to_excel(path)
            self._status_bar.showMessage(f"Exported data quality table to {path}.")
        except Exception as exc:
            QMessageBox.critical(
                self, "Export to Excel", f"Failed to export Excel file:\n{exc}"
            )

    def _export_to_pdf(self) -> None:
        path = self._prompt_save_path(
            "Export Data Quality to PDF", "PDF Files (*.pdf)", "pdf"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            self.export_to_pdf(path)
            self._status_bar.showMessage(f"Exported data quality report to {path}.")
        except Exception as exc:
            QMessageBox.critical(
                self, "Export to PDF", f"Failed to export PDF file:\n{exc}"
            )

    def _export_to_html(self) -> None:
        path = self._prompt_save_path(
            "Export Data Quality to HTML", "HTML Files (*.html)", "html"
        )
        if not path:
            return
        if not path.lower().endswith(".html"):
            path += ".html"
        try:
            self.export_to_html(path)
            self._status_bar.showMessage(f"Exported data quality report to {path}.")
        except Exception as exc:
            QMessageBox.critical(
                self, "Export to HTML", f"Failed to export HTML file:\n{exc}"
            )

    def export_to_excel(self, path: str | Path) -> None:
        used_sheets: set[str] = set()
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for row, rec in enumerate(getattr(self.dataset, "dataset", [])):
                label = _recording_labels(self.dataset)[row]
                sheet = _safe_sheet_name(label, used_sheets)
                metadata = pd.DataFrame(
                    [
                        ["File index", row],
                        ["Subject", _subject_name(rec)],
                        ["Scan number", _scan_number(rec)],
                    ],
                    columns=["Field", "Value"],
                )
                table = self._quality_table_for_recording(row)
                metadata.to_excel(
                    writer, sheet_name=sheet, index=False, header=False, startrow=0
                )
                table.to_excel(writer, sheet_name=sheet, index=False, startrow=5)

    def _quality_table_for_recording(self, row: int) -> pd.DataFrame:
        rec = self.dataset.dataset[row]
        channels = [str(ch) for ch in rec["amp"].channel.values]
        table = pd.DataFrame({"channel": channels})
        for metric, label in QUALITY_METRICS.items():
            _ = label
            scalar = self._metric_for_recording(row, metric)[0]
            values = scalar.sel(channel=rec["amp"].channel.values).to_numpy()
            table[_short_metric_label(metric)] = np.asarray(values, dtype=float)
        return table

    def export_to_pdf(self, path: str | Path) -> None:
        with PdfPages(path) as pdf:
            for section, factory in (
                ("Channel Quality", self._make_export_channel_figure),
                ("Time-series artifacts", self._make_export_heatmap_figure),
            ):
                title_fig = self._make_section_title_figure(section)
                pdf.savefig(title_fig)
                plt.close(title_fig)
                for row in range(len(getattr(self.dataset, "dataset", []))):
                    for metric in QUALITY_METRICS:
                        fig = factory(row, metric)
                        pdf.savefig(fig)
                        plt.close(fig)

    def export_to_html(self, path: str | Path) -> None:
        path = Path(path)
        parts = [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>Data Quality Report</title>",
            "<style>body{font-family:Arial,sans-serif;margin:24px;} "
            "h1,h2,h3{color:#1f2937;} img{max-width:100%;height:auto;"
            "border:1px solid #ddd;margin-bottom:20px;}</style>",
            "</head><body><h1>Data Quality Report</h1>",
        ]
        for section, factory in (
            ("Channel Quality", self._make_export_channel_figure),
            ("Time-series artifacts", self._make_export_heatmap_figure),
        ):
            parts.append(f"<h2>{escape(section)}</h2>")
            for row in range(len(getattr(self.dataset, "dataset", []))):
                rec_label = _recording_labels(self.dataset)[row]
                parts.append(f"<h3>{escape(rec_label)}</h3>")
                for metric in QUALITY_METRICS:
                    fig = factory(row, metric)
                    data_uri = self._figure_to_png_data_uri(fig)
                    plt.close(fig)
                    parts.append(
                        f"<h4>{escape(QUALITY_METRICS[metric])}</h4>"
                        f"<img alt='{escape(section)} {escape(rec_label)} "
                        f"{escape(QUALITY_METRICS[metric])}' src='{data_uri}'>"
                    )
        parts.append("</body></html>")
        path.write_text("\n".join(parts), encoding="utf-8")

    def _make_section_title_figure(self, title: str) -> Figure:
        fig = Figure(figsize=(8.5, 11))
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=24)
        ax.set_axis_off()
        return fig

    def _make_export_channel_figure(self, row: int, metric: str) -> Figure:
        fig = Figure(figsize=(8.5, 6))
        ax = fig.add_subplot(1, 1, 1)
        rec = self.dataset.dataset[row]
        display_mode = self._display_mode("channel")
        scalar = self._metric_display_scalar(row, metric, display_mode)
        vmin, vmax, cmap = _metric_limits(metric, display_mode)
        self._draw_metric_probe_on_figure(fig, rec, scalar, ax, vmin, vmax, cmap)
        ax.set_title(
            f"{_recording_labels(self.dataset)[row]} - {QUALITY_METRICS[metric]}"
        )
        return fig

    def _make_export_heatmap_figure(self, row: int, metric: str) -> Figure:
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        display_mode = self._display_mode("artifact")
        values = self._metric_display_values(row, metric, display_mode)
        vmin, vmax, cmap = _metric_limits(metric, display_mode)
        matrix = values.transpose("time", "channel").to_numpy().T
        image = ax.pcolormesh(
            values.time,
            np.arange(len(values.channel)),
            matrix,
            shading="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_yticks(np.arange(len(values.channel)))
        ax.set_yticklabels([str(ch) for ch in values.channel.values], fontsize=7)
        ax.set_xlabel("time / s")
        ax.set_ylabel("channel")
        ax.set_title(
            f"{_recording_labels(self.dataset)[row]} - {QUALITY_METRICS[metric]}"
        )
        fig.colorbar(image, ax=ax, shrink=0.7)
        return fig

    def _draw_metric_probe_on_figure(
        self,
        fig: Figure,
        rec: Any,
        values: xr.DataArray,
        ax,
        vmin: float | None,
        vmax: float | None,
        cmap_name: str,
    ) -> list[Any]:
        data = rec["amp"]
        channel_lines = draw_probe(rec, data, ax, plot_on_scalp=True)
        values = values.sel(channel=data.channel.values)
        arr = np.asarray(values.to_numpy(), dtype=float)
        if vmin is None:
            vmin = float(np.nanmin(arr))
        if vmax is None:
            vmax = float(np.nanmax(arr))
        if vmin == vmax:
            vmax = vmin + 1.0
        cmap = plt.get_cmap(cmap_name)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for line, value in zip(channel_lines, arr):
            line.set_color(cmap(norm(value)))
            line.set_linewidth(3.0)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, shrink=0.7)
        return channel_lines

    def _figure_to_png_data_uri(self, fig: Figure) -> str:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _draw_artifact_heatmaps(self) -> None:
        self.artifact_figure.clear()
        rows = self._selected_rows_for_prefix("artifact")
        metrics = self._selected_metrics("artifact")
        plots = [(row, metric) for row in rows for metric in metrics]
        if not plots:
            ax = self.artifact_figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No recording selected.", ha="center", va="center")
            ax.set_axis_off()
            self.artifact_canvas.draw_idle()
            return

        display_mode = self._display_mode("artifact")
        grid_rows, grid_cols = self._subplot_grid(len(plots))
        labels = _recording_labels(self.dataset)
        for index, (row, metric) in enumerate(plots, start=1):
            ax = self.artifact_figure.add_subplot(grid_rows, grid_cols, index)
            values = self._metric_display_values(row, metric, display_mode)
            vmin, vmax, cmap = _metric_limits(metric, display_mode)
            matrix = values.transpose("time", "channel").to_numpy().T
            image = ax.pcolormesh(
                values.time,
                np.arange(len(values.channel)),
                matrix,
                shading="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_yticks(np.arange(len(values.channel)))
            ax.set_yticklabels([str(ch) for ch in values.channel.values], fontsize=7)
            ax.set_xlabel("time / s")
            ax.set_ylabel("channel")
            title = _short_metric_label(metric)
            if display_mode == "mask":
                title = f"{title} mask"
            if len(rows) > 1:
                title = f"{labels[row]}\n{title}"
            ax.set_title(title)
            self.artifact_figure.colorbar(image, ax=ax, shrink=0.7)
        self.artifact_canvas.draw_idle()
        self._status_bar.showMessage("Time-series artifacts updated.")


_OPEN_WINDOWS: List[DataQualityManager] = []


def _release_window(window: DataQualityManager) -> None:
    if window in _OPEN_WINDOWS:
        _OPEN_WINDOWS.remove(window)


def _active_ipython():
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def data_quality_manager(
    data: Any,
    block: Optional[bool] = None,
) -> DataQualityManager:
    app = QApplication.instance() or QApplication(sys.argv or ["data_quality_manager"])
    app.setQuitOnLastWindowClosed(False)

    shell = _active_ipython()
    if block is None:
        block = shell is None

    window = DataQualityManager(data)
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

    return window
