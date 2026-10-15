"""Linked time-course and probe viewer for channel/time masks."""
from __future__ import annotations

from itertools import product
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QScrollBar,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from pyBrainAnalyzIR.dataclasses.dataset import DataSet
from pyBrainAnalyzIR.vis.plot_nirs_inline import draw_probe

_CHANNEL_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
)

_MASK_TOOLS: dict[str, dict[str, Any]] = {
    "Peak spectral power": {
        "method": "psp",
        "summary": "Exclude windows with low peak spectral power in the cardiac band.",
        "options": [
            ("window_length", "Window length (s)", "float", 10.0),
            ("psp_thresh", "PSP threshold", "float", 0.1),
            ("cardiac_fmin", "Cardiac minimum (Hz)", "float", 0.5),
            ("cardiac_fmax", "Cardiac maximum (Hz)", "float", 2.5),
        ],
    },
    "Scalp-coupling index": {
        "method": "sci",
        "summary": "Exclude windows whose scalp-coupling index is below threshold.",
        "options": [
            ("window_length", "Window length (s)", "float", 10.0),
            ("sci_thresh", "SCI threshold", "float", 0.8),
            ("cardiac_fmin", "Cardiac minimum (Hz)", "float", 0.5),
            ("cardiac_fmax", "Cardiac maximum (Hz)", "float", 2.5),
        ],
    },
    "Signal-to-noise ratio": {
        "method": "snr",
        "summary": "Exclude channels with signal-to-noise ratio below threshold.",
        "options": [("snr_thresh", "SNR threshold", "float", 2.0)],
    },
    "Mean amplitude range": {
        "method": "mean_amp",
        "summary": "Keep channel/subtype means inside the supplied amplitude range.",
        "options": [
            ("minimum", "Minimum amplitude", "float", 0.0),
            ("maximum", "Maximum amplitude", "float", 1.0),
        ],
    },
    "Source-detector distance": {
        "method": "sd_dist",
        "summary": "Keep channels whose optode distance is inside the supplied range.",
        "options": [
            ("minimum_cm", "Minimum distance (cm)", "float", 0.0),
            ("maximum_cm", "Maximum distance (cm)", "float", 4.5),
        ],
    },
    "Global variance of temporal derivative": {
        "method": "gvtd",
        "summary": "Identify time points with excessive global variance of the temporal derivative.",
        "options": [
            (
                "stat_type",
                "Threshold statistic",
                "choice",
                ["default", "histogram_mode", "kdensity_mode", "parabolic_mode", "median", "mean", "MAD"],
            ),
            ("n_std", "Threshold multiplier", "int", 10),
        ],
    },
    "Motion artifact detection": {
        "method": "id_motion",
        "summary": "Detect rapid signal changes and mask a time window around each artifact.",
        "options": [
            ("t_motion", "Detection interval (s)", "float", 0.5),
            ("t_mask", "Mask interval (s)", "float", 1.0),
            ("stdev_thresh", "Standard-deviation threshold", "float", 50.0),
            ("amp_thresh", "Amplitude threshold", "float", 5.0),
        ],
    },
    "Motion-mask refinement": {
        "method": "id_motion_refine",
        "summary": "Detect motion, then collapse across subtypes per channel or across all channels.",
        "options": [
            ("t_motion", "Detection interval (s)", "float", 0.5),
            ("t_mask", "Mask interval (s)", "float", 1.0),
            ("stdev_thresh", "Standard-deviation threshold", "float", 50.0),
            ("amp_thresh", "Amplitude threshold", "float", 5.0),
            ("operator", "Collapse mode", "choice", ["by_channel", "all"]),
        ],
    },
    "Standard-deviation outliers": {
        "method": "detect_outliers_std",
        "summary": "Exclude samples with unusual rolling standard deviation.",
        "options": [
            ("t_window", "Window length (s)", "float", 5.0),
            ("iqr_threshold", "IQR threshold", "float", 2.0),
        ],
    },
    "Gradient outliers": {
        "method": "detect_outliers_grad",
        "summary": "Exclude samples with unusually large or small signal gradients.",
        "options": [("iqr_threshold", "IQR threshold", "float", 1.5)],
    },
    "Combined outliers": {
        "method": "detect_outliers",
        "summary": "Combine standard-deviation and gradient outlier masks.",
        "options": [
            ("t_window_std", "Standard-deviation window (s)", "float", 5.0),
            ("iqr_threshold_std", "Standard-deviation IQR", "float", 2.0),
            ("iqr_threshold_grad", "Gradient IQR", "float", 1.5),
        ],
    },
    "Baseline shifts": {
        "method": "detect_baselineshift",
        "summary": "Detect baseline shifts using the combined outlier mask as input.",
        "options": [
            ("t_window_std", "Standard-deviation window (s)", "float", 5.0),
            ("iqr_threshold_std", "Standard-deviation IQR", "float", 2.0),
            ("iqr_threshold_grad", "Gradient IQR", "float", 1.5),
        ],
    },
}

_MANUAL_TOOLS = (
    "Manual removal of channels",
    "Manual removal of time points (all channels)",
    "Manual removal of time points (current channel)",
)


def _as_dataset(data: Any) -> DataSet:
    if isinstance(data, DataSet):
        return data
    return DataSet(data)


def _recordings(data: Any) -> list[Any]:
    if hasattr(data, "dataset"):
        return list(data.dataset)
    if data is None:
        return []
    return [data]


def _time_course_keys(recording: Any) -> list[str]:
    return [
        key
        for key, value in getattr(recording, "timeseries", {}).items()
        if isinstance(value, xr.DataArray)
        and "time" in value.dims
        and "channel" in value.dims
    ]


def _plain_array(array: xr.DataArray) -> np.ndarray:
    try:
        array = array.pint.dequantify()
    except (AttributeError, TypeError, ValueError):
        pass
    return np.asarray(array.to_numpy())


def _time_values(array: xr.DataArray) -> np.ndarray:
    values = _plain_array(array["time"])
    return values.astype(float)


def _coordinate_labels(array: xr.DataArray, dim: str) -> list[Any]:
    if dim not in array.coords:
        return list(range(array.sizes[dim]))
    return list(array[dim].values)


def _subtype_options(
    array: xr.DataArray,
) -> list[tuple[dict[str, Any], str]]:
    dimensions = [dim for dim in array.dims if dim not in ("time", "channel")]
    if not dimensions:
        return [({}, "")]
    values = [_coordinate_labels(array, dim) for dim in dimensions]
    options: list[tuple[dict[str, Any], str]] = []
    for combination in product(*values):
        selection = dict(zip(dimensions, combination))
        label = ", ".join(f"{dim}: {value}" for dim, value in selection.items())
        options.append((selection, label))
    return options


def _channel_time_data(array: xr.DataArray) -> xr.DataArray:
    """Order Cedalion's channel/wavelength/time algorithms as they expect."""
    if "channel" not in array.dims or "time" not in array.dims:
        raise ValueError("The selected time course must have channel and time dimensions.")
    dimensions = ["channel"]
    dimensions.extend(dim for dim in array.dims if dim not in ("channel", "time"))
    dimensions.append("time")
    return array.transpose(*dimensions)


def _display_label(data: xr.DataArray, channel_index: int) -> str:
    channel = _coordinate_labels(data, "channel")[channel_index]
    if "source" in data.coords and "detector" in data.coords:
        source, detector = _channel_endpoints(data, channel_index)
        return f"{source}: {detector}"
    return str(channel)


def _channel_endpoints(data: xr.DataArray, channel_index: int) -> tuple[str, str]:
    if "source" in data.coords and "detector" in data.coords:
        source = str(data["source"].isel(channel=channel_index).item())
        detector = str(data["detector"].isel(channel=channel_index).item())
    else:
        channel = str(_coordinate_labels(data, "channel")[channel_index])
        match = re.match(r"^(S\d+)(D\d+)$", channel, flags=re.IGNORECASE)
        if match is None:
            return channel, channel
        source, detector = match.groups()
    source = re.sub(r"^S(\d+)$", r"Source-\1", source, flags=re.IGNORECASE)
    detector = re.sub(r"^D(\d+)$", r"Detector \1", detector, flags=re.IGNORECASE)
    return source, detector


def _channel_identity(data: xr.DataArray, channel_index: int) -> tuple[str, str]:
    source, detector = _channel_endpoints(data, channel_index)
    return source.casefold(), detector.casefold()


def _subject_value(recording: Any) -> str | None:
    metadata = getattr(recording, "meta_data", {}) or {}
    normalized = {str(key).casefold(): value for key, value in metadata.items()}
    for key in ("subject", "subj", "subjid", "subjectid", "id", "name"):
        value = normalized.get(key)
        try:
            missing = value is None or bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = value is None
        if not missing and str(value).strip():
            return str(value).strip().casefold()
    return None


def _probe_label(optode: str) -> str:
    source_match = re.fullmatch(r"Source-(\d+)", optode, flags=re.IGNORECASE)
    if source_match:
        return f"S{source_match.group(1)}"
    detector_match = re.fullmatch(r"Detector\s+(\d+)", optode, flags=re.IGNORECASE)
    if detector_match:
        return f"D{detector_match.group(1)}"
    return optode


def _point_segment_distance(
    point_x: float,
    point_y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> float:
    dx = end_x - start_x
    dy = end_y - start_y
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return float(np.hypot(point_x - start_x, point_y - start_y))
    projection = np.clip(
        ((point_x - start_x) * dx + (point_y - start_y) * dy)
        / length_squared,
        0.0,
        1.0,
    )
    closest_x = start_x + projection * dx
    closest_y = start_y + projection * dy
    return float(np.hypot(point_x - closest_x, point_y - closest_y))


def _subtype_color(selection: dict[str, Any], index: int) -> str:
    for dim, value in selection.items():
        text = str(value).lower().replace(" ", "")
        if dim.lower() in ("chromo", "chromophore", "hemoglobin", "species"):
            if "hbo" in text or "oxy" in text:
                return "#c62828"
            if "hbr" in text or "deoxy" in text:
                return "#2166ac"
            if "hbt" in text or "total" in text:
                return "#238b45"
        if dim.lower() in ("wavelength", "wl"):
            # Keep wavelengths in a red family while making them distinguishable.
            shades = ("#8b0000", "#e53935", "#ff8a80", "#b71c1c")
            return shades[index % len(shades)]
    colors = ("#c62828", "#2166ac", "#238b45", "#6a3d9a", "#e08214")
    return colors[index % len(colors)]


def _mask_values(mask: xr.DataArray, data: xr.DataArray, channel_index: int) -> np.ndarray:
    selected = mask
    if "channel" in selected.dims:
        channel = data["channel"].isel(channel=channel_index).item()
        try:
            selected = selected.sel(channel=channel)
        except (KeyError, ValueError):
            return np.ones(data.sizes["time"], dtype=bool)
    if "time" in selected.dims:
        target_time = data["time"]
        try:
            selected = selected.reindex(time=target_time, method="nearest")
        except (KeyError, ValueError):
            if selected.sizes["time"] != data.sizes["time"]:
                return np.ones(data.sizes["time"], dtype=bool)
        selected = selected.transpose(
            "time", *(dim for dim in selected.dims if dim != "time")
        )
        values = np.asarray(_plain_array(selected), dtype=object)
        good = np.asarray([_is_good(value) for value in values.reshape(-1)])
        good = good.reshape(values.shape)
        if good.ndim > 1:
            good = np.all(good, axis=tuple(range(1, good.ndim)))
        return good
    values = np.asarray(_plain_array(selected), dtype=object).reshape(-1)
    if not values.size:
        return np.ones(data.sizes["time"], dtype=bool)
    good = all(_is_good(value) for value in values)
    return np.full(data.sizes["time"], good, dtype=bool)


def _is_good(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return bool(float(value) != 0)
    except (TypeError, ValueError):
        return True


def _control_icon(kind: str, direction: int) -> QIcon:
    size = 36
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("#333333"), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    center_x = size / 2
    center_y = size / 2

    if kind == "Reset":
        painter.drawArc(6, 6, 24, 24, 35 * 16, 295 * 16)
        painter.drawLine(6, 6, 6, 15)
        painter.drawLine(6, 6, 15, 6)
    elif kind == "Time":
        painter.drawEllipse(5, 4, 21, 21)
        painter.drawLine(22, 22, 32, 32)
        painter.drawLine(10, 14, 21, 14)
        if direction > 0:
            painter.drawLine(15.5, 8.5, 15.5, 19.5)
    elif kind == "Amplitude":
        amplitude = 10 if direction > 0 else 4
        path = QPainterPath()
        path.moveTo(3, center_y)
        for x in range(4, 34):
            y = center_y - amplitude * np.sin((x - 3) * 2 * np.pi / 30)
            path.lineTo(x, float(y))
        painter.drawPath(path)
    else:
        gap = 8 if direction > 0 else 3.5
        for offset in (-gap, 0, gap):
            painter.drawLine(5, int(center_y + offset), 31, int(center_y + offset))
    painter.end()
    return QIcon(pixmap)


class DataMaskingManager(QMainWindow):
    """Browse signals and build channel/time masks from Cedalion quality tools."""

    submitted = Signal()

    def __init__(self, data: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Data Masking Manager")
        self.resize(1320, 850)
        self.dataset = _as_dataset(data)
        self.recordings = _recordings(self.dataset)
        self._subtype_checks: list[QCheckBox] = []
        self._probe_lines: list[Any] = []
        self._channel_labels: list[str] = []
        self._channel_colors: list[str] = []
        self._selected_channel = 0
        self._viewport_height = 24.0
        self._visible_channels = 24
        self._time_zoom = 1.0
        self._amplitude_zoom = 0.7
        self._line_spacing = 1.0
        self._initial_amplitude_zoom = self._amplitude_zoom
        self._time_start = 0.0
        self._plotting = False
        self._applied_masks: list[tuple[int, str, xr.DataArray]] = []
        self._mask_entries: list[dict[str, Any]] = []
        self._next_mask_id = 0
        self._manual_drafts: dict[tuple[int, str], xr.DataArray] = {}
        self._staged_dirty = False
        self._submitted = False
        self._preview_masks: dict[int, xr.DataArray] = {}
        self._manual_mask_draft: xr.DataArray | None = None
        self._manual_drag_start: float | None = None
        self._manual_drag_rect: Rectangle | None = None
        self._displayed_good: np.ndarray | None = None
        self._option_widgets: dict[str, QWidget] = {}

        root = QWidget(self)
        layout = QVBoxLayout(root)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Recording:"))
        self.recording_combo = QComboBox(root)
        for index, recording in enumerate(self.recordings):
            subject = getattr(recording, "meta_data", {}).get("subject", "")
            label = f"[{index}] {subject}" if subject else f"Recording {index + 1}"
            self.recording_combo.addItem(str(label), index)
        self.recording_combo.currentIndexChanged.connect(self._on_recording_changed)
        controls.addWidget(self.recording_combo)

        controls.addWidget(QLabel("Data:"))
        self.data_combo = QComboBox(root)
        self.data_combo.currentIndexChanged.connect(self._on_data_changed)
        controls.addWidget(self.data_combo)
        controls.addWidget(QLabel("Subtypes:"))
        self.subtype_widget = QWidget(root)
        self.subtype_layout = QHBoxLayout(self.subtype_widget)
        self.subtype_layout.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.subtype_widget)

        layout.addLayout(controls)

        zoom_controls = QHBoxLayout()
        self._add_zoom_control(zoom_controls, "Reset", 0, self._reset_view)
        self._add_zoom_control(zoom_controls, "Time", -1, self._change_time_zoom)
        self._add_zoom_control(zoom_controls, "Time", 1, self._change_time_zoom)
        zoom_controls.addSpacing(12)
        self._add_zoom_control(zoom_controls, "Amplitude", -1, self._change_amplitude_zoom)
        self._add_zoom_control(zoom_controls, "Amplitude", 1, self._change_amplitude_zoom)
        zoom_controls.addSpacing(12)
        self._add_zoom_control(zoom_controls, "Spacing", -1, self._change_spacing)
        self._add_zoom_control(zoom_controls, "Spacing", 1, self._change_spacing)
        zoom_controls.addStretch()
        layout.addLayout(zoom_controls)

        body = QGridLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(3)
        self.figure = Figure(figsize=(12, 8), constrained_layout=False)
        self.probe_figure = Figure(figsize=(3, 7), constrained_layout=True)
        self._create_plot_axes()
        self.canvas = FigureCanvas(self.figure)
        self.probe_canvas = FigureCanvas(self.probe_figure)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.canvas.mpl_connect("button_press_event", self._on_label_click)
        self.canvas.mpl_connect("pick_event", self._on_label_pick)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("button_release_event", self._on_plot_release)
        self.probe_canvas.mpl_connect("pick_event", self._on_probe_pick)
        self.probe_canvas.mpl_connect("button_press_event", self._on_probe_click)
        self.vertical_scroll = QScrollBar(Qt.Orientation.Vertical, root)
        self.vertical_scroll.valueChanged.connect(self._on_vertical_scroll)
        self.horizontal_scroll = QScrollBar(Qt.Orientation.Horizontal, root)
        self.horizontal_scroll.valueChanged.connect(self._on_horizontal_scroll)
        self.plot_layout = body
        body.addWidget(self.canvas, 0, 0)
        body.addWidget(self.vertical_scroll, 0, 1)
        right_panel = QWidget(root)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.probe_canvas, 1)
        right_layout.addWidget(self._build_processing_panel(right_panel), 2)
        body.addWidget(right_panel, 0, 2)
        body.addWidget(self.horizontal_scroll, 1, 0)
        body.setColumnStretch(0, 6)
        body.setColumnStretch(2, 2)
        body.setRowStretch(0, 1)
        layout.addLayout(body, stretch=1)

        footer = QHBoxLayout()
        footer.addStretch()
        self.cancel_button = QPushButton("Cancel", root)
        self.cancel_button.clicked.connect(self._cancel)
        footer.addWidget(self.cancel_button)
        self.submit_button = QPushButton("Submit", root)
        self.submit_button.clicked.connect(self._submit)
        footer.addWidget(self.submit_button)
        layout.addLayout(footer)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar(self))

        self.recording_combo.setEnabled(bool(self.recordings))
        self._load_existing_masks()
        self._refresh_applied_mask_list()
        if self.recordings:
            self._refresh_data_types()
            self._on_tool_changed(self.mask_tool_combo.currentText())
        else:
            self.statusBar().showMessage("No recordings are available.")

    def _build_processing_panel(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(2, 2, 2, 2)

        self.mask_tool_combo = QComboBox(panel)
        self.mask_tool_combo.addItems([*_MASK_TOOLS, *_MANUAL_TOOLS])
        self.mask_tool_combo.currentTextChanged.connect(self._on_tool_changed)
        panel_layout.addWidget(self.mask_tool_combo)

        self.tool_summary = QLabel(panel)
        self.tool_summary.setWordWrap(True)
        panel_layout.addWidget(self.tool_summary)

        self.settings_scroll = QScrollArea(panel)
        self.settings_scroll.setWidgetResizable(True)
        self.settings_container = QWidget(self.settings_scroll)
        self.settings_form = QFormLayout(self.settings_container)
        self.settings_form.setContentsMargins(0, 0, 0, 0)
        self.settings_scroll.setWidget(self.settings_container)
        panel_layout.addWidget(self.settings_scroll, 1)

        action_row = QHBoxLayout()
        self.preview_button = QPushButton("Show preview", panel)
        self.preview_button.clicked.connect(self._preview_mask)
        action_row.addWidget(self.preview_button)
        self.add_mask_button = QPushButton("Add data mask", panel)
        self.add_mask_button.setEnabled(False)
        self.add_mask_button.clicked.connect(self._add_preview_mask)
        action_row.addWidget(self.add_mask_button)
        panel_layout.addLayout(action_row)

        masks_group = QGroupBox("Applied masks", panel)
        masks_layout = QHBoxLayout(masks_group)
        self.applied_mask_list = QListWidget(masks_group)
        masks_layout.addWidget(self.applied_mask_list, 1)
        self.remove_mask_button = QPushButton("Remove selected", masks_group)
        self.remove_mask_button.clicked.connect(self._remove_applied_mask)
        masks_layout.addWidget(self.remove_mask_button)
        panel_layout.addWidget(masks_group, 1)
        return panel

    def _add_zoom_control(self, layout, label: str, direction: int, callback) -> None:
        button = QToolButton(self)
        button.setIcon(_control_icon(label, direction))
        button.setIconSize(button.iconSize() * 1.35)
        button.setStyleSheet(
            "QToolButton { background-color: #e6e6e6; border: 1px solid #b8b8b8; "
            "border-radius: 3px; padding: 2px; }"
            "QToolButton:hover { background-color: #f2f2f2; }"
            "QToolButton:pressed { background-color: #d0d0d0; }"
        )
        action = "Increase" if direction > 0 else "Decrease"
        if label == "Reset":
            button.setToolTip("Reset view")
            button.setAccessibleName("Reset view")
        else:
            button.setToolTip(f"{action} {label.lower()}")
            button.setAccessibleName(f"{action} {label.lower()}")
        button.setAutoRaise(True)
        button.clicked.connect(lambda: callback(direction))
        layout.addWidget(button)

    def _current_recording(self) -> Any | None:
        index = self.recording_combo.currentData()
        if index is None or not 0 <= int(index) < len(self.recordings):
            return None
        return self.recordings[int(index)]

    def _current_recording_index(self) -> int | None:
        index = self.recording_combo.currentData()
        if index is None or not 0 <= int(index) < len(self.recordings):
            return None
        return int(index)

    def _load_existing_masks(self) -> None:
        by_key: dict[str, dict[int, xr.DataArray]] = {}
        for index, recording in enumerate(self.recordings):
            for key, mask in getattr(recording, "masks", {}).items():
                by_key.setdefault(str(key), {})[index] = mask
        for key, masks in by_key.items():
            self._mask_entries.append(
                {
                    "id": self._next_mask_id,
                    "name": f"Existing mask: {key}",
                    "masks": masks,
                    "existing_keys": {index: key for index in masks},
                    "existing": True,
                }
            )
            self._next_mask_id += 1

    def _create_plot_axes(self) -> None:
        grid = self.figure.add_gridspec(
            2, 3, height_ratios=(0.65, 4.0), width_ratios=(2.0, 5.5, 1.2),
            hspace=0.03, wspace=0.02,
        )
        self.figure.add_subplot(grid[0, 0]).set_axis_off()
        self.stimulus_ax = self.figure.add_subplot(grid[0, 1])
        self.label_ax = self.figure.add_subplot(grid[1, 0])
        self.waterfall_ax = self.figure.add_subplot(
            grid[1, 1], sharex=self.stimulus_ax, sharey=self.label_ax
        )
        self.summary_ax = self.figure.add_subplot(grid[1, 2], sharey=self.label_ax)
        self.summary_ax.set_axis_off()
        self.probe_ax = self.probe_figure.add_subplot(1, 1, 1)

    def _current_data(self) -> xr.DataArray | None:
        recording = self._current_recording()
        key = self.data_combo.currentData()
        if recording is None or key is None:
            return None
        return recording.timeseries.get(key)

    def _refresh_data_types(self) -> None:
        recording = self._current_recording()
        self.data_combo.blockSignals(True)
        self.data_combo.clear()
        if recording is not None:
            for key in _time_course_keys(recording):
                self.data_combo.addItem(str(key), key)
        self.data_combo.blockSignals(False)
        self._configure_subtypes()
        self._draw()

    def _configure_subtypes(self) -> None:
        while self.subtype_layout.count():
            item = self.subtype_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._subtype_checks.clear()
        data = self._current_data()
        if data is None:
            return
        options = _subtype_options(data)
        for index, (selection, label) in enumerate(options):
            checkbox = QCheckBox(label or "All", self.subtype_widget)
            checkbox.setProperty("selection", selection)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self._draw)
            self.subtype_layout.addWidget(checkbox)
            self._subtype_checks.append(checkbox)
        self.subtype_layout.addStretch()

    def _on_recording_changed(self, _index: int) -> None:
        self._time_start = 0.0
        self._selected_channel = 0
        self._load_manual_draft()
        self._refresh_applied_mask_list()
        self._refresh_data_types()

    def _on_data_changed(self, _index: int) -> None:
        self._configure_subtypes()
        self._time_start = 0.0
        self._draw()

    def _on_tool_changed(self, tool_name: str) -> None:
        self._preview_masks.clear()
        self._load_manual_draft()
        is_manual = tool_name in _MANUAL_TOOLS
        self.preview_button.setEnabled(not is_manual)
        self.add_mask_button.setEnabled(self._has_manual_draft(tool_name))
        self.mask_tool_combo.setToolTip(
            "Right-click a channel or drag a time range to mark data for removal."
            if is_manual
            else ""
        )
        if is_manual:
            self.tool_summary.setText(self._manual_tool_summary(tool_name))
            while self.settings_form.rowCount():
                self.settings_form.removeRow(0)
            self._option_widgets.clear()
            if hasattr(self, "waterfall_ax"):
                self._draw()
            return
        tool = _MASK_TOOLS.get(tool_name)
        if tool is None or not hasattr(self, "settings_form"):
            return
        while self.settings_form.rowCount():
            self.settings_form.removeRow(0)
        self._option_widgets.clear()
        for name, label, kind, default in tool["options"]:
            if kind == "choice":
                widget = QComboBox(self.settings_container)
                widget.addItems(default)
            elif kind == "int":
                widget = QSpinBox(self.settings_container)
                widget.setRange(1, 1000000)
                widget.setValue(default)
            else:
                widget = QDoubleSpinBox(self.settings_container)
                widget.setDecimals(6)
                widget.setRange(-1e12, 1e12)
                widget.setSingleStep(0.1)
                widget.setValue(default)
            self.settings_form.addRow(label, widget)
            self._option_widgets[name] = widget
        self.tool_summary.setText(tool["summary"])
        if hasattr(self, "waterfall_ax"):
            self._draw()

    @staticmethod
    def _manual_tool_summary(tool_name: str) -> str:
        if tool_name == _MANUAL_TOOLS[0]:
            return "Right-click a channel label, trace, or probe channel to choose which recordings should mask that channel."
        if tool_name == _MANUAL_TOOLS[1]:
            return "Right-click and drag across the time-series plot to mask that time range for every channel."
        return "Right-click and drag across the time-series plot to mask that range for the currently selected channel."

    def _tool_options(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, widget in self._option_widgets.items():
            if isinstance(widget, QComboBox):
                values[name] = widget.currentText()
            elif isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                values[name] = widget.value()
        return values

    def _create_mask(self) -> xr.DataArray:
        recording = self._current_recording()
        if recording is None:
            raise ValueError("Select a recording before creating a mask.")
        data = getattr(recording, "timeseries", {}).get("amp")
        if not isinstance(data, xr.DataArray):
            raise ValueError("Mask tools require an 'amp' time series in each recording.")

        from cedalion import units
        from cedalion.sigproc import quality

        method = _MASK_TOOLS[self.mask_tool_combo.currentText()]["method"]
        options = self._tool_options()
        if method in ("psp", "sci"):
            if options["window_length"] <= 0:
                raise ValueError("Window length must be greater than zero.")
            if options["cardiac_fmin"] < 0 or options["cardiac_fmax"] <= options["cardiac_fmin"]:
                raise ValueError("Cardiac frequencies must be non-negative and increasing.")
        if method == "mean_amp" and options["maximum"] <= options["minimum"]:
            raise ValueError("Maximum amplitude must exceed minimum amplitude.")
        if method == "sd_dist" and (
            options["minimum_cm"] < 0
            or options["maximum_cm"] <= options["minimum_cm"]
        ):
            raise ValueError("Distance range must be non-negative and increasing.")
        if method in ("id_motion", "id_motion_refine") and (
            options["t_motion"] <= 0 or options["t_mask"] < 0
        ):
            raise ValueError("Motion intervals must be positive; mask interval may be zero.")
        if method in ("detect_outliers_std", "detect_outliers", "detect_baselineshift"):
            window = options.get("t_window", options.get("t_window_std"))
            if window <= 0:
                raise ValueError("Outlier window length must be greater than zero.")
        if method == "psp":
            ordered_data = _channel_time_data(data)
            _metric, mask = quality.psp(
                ordered_data,
                options["window_length"] * units.s,
                options["psp_thresh"],
                options["cardiac_fmin"] * units.Hz,
                options["cardiac_fmax"] * units.Hz,
            )
        elif method == "sci":
            _metric, mask = quality.sci(
                data,
                options["window_length"] * units.s,
                options["sci_thresh"],
                options["cardiac_fmin"] * units.Hz,
                options["cardiac_fmax"] * units.Hz,
            )
        elif method == "snr":
            _metric, mask = quality.snr(data, options["snr_thresh"])
        elif method == "mean_amp":
            try:
                data_units = data.pint.units
                lower = options["minimum"] * data_units
                upper = options["maximum"] * data_units
            except (AttributeError, TypeError, ValueError):
                lower = options["minimum"]
                upper = options["maximum"]
            _metric, mask = quality.mean_amp(data, (lower, upper))
        elif method == "sd_dist":
            geometry = getattr(recording, "geo3d", None)
            if geometry is None:
                raise ValueError("The selected recording has no 3D probe geometry.")
            _metric, mask = quality.sd_dist(
                data,
                geometry,
                (options["minimum_cm"] * units.cm, options["maximum_cm"] * units.cm),
            )
        elif method == "gvtd":
            _metric, mask = quality.gvtd(
                _channel_time_data(data),
                stat_type=options["stat_type"],
                n_std=options["n_std"],
            )
        elif method == "id_motion":
            mask = quality.id_motion(
                data.pint.dequantify(),
                t_motion=options["t_motion"] * units.s,
                t_mask=options["t_mask"] * units.s,
                stdev_thresh=options["stdev_thresh"],
                amp_thresh=options["amp_thresh"],
            )
        elif method == "id_motion_refine":
            motion_mask = quality.id_motion(
                data.pint.dequantify(),
                t_motion=options["t_motion"] * units.s,
                t_mask=options["t_mask"] * units.s,
                stdev_thresh=options["stdev_thresh"],
                amp_thresh=options["amp_thresh"],
            )
            mask, _info = quality.id_motion_refine(
                motion_mask, operator=options["operator"]
            )
        elif method == "detect_outliers_std":
            mask = quality.detect_outliers_std(
                data, options["t_window"] * units.s, options["iqr_threshold"]
            )
        elif method == "detect_outliers_grad":
            mask = quality.detect_outliers_grad(data, options["iqr_threshold"])
        elif method in ("detect_outliers", "detect_baselineshift"):
            mask = quality.detect_outliers(
                data,
                options["t_window_std"] * units.s,
                options["iqr_threshold_std"],
                options["iqr_threshold_grad"],
            )
            if method == "detect_baselineshift":
                mask = quality.detect_baselineshift(data, mask)
        else:
            raise ValueError(f"Unsupported mask-generating tool: {method}")
        if not isinstance(mask, xr.DataArray):
            raise TypeError(f"{method} did not return an xarray mask.")
        return self._canonical_mask(mask, data)

    def _mask_matrix(self, mask: xr.DataArray, data: xr.DataArray) -> np.ndarray:
        return np.column_stack(
            [_mask_values(mask, data, channel) for channel in range(data.sizes["channel"])]
        )

    def _canonical_mask(self, mask: xr.DataArray, amp: xr.DataArray) -> xr.DataArray:
        good = self._mask_matrix(mask, amp)
        return xr.DataArray(
            good,
            dims=("time", "channel"),
            coords={"time": amp["time"], "channel": amp["channel"]},
        )

    def _base_mask_matrix(self, data: xr.DataArray) -> np.ndarray:
        good = np.ones((data.sizes["time"], data.sizes["channel"]), dtype=bool)
        for _entry_id, _name, mask in self._applied_masks:
            mask_good = self._mask_matrix(mask, data)
            good &= mask_good
        return good

    def _preview_mask(self) -> None:
        self._preview_masks.clear()
        try:
            if not self.recordings:
                raise ValueError("No recordings are available to preview.")
            preview_masks = {
                index: self._create_mask_for_recording(recording)
                for index, recording in enumerate(self.recordings)
            }
            self._preview_masks = preview_masks
            self.add_mask_button.setEnabled(True)
            self.statusBar().showMessage(
                "Mask preview ready for all recordings; purple shows additional data excluded."
            )
            self._draw()
        except Exception as exc:
            self.add_mask_button.setEnabled(False)
            self.statusBar().showMessage(
                f"Could not create mask preview for every recording: {exc}"
            )
            self._draw()

    def _add_preview_mask(self) -> None:
        manual = self.mask_tool_combo.currentText() in _MANUAL_TOOLS
        name = self.mask_tool_combo.currentText()
        masks: dict[int, xr.DataArray] = {}
        if manual:
            for (recording_index, tool_name), draft in self._manual_drafts.items():
                if tool_name == name and not bool(_plain_array(draft).all()):
                    masks[recording_index] = draft.copy(deep=True)
            if not masks:
                return
            for recording_index in list(self._manual_drafts):
                if recording_index[1] == name:
                    del self._manual_drafts[recording_index]
            self._manual_mask_draft = None
        else:
            if len(self._preview_masks) != len(self.recordings):
                return
            masks = {
                index: mask.copy(deep=True)
                for index, mask in self._preview_masks.items()
            }
            self._preview_masks.clear()

        self._mask_entries.append(
            {
                "id": self._next_mask_id,
                "name": name,
                "masks": masks,
                "existing_keys": {},
                "existing": False,
            }
        )
        self._next_mask_id += 1
        self._staged_dirty = True
        self.add_mask_button.setEnabled(False)
        self._refresh_applied_mask_list()
        self.statusBar().showMessage(f"Added {name} for {len(masks)} recording(s).")
        self._draw()

    def _create_mask_for_recording(self, recording: Any) -> xr.DataArray:
        current_index = self._current_recording_index()
        target_index = next(
            (index for index, candidate in enumerate(self.recordings) if candidate is recording),
            None,
        )
        if target_index is None:
            raise ValueError("Recording is not part of this dataset.")
        if target_index == current_index:
            return self._create_mask()

        original_index = self.recording_combo.currentIndex()
        self.recording_combo.blockSignals(True)
        self.recording_combo.setCurrentIndex(target_index)
        try:
            result = self._create_mask()
        finally:
            self.recording_combo.setCurrentIndex(original_index)
            self.recording_combo.blockSignals(False)
        return result

    def _remove_applied_mask(self) -> None:
        row = self.applied_mask_list.currentRow()
        if not 0 <= row < len(self._applied_masks):
            return
        entry_id = self._applied_masks[row][0]
        self._mask_entries = [
            entry for entry in self._mask_entries if entry["id"] != entry_id
        ]
        self._staged_dirty = True
        self._refresh_applied_mask_list()
        self._draw()

    def _refresh_applied_mask_list(self) -> None:
        if not hasattr(self, "applied_mask_list"):
            return
        recording_index = self._current_recording_index()
        self._applied_masks = []
        if recording_index is not None:
            self._applied_masks = [
                (entry["id"], entry["name"], entry["masks"][recording_index])
                for entry in self._mask_entries
                if recording_index in entry["masks"]
            ]
        self.applied_mask_list.clear()
        self.applied_mask_list.addItems([name for _id, name, _mask in self._applied_masks])
        self.remove_mask_button.setEnabled(bool(self._applied_masks))

    def _load_manual_draft(self) -> None:
        recording_index = self._current_recording_index()
        tool_name = self.mask_tool_combo.currentText()
        self._manual_mask_draft = (
            self._manual_drafts.get((recording_index, tool_name))
            if recording_index is not None and tool_name in _MANUAL_TOOLS
            else None
        )

    def _has_manual_draft(self, tool_name: str) -> bool:
        return any(
            draft_tool == tool_name and not bool(_plain_array(draft).all())
            for (_recording_index, draft_tool), draft in self._manual_drafts.items()
        )

    def _ensure_manual_draft(self, recording_index: int | None = None) -> xr.DataArray | None:
        if recording_index is None:
            recording_index = self._current_recording_index()
        if recording_index is None:
            return None
        tool_name = self.mask_tool_combo.currentText()
        draft_key = (recording_index, tool_name)
        if draft_key not in self._manual_drafts:
            amp = getattr(self.recordings[recording_index], "timeseries", {}).get("amp")
            if not isinstance(amp, xr.DataArray):
                raise ValueError("Manual masks require an 'amp' time series.")
            values = np.ones((amp.sizes["time"], amp.sizes["channel"]), dtype=bool)
            self._manual_drafts[draft_key] = xr.DataArray(
                values,
                dims=("time", "channel"),
                coords={"time": amp["time"], "channel": amp["channel"]},
            )
        self._manual_mask_draft = self._manual_drafts[draft_key]
        return self._manual_mask_draft

    def _show_channel_context_menu(self, channel_index: int, global_pos) -> None:
        current_index = self._current_recording_index()
        data = self._current_data()
        if (
            current_index is None
            or data is None
            or not 0 <= channel_index < data.sizes["channel"]
        ):
            return
        menu = QMenu(self)
        actions = {
            "current": menu.addAction("Mask channel in current recording"),
            "subject": menu.addAction("Mask channel in all recordings for this subject"),
            "dataset": menu.addAction("Mask channel in entire dataset"),
        }
        chosen = menu.exec(global_pos)
        scope = next((key for key, action in actions.items() if action is chosen), None)
        if scope is not None:
            try:
                self._mask_channel_for_scope(current_index, channel_index, scope)
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                QMessageBox.warning(self, "Mask channel", str(exc))

    def _mask_channel_for_scope(
        self, source_recording_index: int, channel_index: int, scope: str
    ) -> None:
        source = self.recordings[source_recording_index]
        source_amp = getattr(source, "timeseries", {}).get("amp")
        if not isinstance(source_amp, xr.DataArray):
            raise ValueError("Manual channel masks require an 'amp' time series.")
        if scope == "current":
            targets = [source_recording_index]
        elif scope == "subject":
            source_subject = _subject_value(source)
            if source_subject is None:
                QMessageBox.warning(
                    self,
                    "Mask channel",
                    "This recording has no recognized subject demographic; "
                    "only the current recording can be masked.",
                )
                targets = [source_recording_index]
            else:
                targets = [
                    index
                    for index, recording in enumerate(self.recordings)
                    if _subject_value(recording) == source_subject
                ]
        else:
            targets = list(range(len(self.recordings)))

        source_identity = _channel_identity(source_amp, channel_index)
        current_index = self._current_recording_index()
        unmatched_recordings: list[int] = []
        for target_index in targets:
            target_amp = getattr(self.recordings[target_index], "timeseries", {}).get("amp")
            if not isinstance(target_amp, xr.DataArray):
                unmatched_recordings.append(target_index)
                continue
            target_channel = next(
                (
                    index
                    for index in range(target_amp.sizes["channel"])
                    if _channel_identity(target_amp, index) == source_identity
                ),
                None,
            )
            if target_channel is None:
                unmatched_recordings.append(target_index)
                continue
            draft = self._ensure_manual_draft(target_index)
            if draft is None:
                continue
            channel_coord = draft["channel"].values[target_channel]
            draft.loc[{"channel": channel_coord}] = False
            self._manual_drafts[(target_index, _MANUAL_TOOLS[0])] = draft

        if current_index is not None:
            self._load_manual_draft()
        self._staged_dirty = True
        self.add_mask_button.setEnabled(True)
        if unmatched_recordings:
            labels = ", ".join(str(index + 1) for index in unmatched_recordings)
            self.statusBar().showMessage(
                f"Could not map this source-detector pair in recording(s): {labels}."
            )
        self._draw()

    def _channel_at_event(self, event) -> int | None:
        if event.ydata is None:
            return None
        index = int(round(event.ydata / self._line_spacing))
        return index if 0 <= index < len(self._channel_labels) else None

    def _event_global_position(self, event):
        from PySide6.QtCore import QPoint

        return self.canvas.mapToGlobal(
            QPoint(int(event.x), int(self.canvas.height() - event.y))
        )

    def _selected_subtypes(self) -> list[tuple[dict[str, Any], str, str]]:
        selected = []
        data = self._current_data()
        if data is None:
            return selected
        for index, checkbox in enumerate(self._subtype_checks):
            if checkbox.isChecked():
                selection = checkbox.property("selection") or {}
                selected.append(
                    (selection, checkbox.text(), _subtype_color(selection, index))
                )
        return selected

    def _selected_data_arrays(self) -> list[tuple[xr.DataArray, str, str]]:
        data = self._current_data()
        if data is None:
            return []
        result = []
        for selection, label, color in self._selected_subtypes():
            selected = data.sel(selection, drop=True) if selection else data
            remaining = [dim for dim in selected.dims if dim not in ("time", "channel")]
            if remaining:
                selected = selected.mean(dim=remaining, skipna=True)
            selected = selected.transpose("time", "channel")
            result.append((selected, label, color))
        return result

    def _draw(self, *_args) -> None:
        if not hasattr(self, "waterfall_ax"):
            return
        self._plotting = True
        self.figure.clear()
        self.probe_figure.clear()
        self._create_plot_axes()
        arrays = self._selected_data_arrays()
        recording = self._current_recording()
        if recording is None or not arrays:
            self.stimulus_ax.text(0.5, 0.5, "No time-course data available", ha="center", va="center")
            self.stimulus_ax.set_axis_off()
            self.label_ax.set_axis_off()
            self.waterfall_ax.set_axis_off()
            self.probe_ax.set_axis_off()
            self.canvas.draw_idle()
            self.probe_canvas.draw_idle()
            self._plotting = False
            return

        data, _, _ = arrays[0]
        times = _time_values(data)
        channels = data.sizes["channel"]
        self._channel_labels = [_display_label(data, idx) for idx in range(channels)]
        self._channel_colors = [
            _CHANNEL_COLORS[index % len(_CHANNEL_COLORS)]
            for index in range(channels)
        ]
        self._draw_probe(recording, data)
        self._draw_events(recording, times)
        self._draw_waterfall(arrays, times)
        self._set_scroll_ranges(times, channels)
        if self._displayed_good is not None:
            self._draw_mask_summaries(
                self._displayed_good, self._displayed_good.shape[1]
            )
        self._update_selected_label()
        self.canvas.draw_idle()
        self.probe_canvas.draw_idle()
        self._plotting = False

    def _draw_probe(self, recording: Any, data: xr.DataArray) -> None:
        try:
            self._probe_lines = draw_probe(recording, data, self.probe_ax)
            for text in self.probe_ax.texts:
                text.remove()
            source_positions: dict[str, tuple[float, float]] = {}
            detector_positions: dict[str, tuple[float, float]] = {}
            for index, line in enumerate(self._probe_lines):
                line.set_color(self._channel_colors[index])
                line.set_linewidth(5.0 if index == self._selected_channel else 1.0)
                line.set_picker(8)
                line.set_alpha(1.0 if index == self._selected_channel else 0.55)
                source, detector = _channel_endpoints(data, index)
                source = _probe_label(source)
                detector = _probe_label(detector)
                x_values = line.get_xdata()
                y_values = line.get_ydata()
                source_positions[source] = (float(x_values[0]), float(y_values[0]))
                detector_positions[detector] = (
                    float(x_values[-1]),
                    float(y_values[-1]),
                )
            for positions, color, marker in (
                (source_positions, "#e41a1c", "o"),
                (detector_positions, "#377eb8", "s"),
            ):
                for label, (x, y) in positions.items():
                    self.probe_ax.scatter(
                        [x], [y], color=color, marker=marker, s=28,
                        edgecolors="white", linewidths=0.6, zorder=250,
                    )
                    self.probe_ax.text(
                        x, y, label, color=color, fontsize=8,
                        weight="semibold", ha="center", va="center",
                        zorder=251,
                    )
        except (AttributeError, IndexError, KeyError, ValueError) as exc:
            self.probe_ax.clear()
            self.probe_ax.text(0.5, 0.5, f"Probe unavailable\n{exc}", ha="center", va="center")
            self.probe_ax.set_axis_off()
            self._probe_lines = []

    def _draw_events(self, recording: Any, times: np.ndarray) -> None:
        self.stimulus_ax.clear()
        events = getattr(recording, "stim", None)
        if events is None or not isinstance(events, pd.DataFrame) or events.empty:
            self.stimulus_ax.text(0.5, 0.5, "No stimulus events", ha="center", va="center", transform=self.stimulus_ax.transAxes)
            self.stimulus_ax.set_yticks([])
        else:
            trial_types = events.get("trial_type", pd.Series(["event"] * len(events))).astype(str)
            categories = list(dict.fromkeys(trial_types.tolist()))
            color_map = {name: f"C{i % 10}" for i, name in enumerate(categories)}
            y_lookup = {name: index for index, name in enumerate(categories)}
            for (_, event), event_type in zip(events.iterrows(), trial_types):
                onset = pd.to_numeric(pd.Series([event.get("onset", np.nan)]), errors="coerce").iloc[0]
                if not np.isfinite(onset):
                    continue
                duration = pd.to_numeric(pd.Series([event.get("duration", 0.0)]), errors="coerce").iloc[0]
                duration = max(0.0, float(duration)) if np.isfinite(duration) else 0.0
                row = y_lookup[event_type]
                self.stimulus_ax.barh(
                    row, max(duration, (times[-1] - times[0]) * 0.001),
                    left=float(onset), height=0.7, color=color_map[event_type], alpha=0.75,
                )
            self.stimulus_ax.set_yticks(range(len(categories)), categories, fontsize=8)
            self.stimulus_ax.set_ylim(-0.6, len(categories) - 0.4)
        self.stimulus_ax.set_title("Stimulus", loc="left", fontsize=9)
        self.stimulus_ax.tick_params(axis="x", labelbottom=False)
        self.stimulus_ax.grid(axis="x", alpha=0.2)

    def _draw_waterfall(
        self, arrays: list[tuple[xr.DataArray, str, str]], times: np.ndarray
    ) -> None:
        self.waterfall_ax.clear()
        self.label_ax.clear()
        channel_count = len(self._channel_labels)
        for array, label, color in arrays:
            values = np.asarray(_plain_array(array), dtype=float)
            for channel_index in range(channel_count):
                trace = values[:, channel_index]
                finite = np.isfinite(trace)
                if not finite.any():
                    continue
                center = float(np.nanmedian(trace))
                spread = float(np.nanpercentile(trace[finite], 95) - np.nanpercentile(trace[finite], 5))
                if spread <= 0 or not np.isfinite(spread):
                    spread = float(np.nanstd(trace[finite]) * 6)
                if spread <= 0 or not np.isfinite(spread):
                    spread = 1.0
                normalized = (trace - center) / spread
                baseline = channel_index * self._line_spacing
                vertical = baseline - normalized * self._amplitude_zoom
                width = 2.8 if channel_index == self._selected_channel else 0.75
                alpha = 1.0 if channel_index == self._selected_channel else 0.7
                self.waterfall_ax.plot(
                    times, vertical, color=color, linewidth=width, alpha=alpha,
                    zorder=3 if channel_index == self._selected_channel else 2,
                )

        data = arrays[0][0]
        base_good = self._base_mask_matrix(data)
        shown_good = base_good
        self._draw_mask_matrix(times, ~base_good, "#666666", 0.42)
        recording_index = self._current_recording_index()
        preview = self._preview_masks.get(recording_index)
        if self.mask_tool_combo.currentText() in _MANUAL_TOOLS:
            preview = self._manual_mask_draft
        if preview is not None:
            preview_good = self._mask_matrix(preview, data)
            newly_bad = base_good & ~preview_good
            self._draw_mask_matrix(times, newly_bad, "#8e44ad", 0.38)
            shown_good = base_good & preview_good
        self._displayed_good = shown_good
        self.waterfall_ax.set_xlim(times[0], times[-1] if times[-1] > times[0] else times[0] + 1.0)
        self.waterfall_ax.set_yticks([])
        self.waterfall_ax.grid(axis="x", color="#dddddd", linewidth=0.5)
        self.waterfall_ax.set_xlabel("Time (s)")
        self.label_ax.set_xlim(0, 1)
        self.label_ax.set_yticks(
            np.arange(channel_count) * self._line_spacing,
            self._channel_labels,
            fontsize=8,
        )
        self.label_ax.tick_params(axis="y", length=0, pad=8)
        self.label_ax.set_xticks([])
        self.label_ax.set_frame_on(False)
        self.label_ax.yaxis.tick_right()
        self.label_ax.set_ylim(self.waterfall_ax.get_ylim())
        self.summary_ax.set_ylim(self.waterfall_ax.get_ylim())
        self._draw_mask_summaries(shown_good, channel_count)
        for index, tick in enumerate(self.label_ax.get_yticklabels()):
            tick.set_color(self._channel_colors[index])
            tick.set_fontweight("bold" if index == self._selected_channel else "normal")
            tick.set_fontsize(10 if index == self._selected_channel else 8)
            tick.set_picker(True)

    def _draw_mask_matrix(
        self, times: np.ndarray, bad: np.ndarray, color: str, alpha: float
    ) -> None:
        for channel_index in range(bad.shape[1]):
            starts, stops = _segments(bad[:, channel_index])
            for start, stop in zip(starts, stops):
                left = times[start]
                right = times[stop] if stop < len(times) else times[-1]
                if right <= left:
                    right = left + (times[1] - times[0] if len(times) > 1 else 0.01)
                self.waterfall_ax.add_patch(
                    Rectangle(
                        (
                            left,
                            channel_index * self._line_spacing - self._line_spacing / 2,
                        ),
                        right - left,
                        self._line_spacing,
                        facecolor=color, edgecolor="none", alpha=alpha, zorder=1,
                    )
                )

    def _draw_mask_summaries(self, good: np.ndarray, channel_count: int) -> None:
        for text in list(self.summary_ax.texts):
            text.remove()
        self.summary_ax.set_ylim(self.waterfall_ax.get_ylim())
        y_min, y_max = sorted(self.waterfall_ax.get_ylim())
        total = max(1, good.shape[0])
        for channel_index in range(channel_count):
            y = channel_index * self._line_spacing
            if not y_min <= y <= y_max:
                continue
            removed = 100.0 * (1.0 - float(np.count_nonzero(good[:, channel_index])) / total)
            self.summary_ax.text(
                0.02,
                y,
                f"{removed:.1f}% removed",
                transform=self.summary_ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=8,
                color="#333333",
                clip_on=True,
            )

    def _set_scroll_ranges(self, times: np.ndarray, channel_count: int) -> None:
        self._visible_channels = max(
            1, min(channel_count, int(self._viewport_height / self._line_spacing))
        )
        self.vertical_scroll.blockSignals(True)
        self.vertical_scroll.setRange(0, max(0, channel_count - self._visible_channels))
        self.vertical_scroll.setPageStep(self._visible_channels)
        self.vertical_scroll.setValue(
            max(0, min(channel_count - self._visible_channels, self._selected_channel - self._visible_channels // 2))
        )
        self.vertical_scroll.blockSignals(False)
        duration = max(float(times[-1] - times[0]), 1e-9)
        self.horizontal_scroll.blockSignals(True)
        self.horizontal_scroll.setRange(0, 10000)
        self.horizontal_scroll.setPageStep(max(1, int(10000 / self._time_zoom)))
        self.horizontal_scroll.setValue(
            int(10000 * self._time_start / max(duration * (1 - 1 / self._time_zoom), 1e-9))
            if self._time_zoom > 1 else 0
        )
        self.horizontal_scroll.blockSignals(False)
        self._apply_view_limits(times)

    def _apply_view_limits(self, times: np.ndarray | None = None) -> None:
        data = self._current_data()
        if data is None or data.sizes.get("channel", 0) == 0:
            return
        if times is None:
            times = _time_values(data)
        total = max(float(times[-1] - times[0]), 1e-9)
        span = total / self._time_zoom
        max_start = max(0.0, total - span)
        self._time_start = min(max(self._time_start, 0.0), max_start)
        start = float(times[0]) + self._time_start
        end = min(float(times[-1]), start + span)
        if end <= start:
            end = float(times[-1])
        self.stimulus_ax.set_xlim(start, end)
        self.waterfall_ax.set_xlim(start, end)
        top = self.vertical_scroll.value()
        lower = top * self._line_spacing - self._line_spacing / 2
        upper = lower + self._viewport_height
        self.waterfall_ax.set_ylim(upper, lower)
        self.label_ax.set_ylim(upper, lower)

    def _on_horizontal_scroll(self, value: int) -> None:
        if self._plotting:
            return
        data = self._current_data()
        if data is None:
            return
        times = _time_values(data)
        duration = max(float(times[-1] - times[0]), 1e-9)
        span = duration / self._time_zoom
        self._time_start = (value / 10000) * max(0.0, duration - span)
        self._apply_view_limits(times)
        self.canvas.draw_idle()

    def _on_vertical_scroll(self, _value: int) -> None:
        if self._plotting or self._current_data() is None:
            return
        self._apply_view_limits()
        if self._displayed_good is not None:
            self._draw_mask_summaries(
                self._displayed_good, self._displayed_good.shape[1]
            )
        self.canvas.draw_idle()

    def _change_time_zoom(self, direction: int) -> None:
        self._time_zoom = float(np.clip(self._time_zoom * (1.25 if direction > 0 else 0.8), 1.0, 100.0))
        self._draw()

    def _change_amplitude_zoom(self, direction: int) -> None:
        self._amplitude_zoom = float(
            np.clip(
                self._amplitude_zoom * (1.25 if direction > 0 else 0.8),
                0.05,
                256.0,
            )
        )
        self._draw()

    def _change_spacing(self, direction: int) -> None:
        factor = 1.2 if direction > 0 else 1 / 1.2
        self._line_spacing = float(np.clip(self._line_spacing * factor, 0.5, 10.0))
        self._draw()

    def _reset_view(self, _direction: int = 0) -> None:
        self._time_zoom = 1.0
        self._amplitude_zoom = self._initial_amplitude_zoom
        self._line_spacing = 1.0
        self._time_start = 0.0
        self._viewport_height = 24.0
        self._visible_channels = 24
        self._selected_channel = 0
        if hasattr(self, "horizontal_scroll"):
            self.horizontal_scroll.setValue(0)
        if hasattr(self, "vertical_scroll"):
            self.vertical_scroll.setValue(0)
        self._draw()

    def _select_channel(self, index: int) -> None:
        if not 0 <= index < len(self._channel_labels):
            return
        self._selected_channel = index
        minimum = max(0, len(self._channel_labels) - self._visible_channels)
        target = max(0, min(minimum, index - self._visible_channels // 2))
        self.vertical_scroll.setValue(target)
        self._draw()

    def _on_plot_click(self, event) -> None:
        if event.inaxes is not self.waterfall_ax or event.ydata is None:
            return
        tool = self.mask_tool_combo.currentText()
        if getattr(event, "button", 1) == 3:
            if tool == _MANUAL_TOOLS[0]:
                channel_index = self._channel_at_event(event)
                if channel_index is not None:
                    self._show_channel_context_menu(
                        channel_index, self._event_global_position(event)
                    )
            elif tool in _MANUAL_TOOLS[1:]:
                self._manual_drag_start = float(event.xdata)
                self._manual_drag_rect = Rectangle(
                    (event.xdata, self.waterfall_ax.get_ylim()[1]),
                    0,
                    abs(self.waterfall_ax.get_ylim()[0] - self.waterfall_ax.get_ylim()[1]),
                    facecolor="#8e44ad",
                    edgecolor="#6c3483",
                    alpha=0.22,
                    linestyle="--",
                    zorder=4,
                )
                self.waterfall_ax.add_patch(self._manual_drag_rect)
                self.canvas.draw_idle()
            return
        channel_index = self._channel_at_event(event)
        if channel_index is not None:
            self._select_channel(channel_index)

    def _on_label_click(self, event) -> None:
        if event.inaxes is not self.label_ax or event.ydata is None:
            return
        channel_index = self._channel_at_event(event)
        if channel_index is None:
            return
        if (
            getattr(event, "button", 1) == 3
            and self.mask_tool_combo.currentText() == _MANUAL_TOOLS[0]
        ):
            self._show_channel_context_menu(
                channel_index, self._event_global_position(event)
            )
        elif getattr(event, "button", 1) != 3:
            self._select_channel(channel_index)

    def _on_plot_motion(self, event) -> None:
        if self._manual_drag_start is None or self._manual_drag_rect is None:
            return
        if event.inaxes is not self.waterfall_ax or event.xdata is None:
            return
        left = min(self._manual_drag_start, float(event.xdata))
        width = abs(float(event.xdata) - self._manual_drag_start)
        self._manual_drag_rect.set_x(left)
        self._manual_drag_rect.set_width(width)
        self.canvas.draw_idle()

    def _on_plot_release(self, event) -> None:
        if self._manual_drag_start is None:
            return
        start = self._manual_drag_start
        end = float(event.xdata) if event.inaxes is self.waterfall_ax and event.xdata is not None else start
        if self._manual_drag_rect is not None:
            self._manual_drag_rect.remove()
        self._manual_drag_rect = None
        self._manual_drag_start = None
        if abs(end - start) > 0:
            self._apply_manual_time_range(start, end)
        self.canvas.draw_idle()

    def _apply_manual_time_range(self, start: float, end: float) -> None:
        draft = self._ensure_manual_draft()
        recording_index = self._current_recording_index()
        recording = self._current_recording()
        amp = getattr(recording, "timeseries", {}).get("amp") if recording is not None else None
        if not isinstance(amp, xr.DataArray):
            self.statusBar().showMessage(
                "Manual time masks require an 'amp' time series."
            )
            return
        if draft is None or recording_index is None:
            return
        times = _time_values(amp)
        first = int(np.clip(np.searchsorted(times, min(start, end), side="left"), 0, len(times) - 1))
        last = int(np.clip(np.searchsorted(times, max(start, end), side="right"), 0, len(times)))
        if last <= first:
            return
        if self.mask_tool_combo.currentText() == _MANUAL_TOOLS[1]:
            draft.values[first:last, :] = False
        elif self.mask_tool_combo.currentText() == _MANUAL_TOOLS[2]:
            draft.values[first:last, self._selected_channel] = False
        else:
            return
        self._manual_mask_draft = draft
        self._manual_drafts[(recording_index, self.mask_tool_combo.currentText())] = draft
        self._staged_dirty = True
        self.add_mask_button.setEnabled(True)
        self._draw()

    def _on_label_pick(self, event) -> None:
        if event.artist in self.label_ax.get_yticklabels():
            self._select_channel(self.label_ax.get_yticklabels().index(event.artist))

    def _on_probe_pick(self, event) -> None:
        if event.artist in self._probe_lines and event.artist in event.artist.axes.lines:
            self._select_channel(self._probe_lines.index(event.artist))

    def _on_probe_click(self, event) -> None:
        if event.inaxes is not self.probe_ax or event.x is None or event.y is None:
            return
        nearest_index = -1
        nearest_distance = float("inf")
        for index, line in enumerate(self._probe_lines):
            points = line.get_transform().transform(
                np.column_stack((line.get_xdata(), line.get_ydata()))
            )
            distance = min(
                _point_segment_distance(
                    event.x, event.y, *points[0], *points[-1]
                ),
                nearest_distance,
            )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index
        if nearest_index >= 0 and nearest_distance <= 12:
            if (
                getattr(event, "button", 1) == 3
                and self.mask_tool_combo.currentText() == _MANUAL_TOOLS[0]
            ):
                from PySide6.QtCore import QPoint

                position = self.probe_canvas.mapToGlobal(
                    QPoint(int(event.x), int(self.probe_canvas.height() - event.y))
                )
                self._show_channel_context_menu(nearest_index, position)
            elif getattr(event, "button", 1) != 3:
                self._select_channel(nearest_index)

    def _update_selected_label(self) -> None:
        if not self._channel_labels or not 0 <= self._selected_channel < len(self._channel_labels):
            return
        for index, label in enumerate(self.label_ax.get_yticklabels()):
            label.set_fontweight("bold" if index == self._selected_channel else "normal")
            label.set_fontsize(10 if index == self._selected_channel else 8)

    def _composite_masks(
        self, recording: Any, entries: list[dict[str, Any]], recording_index: int
    ) -> xr.DataArray | None:
        amp = getattr(recording, "timeseries", {}).get("amp")
        if not isinstance(amp, xr.DataArray):
            if any(recording_index in entry["masks"] for entry in entries):
                raise ValueError(
                    f"Recording {recording_index + 1} has masks but no 'amp' time series."
                )
            return None
        good = np.ones((amp.sizes["time"], amp.sizes["channel"]), dtype=bool)
        found = False
        for entry in entries:
            mask = entry["masks"].get(recording_index)
            if mask is None:
                continue
            good &= self._mask_matrix(mask, amp)
            found = True
        if not found:
            return None
        return xr.DataArray(
            good,
            dims=("time", "channel"),
            coords={"time": amp["time"], "channel": amp["channel"]},
        )

    def _choose_submission_mode(self) -> str | None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Submit data masks")
        dialog.setText("How should the submitted masks be saved?")
        buttons = {
            "combine": dialog.addButton(
                "Combine with existing mask", QMessageBox.ButtonRole.AcceptRole
            ),
            "add": dialog.addButton(
                "Add a new mask", QMessageBox.ButtonRole.ActionRole
            ),
            "replace": dialog.addButton(
                "Replace existing masks", QMessageBox.ButtonRole.DestructiveRole
            ),
        }
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        clicked = dialog.clickedButton()
        return next((mode for mode, button in buttons.items() if clicked is button), None)

    def _submit(self) -> bool:
        if self._submitted:
            return True
        pending_manual = [
            {
                "id": self._next_mask_id + index,
                "name": tool_name,
                "masks": {recording_index: mask.copy(deep=True)},
                "existing_keys": {},
                "existing": False,
            }
            for index, ((recording_index, tool_name), mask) in enumerate(
                self._manual_drafts.items()
            )
            if not bool(_plain_array(mask).all())
        ]
        submission_entries = [*self._mask_entries, *pending_manual]
        if not self._staged_dirty and not pending_manual:
            self._submitted = True
            self.close()
            return True
        existing_masks = any(getattr(recording, "masks", {}) for recording in self.recordings)
        mode = "add"
        if existing_masks:
            selected_mode = self._choose_submission_mode()
            if selected_mode is None:
                return False
            mode = selected_mode

        new_name = "quality_mask"
        if mode == "add":
            has_new_masks = any(not entry["existing"] for entry in submission_entries)
            if has_new_masks:
                new_name, accepted = QInputDialog.getText(
                    self,
                    "Name new mask",
                    "Mask name:",
                    text="quality_mask",
                )
                if not accepted:
                    return False
                new_name = new_name.strip()
                if not new_name:
                    QMessageBox.warning(self, "Name new mask", "The mask name cannot be empty.")
                    return False

        plans: list[tuple[Any, set[str], str | None, xr.DataArray | None]] = []
        for recording_index, recording in enumerate(self.recordings):
            recording_masks = getattr(recording, "masks", {})
            kept_existing = [
                entry
                for entry in submission_entries
                if entry["existing"]
                and recording_index in entry["masks"]
                and entry["existing_keys"].get(recording_index) in recording_masks
            ]
            staged_new = [
                entry
                for entry in submission_entries
                if not entry["existing"] and recording_index in entry["masks"]
            ]
            existing_keys = set(recording_masks)
            kept_keys = {
                entry["existing_keys"][recording_index]
                for entry in kept_existing
            }
            remove_keys = existing_keys - kept_keys
            add_key: str | None = None
            add_mask: xr.DataArray | None = None

            if mode == "replace":
                remove_keys = existing_keys
                add_mask = self._composite_masks(
                    recording, submission_entries, recording_index
                )
                if add_mask is not None:
                    add_key = "quality_mask"
            elif mode == "combine":
                combined_entries = kept_existing + staged_new
                add_mask = self._composite_masks(
                    recording, combined_entries, recording_index
                )
                if add_mask is not None:
                    if kept_existing:
                        add_key = kept_existing[0]["existing_keys"][recording_index]
                    else:
                        add_key = "quality_mask"
                    remove_keys = existing_keys - (
                        {add_key} if add_key in existing_keys else set()
                    )
            elif staged_new:
                add_mask = self._composite_masks(recording, staged_new, recording_index)
                add_key = new_name
                suffix = 2
                while add_key in existing_keys and add_key not in remove_keys:
                    add_key = f"{new_name}_{suffix}"
                    suffix += 1

            if add_key is not None and add_key in existing_keys and add_key not in remove_keys:
                remove_keys.add(add_key)
            plans.append((recording, remove_keys, add_key, add_mask))

        snapshots = [
            (recording, dict(getattr(recording, "masks", {})))
            for recording, _remove, _key, _mask in plans
        ]
        try:
            for recording, remove_keys, add_key, add_mask in plans:
                masks = getattr(recording, "masks", {})
                for key in remove_keys:
                    masks.pop(key, None)
                if add_key is not None and add_mask is not None:
                    setter = getattr(recording, "set_mask", None)
                    if setter is None:
                        raise TypeError("A recording does not provide Cedalion's set_mask API.")
                    setter(add_key, add_mask, overwrite=False)
        except Exception as exc:
            for recording, old_masks in snapshots:
                current_masks = getattr(recording, "masks", {})
                current_masks.clear()
                current_masks.update(old_masks)
            QMessageBox.critical(
                self,
                "Submit data masks",
                f"Could not save masks; no recordings were changed:\n{exc}",
            )
            return False

        self._submitted = True
        self._staged_dirty = False
        self._mask_entries = submission_entries
        self._manual_drafts.clear()
        self.submitted.emit()
        self.close()
        return True

    def _cancel(self) -> None:
        self._submitted = False
        self._staged_dirty = False
        self.close()

    def _choose_close_action(self) -> str | None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Save changes")
        dialog.setText("Save changes to the data masks?")
        buttons = {
            "save": dialog.addButton(
                "Save changes", QMessageBox.ButtonRole.AcceptRole
            ),
            "discard": dialog.addButton(
                "Discard Changes", QMessageBox.ButtonRole.DestructiveRole
            ),
        }
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        clicked = dialog.clickedButton()
        return next((action for action, button in buttons.items() if clicked is button), None)

    def closeEvent(self, event) -> None:
        has_pending_changes = self._staged_dirty
        if self._submitted or not has_pending_changes:
            event.accept()
            return
        action = self._choose_close_action()
        if action == "discard":
            self._submitted = False
            self._staged_dirty = False
            event.accept()
        elif action == "save":
            if self._submit():
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()


def _segments(masked: Iterable[bool]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(list(masked), dtype=bool)
    padded = np.concatenate(([False], values, [False])).astype(int)
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


_OPEN_WINDOWS: list[DataMaskingManager] = []


def _release_window(window: DataMaskingManager) -> None:
    if window in _OPEN_WINDOWS:
        _OPEN_WINDOWS.remove(window)


def _active_ipython():
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def data_masking_manager(
    data: Any,
    block: bool | None = False,
    parent: QWidget | None = None,
) -> DataMaskingManager:
    """Open the data masking viewer for a DataSet or a single recording.

    By default, this returns immediately for notebook use and integrates with
    IPython/Jupyter's Qt event loop. Pass ``block=True`` in a standalone
    script to run the Qt event loop until the window closes.
    """
    app = QApplication.instance() or QApplication([])
    shell = _active_ipython()
    if block is None:
        block = shell is None

    manager = DataMaskingManager(data, parent)
    manager.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    _OPEN_WINDOWS.append(manager)
    manager.destroyed.connect(lambda *_: _release_window(manager))
    manager.show()
    manager.raise_()
    manager.activateWindow()

    if not block and shell is not None and shell.active_eventloop != "qt":
        try:
            shell.enable_gui("qt")
        except Exception:
            block = True

    if block:
        app.setQuitOnLastWindowClosed(True)
    elif shell is not None:
        app.setQuitOnLastWindowClosed(False)

    if block:
        app.exec()
    return manager
