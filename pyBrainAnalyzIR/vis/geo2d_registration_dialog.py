"""Dialog registering a 2D probe layout (geo2d) to standard head positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from pyBrainAnalyzIR.vis.geo2d_registration import (
    LABEL_SYSTEMS,
    LABELS_1020,
    MIN_ASSIGNED_LANDMARKS,
    as_scalp,
    canonical_label,
    estimate_mm_per_unit,
    is_standard_label,
    label_display,
    probe_2d_layout,
    register_probe_2d,
    symmetrize_layout,
    symmetrize_points,
    system_labels,
    topographic_frame,
    topographic_projection,
)

NO_POSITION = "\u2014"
"""Combo entry for points without an assigned position."""

KIND_NAMES = {
    "SOURCE": "Source", "DETECTOR": "Detector", "LANDMARK": "Landmark",
}
ANCHOR, DIRECTIONAL = "Anchor", "Directional"
KIND_STYLE = {
    "Source": dict(color="#d62728", marker="o"),
    "Detector": dict(color="#1f77b4", marker="o"),
    "Landmark": dict(color="#2ca02c", marker="^"),
    ANCHOR: dict(color="#9467bd", marker="D"),
}
DIRECTION_COLOR = "#ff7f0e"
ASSIGNED_COLOR = "#ff7f0e"
GROUP_COLORS = ("#000000", "#e377c2", "#17becf", "#bcbd22", "#8c564b", "#7f7f7f")
HIT_PIXELS = 9
OUTLINE_RING = (
    "Fpz Fp2 AF8 F8 FT8 T8 TP8 P8 PO8 O2 Oz O1 PO7 P7 TP7 T7 FT7 F7 AF7 Fp1 Fpz"
).split()
POSITION_COLUMN = 2


@dataclass
class ProbePoint:
    """A row of the dialog: an optode, 2D landmark, anchor or direction arrow.

    ``xy`` is in geo2d units converted to mm (before the dialog's
    "mm per unit" factor); for direction arrows it is the tip and ``tail``
    the start. These are the layout's (or clicked) positions; with "Enforce
    L/R Symmetry", ``shown``/``shown_tail`` hold the symmetrized positions
    that are drawn and registered.
    """

    name: str
    kind: str
    xy: np.ndarray
    label: str | None = None
    tail: np.ndarray | None = None
    group: int = 0
    shown: np.ndarray | None = None
    shown_tail: np.ndarray | None = None

    @property
    def at(self) -> np.ndarray:
        return self.xy if self.shown is None else self.shown

    @property
    def tail_at(self) -> np.ndarray | None:
        return self.tail if self.shown_tail is None else self.shown_tail


class Geo2DRegistrationDialog(QDialog):
    """Assign standard positions to a 2D probe and register it to a head.

    ``head_positions`` maps standard labels to positions on the head (RAS,
    mm) and ``scalp_vertices`` is the head's scalp (RAS, mm): a
    ``ScalpSurface`` (vertices and faces) or bare vertices. After
    ``accept``, ``result`` holds the ``Geo2DRegistration`` and
    ``optode_positions()`` the registered sources and detectors.
    """

    def __init__(self, recording, head_positions: dict, scalp_vertices, parent=None,
                 title: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(title or "Register to Geo2D: probe landmarks")
        self.resize(1250, 820)
        self.head_positions = {canonical_label(k): np.asarray(v, dtype=float)
                               for k, v in head_positions.items()}
        self.scalp_vertices = as_scalp(scalp_vertices)
        labels, kinds, coords, pairs = probe_2d_layout(recording)
        if not labels:
            raise ValueError("the recording has no 2D probe (geo2d)")
        self.channel_pairs = pairs
        self.points: list[ProbePoint] = []
        for label, kind, xy in zip(labels, kinds, coords):
            point = ProbePoint(label, KIND_NAMES[kind], np.asarray(xy, dtype=float))
            if kind == "LANDMARK" and is_standard_label(label):
                key = canonical_label(label)
                point.label = key if key in self.head_positions else None
            self.points.append(point)
        self.result = None
        self.result_current = False
        self.symmetry_pairs, self.symmetry_axis = [], None
        self._result_points: list[ProbePoint] = []
        self._arrow = None
        self._lasso = None
        self._frame = topographic_frame(self.head_positions)

        self._build_ui(estimate_mm_per_unit(labels, coords, pairs))
        self._populate_table()
        self._redraw()

    # UI ----------------------------------------------------------------
    def _build_ui(self, mm_per_unit: float) -> None:
        self.probe_figure = Figure(figsize=(5, 4))
        self.probe_canvas = FigureCanvasQTAgg(self.probe_figure)
        self.probe_axes = self.probe_figure.add_subplot(111)
        self.probe_canvas.setFocusPolicy(Qt.ClickFocus)
        self.probe_canvas.mpl_connect("button_press_event", self._canvas_pressed)
        self.probe_canvas.mpl_connect("button_release_event", self._canvas_released)
        self.probe_canvas.mpl_connect("motion_notify_event", self._canvas_moved)
        self.probe_canvas.mpl_connect("key_press_event", self._canvas_key)
        self.head_figure = Figure(figsize=(5, 4))
        self.head_canvas = FigureCanvasQTAgg(self.head_figure)
        self.head_axes = self.head_figure.add_subplot(111)

        left = QSplitter(Qt.Vertical)
        left.addWidget(self._titled(self.probe_canvas, "2D probe"))
        left.addWidget(self._titled(self.head_canvas, "10-20 view"))

        self.system_combo = QComboBox()
        self.system_combo.addItems(LABEL_SYSTEMS)
        self.system_combo.setCurrentText("10-10")
        self.system_combo.setToolTip(
            "Standard positions offered in the position column. Switching keeps "
            "existing assignments."
        )
        self.system_combo.currentTextChanged.connect(self._system_changed)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(3)
        self.scale_spin.setRange(0.001, 1000.0)
        self.scale_spin.setValue(mm_per_unit)
        self.scale_spin.setToolTip(
            "Factor from the 2D layout's coordinates to mm. Estimated from the "
            "source-detector distances when the stored units are implausible."
        )
        self.scale_spin.valueChanged.connect(self._mark_stale)
        self.distance_check = QCheckBox("Preserve 2D distances")
        self.distance_check.setToolTip(
            "Finally relax nearby points to their 2D distances (MATLAB pushdistances). "
            "A flat layout cannot keep all distances on a curved head."
        )
        self.distance_check.toggled.connect(self._mark_stale)
        self.symmetry_check = QCheckBox("Enforce L/R Symmetry")
        self.symmetry_check.setToolTip(
            "Flip the 2D layout left/right and average points of the same type that "
            "nearly coincide with a flipped point (and points assigned mirror positions "
            "such as T7/T8), so small layout or clicking errors do not make the "
            "registration asymmetric. Moved points are drawn with a grey cross at "
            "their original position."
        )
        self.symmetry_check.toggled.connect(self._mark_stale)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Optode", "Type", "10-10 position"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(POSITION_COLUMN, QHeaderView.Stretch)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu_requested)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)

        system_row = QHBoxLayout()
        system_row.addWidget(QLabel("Position system:"))
        system_row.addWidget(self.system_combo, 1)
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("2D units to mm:"))
        scale_row.addWidget(self.scale_spin, 1)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addLayout(system_row)
        right_layout.addLayout(scale_row)
        right_layout.addWidget(self.distance_check)
        right_layout.addWidget(self.symmetry_check)
        right_layout.addWidget(self.table, 1)
        right_layout.addWidget(self.status_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.compute_button = QPushButton("Compute registration")
        self.compute_button.clicked.connect(self.compute_registration)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.accept_button = self.buttons.button(QDialogButtonBox.Ok)
        self.accept_button.setText("Accept")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        bottom = QHBoxLayout()
        bottom.addWidget(self.compute_button)
        bottom.addStretch(1)
        bottom.addWidget(self.buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, 1)
        layout.addLayout(bottom)

    @staticmethod
    def _titled(widget, title):
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel(f"<b>{title}</b>")
        layout.addWidget(heading)
        layout.addWidget(widget, 1)
        return box

    # Positions ---------------------------------------------------------
    @property
    def system(self) -> str:
        return self.system_combo.currentText()

    def available_labels(self, system: str | None = None) -> list[str]:
        """Labels of ``system`` (default: the selected one) known on the head."""
        return [label for label in system_labels(system or self.system)
                if label in self.head_positions]

    def _fill_position_combo(self, combo: QComboBox, label: str | None) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(NO_POSITION, None)
        labels = self.available_labels()
        if label and label not in labels:
            # Keep assignments made with a larger system.
            combo.addItem(label_display(label), label)
        for name in labels:
            combo.addItem(label_display(name), name)
        combo.setCurrentIndex(max(combo.findData(label), 0) if label else 0)
        combo.setMaxVisibleItems(25)
        combo.blockSignals(False)

    def _position_combo(self, row: int) -> QComboBox:
        combo = QComboBox()
        self._fill_position_combo(combo, self.points[row].label)
        combo.currentIndexChanged.connect(
            lambda _index, c=combo, r=row: self.set_label(r, c.currentData()))
        return combo

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.points))
        for row, point in enumerate(self.points):
            self.table.setItem(row, 0, QTableWidgetItem(point.name))
            self.table.setItem(row, 1, QTableWidgetItem(point.kind))
            self.table.setCellWidget(row, POSITION_COLUMN, self._position_combo(row))
        self._update_status()

    def position_combo(self, row: int) -> QComboBox:
        return self.table.cellWidget(row, POSITION_COLUMN)

    def _system_changed(self, system: str) -> None:
        self.table.setHorizontalHeaderItem(POSITION_COLUMN, QTableWidgetItem(f"{system} position"))
        for row, point in enumerate(self.points):
            self._fill_position_combo(self.position_combo(row), point.label)
        self._redraw()

    def set_label(self, row: int, label: str | None) -> None:
        """Assign standard position ``label`` (None to clear) to point ``row``."""
        label = canonical_label(label) if label else None
        if label is not None and label not in self.head_positions:
            raise ValueError(f"position {label!r} is not available on this head")
        point = self.points[row]
        if point.label == label:
            return
        point.label = label
        combo = self.position_combo(row)
        if combo is not None and combo.currentData() != label:
            self._fill_position_combo(combo, label)
        self._mark_stale()

    def assigned_labels(self) -> set:
        return {point.label for point in self.points if point.label}

    def can_compute(self) -> bool:
        return len(self.assigned_labels()) >= MIN_ASSIGNED_LANDMARKS

    # Points --------------------------------------------------------------
    def _next_name(self, kind: str) -> str:
        prefix = "Anchor" if kind == ANCHOR else "Direction"
        used = {point.name for point in self.points}
        number = 1
        while f"{prefix}-{number}" in used:
            number += 1
        return f"{prefix}-{number}"

    def add_anchor(self, xy, label: str | None = None) -> int:
        """Add a landmark anchor at ``xy`` (2D layout coordinates); returns its row."""
        point = ProbePoint(self._next_name(ANCHOR), ANCHOR, np.asarray(xy, dtype=float))
        point.group = self._group_at(point.xy)
        return self._append(point, label)

    def add_direction(self, tail, tip, label: str | None = None) -> int:
        """Add directional guidance from ``tail`` to ``tip``; returns its row."""
        point = ProbePoint(self._next_name(DIRECTIONAL), DIRECTIONAL,
                           np.asarray(tip, dtype=float), tail=np.asarray(tail, dtype=float))
        return self._append(point, label)

    def _append(self, point: ProbePoint, label) -> int:
        self.points.append(point)
        row = len(self.points) - 1
        self.table.setRowCount(len(self.points))
        self.table.setItem(row, 0, QTableWidgetItem(point.name))
        self.table.setItem(row, 1, QTableWidgetItem(point.kind))
        self.table.setCellWidget(row, POSITION_COLUMN, self._position_combo(row))
        if label:
            self.set_label(row, label)
        else:
            self._mark_stale()
        return row

    def remove_point(self, row: int) -> None:
        """Remove an anchor or direction arrow from the table and the plot."""
        if self.points[row].kind not in (ANCHOR, DIRECTIONAL):
            raise ValueError("only anchors and directional guidance can be removed")
        del self.points[row]
        self.table.removeRow(row)
        for index in range(row, len(self.points)):
            combo = self.position_combo(index)
            combo.currentIndexChanged.disconnect()
            combo.currentIndexChanged.connect(
                lambda _index, c=combo, r=index: self.set_label(r, c.currentData()))
        self._mark_stale()

    def _group_at(self, xy) -> int:
        """Probe part of the point nearest to ``xy``."""
        rows = [p for p in self.points if p.kind != DIRECTIONAL]
        if not rows:
            return 0
        distance = [np.linalg.norm(p.xy - xy) for p in rows]
        return rows[int(np.argmin(distance))].group

    @property
    def groups(self) -> list[int]:
        return [point.group for point in self.points if point.kind != DIRECTIONAL]

    def separate_parts(self, polygon) -> int:
        """Make the points inside ``polygon`` (2D layout coordinates) a probe part.

        Returns the number of points in the new part.
        """
        polygon = np.asarray(polygon, dtype=float)
        if len(polygon) < 3:
            return 0
        path = Path(polygon)
        rows = [p for p in self.points if p.kind != DIRECTIONAL]
        inside = path.contains_points(np.array([p.at for p in rows]))
        if not inside.any():
            return 0
        group = max(p.group for p in rows) + 1
        for point, use in zip(rows, inside):
            if use:
                point.group = group
        self._mark_stale()
        return int(inside.sum())

    def clear_parts(self) -> None:
        for point in self.points:
            point.group = 0
        self._mark_stale()

    # Registration --------------------------------------------------------
    def registration_inputs(self) -> dict:
        """Inputs of ``register_probe_2d`` (without the head) for the current table."""
        scale = self.scale_spin.value()
        rows = [p for p in self.points if p.kind != DIRECTIONAL]
        return dict(
            points2d=np.array([p.at for p in rows]) * scale,
            anchors={i: p.label for i, p in enumerate(rows) if p.label},
            attractors=[(p.tail_at * scale, p.at * scale, p.label)
                        for p in self.points if p.kind == DIRECTIONAL and p.label],
            groups=[p.group for p in rows],
            probe_rows=[i for i, p in enumerate(rows) if p.kind in ("Source", "Detector")],
            preserve_distances=self.distance_check.isChecked(),
        )

    def register_on(self, head_positions: dict, scalp_vertices):
        """Register the probe with the current assignments on another head."""
        return register_probe_2d(head_positions=head_positions, scalp_vertices=scalp_vertices,
                                 **self.registration_inputs())

    def compute_registration(self) -> bool:
        """Register the probe on the dialog's head; returns True on success."""
        try:
            self.result = self.register_on(self.head_positions, self.scalp_vertices)
        except ValueError as exc:
            QMessageBox.warning(self, "Compute registration", f"Cannot register the probe: {exc}.")
            return False
        self._result_points = [p for p in self.points if p.kind != DIRECTIONAL]
        self.result_current = True
        self._update_status()
        self._redraw()
        return True

    def optode_positions(self, result=None) -> dict:
        """``{optode label: RAS mm}`` of the registered sources and detectors."""
        result = result or self.result
        if result is None:
            return {}
        rows = [p for p in self.points if p.kind != DIRECTIONAL]
        return {
            point.name: result.positions[i] for i, point in enumerate(rows)
            if point.kind in ("Source", "Detector")
        }

    def registered_points(self, result=None) -> list[tuple[str, str, np.ndarray]]:
        """``(name, kind, RAS mm)`` of the registered optodes and geo2d landmarks.

        Kinds are ``"SOURCE"``, ``"DETECTOR"`` or ``"LANDMARK"``, in geo2d
        order; spatial anchors and directional guidance are left out.
        """
        result = result or self.result
        if result is None:
            return []
        rows = [p for p in self.points if p.kind != DIRECTIONAL]
        return [
            (point.name, point.kind.upper(), np.asarray(result.positions[i], dtype=float))
            for i, point in enumerate(rows) if point.kind in KIND_NAMES.values()
        ]

    def state(self) -> dict:
        """Assignments, guidance and probe parts, to store with the registration."""
        return dict(
            system=self.system,
            mm_per_unit=self.scale_spin.value(),
            preserve_distances=self.distance_check.isChecked(),
            symmetric=self.symmetry_check.isChecked(),
            symmetry_pairs=list(self.symmetry_pairs),
            assignments={p.name: p.label for p in self.points
                         if p.label and p.kind != DIRECTIONAL},
            anchors={p.name: p.xy.tolist() for p in self.points if p.kind == ANCHOR},
            directions={p.name: (p.tail.tolist(), p.xy.tolist(), p.label)
                        for p in self.points if p.kind == DIRECTIONAL},
            parts={p.name: p.group for p in self.points if p.kind != DIRECTIONAL and p.group},
        )

    def _mark_stale(self, *_args) -> None:
        self.result_current = False
        self._apply_symmetry()
        self._update_status()
        self._redraw()

    def _apply_symmetry(self) -> None:
        """Set the points' ``shown`` positions from "Enforce L/R Symmetry"."""
        self.symmetry_pairs, self.symmetry_axis = [], None
        for point in self.points:
            point.shown = point.shown_tail = None
        if not self.symmetry_check.isChecked() or len(self.points) < 2:
            return
        kinds = ["Anchor" if p.kind in ("Landmark", ANCHOR) else p.kind for p in self.points]
        tips, pairs, axis = symmetrize_layout(
            np.array([p.xy for p in self.points]), kinds, [p.label for p in self.points])
        arrows = [(i, j) for i, j in pairs
                  if self.points[i].kind == DIRECTIONAL and self.points[j].kind == DIRECTIONAL]
        tails = np.array([p.tail if p.kind == DIRECTIONAL else p.xy for p in self.points])
        tails = symmetrize_points(tails, axis, arrows) if arrows else tails
        for point, tip, tail in zip(self.points, tips, tails):
            if not np.allclose(tip, point.xy):
                point.shown = tip
            if point.kind == DIRECTIONAL and not np.allclose(tail, point.tail):
                point.shown_tail = tail
        self.symmetry_axis = axis
        self.symmetry_pairs = [(self.points[i].name, self.points[j].name) for i, j in pairs]

    def _update_status(self) -> None:
        count = len(self.assigned_labels())
        self.compute_button.setEnabled(count >= MIN_ASSIGNED_LANDMARKS)
        self.compute_button.setToolTip(
            "Register the probe to the assigned positions." if count >= MIN_ASSIGNED_LANDMARKS
            else f"Assign at least {MIN_ASSIGNED_LANDMARKS} different positions first."
        )
        self.accept_button.setEnabled(self.result_current)
        text = f"{count} different position(s) assigned"
        if count < MIN_ASSIGNED_LANDMARKS:
            text += f" (at least {MIN_ASSIGNED_LANDMARKS} needed)."
        elif self.result is None:
            text += ". Click Compute registration."
        elif not self.result_current:
            text += ". The registration is out of date: compute it again."
        else:
            parts = len(set(self.groups))
            errors = list(self.result.anchor_errors.values())
            text += (f". Registered {len(self.optode_positions())} optodes"
                     + (f" in {parts} probe parts" if parts > 1 else "")
                     + f". The layout fit its positions to {self.result.rms:.1f} mm RMS "
                     f"(largest {max(errors):.1f} mm) before assigned points were moved "
                     "onto them.")
        if self.symmetry_check.isChecked():
            mirrored = sum(a != b for a, b in self.symmetry_pairs)
            text += (f" L/R symmetry: {mirrored} mirrored pair(s), "
                     f"{len(self.symmetry_pairs) - mirrored} midline point(s).")
        self.status_label.setText(text)

    # Drawing -------------------------------------------------------------
    def _redraw(self) -> None:
        self._draw_probe()
        self._draw_head()

    def _draw_probe(self) -> None:
        axes = self.probe_axes
        axes.clear()
        lookup = {p.name: p.at for p in self.points}
        for source, detector in self.channel_pairs:
            a, b = lookup[source], lookup[detector]
            axes.plot([a[0], b[0]], [a[1], b[1]], color="0.7", lw=1, zorder=1)
        if self.symmetry_axis is not None:
            self._draw_symmetry(axes)
        grouped = len(set(self.groups)) > 1
        for point in self.points:
            if point.kind == DIRECTIONAL:
                self._draw_direction(axes, point)
                continue
            style = KIND_STYLE[point.kind]
            edge = GROUP_COLORS[point.group % len(GROUP_COLORS)] if grouped else "black"
            axes.scatter(*point.at, s=60, c=style["color"], marker=style["marker"],
                         edgecolors=edge, linewidths=2.0 if grouped else 0.6, zorder=3)
            axes.annotate(point.name, point.at, xytext=(5, 4), textcoords="offset points",
                          fontsize=7, zorder=4)
            if point.label:
                axes.annotate(label_display(point.label), point.at, xytext=(5, -10),
                              textcoords="offset points", fontsize=8, fontweight="bold",
                              color=ASSIGNED_COLOR, zorder=4)
        axes.set_aspect("equal", adjustable="datalim")
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title("Right-click: assign positions, add anchors or guidance, split the probe",
                       fontsize=8)
        self.probe_canvas.draw_idle()

    def _draw_symmetry(self, axes) -> None:
        """The mirror line and, for moved points, their original positions."""
        point, normal = self.symmetry_axis
        along = np.array([-normal[1], normal[0]])
        xy = np.array([p.xy for p in self.points])
        extent = np.abs((xy - point) @ along).max() * 1.1 + 1.0
        line = point + np.outer([-extent, extent], along)
        axes.plot(*line.T, color="0.6", lw=0.8, ls="--", zorder=0)
        for p in self.points:
            for old, new in ((p.xy, p.shown), (p.tail, p.shown_tail)):
                if new is not None:
                    axes.plot([old[0], new[0]], [old[1], new[1]], color="0.5", lw=0.8, zorder=2)
                    axes.scatter(*old, s=25, c="0.5", marker="x", linewidths=1, zorder=2)

    @staticmethod
    def _draw_direction(axes, point) -> None:
        arrow = FancyArrowPatch(point.tail_at, point.at, arrowstyle="-|>", mutation_scale=14,
                                linestyle=":", lw=1.5, color=DIRECTION_COLOR, zorder=2)
        axes.add_patch(arrow)
        axes.scatter(*point.at, s=20, c=DIRECTION_COLOR, zorder=3)
        text = point.name + (f" \u2192 {label_display(point.label)}" if point.label else "")
        axes.annotate(text, point.at, xytext=(5, 4), textcoords="offset points", fontsize=7,
                      color=DIRECTION_COLOR, zorder=4)

    def _project(self, points):
        return topographic_projection(points, self._frame)

    def _draw_head(self) -> None:
        axes = self.head_axes
        axes.clear()
        head = self.head_positions
        ring = [label for label in OUTLINE_RING if label in head]
        if len(ring) > 3:
            outline = self._project([head[label] for label in ring])
            axes.plot(*outline.T, color="0.6", lw=1)
        edge = [label for label in ("Nz", "Iz", "LPA", "RPA") if label in head]
        radius = (np.mean(np.linalg.norm(self._project([head[label] for label in edge]), axis=1))
                  if edge else np.pi / 2 * self._frame[2])
        axes.add_patch(Circle((0, 0), radius, fill=False, lw=1.5, color="0.3"))
        axes.add_patch(Polygon([(-0.08 * radius, 0.99 * radius), (0, 1.1 * radius),
                                (0.08 * radius, 0.99 * radius)], closed=False, fill=False,
                               lw=1.5, color="0.3"))
        for side in (-1, 1):
            axes.add_patch(Circle((side * 1.03 * radius, 0), 0.05 * radius, fill=False,
                                  lw=1.5, color="0.3"))
        shown = self.available_labels()
        xy = self._project([head[label] for label in shown])
        axes.scatter(*xy.T, s=6, c="0.55", zorder=2)
        named = set(LABELS_1020)
        for label, point in zip(shown, xy):
            if label in named:
                axes.annotate(label_display(label).split(" ")[0], point, xytext=(2, 2),
                              textcoords="offset points", fontsize=6, color="0.4")
        assigned = sorted(self.assigned_labels())
        if assigned:
            marks = self._project([head[label] for label in assigned])
            axes.scatter(*marks.T, s=40, facecolors="none", edgecolors=ASSIGNED_COLOR,
                         linewidths=1.5, zorder=4)
            for label, point in zip(assigned, marks):
                axes.annotate(label, point, xytext=(3, -9), textcoords="offset points",
                              fontsize=7, fontweight="bold", color=ASSIGNED_COLOR)
        if self.result is not None:
            self._draw_registered(axes)
        axes.set_aspect("equal", adjustable="datalim")
        axes.set_xlim(-1.25 * radius, 1.25 * radius)
        axes.set_ylim(-1.2 * radius, 1.2 * radius)
        axes.axis("off")
        title = "Head and standard positions (nose up)"
        if self.result is not None:
            title = ("Registered probe" if self.result_current
                     else "Registered probe (out of date: compute again)")
        axes.set_title(title, fontsize=8)
        self.head_canvas.draw_idle()

    def _draw_registered(self, axes) -> None:
        alpha = 1.0 if self.result_current else 0.3
        positions = self._project(self.result.positions)
        lookup = {p.name: xy for p, xy in zip(self._result_points, positions)}
        for source, detector in self.channel_pairs:
            if source in lookup and detector in lookup:
                a, b = lookup[source], lookup[detector]
                axes.plot([a[0], b[0]], [a[1], b[1]], color="0.35", lw=1, alpha=alpha, zorder=3)
        for point, xy in zip(self._result_points, positions):
            style = KIND_STYLE[point.kind]
            axes.scatter(*xy, s=22, c=style["color"], marker=style["marker"],
                         edgecolors="black", linewidths=0.4, alpha=alpha, zorder=5)

    # Mouse -----------------------------------------------------------------
    def point_at(self, x: float, y: float) -> int | None:
        """Row of the point (arrow tip) drawn within a few pixels of data ``(x, y)``."""
        if not self.points:
            return None
        transform = self.probe_axes.transData
        clicked = transform.transform((x, y))
        drawn = transform.transform(np.array([p.at for p in self.points]))
        distance = np.linalg.norm(drawn - clicked, axis=1)
        row = int(np.argmin(distance))
        return row if distance[row] <= HIT_PIXELS else None

    def _canvas_pressed(self, event) -> None:
        if event.inaxes is not self.probe_axes or event.xdata is None:
            return
        if self._arrow is not None or self._lasso is not None:
            return
        if event.button == 3:
            row = self.point_at(event.xdata, event.ydata)
            menu = (self.point_menu(row) if row is not None
                    else self.blank_menu((event.xdata, event.ydata)))
            menu.exec(QCursor.pos())

    def _canvas_moved(self, event) -> None:
        if self._arrow is None or event.inaxes is not self.probe_axes or event.xdata is None:
            return
        self._arrow.set_positions(self._arrow_tail, (event.xdata, event.ydata))
        self.probe_canvas.draw_idle()

    def _canvas_released(self, event) -> None:
        if self._arrow is None or event.button != 1 or event.xdata is None:
            return
        self.finish_direction((event.xdata, event.ydata))

    def _canvas_key(self, event) -> None:
        if event.key == "escape":
            self.cancel_interaction()

    def start_direction(self, tail) -> None:
        """Start a direction arrow at ``tail``; it follows the mouse until a click sets it."""
        self.cancel_interaction()
        self._arrow_tail = tuple(np.asarray(tail, dtype=float))
        self._arrow = FancyArrowPatch(self._arrow_tail, self._arrow_tail, arrowstyle="-|>",
                                      mutation_scale=14, linestyle=":", lw=1.5,
                                      color=DIRECTION_COLOR)
        self.probe_axes.add_patch(self._arrow)
        self.probe_canvas.setFocus()
        self.status_label.setText("Move the mouse to aim the arrow and click to set it "
                                  "(Esc cancels).")
        self.probe_canvas.draw_idle()

    def finish_direction(self, tip) -> int | None:
        if self._arrow is None:
            return None
        tail = np.asarray(self._arrow_tail)
        self._arrow.remove()
        self._arrow = None
        if np.linalg.norm(np.asarray(tip, dtype=float) - tail) <= 0:
            self._mark_stale()
            return None
        return self.add_direction(tail, tip)

    def start_lasso(self) -> None:
        """Draw a lasso around optodes to make them a separate probe part."""
        self.cancel_interaction()
        self._lasso = LassoSelector(self.probe_axes, self._lasso_done, button=1,
                                    props=dict(color=GROUP_COLORS[1], lw=1.5))
        self.status_label.setText("Draw a loop around the optodes of one probe part "
                                  "(Esc cancels).")

    def _lasso_done(self, vertices) -> None:
        lasso, self._lasso = self._lasso, None
        if lasso is not None:
            lasso.disconnect_events()
            lasso.set_visible(False)
        if not self.separate_parts(vertices):
            self._update_status()
            self._redraw()

    def cancel_interaction(self) -> None:
        if self._arrow is not None:
            self._arrow.remove()
            self._arrow = None
        if self._lasso is not None:
            self._lasso.disconnect_events()
            self._lasso.set_visible(False)
            self._lasso = None
        self._update_status()
        self.probe_canvas.draw_idle()

    # Menus -----------------------------------------------------------------
    def point_menu(self, row: int) -> QMenu:
        """Context menu of point ``row``: its position and, for anchors/arrows, removal."""
        point = self.points[row]
        menu = QMenu(self)
        heading = menu.addAction(f"{point.name} ({point.kind})")
        heading.setEnabled(False)
        widget = QWidget(menu)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.addWidget(QLabel("Position:"))
        combo = QComboBox(widget)
        self._fill_position_combo(combo, point.label)
        combo.currentIndexChanged.connect(
            lambda _index: (self.set_label(row, combo.currentData()), menu.close()))
        layout.addWidget(combo, 1)
        action = QWidgetAction(menu)
        action.setDefaultWidget(widget)
        menu.addAction(action)
        menu.position_combo = combo
        if point.label:
            menu.addAction("Clear position").triggered.connect(
                lambda checked=False: self.set_label(row, None))
        if point.kind in (ANCHOR, DIRECTIONAL):
            menu.addSeparator()
            menu.addAction("Remove point").triggered.connect(
                lambda checked=False: self.remove_point(row))
        return menu

    def blank_menu(self, xy) -> QMenu:
        """Context menu for empty space at ``xy`` of the 2D probe plot."""
        menu = QMenu(self)
        menu.addAction("Add Landmark Anchor").triggered.connect(
            lambda checked=False: self.add_anchor(xy))
        menu.addAction("Add Directional Guidance").triggered.connect(
            lambda checked=False: self.start_direction(xy))
        menu.addSeparator()
        menu.addAction("Separate probe parts").triggered.connect(
            lambda checked=False: self.start_lasso())
        clear = menu.addAction("Clear probe parts")
        clear.setEnabled(len(set(self.groups)) > 1)
        clear.triggered.connect(lambda checked=False: self.clear_parts())
        return menu

    def _table_menu_requested(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        if row >= 0:
            self.point_menu(row).exec(self.table.viewport().mapToGlobal(pos))
