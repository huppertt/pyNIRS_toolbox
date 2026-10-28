"""Depth maps from 10-20 scalp positions to selected brain parcels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import Ellipse
from matplotlib.tri import Triangulation
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QMenu, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from scipy.spatial import cKDTree

from pyBrainAnalyzIR.sigproc.short_separation import DEFAULT_THRESHOLD
from pyBrainAnalyzIR.vis.geo2d_registration import LABELS_1020, topographic_projection
from pyBrainAnalyzIR.vis.parcel_regions import (
    add_parcellation_actions, larger_region, region_members,
)


@dataclass(frozen=True)
class DepthMapHead:
    """Head model geometry in original-size RAS millimetres."""

    key: str
    labels: tuple[str, ...]
    parcel_points: np.ndarray
    parcel_labels: np.ndarray
    parcel_colors: dict[str, tuple[float, float, float]]
    scalp_labels: tuple[str, ...]
    scalp_points: np.ndarray
    frame: tuple


@dataclass(frozen=True)
class DepthMapScan:
    """Registered optodes and unique source-detector pairs of one scan."""

    label: str
    head_key: str
    optode_labels: tuple[str, ...]
    optode_kinds: tuple[str, ...]
    optode_points: np.ndarray
    pairs: tuple[tuple[str, str], ...]


SHORT_SEPARATION_MM = float(DEFAULT_THRESHOLD.to("mm").magnitude)
SHORT_SEPARATION_LABEL = "N/A- Short Seperation Channel"


def _selected_vertices(head: DepthMapHead, selected: set[str], subregions: bool = True):
    if not selected or len(head.parcel_points) == 0:
        return None, None
    all_labels = (
        head.parcel_labels if subregions
        else np.array([larger_region(label) for label in head.parcel_labels])
    )
    mask = np.isin(all_labels, tuple(selected))
    if not mask.any():
        return None, None
    points = np.asarray(head.parcel_points, dtype=float)[mask]
    labels = np.asarray(all_labels, dtype=str)[mask]
    return cKDTree(points), labels


def nearest_selected_parcels(points, tree, parcel_labels):
    """Nearest selected parcel vertex and Euclidean distance for each point."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if tree is None or len(points) == 0:
        return np.full(len(points), None, dtype=object), np.full(len(points), np.nan)
    distances, indices = tree.query(points)
    return parcel_labels[indices], distances


class DepthMapDialog(QDialog):
    """Interactive scalp depth map, channel table, and registered probes."""

    _outline_labels = (
        "Fp1", "Fpz", "Fp2", "F8", "T8", "P8", "O2", "Oz", "O1", "P7", "T7", "F7",
    )
    _probe_colors = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf")

    def __init__(
        self,
        heads: dict[str, DepthMapHead],
        scans: list[DepthMapScan],
        reference_head: str,
        parent=None,
    ):
        super().__init__(parent)
        if reference_head not in heads:
            raise ValueError(f"Unknown reference head model {reference_head!r}.")
        self.heads = heads
        self.scans = list(scans)
        self.reference_head = reference_head
        self.setWindowTitle(f"Region-of-Interest Depth Map — {reference_head}")
        self.resize(1250, 850)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        left = QWidget(splitter)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        left_layout.addWidget(self.canvas, stretch=3)
        self.table = QTableWidget(left)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setToolTip("Click a channel to highlight it on the depth map.")
        self._highlight_artists = []
        self._channel_names: list[str] = []
        self.table.itemSelectionChanged.connect(self._highlight_selected_channels)
        left_layout.addWidget(self.table, stretch=2)

        right = QWidget(splitter)
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Regions (select one or more):", right))
        self.region_list = QListWidget(right)
        self.region_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.region_list.setUniformItemSizes(True)
        self.show_subregions = False
        self.region_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.region_list.customContextMenuRequested.connect(self._region_context_menu)
        right_layout.addWidget(self.region_list, stretch=3)
        right_layout.addWidget(QLabel("Registered 3D scans (select probes to draw):", right))
        self.scan_list = QListWidget(right)
        self.scan_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.scan_list.setUniformItemSizes(True)
        right_layout.addWidget(self.scan_list, stretch=2)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

        self._region_names = tuple(
            name for name in heads[reference_head].labels
            if name and not name.startswith("Background")
        )
        self._region_groups = region_members(self._region_names)
        self.region_list.addItems(list(self._region_groups))
        for index, scan in enumerate(self.scans):
            item = QListWidgetItem(scan.label)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setForeground(QColor(self._probe_colors[index % len(self._probe_colors)]))
            item.setToolTip(
                f"Head model: {scan.head_key}; select to draw this registered probe."
            )
            self.scan_list.addItem(item)

        self.region_list.itemSelectionChanged.connect(self._selection_changed)
        self.scan_list.itemSelectionChanged.connect(self._probes_changed)
        self._colorbar = None
        self._selection_changed()

    def _selected_regions(self) -> set[str]:
        return {item.text() for item in self.region_list.selectedItems()}

    def _make_region_menu(self) -> QMenu:
        menu = QMenu(self.region_list)
        menu.addAction("Clear Selection", self.region_list.clearSelection)
        menu.addSeparator()
        add_parcellation_actions(menu, self.show_subregions, self._set_parcellation_level)
        return menu

    def _region_context_menu(self, position) -> None:
        menu = self._make_region_menu()
        menu.exec(self.region_list.viewport().mapToGlobal(position))
        menu.deleteLater()

    def _set_parcellation_level(self, subregions: bool) -> None:
        if self.show_subregions == subregions:
            return
        selected = self._selected_regions()
        if subregions:
            selected = {
                member for group in selected for member in self._region_groups[group]
            }
        else:
            selected = {larger_region(label) for label in selected}
        self.show_subregions = subregions
        self.region_list.blockSignals(True)
        self.region_list.clear()
        names = self._region_names if subregions else tuple(self._region_groups)
        self.region_list.addItems(list(names))
        for row in range(self.region_list.count()):
            item = self.region_list.item(row)
            item.setSelected(item.text() in selected)
        self.region_list.blockSignals(False)
        self.region_list.setToolTip(
            "Switching to larger regions selects the whole parent of each selected sub-region."
        )
        self._selection_changed()

    def _selected_scan_indices(self) -> list[int]:
        return sorted(
            int(item.data(Qt.ItemDataRole.UserRole)) for item in self.scan_list.selectedItems()
        )

    def _all_channels(self) -> list[str]:
        channels = set()
        for index in self._selected_scan_indices():
            scan = self.scans[index]
            channels.update(f"{source}-{detector}" for source, detector in scan.pairs)
        return sorted(channels)

    def _create_table(self) -> None:
        indices = self._selected_scan_indices()
        previous = set(self.selected_channels())
        self._channel_names = self._all_channels()
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setVisible(bool(indices))
        self.table.setRowCount(len(self._channel_names))
        headers = (
            ["Channel", "Larger region", "Sub-region parcellation", "Depth (mm)"]
            if len(indices) == 1
            else ["Channel", *(self.scans[index].label for index in indices)]
        )
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        for row, name in enumerate(self._channel_names):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, item)
        # Keep the highlighted channels when regions or probes change.
        model = self.table.selectionModel()
        model.clearSelection()
        flags = model.SelectionFlag.Select | model.SelectionFlag.Rows
        for row, name in enumerate(self._channel_names):
            if name in previous:
                model.select(self.table.model().index(row, 0), flags)
        self.table.blockSignals(False)
        for column in range(1, len(headers)):
            self.table.setColumnWidth(column, 230 if len(indices) == 1 and column < 3 else 150)
            self.table.horizontalHeaderItem(column).setToolTip(
                "Nearest selected parcel to the channel midpoint; depth is Euclidean distance in mm."
            )
        self.table.setColumnWidth(0, 130)

    def _update_table(self, selected: set[str]) -> None:
        """Report only selected probes, using the same parcel mask as the map."""
        self._create_table()
        indices = self._selected_scan_indices()
        single = len(indices) == 1
        for column, index in enumerate(indices, start=1):
            scan = self.scans[index]
            head = self.heads.get(scan.head_key)
            tree, labels = None, None
            if head is not None:
                # Query original labels so a single probe can report both levels
                # for the same nearest vertex, even when the map uses larger ROIs.
                raw_selected = (
                    selected if self.show_subregions else {
                        label for label in head.labels if larger_region(label) in selected
                    }
                )
                tree, labels = _selected_vertices(head, raw_selected)
            lookup = dict(zip(scan.optode_labels, scan.optode_points))
            pairs = {
                f"{source}-{detector}": (source, detector)
                for source, detector in scan.pairs
            }
            valid_names, centers, separations = [], [], []
            for name in self._channel_names:
                pair = pairs.get(name)
                if pair is None or pair[0] not in lookup or pair[1] not in lookup:
                    continue
                valid_names.append(name)
                centers.append((lookup[pair[0]] + lookup[pair[1]]) / 2)
                separations.append(np.linalg.norm(lookup[pair[0]] - lookup[pair[1]]))
            regions, distances = nearest_selected_parcels(centers, tree, labels)
            values = dict(zip(valid_names, zip(regions, distances, separations)))
            for row, name in enumerate(self._channel_names):
                result = values.get(name)
                if result is not None and result[2] < SHORT_SEPARATION_MM:
                    texts = (
                        [SHORT_SEPARATION_LABEL, SHORT_SEPARATION_LABEL, ""]
                        if single else [SHORT_SEPARATION_LABEL]
                    )
                    for offset, text in enumerate(texts):
                        item = QTableWidgetItem(text)
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        item.setForeground(QColor("gray"))
                        item.setToolTip(
                            f"Source-detector distance {result[2]:.1f} mm is below the "
                            f"{SHORT_SEPARATION_MM:g} mm short-separation threshold; "
                            "it samples the scalp rather than the brain."
                        )
                        self.table.setItem(row, column + offset, item)
                    continue
                # A region deeper than the source-detector distance is out of reach.
                valid = (
                    result is not None and result[0] is not None and result[1] < result[2]
                )
                if single:
                    texts = (
                        [larger_region(result[0]), result[0], f"{result[1]:.1f}"]
                        if valid else ["", "", ""]
                    )
                else:
                    region = (
                        result[0] if self.show_subregions else larger_region(result[0])
                    ) if valid else ""
                    texts = [f"{region} ({result[1]:.1f} mm)" if valid else ""]
                for offset, text in enumerate(texts):
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(row, column + offset, item)

    def _outline(self, head: DepthMapHead, frame_radius: float):
        positions = dict(zip(head.scalp_labels, head.scalp_points))
        labels = [label for label in self._outline_labels if label in positions]
        if len(labels) < 3:
            return np.empty((0, 2))
        projected = topographic_projection(np.array([positions[label] for label in labels]), head.frame)
        scale = frame_radius / head.frame[2]
        return projected * scale

    def _draw_head_outline(self, head: DepthMapHead, frame_radius: float) -> np.ndarray:
        outline = self._outline(head, frame_radius)
        if len(outline):
            closed = np.vstack((outline, outline[0]))
            self.axes.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.6, zorder=5)
            top = float(outline[:, 1].max())
        else:
            theta = np.linspace(0, 2 * np.pi, 200)
            outline = frame_radius * np.column_stack((np.cos(theta), np.sin(theta)))
            self.axes.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.6, zorder=5)
            top = frame_radius
        # A simple nose and ears complete the 10-20 head outline.
        self.axes.plot(
            [-frame_radius * 0.05, -frame_radius * 0.025, 0,
             frame_radius * 0.025, frame_radius * 0.05],
            [top - 4, top + 5, top + 22, top + 5, top - 4],
            color="black", linewidth=1.5, zorder=5,
        )
        self.axes.add_patch(Ellipse(
            (-frame_radius * 1.015, 0), frame_radius * 0.12, frame_radius * 0.36,
            fill=False, edgecolor="black", linewidth=1.4, zorder=5,
        ))
        self.axes.add_patch(Ellipse(
            (frame_radius * 1.015, 0), frame_radius * 0.12, frame_radius * 0.36,
            fill=False, edgecolor="black", linewidth=1.4, zorder=5,
        ))
        return outline

    def _draw_depth(self, selected: set[str]) -> None:
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        self.axes.clear()
        self._highlight_artists = []
        head = self.heads[self.reference_head]
        frame_radius = head.frame[2]
        scalp_points = np.asarray(head.scalp_points, dtype=float)
        scalp_xy = topographic_projection(scalp_points, head.frame)
        outline = self._outline(head, frame_radius)
        path = Path(outline) if len(outline) else None
        valid = (
            path.contains_points(scalp_xy, radius=1e-7)
            if path is not None else np.ones(len(scalp_xy), dtype=bool)
        )
        sample_points, sample_xy = scalp_points[valid], scalp_xy[valid]
        tree, parcel_labels = _selected_vertices(head, selected, self.show_subregions)
        depths = np.full(len(sample_xy), np.nan)
        if tree is not None:
            _, depths = nearest_selected_parcels(sample_points, tree, parcel_labels)
            triangulation = Triangulation(sample_xy[:, 0], sample_xy[:, 1])
            if len(triangulation.triangles):
                if path is not None:
                    centroids = sample_xy[triangulation.triangles].mean(axis=1)
                    triangulation.set_mask(~path.contains_points(centroids))
                finite = np.isfinite(depths)
                if finite.sum() >= 3:
                    contours = self.axes.tricontourf(
                        triangulation, np.nan_to_num(depths, nan=0.0),
                        levels=np.linspace(0, 40, 21), cmap="viridis", vmin=0, vmax=40,
                        extend="max",
                    )
                    self._colorbar = self.figure.colorbar(
                        contours, ax=self.axes,
                        label="Distance to nearest selected parcel (mm)",
                    )
        outline = self._draw_head_outline(head, frame_radius)
        visible_1020 = np.array([label in LABELS_1020 for label in head.scalp_labels]) & valid
        self.axes.scatter(
            scalp_xy[visible_1020, 0], scalp_xy[visible_1020, 1],
            s=10, color="black", zorder=6,
        )
        for label, (x, y), show in zip(head.scalp_labels, scalp_xy, visible_1020):
            if show:
                self.axes.annotate(
                    label, (x, y), xytext=(3, 3), textcoords="offset points",
                    fontsize=6, zorder=7,
                )
        self._draw_probes()
        limit = (
            max(float(np.max(np.abs(outline))), frame_radius * 1.12) * 1.15
            if len(outline) else frame_radius * 1.3
        )
        self.axes.set_xlim(-limit, limit)
        self.axes.set_ylim(-limit, limit * 1.08)
        self.axes.set_aspect("equal")
        self.axes.axis("off")
        self.axes.set_title(f"Depth to selected parcels — {self.reference_head}")
        if not selected:
            self.axes.text(0, 0, "Select one or more regions", ha="center", va="center")
        elif tree is None:
            self.axes.text(0, 0, "No selected parcel vertices in this head model",
                           ha="center", va="center")
        self.canvas.draw_idle()

    def _projected_optodes(self, index: int) -> dict[str, np.ndarray]:
        scan = self.scans[index]
        head = self.heads.get(scan.head_key)
        if head is None or len(scan.optode_points) == 0:
            return {}
        points_xy = topographic_projection(np.asarray(scan.optode_points), head.frame)
        points_xy *= self.heads[self.reference_head].frame[2] / head.frame[2]
        return dict(zip(scan.optode_labels, points_xy))

    def selected_channels(self) -> list[str]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        names = self._channel_names
        return [names[row] for row in rows if row < len(names)]

    def _highlight_selected_channels(self) -> None:
        """Overlay the selected table channels on the map without recomputing depth."""
        for artist in self._highlight_artists:
            artist.remove()
        self._highlight_artists = []
        channels = self.selected_channels()
        for index in self._selected_scan_indices() if channels else []:
            scan = self.scans[index]
            lookup = self._projected_optodes(index)
            pairs = {f"{source}-{detector}": (source, detector) for source, detector in scan.pairs}
            for name in channels:
                pair = pairs.get(name)
                if pair is None or pair[0] not in lookup or pair[1] not in lookup:
                    continue
                ends = np.array([lookup[pair[0]], lookup[pair[1]]])
                middle = ends.mean(axis=0)
                self._highlight_artists += [
                    *self.axes.plot(ends[:, 0], ends[:, 1], color="black",
                                    linewidth=7, solid_capstyle="round", zorder=10),
                    *self.axes.plot(ends[:, 0], ends[:, 1], color="#ff00ff",
                                    linewidth=4.5, solid_capstyle="round", zorder=11),
                    self.axes.scatter(middle[0], middle[1], marker="*", s=180,
                                      color="#ff00ff", edgecolors="black",
                                      linewidths=0.8, zorder=12),
                    self.axes.annotate(
                        name, middle, xytext=(6, 6), textcoords="offset points",
                        fontsize=8, fontweight="bold", zorder=13,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="black", linewidth=0.6, alpha=0.9),
                    ),
                ]
        self.canvas.draw_idle()

    def _draw_probes(self) -> None:
        selected = self._selected_scan_indices()
        for index in selected:
            scan = self.scans[index]
            lookup = self._projected_optodes(index)
            if not lookup:
                continue
            color = self._probe_colors[index % len(self._probe_colors)]
            for source, detector in scan.pairs:
                if source in lookup and detector in lookup:
                    self.axes.plot(
                        [lookup[source][0], lookup[detector][0]],
                        [lookup[source][1], lookup[detector][1]],
                        color=color, alpha=0.7, linewidth=1.2, zorder=8,
                    )
            kinds = dict(zip(scan.optode_labels, scan.optode_kinds))
            for label, point in lookup.items():
                marker = "o" if kinds.get(label) == "SOURCE" else "s"
                self.axes.scatter(
                    point[0], point[1], marker=marker, s=22, color=color,
                    edgecolors="black", linewidths=0.4, zorder=9,
                )

    def _selection_changed(self) -> None:
        selected = self._selected_regions()
        self._update_table(selected)
        self._draw_depth(selected)
        self._highlight_selected_channels()

    def _probes_changed(self) -> None:
        self._selection_changed()
