"""Interactive first- and group-level connectivity result viewer."""
from __future__ import annotations

import sys
import re

import numpy as np
import pandas as pd
import patsy
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D
from PySide6.QtCore import Qt
from PySide6.QtGui import QActionGroup, QBrush, QCursor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from pyBrainAnalyzIR.dataclasses.connectivity import Connectivity
from pyBrainAnalyzIR.dataclasses.model_fit import ConnectivityGroupFit
from pyBrainAnalyzIR.vis.plot_nirs_inline import _draw_scalp_outline
from pyBrainAnalyzIR.vis.statistics_viewer import (
    SIGNIFICANT_ROW_COLOR,
    _clipboard_value,
    _display_value,
)


class ConnectivityViewer(QMainWindow):
    """Browse connectivity estimates and the observations underlying a selected edge."""

    def __init__(
        self,
        connectivity: Connectivity,
        data=None,
        input_name: str = "conc",
        *,
        plot_on_scalp: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        if not isinstance(connectivity, Connectivity):
            raise TypeError("ConnectivityViewer requires a Connectivity result.")
        if connectivity.coef.empty:
            raise ValueError("The Connectivity result has no edge estimates.")
        self.connectivity = connectivity
        self.data = data
        self.input_name = input_name
        self.plot_on_scalp = bool(plot_on_scalp)
        self.fit_data = connectivity.fit_data
        layouts_3d, is_10_20_view = connectivity.probe_layouts(True)
        self.probe_3d_available = bool(layouts_3d) and is_10_20_view
        self.map_mode = "3d" if self.plot_on_scalp and self.probe_3d_available else "2d"
        self.plot_on_scalp = self.map_mode == "3d"
        if isinstance(self.fit_data, ConnectivityGroupFit):
            self.is_hyperscanning = bool(self.fit_data.hyperscanning)
        else:
            coef = connectivity.coef
            self.is_hyperscanning = (
                "Subject_to" in coef
                and "Subject_from" in coef
                and (
                    coef["Subject_to"].astype(str)
                    != coef["Subject_from"].astype(str)
                ).any()
            )
        self._selected_edge = None
        self._table_frame = pd.DataFrame()
        self._sort_column = None
        self._sort_ascending = True
        self._show_only_significant = True
        self.setWindowTitle("Connectivity Viewer")
        self.resize(1250, 750)

        central = QWidget()
        layout = QVBoxLayout(central)
        controls = QHBoxLayout()
        self.condition_combo = QComboBox()
        self.condition_combo.addItems(connectivity.conditions)
        self.type_combo = QComboBox()
        self._populate_type_pairs()
        self.threshold = QDoubleSpinBox()
        self.threshold.setDecimals(6)
        self.threshold.setRange(0, 1)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.05)
        self.threshold_type = QComboBox()
        self.threshold_type.addItems(["q-value", "p-value"])
        self.endpoint_combo = QComboBox()
        self.endpoint_combo.addItems(["Inter-subject only", "Intra-subject only", "Both"])
        self.endpoint_combo.setVisible(self.is_hyperscanning)
        for label, widget in (
            ("Task / model term", self.condition_combo),
            ("Data type pair", self.type_combo),
            ("Threshold", self.threshold),
            ("Threshold type", self.threshold_type),
            ("Dyad endpoints", self.endpoint_combo),
        ):
            control_label = QLabel(label)
            control_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            controls.addWidget(control_label)
            controls.addWidget(widget)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)
        self.map_figure = Figure(layout="constrained")
        self.detail_figure = Figure(layout="constrained")
        self.map_canvas = FigureCanvasQTAgg(self.map_figure)
        self.detail_tabs = QTabWidget()
        if isinstance(self.fit_data, ConnectivityGroupFit):
            self.detail_canvas = FigureCanvasQTAgg(self.detail_figure)
            self.detail_tabs.addTab(self.detail_canvas, "Results Plot")
            self.scatter_canvas = None
        else:
            time_panel = QWidget()
            QVBoxLayout(time_panel).addWidget(FigureCanvasQTAgg(self.detail_figure))
            self.detail_canvas = time_panel.layout().itemAt(0).widget()
            self.scatter_figure = Figure(layout="constrained")
            self.scatter_canvas = FigureCanvasQTAgg(self.scatter_figure)
            scatter_panel = QWidget()
            QVBoxLayout(scatter_panel).addWidget(self.scatter_canvas)
            self.detail_tabs.addTab(time_panel, "Time courses")
            self.detail_tabs.addTab(scatter_panel, "Endpoint scatter")
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.verticalHeader().setVisible(False)
        header = self.results_table.horizontalHeader()
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self._sort_results_from_header)
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._show_table_menu)
        self.results_table.itemSelectionChanged.connect(self._on_table_selection)
        self.results_tabs = QTabWidget()
        self.results_tabs.addTab(self.detail_tabs, "Results Plot")
        self.results_tabs.addTab(self.results_table, "Results Table")
        splitter.addWidget(self.map_canvas)
        splitter.addWidget(self.results_tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

        self.condition_combo.currentTextChanged.connect(self._refresh)
        self.type_combo.currentTextChanged.connect(self._refresh)
        self.threshold.valueChanged.connect(self._refresh)
        self.threshold_type.currentTextChanged.connect(self._refresh)
        self.endpoint_combo.currentTextChanged.connect(self._refresh)
        if self.scatter_canvas is not None:
            self.scatter_canvas.mpl_connect("pick_event", self._on_pick)
        self.map_canvas.mpl_connect("pick_event", self._on_pick)
        self.map_canvas.mpl_connect("button_press_event", self._on_map_click)
        self._refresh()

    def _on_map_click(self, event):
        if event.inaxes is not self.map_ax:
            return
        if event.button == 3:
            self._show_map_context_menu()
            return
        if event.button != 1 or self.map_mode != "adjacency":
            return
        if event.xdata is None or event.ydata is None:
            return
        column, row = int(round(event.xdata)), int(round(event.ydata))
        edge = self._matrix_edge_lookup.get((row, column))
        if edge is None:
            edge = self._matrix_edge_lookup.get((column, row))
        if edge is not None:
            self._selected_edge = edge
            self._draw_detail()

    def _show_map_context_menu(self):
        self._build_map_context_menu().exec(QCursor.pos())

    def _build_map_context_menu(self):
        menu = QMenu(self)
        group = QActionGroup(menu)
        group.setExclusive(True)
        options = [("2D probe", "2d", True)]
        if self.probe_3d_available:
            options.append(("3D probe / 10-20 view", "3d", True))
        options.append(("Adjacency matrix", "adjacency", True))
        for label, mode, enabled in options:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.map_mode == mode)
            action.setEnabled(enabled)
            group.addAction(action)
            action.triggered.connect(
                lambda _checked=False, selected_mode=mode: self._set_map_mode(selected_mode)
            )
        return menu

    def _set_map_mode(self, mode):
        if mode not in {"2d", "3d", "adjacency"}:
            raise ValueError(f"Unknown connectivity map view: {mode!r}")
        if mode == "3d" and not self.probe_3d_available:
            raise ValueError("A 3D probe view is not available for this connectivity result.")
        if self.map_mode == mode:
            return
        self.map_mode = mode
        self.plot_on_scalp = mode == "3d"
        self._refresh()

    def _populate_type_pairs(self):
        table = self.connectivity.coef
        pairs = list(dict.fromkeys(zip(table.Type_to.astype(str), table.Type_from.astype(str))))
        self.type_combo.addItems([f"{to} / {from_}" for to, from_ in pairs])
        self._type_pairs = pairs

    @property
    def _current_type_pair(self):
        index = max(0, self.type_combo.currentIndex())
        return self._type_pairs[index] if self._type_pairs else (None, None)

    def _filtered_table(self):
        table = self.connectivity.table()
        type_to, type_from = self._current_type_pair
        selected = table[
            (table["Condition"].astype(str) == self.condition_combo.currentText())
            & (table["Type_to"].astype(str) == type_to)
            & (table["Type_from"].astype(str) == type_from)
        ].copy()
        if "Subject_to" in selected and not self.endpoint_combo.isHidden():
            endpoint_mode = self.endpoint_combo.currentIndex()
            same = selected["Subject_to"].astype(str) == selected["Subject_from"].astype(str)
            if endpoint_mode == 0:
                selected = selected.loc[~same]
            elif endpoint_mode == 1:
                selected = selected.loc[same]
        selected = selected[
            (selected["Channel_to"] != selected["Channel_from"])
            | (selected["Type_to"] != selected["Type_from"])
            | (selected.get("geo_to", pd.Series(index=selected.index)).astype(str)
               != selected.get("geo_from", pd.Series(index=selected.index)).astype(str))
        ]
        return selected

    def _refresh(self, *_):
        self.map_figure.clear()
        ax = self.map_figure.add_subplot()
        self.map_ax = ax
        table = self._filtered_table()
        threshold_column = "Qvalue" if self.threshold_type.currentIndex() == 0 else "Pvalue"
        passing = table[threshold_column].astype(float) < self.threshold.value()
        table["Significant"] = passing.to_numpy(dtype=bool)
        self._set_results_frame(table)
        significant = table.loc[passing]
        visible_values = np.abs(
            significant["Coeff"].astype(float).to_numpy()
        )
        maximum = float(visible_values.max()) if visible_values.size else 1.0
        norm = Normalize(-maximum, maximum)
        cmap = colormaps["RdBu_r"]
        self._artists = {}
        self._matrix_edge_lookup = {}
        if self.map_mode == "adjacency":
            self._draw_adjacency_matrix(ax, significant, norm, cmap)
        else:
            self._draw_probe_map(ax, significant, norm, cmap)
        ax.set_title(f"{self.condition_combo.currentText()} | "
                     f"{self.type_combo.currentText()} | {int(passing.sum())} of "
                     f"{len(table)} edges")
        self.map_figure.colorbar(
            __import__("matplotlib").cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax, label="Correlation",
        )
        self.map_canvas.draw_idle()
        if self._selected_edge is None or not self._edge_exists(self._selected_edge, significant):
            self._selected_edge = significant.iloc[0] if not significant.empty else None
        self._draw_detail()

    def _set_results_frame(self, table):
        frame = table.copy()
        if self._show_only_significant:
            frame = frame.loc[frame["Significant"]]
        columns = ["Condition"]
        if self.is_hyperscanning:
            columns.extend(["Subject_to", "Subject_from"])
        columns.extend([
            "Channel_to", "Type_to", "Channel_from", "Type_from",
            "Coeff", "Pvalue", "Qvalue", "Tstat", "dfe", "ZstdErr",
            "Significant",
        ])
        self._table_frame = frame[
            [column for column in columns if column in frame]
        ].rename(columns={
            "Channel_to": "To_Channel",
            "Type_to": "To_Type",
            "Channel_from": "From_Channel",
            "Type_from": "From_Type",
            "Subject_to": "To_Subject",
            "Subject_from": "From_Subject",
            "Coeff": "Correlation",
            "Pvalue": "P-value",
            "Qvalue": "Q-value",
            "Tstat": "T-statistic",
            "ZstdErr": "Z standard error",
        })
        if self._sort_column in self._table_frame:
            self._table_frame = self._table_frame.sort_values(
                self._sort_column, ascending=self._sort_ascending,
                na_position="last", kind="mergesort",
            ).reset_index(drop=True)
        else:
            self._table_frame = self._table_frame.reset_index(drop=True)
        self._populate_results_table()

    def _populate_results_table(self):
        table = self.results_table
        frame = self._table_frame
        table.blockSignals(True)
        table.clearContents()
        table.setRowCount(len(frame))
        table.setColumnCount(len(frame.columns))
        table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
        bold = QFont()
        bold.setBold(True)
        for row_index, record in enumerate(frame.itertuples(index=False, name=None)):
            significant = bool(record[frame.columns.get_loc("Significant")])
            for column_index, value in enumerate(record):
                item = QTableWidgetItem(_display_value(value))
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, float, np.number)
                ):
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if significant:
                    item.setBackground(QBrush(SIGNIFICANT_ROW_COLOR))
                    item.setFont(bold)
                table.setItem(row_index, column_index, item)
        table.resizeColumnsToContents()
        table.blockSignals(False)

    def results_frame(self):
        return self._table_frame.copy()

    def _sort_results_from_header(self, position):
        column = self.results_table.horizontalHeader().logicalIndexAt(position)
        if column >= 0:
            self._sort_table_column(column)

    def _sort_table_column(self, column):
        if not 0 <= column < len(self._table_frame.columns):
            return
        name = self._table_frame.columns[column]
        if name == self._sort_column:
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_column = name
            self._sort_ascending = True
        self._table_frame = self._table_frame.sort_values(
            name, ascending=self._sort_ascending, na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
        self._populate_results_table()

    def _on_table_selection(self):
        rows = self.results_table.selectionModel().selectedRows()
        if not rows:
            return
        record = self._table_frame.iloc[rows[0].row()]
        source = self.connectivity.coef
        mask = (
            (source["Condition"].astype(str) == str(record["Condition"]))
            & (source["Channel_to"].astype(str) == str(record["To_Channel"]))
            & (source["Channel_from"].astype(str) == str(record["From_Channel"]))
            & (source["Type_to"].astype(str) == str(record["To_Type"]))
            & (source["Type_from"].astype(str) == str(record["From_Type"]))
        )
        if self.is_hyperscanning and "To_Subject" in record:
            mask &= (
                source["Subject_to"].astype(str) == str(record["To_Subject"])
            ) & (
                source["Subject_from"].astype(str) == str(record["From_Subject"])
            )
        matches = source.loc[mask]
        if not matches.empty:
            self._selected_edge = self.connectivity.table().iloc[
                int(source.index.get_loc(matches.index[0]))
            ]
            self._draw_detail()

    def _build_table_menu(self):
        menu = QMenu(self.results_table)
        show_significant = menu.addAction("Show Only Significant Channels")
        show_significant.setCheckable(True)
        show_significant.setChecked(self._show_only_significant)
        show_significant.toggled.connect(self._set_show_only_significant)
        menu.addSeparator()
        menu.addAction("Copy to Clipboard").triggered.connect(
            self.copy_table_to_clipboard
        )
        menu.addAction("Save to Excel...").triggered.connect(
            lambda *_: self.save_table_to_excel()
        )
        return menu

    def _show_table_menu(self, position):
        self._build_table_menu().exec(
            self.results_table.viewport().mapToGlobal(position)
        )

    def _set_show_only_significant(self, checked):
        checked = bool(checked)
        if checked == self._show_only_significant:
            return
        self._show_only_significant = checked
        self._refresh()

    def table_clipboard_text(self):
        frame = self._table_frame
        lines = ["\t".join(str(column) for column in frame.columns)]
        lines.extend(
            "\t".join(_clipboard_value(value) for value in record)
            for record in frame.itertuples(index=False, name=None)
        )
        return "\n".join(lines) + "\n"

    def copy_table_to_clipboard(self):
        QApplication.clipboard().setText(self.table_clipboard_text())
        self.statusBar().showMessage(
            f"Copied {len(self._table_frame)} rows to the clipboard.", 5000
        )

    def save_table_to_excel(self, path=None):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Results Table", "connectivity.xlsx",
                "Excel Workbook (*.xlsx)",
            )
            if not path:
                return None
        path = str(path)
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            self._table_frame.to_excel(
                path, index=False, sheet_name="Connectivity", engine="openpyxl"
            )
        except (OSError, ValueError, ImportError) as exc:
            QMessageBox.critical(
                self, "Save Results Table", f"Could not save the table:\n{exc}"
            )
            return None
        self.statusBar().showMessage(f"Saved results table to {path}", 5000)
        return path

    def _draw_probe_map(self, ax, significant, norm, cmap):
        layouts, on_scalp = self.connectivity.probe_layouts(self.map_mode == "3d")
        if layouts:
            probe_points = []
            for geometry, outline in layouts.values():
                if geometry is None:
                    continue
                labels, positions = self._layout_arrays(geometry)
                self._draw_optodes(ax, labels, positions)
                probe_points.append(positions[:, :2])
                if on_scalp and outline is not None:
                    _draw_scalp_outline(ax, Affine2D(outline), set_limits=False)
                    angles = np.linspace(0, 2 * np.pi, 64)
                    ring = np.c_[
                        1.15 * np.cos(angles),
                        1.15 * np.sin(angles),
                        np.ones_like(angles),
                    ]
                    probe_points.append((ring @ outline.T)[:, :2])
            for _, row in significant.iterrows():
                geo_to = str(row.get("geo_to", next(iter(layouts))))
                geo_from = str(row.get("geo_from", geo_to))
                if geo_to not in layouts or geo_from not in layouts:
                    continue
                p_to = self._channel_midpoint(
                    *self._layout_arrays(layouts[geo_to][0]), str(row["Channel_to"])
                )
                p_from = self._channel_midpoint(
                    *self._layout_arrays(layouts[geo_from][0]), str(row["Channel_from"])
                )
                if p_to is None or p_from is None:
                    continue
                artist, = ax.plot(
                    [p_to[0], p_from[0]], [p_to[1], p_from[1]],
                    linewidth=2, color=cmap(norm(float(row["Coeff"]))), zorder=2,
                )
                artist.set_picker(6)
                self._artists[artist] = row
            if probe_points:
                probe_points = np.vstack(probe_points)
                low = np.nanmin(probe_points, axis=0)
                high = np.nanmax(probe_points, axis=0)
                padding = (
                    0.0 if on_scalp
                    else max(float(np.max(high - low)) * 0.1, 1.0)
                )
                ax.set_xlim(low[0] - padding, high[0] + padding)
                ax.set_ylim(low[1] - padding, high[1] + padding)
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
        else:
            values = significant["Coeff"].astype(float).to_numpy()
            artists = ax.scatter(
                np.arange(len(significant)), values,
                c=[cmap(norm(value)) for value in values], picker=True, zorder=2,
            )
            ax.set_xticks(np.arange(len(significant)), [
                f"{r.Channel_to} <-> {r.Channel_from}" for r in significant.itertuples()
            ], rotation=70)
            self._artists[artists] = significant
            ax.set_ylabel("Correlation")

    def _adjacency_label(self, row, endpoint):
        channel = str(row[f"Channel_{endpoint}"])
        signal_type = str(row[f"Type_{endpoint}"])
        if self.is_hyperscanning and f"Subject_{endpoint}" in row:
            subject = str(row[f"Subject_{endpoint}"])
            return f"{subject}:{channel} ({signal_type})"
        return f"{channel} ({signal_type})"

    def _draw_adjacency_matrix(self, ax, significant, norm, cmap):
        self._matrix_edge_lookup = {}
        if significant.empty:
            ax.text(
                0.5, 0.5, "No edges meet the selected threshold.",
                ha="center", va="center", transform=ax.transAxes,
            )
            return
        edge_names = [
            (self._adjacency_label(row, "to"), self._adjacency_label(row, "from"))
            for _, row in significant.iterrows()
        ]
        nodes = list(dict.fromkeys(name for pair in edge_names for name in pair))
        node_index = {name: index for index, name in enumerate(nodes)}
        matrix = np.full((len(nodes), len(nodes)), np.nan, dtype=float)
        for (_, row), (to_name, from_name) in zip(significant.iterrows(), edge_names):
            to_index, from_index = node_index[to_name], node_index[from_name]
            matrix[to_index, from_index] = float(row["Coeff"])
            matrix[from_index, to_index] = float(row["Coeff"])
            self._matrix_edge_lookup[(from_index, to_index)] = row
            self._matrix_edge_lookup[(to_index, from_index)] = row
        ax.imshow(
            np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, interpolation="none"
        )
        ticks = np.arange(len(nodes))
        ax.set_xticks(ticks, nodes, rotation=90)
        ax.set_yticks(ticks, nodes)
        ax.set_xticks(np.arange(-0.5, len(nodes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(nodes), 1), minor=True)
        ax.grid(which="minor", color="0.35", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlim(-0.5, len(nodes) - 0.5)
        ax.set_ylim(len(nodes) - 0.5, -0.5)
        ax.set_xlabel("Connected channel")
        ax.set_ylabel("Connected channel")
        ax.set_aspect("equal")

    @staticmethod
    def _draw_optodes(ax, labels, positions):
        for label, point in zip(labels, positions):
            label = str(label)
            if not label.startswith(("S", "D")):
                continue
            color = "#e41a1c" if label.startswith("S") else (
                "#377eb8" if label.startswith("D") else "0.25"
            )
            ax.scatter(
                point[0], point[1], s=28, color=color, edgecolor="white",
                linewidth=0.5, zorder=5,
            )
            ax.text(
                point[0], point[1], label, ha="center", va="center",
                fontsize=8, color=color, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5},
            )

    @staticmethod
    def _layout_arrays(geometry):
        labels = np.asarray([str(value) for value in geometry.label.values])
        positions = np.asarray(
            geometry.pint.dequantify().values if hasattr(geometry, "pint")
            else geometry.values, dtype=float,
        )
        return labels, positions

    @staticmethod
    def _channel_midpoint(labels, positions, channel):
        if "D" not in channel:
            return None
        source, detector = channel.split("D", 1)
        detector = "D" + detector
        src = positions[labels == source]
        det = positions[labels == detector]
        if not len(src) or not len(det):
            return None
        return (src[0, :2] + det[0, :2]) / 2

    @staticmethod
    def _edge_exists(edge, table):
        if table.empty:
            return False
        keys = ("Channel_to", "Type_to", "Channel_from", "Type_from", "Condition")
        return any(all(str(row[k]) == str(edge[k]) for k in keys) for _, row in table.iterrows())

    def _on_pick(self, event):
        selected = self._artists.get(event.artist)
        if isinstance(selected, pd.DataFrame):
            index = getattr(event, "ind", [0])[0]
            if len(selected):
                self._selected_edge = selected.iloc[index]
        elif selected is not None:
            self._selected_edge = selected
        self._draw_detail()

    def _draw_detail(self, *_):
        row = self._selected_edge
        if isinstance(self.fit_data, ConnectivityGroupFit):
            self.detail_figure.clear()
            ax = self.detail_figure.add_subplot()
            self.detail_ax = ax
            if row is None:
                ax.text(0.5, 0.5, "No edges meet the selected threshold.",
                        ha="center", va="center", transform=ax.transAxes)
            else:
                self._draw_group_detail(ax, row)
            self.detail_canvas.draw_idle()
        else:
            self.detail_figure.clear()
            ax = self.detail_figure.add_subplot()
            self.detail_ax = ax
            if row is None:
                message = "No edges meet the selected threshold."
                ax.text(0.5, 0.5, message, ha="center", va="center",
                        transform=ax.transAxes)
                self.scatter_figure.clear()
                scatter_ax = self.scatter_figure.add_subplot()
                scatter_ax.text(0.5, 0.5, message, ha="center", va="center",
                                transform=scatter_ax.transAxes)
            else:
                self._draw_first_level_detail(ax, row)
                self._draw_first_level_scatter(row)
            self.detail_canvas.draw_idle()
            self.scatter_canvas.draw_idle()
        if row is not None and not isinstance(self.fit_data, ConnectivityGroupFit):
            self.detail_tabs.setTabText(0, "Time courses")
            self.detail_tabs.setTabText(1, "Endpoint scatter")

    def _draw_group_detail(self, ax, row):
        fit = self.fit_data
        observations = fit.observations
        mask = (
            (observations["Channel_to"].astype(str) == str(row["Channel_to"]))
            & (observations["Type_to"].astype(str) == str(row["Type_to"]))
            & (observations["Channel_from"].astype(str) == str(row["Channel_from"]))
            & (observations["Type_from"].astype(str) == str(row["Type_from"]))
        )
        edge = observations.loc[mask].copy()
        term = str(row["Condition"])
        axis_name, axis_kind = self._coefficient_axis(fit, edge, term)
        if axis_name and axis_kind == "numeric":
            points = edge.dropna(subset=[axis_name, "Beta"])
            x = points[axis_name].to_numpy(dtype=float)
            uncertainty = next(
                (name for name in ("StdErr", "ZstdErr") if name in points),
                None,
            )
            if uncertainty:
                ax.errorbar(
                    x, points["Beta"], yerr=points[uncertainty],
                    fmt="o", capsize=3, label="First-level Fisher-Z estimate",
                )
            else:
                ax.scatter(x, points["Beta"], label="First-level Fisher-Z estimate")
            for condition, condition_points in points.groupby("Condition", sort=False):
                order = np.argsort(condition_points[axis_name].to_numpy(dtype=float))
                ax.plot(
                    condition_points[axis_name].to_numpy(dtype=float)[order],
                    condition_points["FixedFit"].to_numpy(dtype=float)[order],
                    label=f"{condition}: group fit",
                )
            ax.scatter(
                x, points["ConditionalFit"], marker="x", color="red",
                label="Conditional fit",
            )
            ax.set_xlabel(axis_name)
        elif axis_name and axis_name in edge:
            categories = edge[axis_name].dropna().astype(str)
            levels = list(pd.unique(categories))
            samples = [
                edge.loc[edge[axis_name].astype(str) == level, "Beta"].dropna()
                for level in levels
            ]
            ax.boxplot(samples, labels=levels)
            for index, values in enumerate(samples, start=1):
                ax.scatter(np.full(len(values), index), values, zorder=3)
            ax.set_xlabel(axis_name)
        else:
            points = edge["Beta"].dropna()
            ax.boxplot([points], labels=[term])
            ax.scatter(np.ones(len(points)), points, zorder=3)
            ax.set_xlabel("Model term")
        ax.set_ylabel("Fisher-Z connectivity")
        ax.set_title(f"{row['Channel_to']} <-> {row['Channel_from']} | {term}")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="small")

    @staticmethod
    def _coefficient_axis(fit, observations, coefficient):
        design = patsy.dmatrix(
            fit.formula.split("~", 1)[1],
            observations,
            return_type="dataframe",
        )
        if coefficient not in design.columns:
            return None, None
        column_index = design.columns.get_loc(coefficient)
        for model_term in design.design_info.terms:
            term_slice = design.design_info.term_name_slices[model_term.name()]
            if not term_slice.start <= column_index < term_slice.stop:
                continue
            variables = []
            for factor in model_term.factors:
                for name in re.findall(r"\b[A-Za-z_]\w*\b", factor.name()):
                    if name in observations and name not in variables:
                        variables.append(name)
            numeric = [
                name for name in variables
                if pd.api.types.is_numeric_dtype(observations[name])
                and not pd.api.types.is_bool_dtype(observations[name])
            ]
            if numeric:
                return numeric[0], "numeric"
            if variables:
                return variables[0], "categorical"
        return None, None

    def _endpoint_timecourses(self, row):
        if self.data is None:
            return [], "Pass the source DataSet to view endpoint time courses."
        recordings = list(getattr(self.data, "dataset", []))
        if not recordings and hasattr(self.data, "timeseries"):
            recordings = [self.data]
        if not recordings:
            return [], "No source recordings are available."
        demographics = self.data.get_demographics() if hasattr(self.data, "get_demographics") else None
        endpoint_names = (str(row.get("Subject_to", "")), str(row.get("Subject_from", "")))
        selected = []
        for endpoint in endpoint_names:
            if endpoint and demographics is not None and "subject" in demographics:
                matches = np.flatnonzero(demographics["subject"].astype(str).to_numpy() == endpoint)
                selected.append(recordings[int(matches[0])] if len(matches) else recordings[0])
            else:
                selected.append(recordings[0])
        series_data = []
        for side, (recording, channel, signal_type) in enumerate(zip(
            selected, (row["Channel_to"], row["Channel_from"]),
            (row["Type_to"], row["Type_from"]),
        )):
            if self.input_name not in recording.timeseries:
                continue
            data = recording[self.input_name]
            signal_dim = "chromo" if "chromo" in data.dims else "wavelength"
            series = data.sel(channel=str(channel), **{signal_dim: str(signal_type)})
            plain_series = series.pint.dequantify() if hasattr(series, "pint") else series
            time_coord = plain_series.time
            if hasattr(time_coord, "pint"):
                time_coord = time_coord.pint.dequantify()
            time = np.asarray(time_coord.values, dtype=float)
            values = np.asarray(plain_series.values, dtype=float).squeeze()
            label = f"{endpoint_names[side] + ' ' if endpoint_names[side] else ''}{channel} {signal_type}"
            series_data.append((time, values, label))
        if len(series_data) != 2:
            return series_data, (
                f"Expected both endpoint series under {self.input_name!r}; "
                f"found {len(series_data)}."
            )
        return series_data, None

    def _draw_first_level_detail(self, ax, row):
        courses, message = self._endpoint_timecourses(row)
        if message:
            ax.text(0.5, 0.5, message, ha="center", va="center",
                    transform=ax.transAxes, wrap=True)
            return
        for time, values, label in courses:
            ax.plot(time, values, label=label)
        ax.set_xlabel("Time")
        ax.set_ylabel(f"Signal ({self.input_name})")
        ax.set_title(f"Compared endpoint time courses | r={row['Coeff']:.3g}")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="small")

    def _draw_first_level_scatter(self, row):
        self.scatter_figure.clear()
        ax = self.scatter_figure.add_subplot()
        self.scatter_ax = ax
        courses, message = self._endpoint_timecourses(row)
        if message:
            ax.text(0.5, 0.5, message, ha="center", va="center",
                    transform=ax.transAxes, wrap=True)
            return
        (time_to, values_to, label_to), (time_from, values_from, label_from) = courses
        overlap = (
            (time_to >= np.nanmin(time_from))
            & (time_to <= np.nanmax(time_from))
        )
        x = values_to[overlap]
        y = np.interp(time_to[overlap], time_from, values_from)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 2:
            ax.text(0.5, 0.5, "Not enough overlapping samples for an endpoint scatter plot.",
                    ha="center", va="center", transform=ax.transAxes, wrap=True)
            return
        ax.scatter(x, y, s=12, alpha=0.55)
        if np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            fit_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(fit_x, slope * fit_x + intercept, color="C3", label="Linear fit")
            ax.legend(fontsize="small")
        ax.set_xlabel(label_to)
        ax.set_ylabel(label_from)
        ax.set_title(f"Endpoint scatter | r={row['Coeff']:.3g} | n={len(x)}")


_OPEN_WINDOWS = []


def view_connectivity(
    result,
    data=None,
    *,
    result_name: str = "conn",
    input_name: str = "conc",
    block: bool = False,
    plot_on_scalp: bool = True,
) -> ConnectivityViewer:
    """Open a Connectivity result; provide its source DataSet for time-course plots.

    ``result`` may be a Connectivity object or a DataSet. Group-level results
    are resolved from ``DataSet.statistics``; first-level results are resolved
    from the first recording's timeseries. For first-level results, that same
    dataset supplies the source time courses.
    """
    if isinstance(result, Connectivity):
        connectivity = result
    else:
        data = result if data is None else data
        statistics = getattr(result, "statistics", {})
        if result_name in statistics:
            connectivity = statistics[result_name]
        else:
            recordings = getattr(result, "dataset", [])
            if not recordings or result_name not in recordings[0].timeseries:
                available_group = list(statistics)
                available_recording = (
                    list(recordings[0].timeseries) if recordings else []
                )
                raise KeyError(
                    f"Connectivity result {result_name!r} was not found. "
                    f"Dataset-level results: {available_group}; "
                    f"first-recording timeseries: {available_recording}."
                )
            connectivity = recordings[0][result_name]
    app = QApplication.instance() or QApplication(sys.argv or ["connectivity_viewer"])
    try:
        from IPython import get_ipython
    except ImportError:
        shell = None
    else:
        shell = get_ipython()
    if not block and shell is not None and shell.active_eventloop != "qt":
        shell.enable_gui("qt")
    window = ConnectivityViewer(
        connectivity, data=data, input_name=input_name, plot_on_scalp=plot_on_scalp,
    )
    window.setAttribute(Qt.WA_DeleteOnClose, True)
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(lambda *_: _OPEN_WINDOWS.remove(window))
    window.show()
    if block:
        app.exec()
    return window
