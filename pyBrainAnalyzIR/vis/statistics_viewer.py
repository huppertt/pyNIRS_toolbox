"""Interactive statistical maps and original-space model fits."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import patsy
import xarray as xr
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QActionGroup, QBrush, QColor, QCursor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QMenu, QMessageBox, QScrollArea, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)
from scipy.stats import t

from pyBrainAnalyzIR.dataclasses.model_fit import FirstLevelFit, GroupModelFit
from pyBrainAnalyzIR.dataclasses.statistics import Statistics
from pyBrainAnalyzIR.vis.plot_nirs_inline import draw_probe, probe_views_available

SIGNIFICANT_ROW_COLOR = QColor(255, 236, 153)


class StatisticsViewer(QMainWindow):
    """Inspect one Statistics object without changing its estimates or contrasts."""

    probeViewChanged = Signal(bool)

    def __init__(self, statistics: Statistics, parent=None, plot_on_scalp: bool = True):
        super().__init__(parent)
        self.map_ax = None
        self.fit_ax = None
        if not isinstance(statistics, Statistics):
            raise TypeError("StatisticsViewer requires a single Statistics object.")
        if not isinstance(getattr(statistics, "betas", None), xr.DataArray) or not statistics.betas.size:
            raise ValueError("The Statistics object has no estimates to display.")
        self.statistics = statistics
        self.fit_data = getattr(statistics, "fit_data", None)
        self._channel_artists = {}
        self.plot_on_scalp = bool(plot_on_scalp)
        self._table_frame = pd.DataFrame()
        self.setWindowTitle("Statistics Viewer")
        self.resize(1250, 700)

        central = QWidget()
        layout = QVBoxLayout(central)
        controls = QHBoxLayout()
        self.condition_combo = QComboBox()
        self.condition_combo.addItems([str(c) for c in statistics.list_conditions()])
        self.type_combo = QComboBox()
        self.type_combo.addItems([str(c) for c in np.unique(statistics.betas.type)])
        self.threshold = QDoubleSpinBox()
        self.threshold.setDecimals(6)
        self.threshold.setRange(0, 1)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.05)
        self.threshold_type = QComboBox()
        self.threshold_type.addItems(["q-value", "p-value"])
        self.channel_combo = QComboBox()
        for label, widget in (
            ("Condition / model term", self.condition_combo), ("Signal", self.type_combo),
            ("Threshold", self.threshold), ("Threshold type", self.threshold_type),
            ("Channel", self.channel_combo),
        ):
            control_label = QLabel(label)
            control_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            controls.addWidget(control_label)
            controls.addWidget(widget)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)
        self.map_figure = Figure(layout="constrained")
        self.fit_figure = Figure(layout="constrained")
        self.map_canvas = FigureCanvasQTAgg(self.map_figure)
        self.fit_canvas = FigureCanvasQTAgg(self.fit_figure)
        map_panel = QWidget()
        QVBoxLayout(map_panel).addWidget(self.map_canvas)
        splitter.addWidget(map_panel)
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._show_table_menu)
        self.results_table.itemSelectionChanged.connect(self._on_table_selection)
        self.results_tabs = QTabWidget()
        self.results_tabs.addTab(self.fit_canvas, "Results Plots")
        self.results_tabs.addTab(self.results_table, "Results Table")
        splitter.addWidget(self.results_tabs)
        self.plot_splitter = splitter

        self.info_splitter = QSplitter(Qt.Horizontal)
        text_panel = QWidget()
        text_layout = QVBoxLayout(text_panel)
        self.fit_summary = QLabel()
        self.fit_text = QLabel()
        for label in (self.fit_summary, self.fit_text):
            label.setWordWrap(True)
            label.setTextFormat(Qt.PlainText)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            text_layout.addWidget(label)
        text_layout.addStretch(1)
        text_scroll = QScrollArea()
        text_scroll.setWidgetResizable(True)
        text_scroll.setFrameShape(QFrame.NoFrame)
        text_scroll.setWidget(text_panel)
        self.info_splitter.addWidget(text_scroll)
        self.legend_figure = Figure()
        self.legend_canvas = FigureCanvasQTAgg(self.legend_figure)
        self.legend_canvas.setMinimumHeight(90)
        self.fit_legend = None
        self.info_splitter.addWidget(self.legend_canvas)
        for horizontal in (splitter, self.info_splitter):
            horizontal.setStretchFactor(0, 1)
            horizontal.setStretchFactor(1, 1)
        splitter.splitterMoved.connect(lambda *_: self.info_splitter.setSizes(splitter.sizes()))
        self.info_splitter.splitterMoved.connect(lambda *_: splitter.setSizes(self.info_splitter.sizes()))

        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.addWidget(splitter)
        self.main_splitter.addWidget(self.info_splitter)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([480, 170])
        layout.addWidget(self.main_splitter, stretch=1)
        self.setCentralWidget(central)

        self.map_canvas.mpl_connect("pick_event", self._on_pick)
        self.map_canvas.mpl_connect("button_press_event", self._on_map_click)
        self.condition_combo.currentTextChanged.connect(self._refresh)
        self.type_combo.currentTextChanged.connect(self._refresh)
        self.threshold.valueChanged.connect(self._refresh)
        self.threshold_type.currentTextChanged.connect(self._refresh)
        self.channel_combo.currentTextChanged.connect(self._select_channel)
        self.results_tabs.currentChanged.connect(self._on_results_tab_changed)
        self._refresh()

    def _on_results_tab_changed(self, index):
        self.legend_canvas.setVisible(self.results_tabs.widget(index) is self.fit_canvas)

    def showEvent(self, event):
        super().showEvent(event)
        self.info_splitter.setSizes(self.plot_splitter.sizes())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.info_splitter.setSizes(self.plot_splitter.sizes())

    def _refresh(self, *_, preferred_channel=None):
        table = self.statistics.table()
        self._table = table[
            (table["Condition"] == self.condition_combo.currentText())
            & (table["Type"] == self.type_combo.currentText())
        ]
        previous = preferred_channel or self.channel_combo.currentText()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(self._table["Channel"].astype(str).tolist())
        if previous in self._table["Channel"].values:
            self.channel_combo.setCurrentText(previous)
        self.channel_combo.blockSignals(False)
        self._table_frame = self._condition_results(table)
        self._populate_table()
        self._draw_map()
        self._select_channel()

    def _threshold_column(self):
        return "Q-values" if self.threshold_type.currentIndex() == 0 else "P-values"

    def _condition_results(self, table):
        frame = table[table["Condition"] == self.condition_combo.currentText()].reset_index(drop=True)
        frame["Significant"] = (frame[self._threshold_column()] < self.threshold.value()).to_numpy(dtype=bool)
        return frame

    def results_frame(self) -> pd.DataFrame:
        """Rows shown in the Results Table: every signal type for the selected condition."""
        return self._table_frame.copy()

    def _populate_table(self):
        frame = self._table_frame
        table = self.results_table
        table.blockSignals(True)
        table.clearContents()
        table.setRowCount(len(frame))
        table.setColumnCount(len(frame.columns))
        table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
        bold = QFont()
        bold.setBold(True)
        for row, record in enumerate(frame.itertuples(index=False)):
            significant = bool(record[-1])
            for column, value in enumerate(record):
                item = QTableWidgetItem(_display_value(value))
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if significant:
                    item.setBackground(QBrush(SIGNIFICANT_ROW_COLOR))
                    item.setFont(bold)
                table.setItem(row, column, item)
        table.resizeColumnsToContents()
        table.blockSignals(False)

    def _sync_table_selection(self):
        frame = self._table_frame
        matches = np.flatnonzero(
            (frame["Channel"].astype(str) == self.channel_combo.currentText()).to_numpy()
            & (frame["Type"].astype(str) == self.type_combo.currentText()).to_numpy()
        ) if len(frame) else []
        self.results_table.blockSignals(True)
        self.results_table.clearSelection()
        if len(matches):
            self.results_table.selectRow(int(matches[0]))
            self.results_table.scrollToItem(self.results_table.item(int(matches[0]), 0))
        self.results_table.blockSignals(False)

    def _on_table_selection(self):
        rows = self.results_table.selectionModel().selectedRows()
        if not rows:
            return
        record = self._table_frame.iloc[rows[0].row()]
        channel, signal_type = str(record["Channel"]), str(record["Type"])
        if signal_type != self.type_combo.currentText():
            self.type_combo.blockSignals(True)
            self.type_combo.setCurrentText(signal_type)
            self.type_combo.blockSignals(False)
            self._refresh(preferred_channel=channel)
        elif channel != self.channel_combo.currentText():
            self.channel_combo.setCurrentText(channel)

    def _build_table_menu(self) -> QMenu:
        menu = QMenu(self.results_table)
        menu.addAction("Copy to Clipboard").triggered.connect(self.copy_table_to_clipboard)
        menu.addAction("Save to Excel...").triggered.connect(lambda *_: self.save_table_to_excel())
        return menu

    def _show_table_menu(self, position):
        self._build_table_menu().exec(self.results_table.viewport().mapToGlobal(position))

    def table_clipboard_text(self) -> str:
        """Tab-delimited rows with a header, as nirs.util.copytable2clip pastes into Excel."""
        frame = self._table_frame
        lines = ["\t".join(str(column) for column in frame.columns)]
        lines += ["\t".join(_clipboard_value(value) for value in record)
                  for record in frame.itertuples(index=False)]
        return "\n".join(lines) + "\n"

    def copy_table_to_clipboard(self):
        QApplication.clipboard().setText(self.table_clipboard_text())
        self.statusBar().showMessage(f"Copied {len(self._table_frame)} rows to the clipboard.", 5000)

    def save_table_to_excel(self, path=None):
        """Save the Results Table; prompts for a file name when ``path`` is not given."""
        if path is None:
            default = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.condition_combo.currentText())
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Results Table", f"{default or 'statistics'}.xlsx", "Excel Workbook (*.xlsx)",
            )
            if not path:
                return None
        path = str(path)
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            self._table_frame.to_excel(path, index=False, sheet_name="Statistics", engine="openpyxl")
        except (OSError, ValueError, ImportError) as exc:
            QMessageBox.critical(self, "Save Results Table", f"Could not save the table:\n{exc}")
            return None
        self.statusBar().showMessage(f"Saved results table to {path}", 5000)
        return path

    def _draw_map(self):
        self.map_figure.clear()
        self.map_ax = self.map_figure.add_subplot()
        ax = self.map_ax
        self._channel_artists.clear()
        table = self._table
        values = table["T-value"].to_numpy()
        finite = np.abs(values[np.isfinite(values)])
        maximum = max(float(finite.max()), 1e-12) if finite.size else 1.0
        norm = Normalize(-maximum, maximum)
        cmap = colormaps["RdBu_r"]
        column = self._threshold_column()
        significant = table[column].to_numpy() < self.threshold.value()

        geo2d = getattr(self.statistics, "geo2d", None)
        geo3d = getattr(self.statistics, "geo3d", None)
        has_geometry = any(geo is not None and len(geo) for geo in (geo2d, geo3d))
        if has_geometry and getattr(self.statistics, "geoROI", None) is None:
            channels = table["Channel"].to_numpy()
            data = xr.DataArray(np.zeros(len(channels)), dims="channel", coords={"channel": channels})
            selected = self.statistics.betas.where(
                (self.statistics.betas.conditions == self.condition_combo.currentText())
                & (self.statistics.betas.type == self.type_combo.currentText()), drop=True
            )
            for coord in ("source", "detector"):
                if coord in selected.coords:
                    data = data.assign_coords({coord: ("channel", selected[coord].values)})
            try:
                artists = draw_probe(self.statistics, data, ax, plot_on_scalp=self.plot_on_scalp)
            except ValueError as exc:
                ax.clear()
                ax.text(0.5, 0.5, f"Cannot draw probe: {exc}", ha="center", wrap=True)
                self.statusBar().showMessage(str(exc))
                artists = []
        else:
            artists = [
                ax.plot(index, value, "o", markersize=9)[0]
                for index, value in enumerate(values)
            ]
            ax.set_xticks(np.arange(len(table)), table["Channel"], rotation=60)
            ax.set_ylabel("T-statistic")
            ax.set_xlabel("ROI" if getattr(self.statistics, "geoROI", None) else "Channel (no probe geometry)")
        for artist, (_, row), significant_value in zip(artists, table.iterrows(), significant):
            value = row["T-value"]
            artist.set_color(cmap(norm(value)) if significant_value and np.isfinite(value) else "0.75")
            artist.set_picker(6)
            self._channel_artists[artist] = str(row["Channel"])
        ax.set_title(f"{self.condition_combo.currentText()} | {self.type_combo.currentText()}")
        self.map_figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="T-statistic")
        self.map_canvas.draw_idle()

    def _on_pick(self, event):
        mouse = getattr(event, "mouseevent", None)
        if mouse is not None and mouse.button not in (1, None):
            return
        channel = self._channel_artists.get(event.artist)
        if channel is not None:
            self.channel_combo.setCurrentText(channel)

    def _on_map_click(self, event):
        if event.button != 3 or event.inaxes is not self.map_ax:
            return
        if any(artist.contains(event)[0] for artist in self._channel_artists):
            return
        self._show_probe_view_menu()

    def _probe_views_available(self) -> bool:
        if getattr(self.statistics, "geoROI", None) is not None:
            return False
        try:
            return probe_views_available(self.statistics)
        except (ValueError, KeyError, AttributeError):
            return False

    def _build_probe_view_menu(self) -> QMenu:
        menu = QMenu(self)
        available = self._probe_views_available()
        group = QActionGroup(menu)
        group.setExclusive(True)
        for text, on_scalp in (("10-20 View", True), ("2D Probe View", False)):
            action = menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(self.plot_on_scalp == on_scalp)
            action.setEnabled(available)
            group.addAction(action)
            action.triggered.connect(lambda _checked=False, value=on_scalp: self._set_probe_view(value))
        if not available:
            menu.addSeparator()
            note = menu.addAction("Requires both a 2D probe and a 3D probe with head landmarks")
            note.setEnabled(False)
        return menu

    def _show_probe_view_menu(self):
        self._build_probe_view_menu().exec(QCursor.pos())

    def _set_probe_view(self, plot_on_scalp: bool):
        plot_on_scalp = bool(plot_on_scalp)
        if plot_on_scalp == self.plot_on_scalp:
            return
        self.plot_on_scalp = plot_on_scalp
        self._draw_map()
        self._select_channel()
        self.probeViewChanged.emit(plot_on_scalp)
        self.statusBar().showMessage("Probe view: " + ("10-20" if plot_on_scalp else "2D probe"), 5000)

    def _select_channel(self, *_):
        channel = self.channel_combo.currentText()
        for artist, label in self._channel_artists.items():
            artist.set_linewidth(4 if label == channel else 2)
            artist.set_markeredgecolor("black" if label == channel else artist.get_color())
        self.map_canvas.draw_idle()
        self._sync_table_selection()
        self._draw_fit()

    def _draw_fit(self, *_):
        self.fit_summary.clear()
        self._show_legend([])
        self.fit_figure.clear()
        self.fit_ax = self.fit_figure.add_subplot()
        channel = self.channel_combo.currentText()
        signal_type = self.type_combo.currentText()
        self.fit_text.setText("")
        try:
            if isinstance(self.fit_data, FirstLevelFit):
                self._draw_first_level(channel, signal_type)
            elif isinstance(self.fit_data, GroupModelFit):
                self._draw_group(channel, signal_type)
            else:
                self.fit_ax.text(
                    0.5, 0.5, "Fit data were not retained.\nRerun the GLM or MixedEffects model.",
                    ha="center", va="center", transform=self.fit_ax.transAxes,
                )
                self.fit_text.setText("Statistical maps remain available for older results.")
        except (ValueError, KeyError, patsy.PatsyError) as exc:
            self.fit_summary.clear()
            self._show_legend([])
            self.fit_ax.clear()
            self.fit_ax.text(0.5, 0.5, f"Fit unavailable: {exc}", ha="center",
                             va="center", wrap=True, transform=self.fit_ax.transAxes)
            self.fit_text.setText(f"Fit unavailable: {exc}")
        self.fit_canvas.draw_idle()

    def _show_legend(self, handles):
        """Draw detail-plot legend entries outside the data axes."""
        self.legend_figure.clear()
        self.fit_legend = None
        if handles:
            self.fit_legend = self.legend_figure.legend(
                handles=handles, labels=[handle.get_label() for handle in handles],
                loc="center", frameon=False, fontsize="small",
            )
        self.legend_canvas.draw_idle()

    def _draw_first_level(self, channel, signal_type):
        data = self.fit_data.data.sel(channel=channel, chromo=signal_type)
        valid = data.valid.values
        time = data.time.values
        ax = self.fit_ax
        ax.plot(time, data.observed.where(data.valid), label="Observed")
        ax.plot(time, data.fitted.where(data.valid), label="Full model fit")
        ax.set_xlabel(f"Time ({data.time.attrs.get('units', 's')})")
        ax.set_ylabel(f"Signal ({self.fit_data.units})" if self.fit_data.units else "Signal")
        ax.set_title(f"{channel} | {signal_type}: original-space time course")
        self._show_legend(ax.get_legend_handles_labels()[0])

        def display(name):
            value = float(data[name])
            return f"{value:.4g}" if np.isfinite(value) else "unavailable"

        self.fit_text.setText(
            f"Full original model, including nuisance regressors; {valid.sum()} valid samples. "
            f"R^2 (original-space, descriptive) = {display('r_squared')}; "
            f"Wald F({display('df_num')}, {display('df_den')}) = {display('f_statistic')}; "
            f"p = {display('f_pvalue')}. Masked samples are excluded."
        )

    def _draw_group(self, channel, signal_type):
        term = self.condition_combo.currentText()
        points, curve, covariate = self.fit_data.partial_effect(
            channel, signal_type, term,
        )
        ax = self.fit_ax
        numeric = (
            covariate is not None
            and pd.api.types.is_numeric_dtype(points[covariate])
            and not pd.api.types.is_bool_dtype(points[covariate])
        )
        if numeric:
            x = points[covariate].to_numpy()
            ax.errorbar(x, points["Beta"], yerr=points["StdErr"], fmt="o",
                        capsize=3, label="Partial beta +/- 1 SE", zorder=5)
            self._draw_weight_errors(points, x)
            curve_x = curve[covariate].to_numpy()
            fitted = curve["Fit"].to_numpy()
            se = curve["FitSE"].to_numpy()
            ax.plot(curve_x, fitted, color="C3", label="Population fit")
            if self.statistics.dof > 0:
                critical = t.ppf(0.975, self.statistics.dof)
                total_width = critical * curve["TotalFitSE"].to_numpy()
                ax.fill_between(
                    curve_x, fitted - total_width, fitted + total_width,
                    color="C1", alpha=0.15, label="95% full fixed-effects mean CI (translated)",
                )
                ax.plot(curve_x, fitted - total_width, "--", color="C1", linewidth=1)
                ax.plot(curve_x, fitted + total_width, "--", color="C1", linewidth=1)
                width = critical * se
                ax.fill_between(curve_x, fitted - width, fitted + width,
                                color="C3", alpha=0.15, label="95% population mean CI")
            ax.plot(x, points["ConditionalFit"], "x", color="red", markersize=8,
                    label="Individual fit (fixed + random)", zorder=5)
        else:
            self._draw_categorical_group(points, curve, covariate, term)
        ax.set_xlabel(covariate or "Condition intercept")
        ax.set_ylabel("Partial first-level beta (original units)")
        ax.set_title(f"{channel} | {signal_type} | {term}")
        if numeric:
            self._show_legend(ax.get_legend_handles_labels()[0])
        selected = self._table[self._table["Channel"] == channel].iloc[0]
        uncertainty = (
            "Subject points: betas +/- 1 first-level SE (horizontal jitter only). "
            "Red whiskers: selected-term 95% mean CI; orange whiskers: full fixed-effects 95% mean CI."
            if not numeric else "Error bars: original first-level +/- 1 SE. "
            "Red band: selected-term 95% mean CI; orange dashed band: full fixed-effects 95% mean CI."
        )
        if "WeightStdErr" in points:
            unbounded = int(np.isinf(points["WeightStdErr"]).sum())
            weighting = self.fit_data.observations.attrs.get("observation_weighting", "unknown")
            weight_text = (
                "Purple bars: +/- 1 weight-implied SD in beta units, from the final robust multipliers "
                "and inverse whitening; not a calibrated confidence or prediction interval. "
                "They omit residual-scale, random-effect and adjustment uncertainty. "
                "Correlated observations cannot be summarized fully by marginal bars. "
                + ("Unweighted fit: the pre-robust reference is unit variance, not first-level covariance. "
                   if weighting == "unweighted" else "")
                + (f"{unbounded} points have unbounded weight-implied uncertainty (no finite purple bar). "
                   if unbounded else "")
            )
        else:
            weight_text = "Final observation weights were not retained; rerun MixedEffects for purple error bars. "
        self.fit_text.setText(
            f"{self.fit_data.formula}\n"
            "All other fitted fixed-effect contributions are subtracted, including the condition intercept "
            "when displaying a slope. Centering uses the original training means. Single-variable axes use "
            "original units; multivariable interactions use design values. "
            "Residuals and random effects are retained. SE bars exclude uncertainty in the fitted subtraction. "
            f"{uncertainty} {weight_text} Red X's: partial individual fits (fixed + random). "
            f"{len(points)} observations (one per subject when there is one recording per subject). "
            f"Selected term: beta={selected['Beta']:.4g}, t={selected['T-value']:.4g}, "
            f"p={selected['P-values']:.4g}, q={selected['Q-values']:.4g}. "
            "The full-model interval includes all coefficient covariances and is translated onto the partial-fit "
            "line using a fixed fitted offset, not a propagated partial-residual interval. Other covariates are "
            "held at mean/modal values; multivariable interactions use observed predictor combinations. "
            "Neither interval includes subject/residual variation or is expected to contain 95% of subjects. "
            "Covariance can make the full-model interval narrower at some x values."
        )
        if numeric:
            self.fit_summary.setText(
                "Points: partial subject observations, not independent equal-weight inputs to this line. "
                "Red line: selected fixed effect from the complete mixed model. "
                "Point spread and confidence in the fitted mean are different quantities."
            )

    def _draw_weight_errors(self, points, x):
        if "WeightStdErr" not in points:
            return None
        errors = points["WeightStdErr"].to_numpy(dtype=float)
        if np.isnan(errors).any() or np.any(errors < 0):
            raise ValueError("Invalid retained weight-implied observation uncertainty.")
        return self.fit_ax.errorbar(
            x, points["Beta"].to_numpy(dtype=float),
            yerr=np.where(np.isfinite(errors), errors, np.nan),
            fmt="none", capsize=6, elinewidth=3, color="C4", alpha=0.55,
            label="Final weighting +/- 1 implied SD", zorder=4,
        )

    def _draw_categorical_group(self, points, curve, covariate, term):
        ax = self.fit_ax
        values = points[covariate] if covariate is not None else pd.Series(term, index=points.index)
        if values.isna().any():
            raise ValueError(f"Missing category labels for {covariate}.")
        if isinstance(values.dtype, pd.CategoricalDtype):
            levels = [level for level in values.cat.categories if (values == level).any()]
        else:
            levels = list(pd.unique(values))
        positions = np.arange(len(levels))

        if covariate is not None:
            curve = curve.set_index(covariate).reindex(levels)
        box = ax.boxplot(
            [points.loc[values == level, "Beta"].to_numpy(dtype=float) for level in levels],
            positions=positions, widths=0.5, patch_artist=True, manage_ticks=False,
            boxprops={"facecolor": "0.5", "alpha": 0.15}, medianprops={"color": "0.3"},
            showfliers=False,
        )
        width = None
        label = "Population fit"
        if self.statistics.dof > 0:
            width = t.ppf(0.975, self.statistics.dof) * curve["FitSE"].to_numpy()
            label += " (95% mean CI)"
        fit = ax.errorbar(
            positions, curve["Fit"].to_numpy(), yerr=width,
            fmt="D", markersize=8, capsize=10, elinewidth=3, color="C3", label=label,
            zorder=4,
        )
        handles = []
        if self.statistics.dof > 0:
            total_width = t.ppf(0.975, self.statistics.dof) * curve["TotalFitSE"].to_numpy()
            total = ax.errorbar(
                positions, curve["Fit"].to_numpy(), yerr=total_width,
                fmt="none", capsize=14, elinewidth=6, color="C1", alpha=0.5,
                label="95% full fixed-effects mean CI (translated)", zorder=3,
            )
            handles.append(total)
        handles.append(fit)
        box["boxes"][0].set_label("Unweighted data: median and middle 50%")
        handles.append(box["boxes"][0])
        x = np.empty(len(points), dtype=float)
        for position, level in enumerate(levels):
            selected = (values == level).to_numpy()
            count = int(selected.sum())
            offsets = np.linspace(-0.15, 0.15, count) if count > 1 else np.zeros(count)
            x[selected] = position + offsets
        observed = ax.errorbar(
            x, points["Beta"].to_numpy(dtype=float), yerr=points["StdErr"].to_numpy(dtype=float),
            fmt="o", markersize=5, capsize=3, color="C0", alpha=0.8,
            label="Partial beta +/- 1 SE", zorder=5,
        )
        handles.append(observed)
        weighted = self._draw_weight_errors(points, x)
        if weighted is not None:
            handles.append(weighted)
        individual, = ax.plot(
            x, points["ConditionalFit"].to_numpy(dtype=float), "x", color="red",
            markersize=8, label="Individual fit (fixed + random)", zorder=6,
        )
        handles.append(individual)
        ax.set_xticks(positions, [str(level) for level in levels])
        ax.axhline(0, color="0.6", linewidth=0.8, zorder=0)
        self._show_legend(handles)
        summaries = []
        sign_mismatch = False
        for level, fitted in zip(levels, curve["Fit"]):
            median = float(points.loc[values == level, "Beta"].median())
            summaries.append(f"{level}: unweighted median={median:.3g}, fitted mean={fitted:.3g}, "
                             f"fit-minus-median={fitted - median:.3g}")
            sign_mismatch |= median * fitted < 0
        mode = self.fit_data.observations.attrs.get("observation_weighting", "unknown")
        robust = self.statistics.covariance.attrs.get("robust", "unknown")
        self.fit_summary.setText(
            "; ".join(summaries) + f". Weighting: {mode}; robust: {robust}. "
            "Gray box/median describe unweighted subject spread, not model uncertainty. "
            + ("SIGN DIFFERENCE: inspect covariance/robust sensitivity before interpreting this estimate. "
               if sign_mismatch else "")
            + "A discrepancy alone neither validates nor invalidates the fit; marginal error bars do not show "
            "the cross-observation influence used by the full model."
        )


_OPEN_WINDOWS = []


def _display_value(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    return str(value)


def _clipboard_value(value) -> str:
    """Full-precision cell text; tabs/newlines would split Excel cells."""
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (float, np.floating)):
        return repr(float(value)) if np.isfinite(value) else ("" if np.isnan(value) else str(float(value)))
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def view_statistics(statistics: Statistics, block: bool = False, plot_on_scalp: bool = True) -> StatisticsViewer:
    """Open a model viewer; retain the returned window for notebook interaction.

    Use ``block=True`` in standalone scripts. Results made before fit retention
    was introduced can still show maps, but must be refit for the detail panel.
    ``plot_on_scalp`` starts in the 10-20 view when geo3d head landmarks exist;
    right-click empty probe space to switch between the 10-20 and 2D views.
    """
    app = QApplication.instance() or QApplication(sys.argv or ["statistics_viewer"])
    try:
        from IPython import get_ipython
    except ImportError:
        shell = None
    else:
        shell = get_ipython()
    if not block and shell is not None and shell.active_eventloop != "qt":
        shell.enable_gui("qt")
    window = StatisticsViewer(statistics, plot_on_scalp=plot_on_scalp)
    window.setAttribute(Qt.WA_DeleteOnClose, True)
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(lambda *_: _OPEN_WINDOWS.remove(window))
    window.show()
    if block:
        app.exec()
    return window
