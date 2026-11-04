"""Headless tests of statistics maps, selection, and model-fit panels."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.backend_bases import MouseEvent

pytest.importorskip("cedalion")
pytest.importorskip("PySide6.QtWidgets")

from tests.test_model_fit_data import make_dataset, small_model  # noqa: E402
from pyBrainAnalyzIR.dataclasses.statistics import Statistics  # noqa: E402
from pyBrainAnalyzIR.math.glm.solve import fit  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.mixedeffects import MixedEffects  # noqa: E402
from pyBrainAnalyzIR.vis.statistics_viewer import StatisticsViewer  # noqa: E402

pytestmark = [
    pytest.mark.requires_cedalion, pytest.mark.requires_qt,
    pytest.mark.usefixtures("qt_app"),
]


@pytest.fixture
def small_dataset():
    return make_dataset()


def _geometry(stats):
    stats.geo2d = xr.DataArray(
        [[0., 0.], [1., 1.], [1., 0.], [0., 1.]], dims=("label", "pos"),
        coords={"label": ["S1", "D1", "S2", "D2"]},
    )
    return stats


def _first_level():
    ts, design = small_model()
    return _geometry(Statistics(fit(ts, design, verbose=False)))


def test_channel_pick_updates_original_timecourse_and_diagnostics(qt_app):
    stats = _first_level()
    before = stats.copy()
    window = StatisticsViewer(stats)
    window.show()
    qt_app.processEvents()
    artist = next(a for a, channel in window._channel_artists.items() if channel == "S2D2")
    window.map_canvas.draw()
    midpoint = [np.mean(artist.get_xdata()), np.mean(artist.get_ydata())]
    px, py = window.map_ax.transData.transform(midpoint)
    artist.pick(MouseEvent("button_press_event", window.map_canvas, px, py, button=1))
    assert window.channel_combo.currentText() == "S2D2"
    local = stats.fit_data.data.sel(channel="S2D2", chromo="HbO")
    np.testing.assert_allclose(window.fit_ax.lines[0].get_ydata(), local.observed)
    np.testing.assert_allclose(window.fit_ax.lines[1].get_ydata(), local.fitted)
    assert "R^2" in window.fit_text.text()
    assert "Wald F(" in window.fit_text.text()
    assert "p =" in window.fit_text.text()
    assert window.fit_ax.get_legend() is None
    assert [item.get_text() for item in window.fit_legend.get_texts()] == ["Observed", "Full model fit"]
    window.type_combo.setCurrentText("HbR")
    window.condition_combo.setCurrentText("HRF B")
    assert window.channel_combo.currentText() == "S2D2"
    expected = stats.fit_data.data.fitted.sel(channel="S2D2", chromo="HbR")
    np.testing.assert_allclose(window.fit_ax.lines[1].get_ydata(), expected)
    xr.testing.assert_identical(stats.betas, before.betas)
    xr.testing.assert_identical(stats.covariance, before.covariance)
    window.close()


def test_threshold_p_q_controls_and_nonsignificant_channels_are_pickable():
    stats = _geometry(Statistics())
    stats.betas = xr.DataArray(
        [2.2, 0.2], dims="indices",
        coords={
            "channels": ("indices", ["S1D1", "S2D2"]),
            "type": ("indices", ["HbO", "HbO"]),
            "conditions": ("indices", ["A", "A"]),
        },
    )
    stats.covariance = xr.DataArray(np.eye(2), dims=("rows", "cols"))
    stats.dof = 100
    window = StatisticsViewer(stats)
    assert all(artist.get_color() == "0.75" for artist in window._channel_artists)
    window.threshold_type.setCurrentText("p-value")
    artist = next(a for a, channel in window._channel_artists.items() if channel == "S1D1")
    assert artist.get_color() != "0.75"
    window.threshold.setValue(0)
    assert all(artist.get_color() == "0.75" for artist in window._channel_artists)
    artist = next(a for a, channel in window._channel_artists.items() if channel == "S2D2")
    assert artist.get_picker()
    window._on_pick(SimpleNamespace(artist=artist))
    assert window.channel_combo.currentText() == "S2D2"
    assert "Rerun the GLM" in window.fit_ax.texts[0].get_text()
    window.close()


def test_group_scatter_subject_errors_and_overlaid_predictions(small_dataset):
    job = MixedEffects()
    job.options["FE_formula"] = "Beta ~ 0 + Condition + Condition:age"
    job.options["robust"] = False
    stats = _geometry(job.run(small_dataset)["groupstats"])
    window = StatisticsViewer(stats)
    window.condition_combo.setCurrentText("Condition[HRF B]:age")
    artist = next(a for a, channel in window._channel_artists.items() if channel == "S2D2")
    window._on_pick(SimpleNamespace(artist=artist))
    points, curve, axis = stats.fit_data.partial_effect("S2D2", "HbO", "Condition[HRF B]:age")
    assert axis == window.fit_ax.get_xlabel() == "age"
    observed = next(line for line in window.fit_ax.lines if len(line.get_ydata()) == 8)
    np.testing.assert_allclose(observed.get_ydata(orig=False), points.Beta)
    np.testing.assert_allclose(observed.get_xdata(orig=False), points.age)
    bars = window.fit_ax.containers[0].lines[2][0].get_segments()
    np.testing.assert_allclose([segment[1, 1] - segment[0, 1] for segment in bars], 2 * points.StdErr)
    weighted = next(container for container in window.fit_ax.containers
                    if container.get_label() == "Final weighting +/- 1 implied SD")
    np.testing.assert_allclose(
        [segment[1, 1] - segment[0, 1] for segment in weighted.lines[2][0].get_segments()],
        2 * points.WeightStdErr,
    )
    population = next(line for line in window.fit_ax.lines if line.get_label() == "Population fit")
    assert len(population.get_xdata()) == 100
    individual = next(
        line for line in window.fit_ax.lines if line.get_label() == "Individual fit (fixed + random)"
    )
    np.testing.assert_allclose(individual.get_ydata(), points.ConditionalFit)
    assert individual.get_marker() == "x"
    assert individual.get_color() == "red"
    assert population in window.fit_ax.lines
    assert window.fit_ax.lines.index(population) < window.fit_ax.lines.index(individual)
    np.testing.assert_allclose(population.get_ydata(), curve.Fit)
    from scipy.stats import t
    bounds = [line for line in window.fit_ax.lines if line.get_color() == "C1"]
    width = t.ppf(.975, stats.dof) * curve.TotalFitSE
    np.testing.assert_allclose(bounds[0].get_ydata(), curve.Fit - width)
    np.testing.assert_allclose(bounds[1].get_ydata(), curve.Fit + width)
    bands = {collection.get_label() for collection in window.fit_ax.collections}
    assert "95% population mean CI" in bands
    assert "95% full fixed-effects mean CI (translated)" in bands
    assert "Neither interval includes subject/residual variation" in window.fit_text.text()
    assert "including the condition intercept" in window.fit_text.text()
    assert window.fit_ax.get_legend() is None
    assert window.fit_legend.figure is window.legend_figure
    assert window.info_splitter.widget(1) is window.legend_canvas
    assert {
        "Partial beta +/- 1 SE", "Final weighting +/- 1 implied SD", "Population fit",
        "95% population mean CI", "95% full fixed-effects mean CI (translated)",
        "Individual fit (fixed + random)",
    } == {item.get_text() for item in window.fit_legend.get_texts()}
    window.condition_combo.setCurrentText("Condition[HRF B]")
    assert window.fit_ax.get_xlabel() == "Condition intercept"
    assert len(window.fit_ax.patches) == 1
    assert window.fit_ax.get_legend() is None
    assert "Unweighted data: median and middle 50%" in [
        item.get_text() for item in window.fit_legend.get_texts()
    ]
    window.close()


def test_missing_geometry_and_fit_data_are_explicit():
    stats = _first_level()
    del stats.geo2d
    window = StatisticsViewer(stats)
    assert "no probe geometry" in window.map_ax.get_xlabel()
    assert len(window._channel_artists) == 2
    window.close()
    with pytest.raises(TypeError, match="single Statistics"):
        StatisticsViewer([stats])
    with pytest.raises(ValueError, match="no estimates"):
        StatisticsViewer(Statistics())


def test_intercept_only_model_has_no_demographic_axes(small_dataset):
    job = MixedEffects()
    job.options["robust"] = False
    stats = job.run(small_dataset)["groupstats"]
    window = StatisticsViewer(stats)
    assert window.fit_ax.get_xlabel() == "Condition intercept"
    assert len(window.fit_ax.patches) == 1
    assert not hasattr(window, "covariate_combo")
    window.close()


def test_descriptive_box_and_population_sign_discrepancy_are_explicit(small_dataset):
    job = MixedEffects()
    job.options["robust"] = False
    stats = job.run(small_dataset)["groupstats"]
    term = "Condition[HRF A]"
    stats.fit_data.coefficients.loc[("HbO", "S1D1"), term] = -.5
    frame = stats.fit_data.observations
    mask = (frame.Channel == "S1D1") & (frame.Type == "HbO") & (frame.Condition == "HRF A")
    frame.loc[mask, "Beta"] = 3.
    frame.loc[mask, "FixedFit"] = -.5
    frame.loc[mask, "ConditionalFit"] = -.5
    window = StatisticsViewer(stats)
    window.condition_combo.setCurrentText(term)
    text = window.fit_summary.text()
    assert "unweighted median=3" in text
    assert "fitted mean=-0.5" in text
    assert "fit-minus-median=-3.5" in text
    assert "SIGN DIFFERENCE" in text
    assert "Weighting: full; robust: False" in text
    assert window.fit_ax.get_legend() is None
    assert "Unweighted data: median and middle 50%" in [
        item.get_text() for item in window.fit_legend.get_texts()
    ]
    assert stats.fit_data.coefficients.loc[("HbO", "S1D1"), term] == -.5
    window.close()


@pytest.mark.parametrize("category_kind", ["string", "ordered", "boolean", "numeric"])
def test_categorical_fit_whiskers_with_subject_errorbars(small_dataset, category_kind):
    from matplotlib.container import BarContainer
    from scipy.stats import t

    if category_kind in ("boolean", "numeric"):
        for rec in small_dataset.dataset:
            rec.meta_data["gender"] = rec.meta_data["gender"] == "F"
            if category_kind == "numeric":
                rec.meta_data["gender"] = int(rec.meta_data["gender"])
    job = MixedEffects()
    job.options["FE_formula"] = "Beta ~ 0 + Condition + Condition:age + Condition:C(gender)"
    job.options["robust"] = False
    stats = _geometry(job.run(small_dataset)["groupstats"])
    if category_kind == "ordered":
        for frame in (stats.fit_data.observations, stats.fit_data.demographics):
            frame["gender"] = pd.Categorical(frame["gender"], categories=["F", "M"], ordered=True)
    covariate = "gender"
    window = StatisticsViewer(stats)
    term = next(name for name in stats.list_conditions() if "HRF A" in name and "gender" in name)
    window.condition_combo.setCurrentText(str(term))
    points, curve, axis = stats.fit_data.partial_effect("S1D1", "HbO", str(term))
    assert axis == "gender"
    levels = ["F", "M"] if category_kind == "ordered" else list(pd.unique(points[covariate]))
    assert [tick.get_text() for tick in window.fit_ax.get_xticklabels()] == [str(level) for level in levels]
    curve = curve.set_index(covariate).reindex(levels)
    model = next(container for container in window.fit_ax.containers
                 if container.get_label() == "Population fit (95% mean CI)")
    np.testing.assert_allclose(model.lines[0].get_ydata(), curve.Fit)
    segments = model.lines[2][0].get_segments()
    expected_width = 2 * t.ppf(0.975, stats.dof) * curve.FitSE.to_numpy()
    np.testing.assert_allclose([segment[1, 1] - segment[0, 1] for segment in segments], expected_width)
    total = next(container for container in window.fit_ax.containers
                 if container.get_label() == "95% full fixed-effects mean CI (translated)")
    total_segments = total.lines[2][0].get_segments()
    np.testing.assert_allclose(
        [segment[1, 1] - segment[0, 1] for segment in total_segments],
        2 * t.ppf(.975, stats.dof) * curve.TotalFitSE.to_numpy(),
    )
    assert len(window.fit_ax.patches) == len(levels)

    def check_subjects():
        observed = next(container for container in window.fit_ax.containers
                        if container.get_label() == "Partial beta +/- 1 SE")
        np.testing.assert_allclose(observed.lines[0].get_ydata(), points.Beta)
        x = observed.lines[0].get_xdata()
        for level, position in zip(levels, range(len(levels))):
            assert np.all(np.abs(x[(points[covariate] == level).to_numpy()] - position) <= 0.151)
        segments = observed.lines[2][0].get_segments()
        np.testing.assert_allclose(
            [segment[1, 1] - segment[0, 1] for segment in segments], 2 * points.StdErr,
        )
        weighted = next(container for container in window.fit_ax.containers
                        if container.get_label() == "Final weighting +/- 1 implied SD")
        np.testing.assert_allclose(
            weighted.lines[2][0].get_segments(),
            [np.array([[xx, yy - error], [xx, yy + error]])
             for xx, yy, error in zip(x, points.Beta, points.WeightStdErr)],
        )
        return x.copy()

    original_x = check_subjects()
    assert "95% mean CI" in window.fit_text.text()
    assert not any(isinstance(container, BarContainer) for container in window.fit_ax.containers)
    assert len(window.fit_ax.patches) == len(levels)
    np.testing.assert_allclose(check_subjects(), original_x)
    individual = next(line for line in window.fit_ax.lines
                      if line.get_label() == "Individual fit (fixed + random)")
    np.testing.assert_allclose(individual.get_ydata(), points.ConditionalFit)
    np.testing.assert_allclose(individual.get_xdata(), original_x)
    assert individual.get_marker() == "x"
    assert individual.get_color() == "red"
    assert model in window.fit_ax.containers
    window.condition_combo.setCurrentText("Condition[HRF A]:age")
    assert not window.fit_ax.patches
    assert "Individual fit (fixed + random)" in [line.get_label() for line in window.fit_ax.lines]
    window.close()


def test_public_launcher_exports_and_window_lifecycle(qt_app):
    from pyBrainAnalyzIR.vis import StatisticsViewer as PublicViewer, view_statistics
    from pyBrainAnalyzIR.vis.statistics_viewer import _OPEN_WINDOWS

    window = view_statistics(_first_level())
    assert isinstance(window, PublicViewer)
    assert window in _OPEN_WINDOWS
    window.close()
    from PySide6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qt_app.processEvents()
    assert window not in _OPEN_WINDOWS


def test_group_display_contrast_does_not_show_unrelated_fit(small_dataset):
    job = MixedEffects()
    job.options["robust"] = False
    stats = job.run(small_dataset)["groupstats"]
    stats.ttest(["Condition[HRF A]-Condition[HRF B]"], names=["A minus B"])
    window = StatisticsViewer(stats)
    assert len(window._channel_artists) == 2
    assert "display contrasts are unavailable" in window.fit_text.text()
    assert not window.fit_ax.lines
    window.close()


def test_missing_and_unbounded_observation_weights_are_explicit(small_dataset):
    job = MixedEffects()
    job.options["robust"] = False
    stats = job.run(small_dataset)["groupstats"]
    stats.fit_data.observations["WeightStdErr"] = np.inf
    window = StatisticsViewer(stats)
    assert "unbounded weight-implied uncertainty" in window.fit_text.text()
    assert "no finite purple bar" in window.fit_text.text()
    assert "Partial beta +/- 1 SE" in [container.get_label() for container in window.fit_ax.containers]
    del stats.fit_data.observations["WeightStdErr"]
    window._draw_fit()
    assert "rerun MixedEffects" in window.fit_text.text()
    assert "Final weighting +/- 1 implied SD" not in [
        container.get_label() for container in window.fit_ax.containers
    ]
    window.close()


@pytest.mark.parametrize("covariate", ["age", "gender"])
def test_selected_term_always_partials_subjects_and_individual_fits(small_dataset, covariate):
    job = MixedEffects()
    job.options["FE_formula"] = "Beta ~ 0 + Condition + Condition:age + Condition:gender"
    job.options["robust"] = False
    stats = _geometry(job.run(small_dataset)["groupstats"])
    window = StatisticsViewer(stats)
    term = next(str(name) for name in stats.list_conditions() if "HRF A" in name and covariate in name)
    window.condition_combo.setCurrentText(term)
    expected, curve, axis = stats.fit_data.partial_effect("S1D1", "HbO", term)
    assert axis == covariate
    raw = stats.fit_data.points("S1D1", "HbO", "HRF A")
    assert not np.allclose(expected.Beta, raw.Beta)

    def observed(label):
        return next(container for container in window.fit_ax.containers if container.get_label() == label)

    bars = observed("Partial beta +/- 1 SE")
    np.testing.assert_allclose(bars.lines[0].get_ydata(orig=False), expected.Beta)
    segments = bars.lines[2][0].get_segments()
    np.testing.assert_allclose(
        [segment[1, 1] - segment[0, 1] for segment in segments], 2 * raw.StdErr,
    )
    assert "SE bars exclude uncertainty" in window.fit_text.text()
    fitted = next(line for line in window.fit_ax.lines if line.get_label() == "Individual fit (fixed + random)")
    np.testing.assert_allclose(fitted.get_ydata(), expected.ConditionalFit)
    assert not hasattr(window, "adjust_covariates")
    window.close()


def test_control_labels_are_right_aligned_and_plot_toolbars_removed():
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QCheckBox, QLabel

    window = StatisticsViewer(_first_level())
    labels = {label.text(): label for label in window.findChildren(QLabel)}
    for text in ("Signal", "Threshold", "Threshold type", "Channel"):
        assert labels[text].alignment() & Qt.AlignRight
        assert labels[text].alignment() & Qt.AlignVCenter
    assert not window.findChildren(NavigationToolbar2QT)
    assert window.map_canvas.toolbar is None
    assert window.fit_canvas.toolbar is None
    assert not any("individual fits" in box.text().lower() for box in window.findChildren(QCheckBox))
    assert not window.findChildren(QCheckBox)
    assert "Observed task" not in labels
    assert "X axis" not in labels
    window.close()


def _head_geometry(stats):
    from cedalion import units
    from cedalion.dataclasses.geometry import PointType

    _geometry(stats)
    stats.geo3d = xr.DataArray(
        [[-30., 60., 70.], [30., 60., 70.], [-30., 20., 90.], [30., 20., 90.],
         [0., 100., 0.], [-100., 0., 0.], [100., 0., 0.]],
        dims=("label", "pos"),
        coords={
            "label": ["S1", "D1", "S2", "D2", "Nz", "LPA", "RPA"],
            "type": ("label", [PointType.SOURCE, PointType.DETECTOR] * 2 + [PointType.LANDMARK] * 3),
        },
    ).pint.quantify(units.mm)
    return stats


def _table_rows(window):
    table = window.results_table
    return [[table.item(row, column).text() for column in range(table.columnCount())]
            for row in range(table.rowCount())]


def test_results_table_tab_matches_condition_and_highlights_threshold(small_dataset):
    from pyBrainAnalyzIR.vis.statistics_viewer import SIGNIFICANT_ROW_COLOR

    job = MixedEffects()
    job.options["robust"] = False
    stats = _geometry(job.run(small_dataset)["groupstats"])
    window = StatisticsViewer(stats)
    tabs = window.results_tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Results Plots", "Results Table"]
    assert tabs.widget(0) is window.fit_canvas and tabs.widget(1) is window.results_table
    assert window.plot_splitter.widget(1) is tabs
    window.show()
    assert window.legend_canvas.isVisible()
    tabs.setCurrentWidget(window.results_table)
    assert not window.legend_canvas.isVisible()
    window.condition_combo.setCurrentText("Condition[HRF A]")
    assert not window.legend_canvas.isVisible()
    tabs.setCurrentWidget(window.fit_canvas)
    assert window.legend_canvas.isVisible() and window.fit_legend is not None
    condition = "Condition[HRF B]"
    window.condition_combo.setCurrentText(condition)
    expected = stats.table()
    expected = expected[expected.Condition == condition].reset_index(drop=True)
    frame = window.results_frame()
    pd.testing.assert_frame_equal(frame.drop(columns="Significant"), expected)
    assert set(frame.Type) == set(expected.Type) and len(set(frame.Type)) > 1
    headers = [window.results_table.horizontalHeaderItem(i).text()
               for i in range(window.results_table.columnCount())]
    assert headers == list(expected.columns) + ["Significant"]
    assert len(_table_rows(window)) == len(expected)

    def highlighted():
        return [window.results_table.item(row, 0).background().color() == SIGNIFICANT_ROW_COLOR
                for row in range(window.results_table.rowCount())]

    window.threshold_type.setCurrentText("p-value")
    cutoff = float(np.sort(expected["P-values"])[len(expected) // 2])
    window.threshold.setValue(cutoff)
    cutoff = window.threshold.value()
    significant = (expected["P-values"] < cutoff).tolist()
    assert any(significant) and not all(significant)
    assert highlighted() == significant
    assert window.results_frame().Significant.tolist() == significant
    window.threshold_type.setCurrentText("q-value")
    assert highlighted() == (expected["Q-values"] < cutoff).tolist()
    window.close()


def test_results_table_row_selects_channel_and_signal(small_dataset):
    job = MixedEffects()
    job.options["robust"] = False
    stats = _geometry(job.run(small_dataset)["groupstats"])
    window = StatisticsViewer(stats)
    frame = window.results_frame()
    target = next(i for i, row in frame.iterrows()
                  if row.Type != window.type_combo.currentText() and row.Channel != window.channel_combo.currentText())
    window.results_table.selectRow(int(target))
    assert window.type_combo.currentText() == frame.Type[target]
    assert window.channel_combo.currentText() == frame.Channel[target]
    assert f"{frame.Channel[target]} | {frame.Type[target]}" in window.fit_ax.get_title()
    assert [index.row() for index in window.results_table.selectionModel().selectedRows()] == [target]
    window.close()


def test_results_table_copies_excel_text_and_saves_xlsx(qt_app, small_dataset, tmp_path, monkeypatch):
    job = MixedEffects()
    job.options["robust"] = False
    stats = _geometry(job.run(small_dataset)["groupstats"])
    window = StatisticsViewer(stats)
    frame = window.results_frame()
    window.copy_table_to_clipboard()
    text = qt_app.clipboard().text()
    assert text == window.table_clipboard_text() and text.endswith("\n")
    lines = text.rstrip("\n").split("\n")
    assert lines[0].split("\t") == list(frame.columns)
    cells = [line.split("\t") for line in lines[1:]]
    assert len(cells) == len(frame) and all(len(row) == len(frame.columns) for row in cells)
    assert {row[-1] for row in cells} <= {"TRUE", "FALSE"}
    np.testing.assert_array_equal([float(row[3]) for row in cells], frame.Beta)

    saved = window.save_table_to_excel(tmp_path / "results")
    assert saved == str(tmp_path / "results.xlsx")
    pd.testing.assert_frame_equal(pd.read_excel(saved, sheet_name="Statistics"), frame, check_dtype=False)

    menu = window._build_table_menu()
    assert [action.text() for action in menu.actions()] == ["Copy to Clipboard", "Save to Excel..."]
    qt_app.clipboard().clear()
    menu.actions()[0].trigger()
    assert qt_app.clipboard().text() == text
    prompted = tmp_path / "prompted.xlsx"
    monkeypatch.setattr(
        "pyBrainAnalyzIR.vis.statistics_viewer.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(prompted), "Excel Workbook (*.xlsx)"),
    )
    menu.actions()[1].trigger()
    assert prompted.exists()
    window.close()


def _right_click(window, point, coords="axes"):
    canvas, ax = window.map_canvas, window.map_ax
    canvas.draw()
    transform = ax.transAxes if coords == "axes" else ax.transData
    x, y = transform.transform(point)
    canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", canvas, x, y, button=3))


def test_probe_context_menu_toggles_1020_and_2d_views(monkeypatch):
    from matplotlib.patches import Circle

    stats = _head_geometry(_first_level())
    menus = []
    monkeypatch.setattr(StatisticsViewer, "_show_probe_view_menu",
                        lambda self: menus.append(self._build_probe_view_menu()))
    window = StatisticsViewer(stats)
    assert window.plot_on_scalp and any(isinstance(p, Circle) for p in window.map_ax.patches)
    channel = window.channel_combo.currentText()
    line = next(a for a, label in window._channel_artists.items() if label != channel)
    _right_click(window, np.mean(np.asarray(line.get_data(), float), axis=1), coords="data")
    assert not menus and window.channel_combo.currentText() == channel

    _right_click(window, (0.02, 0.98))
    assert len(menus) == 1
    actions = {action.text(): action for action in menus[0].actions()}
    assert actions["10-20 View"].isChecked() and actions["2D Probe View"].isEnabled()
    actions["2D Probe View"].trigger()
    assert window.plot_on_scalp is False
    assert not any(isinstance(p, Circle) for p in window.map_ax.patches)
    assert len(window._channel_artists) == 2
    _right_click(window, (0.02, 0.98))
    actions = {action.text(): action for action in menus[1].actions()}
    assert actions["2D Probe View"].isChecked()
    actions["10-20 View"].trigger()
    assert window.plot_on_scalp and any(isinstance(p, Circle) for p in window.map_ax.patches)
    window.close()


def test_probe_view_menu_disabled_without_head_landmarks(monkeypatch):
    menus = []
    monkeypatch.setattr(StatisticsViewer, "_show_probe_view_menu",
                        lambda self: menus.append(self._build_probe_view_menu()))
    window = StatisticsViewer(_first_level())
    _right_click(window, (0.02, 0.98))
    assert len(menus) == 1
    assert not any(action.isEnabled() for action in menus[0].actions() if not action.isSeparator())
    window.close()
