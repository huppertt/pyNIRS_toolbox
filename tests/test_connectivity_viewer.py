"""Headless tests for connectivity result filtering and detail plots."""
from __future__ import annotations

import pandas as pd
import pint
import pytest
import numpy as np
import xarray as xr
import warnings
from types import SimpleNamespace

pytest.importorskip("cedalion")
pytest.importorskip("PySide6.QtWidgets")

from pyBrainAnalyzIR.dataclasses.connectivity import Connectivity  # noqa: E402
from pyBrainAnalyzIR.dataclasses.model_fit import ConnectivityGroupFit  # noqa: E402
from pyBrainAnalyzIR.vis.connectivity_viewer import (  # noqa: E402
    ConnectivityViewer,
    view_connectivity,
)

pytestmark = [
    pytest.mark.requires_cedalion,
    pytest.mark.requires_qt,
    pytest.mark.usefixtures("qt_app"),
]


def _result():
    result = Connectivity()
    result.coef = pd.DataFrame([
        {
            "Condition": "task", "Channel_to": "S1D1", "Channel_from": "S2D2",
            "Type_to": "HbO", "Type_from": "HbO", "geo_to": "subject",
            "geo_from": "subject", "Coeff": .5, "Pvalue": .01, "dfe": 20,
            "ZstdErr": .2,
        },
        {
            "Condition": "task", "Channel_to": "S1D1", "Channel_from": "S2D2",
            "Type_to": "HbO", "Type_from": "HbR", "geo_to": "subject",
            "geo_from": "subject", "Coeff": -.3, "Pvalue": .04, "dfe": 20,
            "ZstdErr": .2,
        },
    ])
    return result


def test_data_type_pair_selector_includes_cross_types():
    window = ConnectivityViewer(_result())
    assert window.type_combo.count() == 2
    assert len(window._filtered_table()) == 1
    window.type_combo.setCurrentIndex(1)
    assert window._filtered_table().iloc[0]["Type_from"] == "HbR"
    window.close()


def test_hyperscanning_endpoint_filter_defaults_to_inter_subject():
    result = _result()
    result.coef = pd.concat([
        result.coef.iloc[[0]].assign(Subject_to="participant-1", Subject_from="participant-2"),
        result.coef.iloc[[0]].assign(Subject_to="participant-1", Subject_from="participant-1"),
    ], ignore_index=True)
    window = ConnectivityViewer(result)
    assert window.endpoint_combo.currentText() == "Inter-subject only"
    assert len(window._filtered_table()) == 1
    window.endpoint_combo.setCurrentText("Intra-subject only")
    assert len(window._filtered_table()) == 1
    assert (
        window._filtered_table().iloc[0]["Subject_to"]
        == window._filtered_table().iloc[0]["Subject_from"]
    )
    window.close()


def test_group_detail_plots_age_scatter_and_model_fit():
    result = _result()
    result.coef = result.coef.iloc[[0]].copy()
    result.coef.loc[:, "Condition"] = "age"
    result.coef.loc[:, "Subject_to"] = "subject"
    result.coef.loc[:, "Subject_from"] = "subject"
    observations = pd.DataFrame({
        "Condition": ["task"] * 3,
        "Channel_to": ["S1D1"] * 3,
        "Channel_from": ["S2D2"] * 3,
        "Type_to": ["HbO"] * 3,
        "Type_from": ["HbO"] * 3,
        "Beta": [.1, .2, .3],
        "ZstdErr": [.05] * 3,
        "age": [20., 30., 40.],
        "FixedFit": [.11, .2, .29],
        "ConditionalFit": [.1, .21, .3],
    })
    result.fit_data = ConnectivityGroupFit(
        observations=observations,
        coefficients=result.coef.copy(),
        covariance={},
        formula="Beta ~ 0 + Condition + age",
        random_effects_formula="subject",
        grouping_column="subject",
        centers={},
        hyperscanning=False,
        participant_roles={},
    )
    window = ConnectivityViewer(result)
    assert window.endpoint_combo.isHidden()
    assert window.detail_ax.get_xlabel() == "age"
    assert len(window.detail_ax.lines) >= 2
    assert len(window.detail_ax.containers) == 1
    window.close()


def test_first_level_detail_overlays_both_edge_time_courses():
    class Recording:
        data = xr.DataArray(
            np.arange(12, dtype=float).reshape(2, 1, 6),
            dims=("channel", "chromo", "time"),
            coords={
                "channel": ["S1D1", "S2D2"],
                "chromo": ["HbO"],
                "time": np.arange(6, dtype=float),
            },
            attrs={"units": "molar"},
        )
        data.time.attrs["units"] = "second"
        timeseries = {"conc": data.pint.quantify()}

        def __getitem__(self, name):
            return self.timeseries[name]

    data = SimpleNamespace(
        dataset=[Recording()],
        get_demographics=lambda: pd.DataFrame({"subject": ["s1"]}),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", pint.UnitStrippedWarning)
        window = ConnectivityViewer(_result(), data=data)
    assert len(window.detail_ax.lines) == 2
    assert window.detail_tabs.count() == 2
    assert window.detail_tabs.tabText(1) == "Endpoint scatter"
    assert len(window.scatter_ax.collections) == 1
    window.detail_tabs.setCurrentIndex(1)
    assert "r=0.5" in window.detail_ax.get_title()
    window.close()


def test_map_draws_only_significant_edges_and_labels_optodes():
    result = _result()
    result.coef = pd.concat([
        result.coef.iloc[[0]],
        result.coef.iloc[[0]].assign(
            Channel_from="S2D2", Pvalue=.8, Coeff=.1,
        ),
    ], ignore_index=True)
    geometry = xr.DataArray(
        [[0., 0.], [1., 0.], [0., 1.], [1., 1.]],
        dims=("label", "pos"),
        coords={"label": ["S1", "D1", "S2", "D2"]},
    )
    result.probe_layouts = lambda _plot_on_scalp: ({"subject": (geometry, None)}, False)
    window = ConnectivityViewer(result)
    assert len(window._artists) == 1
    assert len(window.map_ax.lines) == 1
    assert {text.get_text() for text in window.map_ax.texts} == {"S1", "D1", "S2", "D2"}
    assert "1 of 2 edges" in window.map_ax.get_title()
    window.close()


def test_context_menu_switches_to_labeled_grid_adjacency_matrix():
    window = ConnectivityViewer(_result())
    menu = window._build_map_context_menu()
    labels = [action.text() for action in menu.actions()]
    assert labels == ["2D probe", "Adjacency matrix"]
    menu.actions()[1].trigger()
    assert window.map_mode == "adjacency"
    assert len(window.map_ax.get_xticklabels()) == 2
    assert len(window.map_ax.get_yticklabels()) == 2
    assert len(window.map_ax.get_xgridlines()) > 0
    window._on_map_click(SimpleNamespace(
        inaxes=window.map_ax, button=1, xdata=0, ydata=1,
    ))
    assert window._selected_edge is not None
    window.close()


def test_results_table_filter_highlight_sort_and_export(qt_app, tmp_path):
    from pyBrainAnalyzIR.vis.connectivity_viewer import SIGNIFICANT_ROW_COLOR

    result = _result()
    result.coef.loc[1, "Pvalue"] = .8
    result.coef.loc[1, "Type_from"] = "HbO"
    window = ConnectivityViewer(result)
    assert [window.results_tabs.tabText(i) for i in range(window.results_tabs.count())] == [
        "Results Plot", "Results Table"
    ]
    assert len(window.results_frame()) == 1
    menu = window._build_table_menu()
    assert [action.text() for action in menu.actions()] == [
        "Show Only Significant Channels", "", "Copy to Clipboard", "Save to Excel..."
    ]
    menu.actions()[0].trigger()
    frame = window.results_frame()
    assert len(frame) == 2
    assert list(frame.columns[:5]) == [
        "Condition", "To_Channel", "To_Type", "From_Channel", "From_Type"
    ]
    assert [
        window.results_table.item(row, 0).background().color() == SIGNIFICANT_ROW_COLOR
        for row in range(window.results_table.rowCount())
    ] == frame.Significant.tolist()
    correlation_column = list(frame.columns).index("Correlation")
    window._sort_table_column(correlation_column)
    assert window.results_frame().Correlation.is_monotonic_increasing
    window._sort_table_column(correlation_column)
    assert window.results_frame().Correlation.is_monotonic_decreasing
    window.copy_table_to_clipboard()
    assert qt_app.clipboard().text().splitlines()[0].split("\t") == list(frame.columns)
    saved = window.save_table_to_excel(tmp_path / "edges")
    assert saved.endswith("edges.xlsx")
    assert len(pd.read_excel(saved, sheet_name="Connectivity")) == 2
    window.close()


def test_symmetric_connectivity_q_values_correct_once_per_undirected_edge():
    from statsmodels.stats.multitest import multipletests

    result = _result()
    result.coef = pd.DataFrame([
        {
            "Condition": "rest", "Channel_to": "S1D1", "Channel_from": "S2D2",
            "Type_to": "HbO", "Type_from": "HbO", "Coeff": .2,
            "Pvalue": .01, "dfe": 20, "ZstdErr": .2,
        },
        {
            "Condition": "rest", "Channel_to": "S2D2", "Channel_from": "S1D1",
            "Type_to": "HbO", "Type_from": "HbO", "Coeff": .2,
            "Pvalue": .01, "dfe": 20, "ZstdErr": .2,
        },
        {
            "Condition": "rest", "Channel_to": "S1D1", "Channel_from": "S3D3",
            "Type_to": "HbO", "Type_from": "HbO", "Coeff": .3,
            "Pvalue": .04, "dfe": 20, "ZstdErr": .2,
        },
        {
            "Condition": "rest", "Channel_to": "S3D3", "Channel_from": "S1D1",
            "Type_to": "HbO", "Type_from": "HbO", "Coeff": .3,
            "Pvalue": .04, "dfe": 20, "ZstdErr": .2,
        },
    ])
    qvalues = result.table().Qvalue.to_numpy()
    expected = multipletests([.01, .04], method="fdr_bh")[1]
    np.testing.assert_allclose(qvalues, [expected[0], expected[0], expected[1], expected[1]])


def test_directed_hyperscanning_q_values_keep_both_directions():
    from statsmodels.stats.multitest import multipletests

    result = _result()
    result.coef = pd.DataFrame([
        {
            "Condition": "rest", "Channel_to": "S1D1", "Channel_from": "S2D2",
            "Type_to": "HbO", "Type_from": "HbO",
            "Subject_to": "parent", "Subject_from": "child",
            "Coeff": .2, "Pvalue": .01, "dfe": 20, "ZstdErr": .2,
        },
        {
            "Condition": "rest", "Channel_to": "S2D2", "Channel_from": "S1D1",
            "Type_to": "HbO", "Type_from": "HbO",
            "Subject_to": "child", "Subject_from": "parent",
            "Coeff": .2, "Pvalue": .04, "dfe": 20, "ZstdErr": .2,
        },
    ])
    expected = multipletests([.01, .04], method="fdr_bh")[1]
    np.testing.assert_allclose(result.table().Qvalue, expected)


def test_3d_probe_option_uses_10_20_projection_and_head_outline():
    result = _result()
    geometry = xr.DataArray(
        [[-.5, 0.], [0., -.5], [.5, 0.], [0., .5]],
        dims=("label", "pos"),
        coords={"label": ["S1", "D1", "S2", "D2"]},
    )
    outline = np.eye(3)

    def probe_layouts(plot_on_scalp):
        return {"default": (geometry, outline if plot_on_scalp else None)}, plot_on_scalp

    result.probe_layouts = probe_layouts
    window = ConnectivityViewer(result)
    window._set_map_mode("2d")
    menu = window._build_map_context_menu()
    action = next(
        action for action in menu.actions()
        if action.text() == "3D probe / 10-20 view"
    )
    action.trigger()
    assert window.map_mode == "3d"
    assert len(window.map_ax.patches) == 3
    assert window.map_ax.get_xlim() == pytest.approx((-1.15, 1.15), abs=.002)
    assert window.map_ax.get_ylim() == pytest.approx((-1.15, 1.15), abs=.002)
    window.close()


def test_launcher_resolves_first_level_result_from_recording_timeseries():
    class Recording:
        def __init__(self):
            self.timeseries = {"conn": _result()}

        def __getitem__(self, name):
            return self.timeseries[name]

    data = SimpleNamespace(
        dataset=[Recording()],
        statistics={},
        get_demographics=lambda: pd.DataFrame({"subject": ["s1"]}),
    )
    window = view_connectivity(data, result_name="conn")
    assert window.connectivity is data.dataset[0].timeseries["conn"]
    window.close()
