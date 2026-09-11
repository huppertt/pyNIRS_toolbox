"""Tests for the data quality manager GUI."""
from __future__ import annotations

import importlib
from collections import OrderedDict

import pandas as pd
import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("cedalion")

import cedalion.sigproc.quality as quality  # noqa: E402
from pyBrainAnalyzIR.dataclasses.dataset import DataSet  # noqa: E402
from pyBrainAnalyzIR.testing import simData  # noqa: E402
from pyBrainAnalyzIR.vis.data_quality_manager import (  # noqa: E402
    DataQualityManager,
    compute_quality_metric,
    metric_mask,
)

pytestmark = [pytest.mark.requires_qt, pytest.mark.requires_cedalion]
pytestmark.append(pytest.mark.usefixtures("qt_app"))


def _recording(subject: str = "subj_1"):
    rec, _ = simData.Data(snr=10)
    rec.meta_data["subject"] = subject
    rec.meta_data["scan"] = "1"
    return rec


def test_compute_quality_metric_returns_channel_and_time_series(monkeypatch):
    rec = _recording()

    def broken_psp(*args, **kwargs):
        _ = (args, kwargs)
        raise ValueError("forced PSP failure")

    monkeypatch.setattr(quality, "psp", broken_psp)

    for metric in ("snr", "sci", "psp", "gvtd", "clean_percent"):
        scalar, timeseries = compute_quality_metric(rec, metric)
        assert scalar.dims == ("channel",)
        assert timeseries.dims == ("time", "channel")
        assert scalar.sizes["channel"] == rec["amp"].sizes["channel"]
        assert timeseries.sizes["channel"] == rec["amp"].sizes["channel"]


def test_metric_mask_uses_thresholds():
    import numpy as np

    rec = _recording()
    _, sci = compute_quality_metric(rec, "sci")
    mask = metric_mask(sci, "sci", {"sci": 0.75})

    values = np.unique(np.asarray(mask))
    assert set(values.tolist()) <= {0.0, 1.0}
    assert mask.dims == sci.dims

    # GVTD is clean *below* threshold, so a huge threshold marks everything clean.
    _, gvtd = compute_quality_metric(rec, "gvtd")
    all_clean = metric_mask(gvtd, "gvtd", {"gvtd": 1e12})
    assert float(np.nanmin(np.asarray(all_clean))) == 1.0


def test_gui_exposes_gvtd_thresholds_window_and_display_mode():
    rec = _recording()
    window = DataQualityManager(rec)

    assert "gvtd" in window._channel_metric_checks
    assert "gvtd" in window._artifact_metric_checks
    assert set(window._channel_threshold_spins) == set(window.thresholds)
    assert window._channel_window_spin.value() == pytest.approx(window.window_length_s)

    # Editing a threshold on one tab keeps both tabs and the model in sync.
    window._channel_threshold_spins["sci"].setValue(0.5)
    assert window.thresholds["sci"] == pytest.approx(0.5)
    assert window._artifact_threshold_spins["sci"].value() == pytest.approx(0.5)

    window._artifact_window_spin.setValue(10.0)
    assert window.window_length_s == pytest.approx(10.0)
    assert window._channel_window_spin.value() == pytest.approx(10.0)
    assert window.window_length.magnitude == pytest.approx(10.0)

    assert window._display_mode("channel") == "value"
    window._channel_display_mode.setCurrentIndex(1)
    assert window._display_mode("channel") == "mask"


def test_mask_display_mode_applies_to_channel_detail_and_heatmap():
    import numpy as np

    rec = _recording()
    window = DataQualityManager(rec)
    window._channel_display_mode.setCurrentIndex(1)

    values = window._metric_display_values(0, "snr", "mask")
    assert set(np.unique(np.asarray(values)).tolist()) <= {0.0, 1.0}

    line = next(iter(window._channel_line_map))
    xdata, ydata = line.get_data()
    event = type("Event", (), {})()
    event.button = 3
    event.inaxes = line.axes
    event.xdata = float((xdata[0] + xdata[-1]) / 2)
    event.ydata = float((ydata[0] + ydata[-1]) / 2)
    window._on_channel_plot_click(event)

    detail_ax, metric_ax = window.channel_detail_figure.axes
    assert "mask" in detail_ax.get_title()
    assert [label.get_text() for label in metric_ax.get_yticklabels()] == [
        "tainted",
        "clean",
    ]


def test_data_quality_manager_accepts_single_recording_and_draws_tabs():
    from PySide6.QtWidgets import QMenuBar  # noqa: E402

    rec = _recording()
    window = DataQualityManager(rec)

    assert len(window.dataset.dataset) == 1
    assert window.file_tree.topLevelItemCount() == 1
    assert window.tabs.tabText(0) == "Channel Quality"
    assert window.tabs.tabText(1) == "Time-series artifacts"
    assert len(window.channel_figure.axes) >= 1
    menu_bar = window.findChild(QMenuBar)
    export_menu = next(
        action.menu() for action in menu_bar.actions()
        if action.text() == "Export Results"
    )
    assert [action.text() for action in export_menu.actions()] == [
        "Export to Excel",
        "Export to PDF",
        "Export to HTML",
    ]


def test_all_files_mode_limits_selection_to_one_metric():
    dset = DataSet([_recording("one"), _recording("two")])
    window = DataQualityManager(dset)

    window._channel_metric_checks["sci"].setChecked(True)
    window._channel_all_files.setChecked(True)

    checked_metrics = [
        metric for metric, checkbox in window._channel_metric_checks.items()
        if checkbox.isChecked()
    ]

    assert len(checked_metrics) == 1
    assert window._channel_metric_checks[checked_metrics[0]].isEnabled()
    assert all(
        not checkbox.isEnabled()
        for metric, checkbox in window._channel_metric_checks.items()
        if metric != checked_metrics[0]
    )


def test_right_click_single_channel_plot_draws_channel_detail():
    rec = _recording()
    window = DataQualityManager(rec)
    line = next(iter(window._channel_line_map))
    xdata, ydata = line.get_data()

    event = type("Event", (), {})()
    event.button = 3
    event.inaxes = line.axes
    event.xdata = float((xdata[0] + xdata[-1]) / 2)
    event.ydata = float((ydata[0] + ydata[-1]) / 2)

    window._on_channel_plot_click(event)

    assert window._channel_detail_selection == (0, "snr", 0)
    assert not window.channel_detail_canvas.isHidden()
    assert len(window.channel_detail_figure.axes) == 2
    detail_ax, metric_ax = window.channel_detail_figure.axes
    assert "normalized amplitude and SNR" in detail_ax.get_title()
    assert len(detail_ax.lines) == 2
    assert all(line.get_linewidth() <= 0.9 for line in detail_ax.lines)

    assert metric_ax.yaxis.get_ticks_position() in ("right", "default")
    assert len(metric_ax.lines) == 1
    metric_line = metric_ax.lines[0]
    assert metric_line.get_color() == "magenta"
    assert metric_line.get_linewidth() <= 0.9
    assert metric_ax.get_ylabel() == "SNR"


def test_channel_detail_disabled_for_multiple_channel_quality_plots():
    rec = _recording()
    window = DataQualityManager(rec)
    window._channel_metric_checks["sci"].setChecked(True)

    assert window._channel_line_map == {}
    assert window.channel_detail_canvas.isHidden()


def test_export_to_excel_writes_one_sheet_per_recording(tmp_path):
    dset = DataSet([_recording("one"), _recording("two")])
    window = DataQualityManager(dset)
    path = tmp_path / "quality.xlsx"

    window.export_to_excel(path)

    workbook = pd.ExcelFile(path)
    assert len(workbook.sheet_names) == 2
    first = pd.read_excel(path, sheet_name=workbook.sheet_names[0], header=None)
    assert first.iloc[1, 0] == "Subject"
    assert first.iloc[1, 1] == "one"
    assert first.iloc[2, 0] == "Scan number"
    assert first.iloc[2, 1] == "1"
    table = pd.read_excel(path, sheet_name=workbook.sheet_names[0], skiprows=5)
    assert list(table.columns) == [
        "channel",
        "SNR",
        "SCI",
        "PSP",
        "GVTD",
        "Percent of Clean time",
    ]
    assert len(table) == dset.dataset[0]["amp"].sizes["channel"]


def test_export_to_pdf_and_html_write_reports(tmp_path):
    window = DataQualityManager(_recording())
    pdf_path = tmp_path / "quality.pdf"
    html_path = tmp_path / "quality.html"

    window.export_to_pdf(pdf_path)
    window.export_to_html(html_path)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
    html = html_path.read_text(encoding="utf-8")
    assert "Data Quality Report" in html
    assert "Channel Quality" in html
    assert "Time-series artifacts" in html
    assert "data:image/png;base64," in html


class Recording:
    def __init__(self, subject: str = "subj_1"):
        self.meta_data = OrderedDict({"subject": subject})
        self.timeseries = {}
        self.stim = None


def test_nirsviewir_data_review_menu_opens_data_quality_manager(monkeypatch):
    from PySide6.QtWidgets import QMenuBar  # noqa: E402

    nirsview = importlib.import_module("pyBrainAnalyzIR.vis.NIRSviewIR")
    manager_module = importlib.import_module("pyBrainAnalyzIR.vis.data_quality_manager")

    window = nirsview.NIRSviewIRWindow(DataSet([Recording()]))
    calls = []

    def open_manager(dataset, block=False):
        calls.append((dataset, block))

    monkeypatch.setattr(manager_module, "data_quality_manager", open_manager)

    menu_bar = window.findChild(QMenuBar)
    data_review_menu = next(
        action.menu() for action in menu_bar.actions()
        if action.text() == "Data Review"
    )
    data_review_menu.actions()[0].trigger()

    assert calls == [(window.dataset, False)]
