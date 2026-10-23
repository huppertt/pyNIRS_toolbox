"""Opt-in end-to-end comparisons against the local MATLAB nirs-toolbox."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyBrainAnalyzIR.testing.matlab_comparison import (
    ROCCurve,
    _matlab_canonical_basis,
    _metric_pair,
    _merge_estimates,
    _plot_roc_comparison,
    _prepare_recording_for_python_mbll,
    _satterthwaite_type_one_curves,
    run_matlab_python_roc_comparison,
)
from pyBrainAnalyzIR.math.glm.solve import _satterthwaite_dof


def test_metric_pair_reports_numerical_difference():
    result = _metric_pair(
        np.array([1.0, 2.0, np.nan]),
        np.array([1.5, 2.0, 9.0]),
    )

    assert result["count"] == 2
    assert result["mean_absolute_error"] == pytest.approx(0.25)
    assert result["max_absolute_error"] == pytest.approx(0.5)
    assert result["root_mean_square_error"] == pytest.approx(np.sqrt(0.125))
    assert result["pearson_correlation"] == pytest.approx(1.0)


def test_merge_estimates_requires_exact_alignment():
    keys = {
        "iteration": [1],
        "source": [1],
        "detector": [1],
        "signal_type": ["hbo"],
        "condition": ["A"],
    }
    reference = pd.DataFrame({
        **keys,
        "beta": [1.0],
        "t_value": [2.0],
        "p_value": [0.05],
        "truth": [1.0],
    })
    python = pd.DataFrame({
        **keys,
        "beta_python": [1.5],
        "t_value_python": [2.5],
        "p_value_python": [0.1],
    })

    merged = _merge_estimates(reference, python)

    assert merged.loc[0, "beta_matlab"] == 1.0
    assert merged.loc[0, "beta_python"] == 1.5
    assert merged.loc[0, "truth_matlab"] == 1.0


def test_prepare_recording_adds_planar_geometry_and_seconds():
    pytest.importorskip("cedalion")
    from cedalion import units

    geo2d = xr.DataArray(
        [[0.0, 0.0], [30.0, 0.0]],
        dims=("label", "pos"),
        coords={
            "label": ["S1", "D1"],
            "type": ("label", ["source", "detector"]),
        },
        attrs={"units": "mm"},
    ).pint.quantify()
    geo3d = xr.DataArray(
        np.empty((0, 3)),
        dims=("label", "pos"),
        coords={"label": []},
        attrs={"units": "mm"},
    ).pint.quantify()
    amp = xr.DataArray(
        np.ones((2, 1, 4)),
        dims=("channel", "wavelength", "time"),
        coords={
            "channel": ["S1D1", "S1D2"],
            "wavelength": [760],
            "time": [0.0, 0.1, 0.2, 0.3],
        },
    )

    class TestRecording(SimpleNamespace):
        def __getitem__(self, name):
            return self.timeseries[name]

    recording = TestRecording(
        geo2d=geo2d,
        geo3d=geo3d,
        timeseries={"amp": amp},
    )

    _prepare_recording_for_python_mbll(recording)

    assert recording.geo3d.pint.to("mm").pint.dequantify().values.tolist() == [
        [0.0, 0.0, 0.0],
        [30.0, 0.0, 0.0],
    ]
    assert recording["amp"].time.attrs["units"] == units.s


@pytest.mark.requires_cedalion
def test_matlab_canonical_basis_matches_reference_defaults():
    pytest.importorskip("cedalion")

    time_series = xr.DataArray(
        np.zeros((2, 1, 1)),
        dims=("time", "channel", "chromo"),
        coords={
            "time": [0.0, 0.25],
            "channel": ["S1D1"],
            "chromo": ["HbO"],
            "samples": ("time", [0, 1]),
        },
    )
    time_series.time.attrs["units"] = "s"
    time_series = time_series.pint.quantify()

    basis = _matlab_canonical_basis()(time_series)

    assert basis.shape == (129, 1)
    assert basis.time.values[0] == 0
    assert basis.time.values[-1] == 32
    assert basis.time.values[basis[:, 0].argmax().item()] == 3
    assert basis[:, 0].sum().item() == pytest.approx(1)


def test_matlab_python_comparison_rejects_unknown_simulation():
    with pytest.raises(ValueError, match="simulation must be one of"):
        run_matlab_python_roc_comparison(simulation="unknown")


def test_satterthwaite_dof_matches_explicit_hat_matrix():
    rng = np.random.default_rng(13)
    design = rng.normal(size=(40, 4))
    weights = rng.uniform(0.2, 1.0, size=40)
    weighted_design = weights[:, None] * design
    hat_residual = (
        np.diag(weights)
        - weighted_design
        @ np.linalg.pinv(weighted_design.T @ weighted_design)
        @ weighted_design.T
    )
    expected = (
        np.sum(hat_residual * hat_residual.T) ** 2
        / np.sum((hat_residual.T @ hat_residual) ** 2)
    )

    assert _satterthwaite_dof(design, weights) == pytest.approx(expected)


def test_type_one_plot_includes_nominal_diagonal():
    curve = ROCCurve(
        true_positive_rate=np.array([0.2, 1.0]),
        false_positive_rate=np.array([0.1, 1.0]),
        thresholds=np.array([0.01, 0.5]),
    )
    figure = _plot_roc_comparison({
        toolbox: {"hbo": curve, "hbr": curve}
        for toolbox in ("matlab", "python")
    })

    diagonal = [
        line
        for line in figure.axes[1].lines
        if line.get_label() == "Nominal FPR = threshold"
    ]
    assert len(diagonal) == 1
    assert diagonal[0].get_xdata().tolist() == [0, 1]
    assert diagonal[0].get_ydata().tolist() == [0, 1]


def test_satterthwaite_type_one_curves_use_null_channel_pvalues():
    estimates = pd.DataFrame({
        "signal_type": ["hbo", "hbo", "hbr", "hbr"],
        "truth_matlab": [0, 1, 0, 1],
        "p_value_python_satterthwaite": [0.1, 0.01, 0.2, 0.02],
    })

    curves = _satterthwaite_type_one_curves(estimates)

    assert curves["hbo"].thresholds.tolist() == [0.1]
    assert curves["hbo"].false_positive_rate.tolist() == [1.0]
    assert curves["hbr"].thresholds.tolist() == [0.2]
