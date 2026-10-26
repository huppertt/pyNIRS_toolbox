"""Opt-in comparisons of MATLAB and Python GLM/ROI processing order."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("cedalion")

from pyBrainAnalyzIR.testing.matlab_comparison import find_matlab_executable
from pyBrainAnalyzIR.testing.matlab_roi_comparison import (
    run_matlab_python_roi_glm_comparison,
)


@pytest.mark.requires_matlab
def test_matlab_python_roi_glm_processing_orders(tmp_path):
    try:
        matlab = find_matlab_executable()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    summary = run_matlab_python_roi_glm_comparison(
        num_examples=50,
        seed=29,
        output_dir=tmp_path,
        matlab_executable=matlab,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    assert summary["num_examples"] >= 50
    assert summary["single_condition_examples"] > 0
    assert summary["multiple_condition_examples"] > 0
    assert summary["distinct_roi_definitions"] > 1
    for result in summary["matlab_vs_python"].values():
        assert result["estimate_error"]["max_absolute_error"] < 1e-8
        assert result["covariance_matrix_difference"]["max_absolute_error"] < 1e-9
    assert (tmp_path / "roi_glm_per_example_errors.csv").is_file()
    assert (tmp_path / "roi_glm_comparison_summary.json").is_file()


@pytest.mark.requires_matlab
def test_matlab_python_roi_glm_ar_irls_weighted_processing_orders(tmp_path):
    """AR-IRLS with inverse-variance ROI weights: MATLAB and Python agree per path."""
    try:
        matlab = find_matlab_executable()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    summary = run_matlab_python_roi_glm_comparison(
        num_examples=50,
        seed=29,
        output_dir=tmp_path,
        matlab_executable=matlab,
        noise_model="ar_irls",
        ar_order=8,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    assert summary["noise_model"] == "AR-IRLS"
    assert summary["single_condition_examples"] > 0
    assert summary["multiple_condition_examples"] > 0
    assert summary["distinct_roi_definitions"] > 1
    for result in summary["matlab_vs_python"].values():
        assert result["estimate_error"]["max_absolute_error"] < 1e-8
        assert result["covariance_matrix_difference"]["max_absolute_error"] < 1e-8
