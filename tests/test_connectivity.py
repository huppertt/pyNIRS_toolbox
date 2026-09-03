"""Tests for connectivity calculations."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("cedalion")

import cedalion  # noqa: E402
import pint  # noqa: E402
import xarray as xr  # noqa: E402

from pyBrainAnalyzIR.math.connectivity import compute_hyperscanning  # noqa: E402


pytestmark = pytest.mark.requires_cedalion


def _quantified_signal(offset):
    values = np.arange(8, dtype=float).reshape(4, 1, 2) + offset
    signal = xr.DataArray(
        values,
        dims=("time", "channel", "chromo"),
        coords={
            "time": np.arange(4, dtype=float),
            "channel": ["S1D1"],
            "chromo": ["HbO", "HbR"],
        },
    )
    return signal.pint.quantify(cedalion.units.micromolar)


def test_hyperscanning_accepts_quantified_data_without_unit_stripping_warning():
    signals = {"subject-1": _quantified_signal(0), "subject-2": _quantified_signal(1)}

    with warnings.catch_warnings():
        warnings.simplefilter("error", pint.UnitStrippedWarning)
        result = compute_hyperscanning(signals, robust=False, AR=1)

    assert len(result) == 16
