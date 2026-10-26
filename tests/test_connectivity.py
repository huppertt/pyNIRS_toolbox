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


def _quantified_signal(seed, n_samples):
    values = np.random.default_rng(seed).normal(size=(n_samples, 1, 2))
    signal = xr.DataArray(
        values,
        dims=("time", "channel", "chromo"),
        coords={
            "time": np.arange(n_samples, dtype=float),
            "channel": ["S1D1"],
            "chromo": ["HbO", "HbR"],
        },
    )
    return signal.pint.quantify(cedalion.units.micromolar)


@pytest.mark.parametrize("ar_order", [1, 4, 8])
def test_hyperscanning_accepts_quantified_data_without_unit_stripping_warning(ar_order):
    n_samples = max(32, 8 * ar_order)
    signals = {
        f"subject-{seed + 1}": _quantified_signal(seed, n_samples)
        for seed in range(2)
    }
    assert all(signal.sizes["time"] >= 8 * ar_order for signal in signals.values())

    with warnings.catch_warnings():
        warnings.simplefilter("error", pint.UnitStrippedWarning)
        result = compute_hyperscanning(signals, robust=False, AR=ar_order)

    assert len(result) == 16
    assert np.isfinite(result["Coeff"]).all()
