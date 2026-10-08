"""Tests for the connectivity simulations and the analytic slab forward model."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("cedalion")

import cedalion.nirs  # noqa: E402

from pyBrainAnalyzIR.forward import ApproxSlab, slab_mesh  # noqa: E402
from pyBrainAnalyzIR.math.connectivity import compute_connectivity  # noqa: E402
from pyBrainAnalyzIR.math.corrcoef_robust import corrcoef as robust_corrcoef  # noqa: E402
from pyBrainAnalyzIR.media import getspectra, tissues  # noqa: E402
import pyBrainAnalyzIR.testing.simData as simData  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_getspectra_matches_cedalion_hemoglobin():
    ext = getspectra([760, 850])
    ref = cedalion.nirs.get_extinction_coefficients("prahl", [760, 850]).pint.dequantify()
    np.testing.assert_allclose(ext.sel(chromo=["HbO", "HbR"]).values,
                               ref.transpose("chromo", "wavelength").values)
    assert np.all(ext.sel(chromo=["H2O", "Lipid"]).values > 0)


def test_brain_mua_is_physiological():
    mua = tissues.brain([760, 850], 0.7, 50).mua.pint.to("1/mm").pint.dequantify().values
    assert np.all((mua > 0.005) & (mua < 0.05))


def test_approx_slab_jacobian_layout():
    geo2d, meas = simData.defaultProbe2D(shortsep=True)
    rec = simData.ARnoise(geo2d, meas, sigma=0)
    mesh = slab_mesh(dim=(10, 10, 5), shape=(21, 21, 5))
    fwd = ApproxSlab.from_recording(rec, mesh, tissues.brain(meas["wavelength"]))

    J = fwd.jacobian()
    assert J.dims == ("channel", "vertex", "wavelength")
    assert J.shape == (25, mesh.nvertices, 2)
    assert np.all(np.isfinite(J.values)) and np.all(J.values > 0)
    np.testing.assert_array_equal(J.source.values, rec["amp"].source.values)

    Js = fwd.jacobian("spectral")
    assert set(Js.dims) == {"channel", "vertex", "wavelength", "chromo"}

    meas_model = fwd.measurement()
    assert meas_model.dims == ("channel", "wavelength")
    # short-separation channels have a larger modeled signal than long ones
    assert meas_model.sel(channel="S1D9").values[0] > meas_model.sel(channel="S1D1").values[0]


def test_default_probe_shortsep():
    geo2d, meas = simData.defaultProbe2D(shortsep=True)
    assert len(meas["channel"]) == 25
    assert "D17" in geo2d.label.values
    rec = simData.ARnoise(geo2d, meas, sigma=0)
    _, ts_short = cedalion.nirs.split_long_short_channels(rec["amp"], rec.geo3d)
    assert len(ts_short.channel) == 9


def test_connectivity_shortsep_structure():
    np.random.seed(0)
    rec, truth = simData.connectivity_shortsep(t=np.arange(600) * 0.1)
    amp = rec["amp"]
    assert amp.dims == ("time", "channel", "wavelength")
    assert amp.shape == (600, 25, 2)
    assert np.all(amp.pint.dequantify().values > 0)
    assert truth.dims == ("channel_to", "channel_from")
    assert truth.shape == (16, 16)
    np.testing.assert_allclose(truth.values, truth.values.T)
    assert np.all(np.diag(truth.values) == 0)


def test_connectivity_shortsep_truth_is_recovered_after_short_regression():
    np.random.seed(3)
    rec, truth = simData.connectivity_shortsep(sigma=0, snr=2)
    ts_long, _ = cedalion.nirs.split_long_short_channels(cedalion.nirs.int2od(rec["amp"]), rec.geo3d)
    df = compute_connectivity(ts_long, robust=False, AR=10)
    df = df[(df.Channel_to != df.Channel_from) & (df.Type_to == df.Type_from)]
    is_true = np.array([truth.sel(channel_to=r.Channel_to, channel_from=r.Channel_from).item() != 0
                        for r in df.itertuples()])
    assert df.Pvalue[is_true].median() < df.Pvalue[~is_true].median()
    assert (df.Pvalue[is_true] < 0.05).mean() > 0.8


def test_connectivity_mvar_structure():
    np.random.seed(1)
    noise = simData.ARnoise(sigma=0, t=np.arange(1, 601) * 0.1)
    rec, truth = simData.connectivity(noise=noise, pmax=3)
    assert rec["amp"].shape == (600, 16, 2)
    assert np.all(np.isfinite(rec["amp"].pint.dequantify().values))
    assert truth.dims == ("channel_to", "wavelength_to", "channel_from", "wavelength_from", "lag")
    assert truth.shape == (16, 2, 16, 2, 4)
    assert np.all(truth.isel(lag=-1).values == 0)

    # re-using the returned truth works
    rec2, truth2 = simData.connectivity(noise=noise, truth=truth, pmax=3)
    np.testing.assert_array_equal(truth.values, truth2.values)


def test_compute_connectivity_labels_match_data():
    rec = simData.ARnoise(sigma=0, t=np.arange(1, 501) * 0.1)
    amp = rec["amp"].pint.dequantify()
    amp[:, 1, 0] = amp[:, 0, 0]  # S2D1 @ wl0 == S1D1 @ wl0
    df = compute_connectivity(amp, robust=False, AR=0)
    hits = df[(df.Coeff > 0.999) & (df.Channel_to != df.Channel_from)]
    assert set(zip(hits.Channel_to, hits.Channel_from)) == {("S1D1", "S2D1"), ("S2D1", "S1D1")}
    assert set(hits.Type_to) == {amp.wavelength.values[0]}


def test_robust_corrcoef_has_no_nans_for_weak_correlations():
    rng = np.random.default_rng(0)
    r, p = robust_corrcoef(rng.standard_normal((300, 12)))
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(p))
