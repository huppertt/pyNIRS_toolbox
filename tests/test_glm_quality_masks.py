from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cedalion.math.resample import resample as cedalion_resample

pytest.importorskip("cedalion")

from cedalion.models.glm.design_matrix import DesignMatrix
from cedalion.models.glm.solve import fit as cedalion_fit
from cedalion.models.glm.solve import predict as cedalion_predict
from cedalion.models.glm.solve import _channel_fit as cedalion_channel_fit

import pyBrainAnalyzIR.math.glm as local_glm
import pyBrainAnalyzIR.pipelines.modules.glm as glm_module
from pyBrainAnalyzIR.pipelines.modules.motion_correction import (
    _time_inclusion_mask,
)
import pyBrainAnalyzIR.pipelines.modules.preproccessing as prep
import pyBrainAnalyzIR.testing.simData as simdata
from pyBrainAnalyzIR.dataclasses.statistics import Statistics
from pyBrainAnalyzIR.math.resampling import resample_with_quality_masks

pytestmark = pytest.mark.requires_cedalion


def _synthetic_ts_and_design():
    rng = np.random.default_rng(481)
    time = np.arange(160, dtype=float)
    channels = ["S1D1", "S2D2"]
    chromo = ["HbO", "HbR"]
    regressors = ["intercept", "task", "drift"]
    common = np.column_stack(
        [np.ones(time.size), np.sin(time / 12), np.cos(time / 27)]
    )
    design = np.broadcast_to(
        common[:, :, None], (time.size, len(regressors), len(chromo))
    ).copy()
    values = np.empty((time.size, len(channels), len(chromo)))
    for channel_index in range(len(channels)):
        for chromo_index in range(len(chromo)):
            beta = np.array(
                [0.1 * channel_index, 0.5 + 0.1 * chromo_index, 0.2]
            )
            values[:, channel_index, chromo_index] = (
                common @ beta + rng.normal(0, 0.12, time.size)
            )

    ts = xr.DataArray(
        values,
        dims=("time", "channel", "chromo"),
        coords={"time": time, "channel": channels, "chromo": chromo},
    )
    design_matrix = DesignMatrix(
        common=xr.DataArray(
            design,
            dims=("time", "regressor", "chromo"),
            coords={
                "time": time,
                "regressor": regressors,
                "chromo": chromo,
            },
        )
    )
    return ts, design_matrix


@pytest.mark.parametrize(
    "noise_model", ["ols", "rls", "wls", "glsar", "ar_irls"]
)
def test_local_glm_matches_cedalion_without_missing_data(noise_model):
    ts, design = _synthetic_ts_and_design()

    expected = cedalion_fit(
        ts, design, noise_model=noise_model, ar_order=8, max_jobs=1
    )
    actual = local_glm.fit(
        ts, design, noise_model=noise_model, ar_order=8, max_jobs=1
    )

    for channel in ts.channel.values:
        for chromo in ts.chromo.values:
            expected_result = expected.sel(channel=channel, chromo=chromo).item()
            actual_result = actual.sel(channel=channel, chromo=chromo).item()
            np.testing.assert_array_equal(actual_result.params, expected_result.params)
            np.testing.assert_array_equal(actual_result.bse, expected_result.bse)
            np.testing.assert_array_equal(actual_result.resid, expected_result.resid)
            np.testing.assert_array_equal(
                actual_result.cov_params(), expected_result.cov_params()
            )


def test_local_glm_prediction_matches_cedalion():
    ts, design = _synthetic_ts_and_design()
    coefficients = np.empty((ts.sizes["channel"], 3, ts.sizes["chromo"]))
    for channel_index, channel in enumerate(ts.channel.values):
        for chromo_index, chromo in enumerate(ts.chromo.values):
            fit_result = local_glm.fit(
                ts.sel(channel=[channel], chromo=[chromo]),
                DesignMatrix(common=design.common.sel(chromo=[chromo])),
                max_jobs=1,
            ).item()
            coefficients[channel_index, :, chromo_index] = fit_result.params

    theta = xr.DataArray(
        coefficients,
        dims=("channel", "regressor", "chromo"),
        coords={
            "channel": ts.channel,
            "regressor": design.common.regressor,
            "chromo": ts.chromo,
        },
    )
    expected = cedalion_predict(ts, theta, design)
    actual = local_glm.predict(ts, theta, design)

    xr.testing.assert_equal(actual, expected)


def test_local_gls_matches_cedalion_channel_fit_without_missing_data():
    ts, design = _synthetic_ts_and_design()
    x = pd.DataFrame(
        design.common.sel(chromo="HbO").values,
        columns=design.common.regressor.values,
    )
    y = ts.sel(channel="S1D1", chromo="HbO").to_numpy()

    expected = cedalion_channel_fit(y, x.to_numpy(), "gls", ar_order=8)
    actual = local_glm.solve._channel_fit(y, x.to_numpy(), "gls", ar_order=8)

    np.testing.assert_array_equal(actual.params, expected.params)
    np.testing.assert_array_equal(actual.bse, expected.bse)
    np.testing.assert_array_equal(actual.resid, expected.resid)


@pytest.mark.parametrize(
    "noise_model", ["ols", "rls", "wls", "gls", "glsar", "ar_irls"]
)
def test_glm_drops_quality_masked_samples(noise_model):
    ts, design = _synthetic_ts_and_design()
    quality_mask = xr.DataArray(
        np.ones((ts.sizes["time"], ts.sizes["channel"]), dtype=bool),
        dims=("time", "channel"),
        coords={"time": ts.time, "channel": ts.channel},
    )
    quality_mask.values[[10, 11, 72], 0] = False
    quality_mask.values[[35, 104], 1] = False

    result = local_glm.fit(
        ts,
        design,
        noise_model=noise_model,
        ar_order=8,
        max_jobs=1,
        quality_mask=quality_mask,
    )

    for channel_index, channel in enumerate(ts.channel.values):
        for chromo in ts.chromo.values:
            channel_result = result.sel(channel=channel, chromo=chromo).item()
            assert np.isfinite(channel_result.params).all()
            assert channel_result._pybrain_valid_count == int(
                quality_mask.values[:, channel_index].sum()
            )
            assert channel_result._pybrain_full_resid.shape == (ts.sizes["time"],)
            assert np.isnan(
                channel_result._pybrain_full_resid[
                    ~quality_mask.values[:, channel_index]
                ]
            ).all()

    stats = Statistics(result)
    assert stats.dof > 0
    assert np.isfinite(stats.covariance).all()


def test_recording_masks_are_combined_and_collapsed_across_subtypes():
    ts, _ = _synthetic_ts_and_design()
    one_subtype_mask = xr.DataArray(
        np.ones((ts.sizes["time"], ts.sizes["channel"]), dtype=bool),
        dims=("time", "channel"),
        coords={"time": ts.time, "channel": ts.channel},
    )
    subtype_mask = xr.DataArray(
        np.ones(ts.shape, dtype=bool),
        dims=ts.dims,
        coords=ts.coords,
    )
    one_subtype_mask.values[7, 0] = False
    subtype_mask.values[9, 1, 1] = False

    combined = local_glm.solve._recording_quality_mask(
        ts, {"time_channel": one_subtype_mask, "subtype_specific": subtype_mask}
    )

    assert combined.dims == ("time", "channel")
    assert not combined.values[7, 0]
    assert not combined.values[9, 1]
    assert combined.values.sum() == (
        ts.sizes["time"] * ts.sizes["channel"] - 2
    )


def test_pca_motion_mask_excludes_time_if_any_channel_or_subtype_is_bad():
    time = np.arange(4, dtype=float)
    mask = xr.DataArray(
        np.ones((4, 2, 2), dtype=bool),
        dims=("time", "channel", "wavelength"),
        coords={
            "time": time,
            "channel": ["S1D1", "S2D2"],
            "wavelength": [760, 850],
        },
    )
    mask.values[1, 0, 0] = False
    mask.values[3, 1, 1] = False

    collapsed = _time_inclusion_mask(mask)

    assert collapsed.dims == ("time",)
    np.testing.assert_array_equal(collapsed.values, [True, False, True, False])


def test_fully_masked_channel_is_retained_as_nan_statistics():
    ts, design = _synthetic_ts_and_design()
    quality_mask = xr.DataArray(
        np.ones((ts.sizes["time"], ts.sizes["channel"]), dtype=bool),
        dims=("time", "channel"),
        coords={"time": ts.time, "channel": ts.channel},
    )
    quality_mask.values[:, 0] = False

    result = local_glm.fit(
        ts, design, max_jobs=1, quality_mask=quality_mask
    )
    stats = Statistics(result)

    masked_betas = stats.betas.to_numpy()[
        stats.betas.channels.values == "S1D1"
    ]
    assert np.isnan(masked_betas).all()
    assert stats.dof > 0
    assert np.isfinite(
        stats.covariance.to_numpy()[
            stats.covariance.channels_rows.values != "S1D1", :
        ][:, stats.covariance.channels_cols.values != "S1D1"]
    ).all()


def test_masked_resampling_interpolates_before_filtering_then_restores_nan():
    time = np.arange(200, dtype=float) / 10
    values = np.column_stack(
        [
            np.sin(time) + 0.01 * np.random.default_rng(2).normal(size=time.size),
            np.cos(time),
        ]
    )
    data = xr.DataArray(
        values.reshape(time.size, 1, 2),
        dims=("time", "channel", "wavelength"),
        coords={
            "time": ("time", time, {"units": "s"}),
            "samples": ("time", np.arange(time.size)),
            "channel": ["S1D1"],
            "wavelength": [760, 850],
        },
    )
    quality_mask = xr.DataArray(
        np.ones((time.size, 1), dtype=bool),
        dims=("time", "channel"),
        coords={"time": data.time, "channel": data.channel},
    )
    bad_source_index = 23
    data.values[bad_source_index, 0, 0] = 1e5
    naturally_missing_index = 70
    data.values[naturally_missing_index, 0, 1] = np.nan
    quality_mask.values[bad_source_index, 0] = False

    resampled, masks = resample_with_quality_masks(
        data, 2, {"motion": quality_mask}
    )

    projected_index = int(
        np.argmin(np.abs(resampled.time.values - time[bad_source_index]))
    )
    natural_projected_index = int(
        np.argmin(np.abs(resampled.time.values - time[naturally_missing_index]))
    )
    projected_mask = masks["motion"]
    assert not projected_mask.values[projected_index, 0]
    assert np.isnan(resampled.values[projected_index, 0, :]).all()
    assert not masks["missing_data"].values[natural_projected_index, 0]
    assert np.isnan(resampled.values[natural_projected_index, 0, :]).all()
    assert np.isfinite(resampled.values[:, 0, :]).any()
    assert np.nanmax(np.abs(resampled.values)) < 2


def test_unmasked_resampling_preserves_cedalion_behavior():
    time = np.arange(100, dtype=float) / 10
    data = xr.DataArray(
        np.ones((time.size, 1, 2)),
        dims=("time", "channel", "wavelength"),
        coords={
            "time": ("time", time, {"units": "s"}),
            "samples": ("time", np.arange(time.size)),
            "channel": ["S1D1"],
            "wavelength": [760, 850],
        },
    )

    actual, masks = resample_with_quality_masks(data, 2, {})
    expected = cedalion_resample(data, 2)

    xr.testing.assert_identical(actual, expected)
    assert masks == {}


def _roc_auc(stats, truth):
    pvalues = stats.get_pvalues()
    keep = np.asarray(
        [str(condition) == "HRF A" for condition in pvalues.conditions.values]
    )
    pvalues = pvalues[keep]
    truth_lookup = {
        (str(channel), str(chromo)): bool(truth.sel(channel=channel, chromo=chromo))
        for channel in truth.channel.values
        for chromo in truth.chromo.values
    }
    labels = np.asarray(
        [
            truth_lookup[(str(channel), str(chromo))]
            for channel, chromo in zip(pvalues.channels.values, pvalues.type.values)
        ]
    )
    scores = -np.log10(np.clip(pvalues.to_numpy(), 1e-300, 1))
    positive = scores[labels]
    negative = scores[~labels]
    return np.mean(
        (positive[:, None] > negative[None, :])
        + 0.5 * (positive[:, None] == negative[None, :])
    )


def test_small_quality_mask_preserves_first_level_glm_roc_performance():
    recording, truth = simdata.Data(snr=5, random_seed=20261015)
    unmasked = copy.deepcopy(recording)
    masked = copy.deepcopy(recording)
    amp = masked["amp"]
    quality_mask = xr.ones_like(
        amp.isel(wavelength=0, drop=True), dtype=bool
    )
    indices = np.arange(50, amp.sizes["time"], 485)
    quality_mask.values[indices, :] = False
    masked.set_mask("sparse_quality", quality_mask)

    def run_glm(rec):
        job = prep.intensity_opticaldensity()
        job = prep.mbll(job)
        job = prep.resample(job)
        job = glm_module.GLM(job)
        job.set_all_options(
            {"Fs": 2, "noise_model": "ols", "verbose": False}
        )
        return job.run(rec)["stats"]

    baseline = run_glm(unmasked)
    with_mask = run_glm(masked)
    baseline_auc = _roc_auc(baseline, truth)
    masked_auc = _roc_auc(with_mask, truth)

    assert abs(masked_auc - baseline_auc) <= 0.12
