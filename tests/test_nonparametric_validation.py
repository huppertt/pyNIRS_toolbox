import numpy as np
import pandas as pd
import pytest
from importlib import import_module

pytest.importorskip("cedalion")

from pyBrainAnalyzIR.pipelines.modules.glm import (  # noqa: E402
    GLM,
)
from pyBrainAnalyzIR.testing.simulations.simData import (  # noqa: E402
    ARnoise,
    Data,
)

nonparametric_statistics = import_module(
    "pyBrainAnalyzIR.pipelines.modules.nonparametric_statistics"
)
nonparametric_validation = import_module(
    "pyBrainAnalyzIR.testing.helpers.nonparametric_validation"
)
_circular_shift_residual = nonparametric_statistics._circular_shift_residual
_one_sample_fit = nonparametric_statistics._one_sample_fit
draw_ols_null_betas = nonparametric_validation.draw_ols_null_betas
fit_randomized_null_context = (
    nonparametric_validation.fit_randomized_null_context
)
make_randomized_stimulus = nonparametric_validation.make_randomized_stimulus
residual_stationarity_metrics = (
    nonparametric_validation.residual_stationarity_metrics
)

pytestmark = pytest.mark.requires_cedalion


def test_randomized_stimulus_has_jittered_rest_intervals():
    schedule = make_randomized_stimulus(np.arange(0.0, 180.0, 0.25), seed=12)

    intervals = (
        np.diff(schedule.onset.to_numpy()) - schedule.duration.to_numpy()[:-1]
    )
    assert len(schedule) > 2
    assert schedule.trial_type.unique().tolist() == ["validation_task"]
    assert schedule.duration.eq(10.0).all()
    assert (intervals >= 10.0).all()
    assert (intervals <= 20.0).all()
    pd.testing.assert_frame_equal(
        schedule,
        make_randomized_stimulus(np.arange(0.0, 180.0, 0.25), seed=12),
    )


def test_fast_ols_projection_matches_full_glm_refit():
    noise = ARnoise(t=np.arange(0.0, 180.0, 0.25), random_seed=81)
    noise["amp"] = noise["amp"].isel(channel=slice(0, 2))
    stimulus = pd.DataFrame(
        {
            "onset": [20.0, 60.0, 100.0],
            "duration": 10.0,
            "value": 1.0,
            "trial_type": "task",
        }
    )
    recording, _ = Data(
        noise=noise,
        stim=stimulus,
        snr=1.5,
        amplitudes={"task": np.array([[1.8, 1.8], [0.0, 0.0]])},
        random_seed=82,
    )
    context = fit_randomized_null_context(recording, seed=84)
    expected, _ = draw_ols_null_betas(
        context, iterations=1, seed=85, window_seconds=30.0
    )

    rng = np.random.default_rng(85)
    shifted = _circular_shift_residual(
        context.residual, rng, window_samples=120
    ).reshape(context.signal_values.shape)
    null_signal = recording["conc"].copy(deep=True)
    null_signal.pint.magnitude[:] = context.baseline + shifted
    glm = GLM()
    glm.options["noise_model"] = "ols"
    refit = _one_sample_fit(glm, recording, null_signal, iteration=0)
    refit = refit.loc[refit.Condition == context.condition]
    refit = refit.pivot(index="Channel", columns="Type", values="Beta")

    np.testing.assert_allclose(
        expected[0],
        refit.loc[
            list(context.channels), list(context.signal_types)
        ].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_stationarity_metrics_compare_candidate_windows():
    noise = ARnoise(t=np.arange(0.0, 180.0, 0.25), random_seed=81)
    noise["amp"] = noise["amp"].isel(channel=slice(0, 2))
    stimulus = pd.DataFrame(
        {
            "onset": [20.0, 60.0, 100.0],
            "duration": 10.0,
            "value": 1.0,
            "trial_type": "task",
        }
    )
    recording, _ = Data(
        noise=noise,
        stim=stimulus,
        snr=1.5,
        amplitudes={"task": np.array([[1.8, 1.8], [0.0, 0.0]])},
        random_seed=82,
    )
    context = fit_randomized_null_context(recording, seed=84)

    metrics = residual_stationarity_metrics(context, [10.0, 30.0, 60.0, None])

    assert len(metrics) == 2 * 2 * 4
    local_metrics = metrics.loc[metrics.WindowSeconds.notna()]
    assert local_metrics.BetweenWindowVarianceCV.notna().all()
    assert metrics.BetweenWindowTailRatioCV.dropna().ge(0).all()
    assert metrics.MedianWindowExcessKurtosis.notna().all()
    assert metrics.WithinWindowHalfVarianceRatioMedian.notna().all()
    assert metrics.AdjacentWindowWassersteinSD.dropna().ge(0).all()
    assert local_metrics.MedianWindowACF1s.notna().all()
    assert local_metrics.MedianWindowACF5s.notna().all()
    global_metrics = metrics.loc[metrics.WindowSeconds.isna()]
    assert global_metrics.BetweenWindowVarianceCV.isna().all()
    assert global_metrics.AdjacentWindowWassersteinSD.isna().all()
    assert global_metrics.BetweenWindowACF1sSD.isna().all()


def test_randomized_null_context_supports_recordings_without_stimulus():
    noise = ARnoise(t=np.arange(0.0, 180.0, 0.25), random_seed=81)
    noise["amp"] = noise["amp"].isel(channel=slice(0, 2))
    recording, _ = Data(
        noise=noise,
        stim=pd.DataFrame(
            {
                "onset": [20.0, 60.0, 100.0],
                "duration": 10.0,
                "value": 1.0,
                "trial_type": "task",
            }
        ),
        snr=1.5,
        amplitudes={"task": np.array([[1.8, 1.8], [0.0, 0.0]])},
        random_seed=82,
    )
    recording.stim = pd.DataFrame(
        columns=["onset", "duration", "value", "trial_type"]
    )

    context = fit_randomized_null_context(recording, seed=84)

    assert context.condition == "HRF validation_task"
    assert context.observed_betas.shape == (2, 2)


def test_randomized_stimulus_rejects_too_short_recording():
    with pytest.raises(ValueError, match="too short"):
        make_randomized_stimulus(np.arange(0.0, 60.0, 0.25), seed=12)
