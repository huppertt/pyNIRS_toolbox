"""Physical injection, empirical ROC thresholds, inversion and first-level wiring."""
import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("cedalion")

from pyBrainAnalyzIR.testing.power_analysis import (  # noqa: E402
    PowerCalibration, inject_activation, power_analysis,
)

pytestmark = pytest.mark.requires_cedalion


def calibration():
    rows = []
    for channel in ['S1D1', 'S2D1']:
        for ty in ['HbO', 'HbR']:
            for i in range(100):
                null = (i + 1) / 101
                rows.append(dict(Channel=channel, Type=ty, EffectSize=0.,
                                 P=null, Beta=0., Iteration=i, Pilot=i % 2))
                for size, sensitivity in [(0.2, .2), (.4, .9), (.6, .7)]:
                    rows.append(dict(Channel=channel, Type=ty, EffectSize=size,
                                     P=.001 if i < 100 * sensitivity else .5,
                                     Beta=size, Iteration=i, Pilot=i % 2))
    return PowerCalibration(pd.DataFrame(rows), np.array([.2, .4, .6]),
                            .05, 'HRF A', {}, 'gamma', 2)


def test_threshold_curve_roc_ties_and_monotonicity():
    cal = calibration()
    curve = cal.curve
    np.testing.assert_allclose(curve.Sensitivity[:3], [.2, .9, .7])
    np.testing.assert_allclose(curve.Power[:3], [.2, .8, .8])
    np.testing.assert_allclose(curve.NullFPR, .05)
    tied = copy.deepcopy(cal)
    tied.records.loc[tied.records.EffectSize == 0, 'P'] = .01
    assert (tied.curve.NullFPR == 0).all()
    for _, group in cal.roc_table().groupby(['Channel', 'Type', 'EffectSize']):
        assert tuple(group.iloc[0][['FPR', 'TPR']]) == (0., 0.)
        assert tuple(group.iloc[-1][['FPR', 'TPR']]) == (1., 1.)
        assert (np.diff(group.FPR) >= 0).all()
        assert (np.diff(group.TPR) >= 0).all()


def test_interpolation_inverse_bounds_and_persistence():
    cal = calibration()
    np.testing.assert_allclose(cal.power(.3).Power, .5)
    assert cal.power(.1).Power.isna().all()
    assert cal.power(.8).Power.isna().all()
    np.testing.assert_allclose(cal.minimum_detectable_change().MDC, .4)
    assert (cal.minimum_detectable_change().MDCStatus == 'interpolated').all()
    assert cal.minimum_detectable_change(.95).MDC.isna().all()
    assert (cal.minimum_detectable_change(.1).MDCStatus == 'at_or_below_range').all()
    pd.testing.assert_frame_equal(pickle.loads(pickle.dumps(cal)).curve, cal.curve)


def test_channels_are_not_pooled_for_thresholds_or_power():
    cal = calibration()
    channel_two = cal.records.Channel == 'S2D1'
    cal.records.loc[channel_two & (cal.records.EffectSize == 0), 'P'] *= .5
    cal.records.loc[channel_two & (cal.records.EffectSize > 0), 'P'] = .9
    thresholds = cal.curve.groupby('Channel').Threshold.first()
    assert thresholds['S2D1'] == pytest.approx(thresholds['S1D1'] / 2)
    power = cal.power(.4).set_index(['Channel', 'Type'])
    assert power.loc[('S1D1', 'HbO'), 'Power'] == pytest.approx(.8)
    assert power.loc[('S2D1', 'HbO'), 'Power'] == 0.
    mdc = cal.minimum_detectable_change().set_index(['Channel', 'Type'])
    assert mdc.loc[('S1D1', 'HbO'), 'MDC'] == pytest.approx(.4)
    assert np.isnan(mdc.loc[('S2D1', 'HbO'), 'MDC'])


def test_label_aligned_measured_lookup():
    cal = calibration()
    beta = xr.DataArray(
        [-.3, .3, -.9, -.3], dims=['indices'],
        coords={'channels': ('indices', ['S2D1', 'S1D1', 'S1D1', 'S2D1']),
                'type': ('indices', ['HbR', 'HbO', 'HbR', 'HbO']),
                'conditions': ('indices', ['HRF A'] * 4)},
    )
    stats = SimpleNamespace(betas=beta, get_pvalues=lambda: xr.ones_like(beta) * .1)
    table = cal.estimate_power(stats).set_index(['Channel', 'Type'])
    assert table.loc[('S1D1', 'HbO'), 'Power'] == pytest.approx(.5)
    assert table.loc[('S2D1', 'HbR'), 'Power'] == pytest.approx(.5)
    assert table.loc[('S1D1', 'HbR'), 'PowerStatus'] == 'outside_range'
    assert table.loc[('S2D1', 'HbO'), 'PowerStatus'] == 'opposite_direction'
    assert np.isnan(table.loc[('S1D1', 'HbR'), 'Power'])


def pilot_and_design():
    from pyBrainAnalyzIR.testing.simulations.simData import ARnoise

    rec = ARnoise(t=np.arange(0., 120., .25), random_seed=81)
    rec['amp'] = rec['amp'].isel(channel=slice(0, 2))
    stim = pd.DataFrame({'onset': [10., 40., 70.], 'duration': 10.,
                         'value': 1., 'trial_type': 'task'})
    return rec, stim


def test_injection_recovers_exact_physical_beta_and_preserves_input():
    from pyBrainAnalyzIR.pipelines.default_pipelines import default_pipelines

    rec, stim = pilot_and_design()
    before = copy.deepcopy(rec)
    pipeline = default_pipelines.first_level_analysis.pipeline()
    pipeline.set_all_options({'noise_model': 'ols', 'verbose': False})
    baseline = pipeline.run(inject_activation(rec, stim, 0.))['stats']
    injected = pipeline.run(inject_activation(rec, stim, 1.5))['stats']
    difference = injected.betas - baseline.betas
    for ty, expected in [('HbO', 1.5), ('HbR', -1.5)]:
        rows = (difference.conditions == 'HRF task') & (difference.type == ty)
        np.testing.assert_allclose(difference.where(rows, drop=True), expected, atol=1e-10)
    xr.testing.assert_identical(rec['amp'], before['amp'])
    pd.testing.assert_frame_equal(rec.stim, before.stim)
    assert list(rec.timeseries) == ['amp']
    zero = inject_activation(rec, stim, 0.)
    xr.testing.assert_identical(zero['amp'], rec['amp'])


def test_first_level_calibration_end_to_end_and_reproducibility():
    from pyBrainAnalyzIR.pipelines.default_pipelines import default_pipelines

    rec, stim = pilot_and_design()
    before = rec['amp'].copy(deep=True)
    pipeline = default_pipelines.first_level_analysis.pipeline()
    pipeline.set_all_options({'noise_model': 'ols', 'verbose': False})
    with pytest.warns(UserWarning, match='exploratory'):
        cal = power_analysis([rec], stim, pipeline, [.2, 1.], 20,
                             random_seed=13, verbose=False)
    assert cal.records.shape[0] == 20 * 3 * 2 * 2
    assert len(cal.curve) == 2 * 2 * 2
    assert (cal.curve.N == 20).all()
    assert (cal.curve.NullFPR <= .05).all()
    assert set(cal.records.EffectSize) == {0., .2, 1.}
    xr.testing.assert_identical(rec['amp'], before)
    with pytest.warns(UserWarning, match='exploratory'):
        second = power_analysis([rec], stim, pipeline, [.2, 1.], 20,
                                random_seed=13, verbose=False, n_jobs=2)
    pd.testing.assert_frame_equal(cal.records, second.records)
    result = pipeline.run(inject_activation(rec, stim, .5))['stats']
    assert len(cal.estimate_power(result)) == 4
    fig = cal.draw()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_power_calibration_can_resume_from_pickled_rng_state():
    from pyBrainAnalyzIR.pipelines.default_pipelines import default_pipelines

    rec, stim = pilot_and_design()
    pipeline = default_pipelines.first_level_analysis.pipeline()
    pipeline.set_all_options({'noise_model': 'ols', 'verbose': False})
    with pytest.warns(UserWarning, match='exploratory'):
        first_batch = power_analysis(
            [rec], stim, pipeline, [.2, 1.], 10,
            alpha=.1, verbose=False, random_seed=13,
        )

    restored = pickle.loads(pickle.dumps(first_batch))
    with pytest.warns(UserWarning, match='exploratory'):
        continued = power_analysis(
            [rec], stim, pipeline, [.2, 1.], 10,
            alpha=.1, verbose=False, initial_calibration=restored,
        )
    with pytest.warns(UserWarning, match='exploratory'):
        uninterrupted = power_analysis(
            [rec], stim, pipeline, [.2, 1.], 20,
            alpha=.1, verbose=False, random_seed=13,
        )

    pd.testing.assert_frame_equal(continued.records, uninterrupted.records)
    assert continued.records.Iteration.nunique() == 20
    with pytest.raises(ValueError, match='Do not pass random_seed'):
        power_analysis(
            [rec], stim, pipeline, [.2, 1.], 10, alpha=.1,
            verbose=False, initial_calibration=restored, random_seed=13,
        )
    with pytest.raises(ValueError, match='effect sizes, alpha'):
        power_analysis(
            [rec], stim, pipeline, [.2, .8], 10, alpha=.1,
            verbose=False, initial_calibration=restored,
        )
    with pytest.raises(ValueError, match='pilots, stimulus'):
        power_analysis(
            [rec], stim.assign(onset=stim.onset + 1.), pipeline, [.2, 1.], 10,
            alpha=.1, verbose=False, initial_calibration=restored,
        )


@pytest.mark.parametrize('kwargs', [
    {'effect_sizes': []}, {'effect_sizes': [.2, .2]},
    {'effect_sizes': [np.nan]}, {'effect_sizes': [-1]},
    {'num_iterations': 0}, {'num_iterations': 1.5},
    {'num_iterations': 19}, {'alpha': 0}, {'alpha': 1},
    {'n_jobs': 0}, {'n_jobs': 1.5}, {'n_jobs': True},
])
def test_reject_invalid_calibration_parameters(kwargs):
    with pytest.raises(ValueError):
        power_analysis([], None, None, **kwargs)


def test_reject_invalid_stimulus_and_amplitude():
    rec, stim = pilot_and_design()
    with pytest.raises(ValueError, match='nonnegative'):
        inject_activation(rec, stim, -1.)
    with pytest.raises(ValueError, match='inside'):
        inject_activation(rec, stim.assign(onset=1000.), .2)
    with pytest.raises(ValueError, match='one nonempty'):
        inject_activation(rec, pd.concat([stim, stim.assign(trial_type='B')]), .2)
    alias = stim.rename(columns={'value': 'amp'})
    assert 'value' in inject_activation(rec, alias, 0.).stim
