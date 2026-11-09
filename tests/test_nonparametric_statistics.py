import copy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cedalion")

from pyBrainAnalyzIR.pipelines.default_pipelines import default_pipelines  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.nonparametric_statistics import (  # noqa: E402
    _circular_shift_residual,
    Statistics_Bootstrapping,
)
from pyBrainAnalyzIR.testing.simulations.simData import ARnoise, Data  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


def test_residual_shift_bootstrap_reports_empirical_inference_per_signal():
    noise = ARnoise(t=np.arange(0., 120., .25), random_seed=81)
    noise['amp'] = noise['amp'].isel(channel=slice(0, 2))
    stimulus = pd.DataFrame({
        'onset': [10., 40., 70.],
        'duration': 10.,
        'value': 1.,
        'trial_type': 'task',
    })
    recording, truth = Data(
        noise=noise,
        stim=stimulus,
        snr=1.5,
        amplitudes={'task': np.array([[1.8, 1.8], [0., 0.]])},
        random_seed=82,
    )
    pipeline = default_pipelines.first_level_analysis.pipeline()
    pipeline.set_all_options({'noise_model': 'ols', 'verbose': False})
    bootstrap = Statistics_Bootstrapping(pipeline)
    bootstrap.options['iterations'] = 19
    bootstrap.options['random_seed'] = 12

    result = bootstrap.run(copy.deepcopy(recording))
    estimates = result.aux_obj['bootstrap']
    table = estimates.table.set_index(['Channel', 'Type'])

    assert estimates.iterations == 19
    assert estimates.resampling_method == 'local_circular_shift'
    assert set(table.index) == {
        ('S1D1', 'HbO'), ('S1D1', 'HbR'),
        ('S2D1', 'HbO'), ('S2D1', 'HbR'),
    }
    assert table.loc[('S1D1', 'HbO'), 'PowerAtObservedBeta'] > table.loc[
        ('S2D1', 'HbO'), 'PowerAtObservedBeta'
    ]
    assert table.EmpiricalP.between(0, 1).all()
    assert table.NullTimecourseVarianceRatio.gt(0).all()
    assert (table.NullFPR <= estimates.alpha).all()
    assert len(estimates.null_betas.Iteration.unique()) == 19
    assert len(estimates.alternative_betas.Iteration.unique()) == 19
    assert bool(truth.sel(channel='S1D1', chromo='HbO'))


def test_local_circular_shift_preserves_each_window_joint_samples():
    rng = np.random.default_rng(42)
    residual = rng.standard_t(3, size=(120, 2))
    residual[:, 1] = .7 * residual[:, 0] + .3 * rng.standard_t(3, size=120)
    residual[:40] *= .25
    residual[80:] *= 2.
    shifted = _circular_shift_residual(
        residual, np.random.default_rng(43), window_samples=40
    )

    windows = np.array_split(np.arange(len(residual)), 3)
    for window in windows:
        np.testing.assert_array_equal(
            np.sort(shifted[window], axis=0),
            np.sort(residual[window], axis=0),
        )
        assert {
            tuple(row) for row in shifted[window]
        } == {
            tuple(row) for row in residual[window]
        }
    np.testing.assert_allclose(
        shifted.var(axis=0, ddof=1),
        residual.var(axis=0, ddof=1),
    )


def test_global_circular_shift_preserves_complete_residual_rows():
    residual = np.arange(24, dtype=float).reshape(12, 2)
    shifted = _circular_shift_residual(
        residual, np.random.default_rng(43), window_samples=None
    )

    assert {tuple(row) for row in shifted} == {tuple(row) for row in residual}


def test_bootstrap_requires_glm_as_immediate_predecessor():
    with pytest.raises(ValueError, match='placed directly after'):
        Statistics_Bootstrapping()._glm()


def test_bootstrap_window_defaults_to_30_seconds_and_is_configurable():
    bootstrap = Statistics_Bootstrapping()

    assert bootstrap.options['shift_window_seconds'] == 30.
    bootstrap.options['shift_window_seconds'] = 45.
    assert bootstrap.options['shift_window_seconds'] == 45.
