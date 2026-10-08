"""Tests for the ``divide_events`` options of the connectivity / hyperscanning modules and the
multi-condition support (contrasts / t-tests) of the Connectivity data class."""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cedalion")

import pyBrainAnalyzIR.dataclasses.dataset as dataset  # noqa: E402
import pyBrainAnalyzIR.testing.simData as simData  # noqa: E402
from pyBrainAnalyzIR.math.connectivity import compute_connectivity, event_windows  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.connectivity import (  # noqa: E402
    hyperscanning, resting_state_connectivity)

pytestmark = pytest.mark.requires_cedalion

TASK = pd.DataFrame({'onset': [30.0, 150.0], 'duration': [60.0, 60.0], 'value': 1.0,
                     'trial_type': 'task'})


def _task_rec(seed=0, coupling=0.0):
    """ARnoise recording with two 60 s task blocks. With ``coupling`` > 0 a common signal is added
    to every channel during the task blocks only (i.e. higher connectivity during task)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(seed)
        rec = simData.ARnoise()
    rec.stim = TASK.copy()
    if coupling:
        amp = rec['amp']
        t = amp.time.values
        mask = np.zeros_like(t)
        for on, dur in zip(TASK.onset, TASK.duration):
            mask[(t >= on) & (t <= on + dur)] = 1
        common = np.random.randn(len(t)) * mask
        factor = amp.copy(data=np.ones(amp.shape))
        factor = factor + coupling * factor * common[:, None, None] if amp.dims[0] == 'time' else \
            factor + coupling * factor * common[None, None, :]
        rec['amp'] = amp * factor.values
    return rec


def _conn(rec, **options):
    job = resting_state_connectivity()
    job.inputName = 'amp'
    job.options['AR'] = 4
    job.options['robust'] = False
    job.options['short_separation_correction'] = False
    for k, v in options.items():
        job.options[k] = v
    return job.run(rec)['conn']


# ---------------------------------------------------------------------------
# event windows
# ---------------------------------------------------------------------------

def test_event_windows_trim_transitions_and_add_rest():
    windows, skipped = event_windows(TASK, 0.1, 300.0, min_event_duration=30, ignore=10)
    assert list(windows) == ['task', 'rest']
    assert windows['task'] == [(40.0, 80.0), (160.0, 200.0)]
    # the 29.9 s of rest at the start is too short; 90-150 and 210-300 are used
    assert windows['rest'] == [(100.0, 140.0), (220.0, 290.0)]
    assert skipped == []


def test_event_windows_skip_short_conditions():
    stim = pd.concat([TASK, pd.DataFrame({'onset': [100.0], 'duration': [5.0], 'value': 1.0,
                                          'trial_type': 'blip'})], ignore_index=True)
    windows, skipped = event_windows(stim, 0.1, 300.0, min_event_duration=30, ignore=10)
    assert 'blip' not in windows and skipped == ['blip']
    # the short event still counts as labeled time: rest is split around it
    assert windows['rest'] == [(220.0, 290.0)]


def test_event_windows_without_stim_is_one_rest_window():
    windows, skipped = event_windows(None, 0.0, 300.0, min_event_duration=30, ignore=10)
    assert dict(windows) == {'rest': [(10.0, 290.0)]}


def test_single_window_matches_correlation_of_the_window():
    rec = _task_rec()
    df = compute_connectivity(rec['amp'], robust=False, AR=0, windows={'w': [(40.0, 80.0)]})
    amp = rec['amp'].transpose('time', 'channel', 'wavelength')
    sel = amp.sel(time=(amp.time > 40.0) & (amp.time < 80.0)).pint.dequantify().values
    r = np.corrcoef(sel.reshape(sel.shape[0], -1), rowvar=False)
    np.testing.assert_allclose(df['Coeff'].to_numpy(), r.T.flatten(), atol=1e-10)
    assert (df['dfe'] == sel.shape[0] - 2).all()


def test_windows_are_averaged_in_fisher_z():
    rec = _task_rec()
    kw = dict(robust=False, AR=0)
    a = compute_connectivity(rec['amp'], windows={'w': [(40.0, 80.0)]}, **kw)
    b = compute_connectivity(rec['amp'], windows={'w': [(160.0, 200.0)]}, **kw)
    ab = compute_connectivity(rec['amp'], windows={'w': [(40.0, 80.0), (160.0, 200.0)]}, **kw)
    z = (np.arctanh(np.clip(a['Coeff'], -np.tanh(6), np.tanh(6)))
         + np.arctanh(np.clip(b['Coeff'], -np.tanh(6), np.tanh(6)))) / 2
    off = a['Channel_to'] != a['Channel_from']
    np.testing.assert_allclose(ab['Coeff'][off], np.tanh(z)[off], atol=1e-10)
    assert ab['dfe'].iloc[0] == a['dfe'].iloc[0] + b['dfe'].iloc[0]
    np.testing.assert_allclose(ab['ZstdErr'].iloc[0],
                               np.sqrt(a['ZstdErr'].iloc[0] ** 2 + b['ZstdErr'].iloc[0] ** 2) / 2)


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------

def test_module_options_exist_with_matlab_defaults():
    for job in (resting_state_connectivity(), hyperscanning()):
        assert job.options['divide_events'] is False
        assert job.options['min_event_duration'] == 30
        assert job.options['ignore_event_transition_time'] == 10


def test_without_divide_events_there_is_a_single_rest_condition():
    conn = _conn(_task_rec())
    assert conn.conditions == ['rest']
    assert {'dfe', 'ZstdErr'} <= set(conn.coef.columns)


def test_divide_events_computes_one_map_per_condition():
    conn = _conn(_task_rec(), divide_events=True)
    assert conn.conditions == ['task', 'rest']
    per_cond = conn.coef.groupby('Condition').size()
    assert per_cond['task'] == per_cond['rest']
    assert conn.coef.groupby('Condition')['dfe'].first().to_dict() == {'task': 794.0, 'rest': 1094.0}


def test_divide_events_options_control_the_windows():
    conn = _conn(_task_rec(), divide_events=True, min_event_duration=45, ignore_event_transition_time=5)
    # task blocks: 60 - 10 = 50 s > 45; rest: 90-150 (50 s) and 210-300 (80 s)
    assert conn.conditions == ['task', 'rest']
    with pytest.warns(UserWarning, match="skipping condition"):
        conn = _conn(_task_rec(), divide_events=True, min_event_duration=45,
                     ignore_event_transition_time=10)
    assert conn.conditions == ['rest']


def test_task_minus_rest_detects_the_coupling():
    conn = _conn(_task_rec(coupling=0.05), divide_events=True)
    tt = conn.ttest('task - rest', remove_self_connections=True)
    assert tt.conditions == ['task - rest']
    assert tt.description == 'T-test'
    assert (tt.coef['Coeff'] > 0).mean() > 0.95
    assert (tt.coef['Pvalue'] < 0.05).mean() > 0.9
    # without the coupling, the difference is null
    null = _conn(_task_rec(), divide_events=True).ttest('task - rest', remove_self_connections=True)
    assert (null.coef['Pvalue'] < 0.05).mean() < 0.15


def test_hyperscanning_divide_events():
    dset = dataset.DataSet()
    for seed in (0, 1):
        dset.import_data(_task_rec(seed))
    dset.add_demographics_by_index(pd.DataFrame({'subject': ['A', 'B'], 'pair': ['d1', 'd1']}))
    job = hyperscanning()
    job.inputName = 'amp'
    job.options['AR'] = 4
    job.options['robust'] = False
    job.options['short_separation_correction'] = False
    job.options['divide_events'] = True
    dset = job.run(dset)
    conn = dset.dataset[0]['hyperconn']
    assert conn.conditions == ['task', 'rest']
    assert (conn.coef['geo_to'] != conn.coef['geo_from']).all()
    tt = conn.ttest('task - rest')
    assert len(tt.coef) == (conn.coef['Condition'] == 'task').sum()


# ---------------------------------------------------------------------------
# Connectivity.ttest / contrasts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def conn():
    return _conn(_task_rec(), divide_events=True)


def test_selecting_conditions_keeps_the_original_values(conn):
    pd.testing.assert_frame_equal(conn.ttest(['task', 'rest']).coef, conn.coef)
    rest = conn.ttest('rest')
    pd.testing.assert_frame_equal(rest.coef, conn.coef[conn.coef.Condition == 'rest']
                                  .reset_index(drop=True))


def test_numeric_and_string_contrasts_agree(conn):
    a = conn.ttest('task - rest').coef
    b = conn.ttest([1, -1], names='task - rest').coef
    pd.testing.assert_frame_equal(a, b)
    assert conn.ttest([[1, -1]]).conditions == ['task-rest']


def test_contrast_statistics(conn):
    tt = conn.ttest('task - rest').coef
    task = conn.coef[conn.coef.Condition == 'task'].reset_index(drop=True)
    rest = conn.coef[conn.coef.Condition == 'rest'].reset_index(drop=True)
    z = np.arctanh(np.clip(task.Coeff, -np.tanh(6), np.tanh(6))) - \
        np.arctanh(np.clip(rest.Coeff, -np.tanh(6), np.tanh(6)))
    np.testing.assert_allclose(tt.Coeff, np.tanh(z), atol=1e-12)
    np.testing.assert_allclose(tt.ZstdErr, np.sqrt(task.ZstdErr ** 2 + rest.ZstdErr ** 2))
    np.testing.assert_allclose(tt.dfe, (task.dfe + rest.dfe) / 2)
    # null hypothesis offset b (Fisher-Z units)
    shifted = conn.ttest('task - rest', b=0.1).coef
    np.testing.assert_allclose(np.arctanh(shifted.Coeff), z - 0.1, atol=1e-10)


def test_contrast_parser(conn):
    w = conn._parse_contrast('0.5*task + 0.5 * rest')
    np.testing.assert_allclose(w, [0.5, 0.5])
    np.testing.assert_allclose(conn._parse_contrast('-rest+task'), [1, -1])
    with pytest.raises(ValueError, match="Unknown condition"):
        conn.ttest('task - nope')
    with pytest.raises(ValueError, match="columns"):
        conn.ttest([1, -1, 0])


def test_table_and_matrix(conn):
    tbl = conn.table()
    assert {'Z', 'Tstat', 'Qvalue'} <= set(tbl.columns)
    mat = conn.matrix('task', type_to=conn.coef.Type_to.iloc[0])
    nchan = conn.coef.Channel_to.nunique()
    assert mat.shape == (nchan, nchan)
    np.testing.assert_allclose(np.diag(mat.to_numpy()), 1.0)
    np.testing.assert_allclose(mat.to_numpy(), mat.to_numpy().T, atol=1e-10)


def test_show_draws_a_row_per_condition(conn):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.close('all')
    conn.show()
    fig = plt.gcf()
    titles = [a.get_title() for a in fig.axes if a.get_title()]
    assert any(t.startswith('task:') for t in titles)
    assert any(t.startswith('rest:') for t in titles)
    plt.close('all')
    conn.ttest('task - rest').show(type='t', threshold_type='q')
    plt.close('all')


def test_discard_stims_works_on_connectivity_conditions(conn):
    from pyBrainAnalyzIR.pipelines.modules.events import DiscardStims
    rec = _task_rec()
    rec['conn'] = copy.deepcopy(conn)
    job = DiscardStims()
    job.options['listOfStims'] = ['rest']
    rec = job.run(rec)
    assert rec['conn'].conditions == ['task']
