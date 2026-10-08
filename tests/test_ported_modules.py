"""Tests for the modules ported from the MATLAB nirs-toolbox: RemoveStimGaps, RemoveShortStims,
DiscardStims, KeepStims (events), DiscardTypes, KeepTypes, RemoveTooLongDistance (utilities) and
TrimBaseline (preprocessing)."""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cedalion")

import cedalion.nirs  # noqa: E402

import pyBrainAnalyzIR.dataclasses.dataset as dataset  # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.events as events  # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.preproccessing as prep  # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.utilities as util  # noqa: E402
import pyBrainAnalyzIR.testing.simData as simData  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.connectivity import resting_state_connectivity  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.glm import GLM  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


def _stim(rows):
    return pd.DataFrame(rows, columns=['onset', 'duration', 'trial_type']).assign(value=1.0)


@pytest.fixture(scope="module")
def base_rec():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(1)
        rec = simData.ARnoise()  # 10 Hz, t = 0.1 .. 300 s
        job = prep.intensity_opticaldensity()
        job = prep.mbll(job)
        return job.run(rec)


@pytest.fixture
def rec(base_rec):
    return copy.deepcopy(base_rec)


# --------------------------------------------------------------------------- events

def test_remove_stim_gaps_merges_blocks_per_condition(rec):
    # A: [10,15] gap 0.1s (=1 sample) [15.1,20] gap 0.1s [20.1,25]; B overlaps timing but is separate
    rec.stim = _stim([(10, 5, 'A'), (15.1, 4.9, 'A'), (20.1, 4.9, 'A'), (60, 5, 'A'),
                      (15.05, 1, 'B'), (40, 5, 'B')])
    job = events.RemoveStimGaps()
    rec = job.run(rec)
    a = rec.stim[rec.stim.trial_type == 'A'].reset_index(drop=True)
    np.testing.assert_allclose(a.onset, [10, 60])
    np.testing.assert_allclose(a.duration, [15.0, 5])
    assert (rec.stim.trial_type == 'B').sum() == 2


def test_remove_stim_gaps_across_conditions_and_min_isi(rec):
    rec.stim = _stim([(10, 2, 'A'), (14, 2, 'B'), (40, 2, 'A')])
    job = events.RemoveStimGaps()
    job.options['seperate_conditions'] = False
    job.options['maxDuration'] = None
    job.options['minISI'] = 5
    rec = job.run(rec)
    assert rec.stim.trial_type.tolist() == ['A', 'A']
    np.testing.assert_allclose(rec.stim.duration, [6, 2])


def test_remove_short_stims(rec):
    rec.stim = _stim([(10, 5, 'A'), (30, 40, 'A'), (100, 29.9, 'B')])
    rec = events.RemoveShortStims().run(rec)
    assert rec.stim.onset.tolist() == [30]


def test_discard_and_keep_stims_on_recordings(rec):
    rec.stim = _stim([(10, 5, 'A'), (30, 5, 'b'), (50, 5, 'C:01'), (70, 5, '1back')])
    job = events.DiscardStims()
    job.options['listOfStims'] = ['B']  # case-insensitive
    r = job.run(copy.deepcopy(rec))
    assert sorted(r.stim.trial_type) == ['1back', 'A', 'C:01']

    with pytest.warns(UserWarning, match='No stims discarded'):
        job.options['listOfStims'] = ['nope']
        job.run(copy.deepcopy(rec))

    job = events.KeepStims()
    job.options['listOfStims'] = ['a', 'C:01']
    r = job.run(copy.deepcopy(rec))
    assert sorted(r.stim.trial_type) == ['A', 'C:01']

    job.options['listOfStims'] = [':01$', '^[012]back$']
    job.options['regex'] = True
    r = job.run(copy.deepcopy(rec))
    assert sorted(r.stim.trial_type) == ['1back', 'C:01']


def test_keep_stims_required_removes_recordings(rec):
    r1, r2 = copy.deepcopy(rec), copy.deepcopy(rec)
    r1.stim = _stim([(10, 5, 'A'), (30, 5, 'B')])
    r2.stim = _stim([(10, 5, 'A')])
    dset = dataset.DataSet([r1, r2])
    job = events.KeepStims()
    job.options['listOfStims'] = ['A', 'B']
    job.options['required'] = True
    with pytest.warns(UserWarning, match='missing required'):
        dset = job.run(dset)
    assert len(dset.dataset) == 1


def test_stim_modules_filter_glm_statistics(rec):
    rec.stim = _stim([(t, 5, c) for t, c in zip(np.arange(10, 280, 20), 'ABC' * 5)])
    job = GLM()
    job.inputName = 'conc'
    job.options['verbose'] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = job.run(rec)
    stats = rec[job.outputName]
    conds = set(stats.list_conditions())
    assert {'HRF A', 'HRF B', 'HRF C'} <= conds
    nuisance = {c for c in conds if not c.startswith('HRF ')}
    assert nuisance  # drift regressors are never removed by the stim modules

    job = events.DiscardStims()
    job.options['listOfStims'] = ['C']
    r = job.run(copy.deepcopy(rec))
    assert set(r[GLM().outputName].list_conditions()) == conds - {'HRF C'}
    s = r[GLM().outputName]
    assert s.covariance.shape == (s.betas.size, s.betas.size)

    job = events.KeepStims()
    job.options['listOfStims'] = ['A']
    r = job.run(copy.deepcopy(rec))
    assert set(r[GLM().outputName].list_conditions()) == nuisance | {'HRF A'}


# --------------------------------------------------------------------------- utilities

def test_keep_and_discard_types(rec):
    job = util.KeepTypes()
    job.options['types'] = ['hbo']
    r = job.run(copy.deepcopy(rec))
    assert r['conc'].chromo.values.tolist() == ['HbO']
    assert r['amp'].sizes['wavelength'] == 2  # no matching type: unchanged

    job = util.DiscardTypes()
    job.options['types'] = [760]
    r = job.run(copy.deepcopy(rec))
    assert r['amp'].wavelength.values.tolist() == [850.0]
    assert r['od'].wavelength.values.tolist() == [850.0]
    assert r['conc'].sizes['chromo'] == 2

    with pytest.warns(UserWarning, match='No types'):
        util.KeepTypes().run(copy.deepcopy(rec))


def test_types_on_connectivity(rec):
    job = resting_state_connectivity()
    job.inputName = 'conc'
    job.options['AR'] = 2
    job.options['short_separation_correction'] = False
    rec = job.run(rec)
    rec['conn'].coef = rec['conn'].coef  # conn after ttest has only same-type pairs
    job = util.DiscardTypes()
    job.options['types'] = ['HbR']
    r = job.run(copy.deepcopy(rec))
    assert set(r['conn'].coef.Type_to) == {'HbO'} and set(r['conn'].coef.Type_from) == {'HbO'}


def test_remove_too_long_distance(rec):
    dist = cedalion.nirs.channel_distances(rec['amp'], rec.geo3d).pint.to('mm').pint.dequantify()
    job = util.RemoveTooLongDistance()
    job.options['min_distance'] = 35
    r = job.run(copy.deepcopy(rec))
    expected = dist.channel.values[dist.values < 35].tolist()
    for key in ('amp', 'od', 'conc'):
        assert r[key].channel.values.tolist() == expected
    used = set(r['amp'].source.values) | set(r['amp'].detector.values)
    assert set(r.geo3d.label.values) == used

    job.options['min_distance'] = 100
    r = job.run(copy.deepcopy(rec))
    assert r['amp'].sizes['channel'] == rec['amp'].sizes['channel']


# --------------------------------------------------------------------------- preprocessing

def test_trim_baseline(rec):
    rec.stim = _stim([(100, 10, 'A'), (150, 20, 'B')])
    job = prep.TrimBaseline()
    job.options['preBaseline'] = 20
    job.options['postBaseline'] = 10
    r = job.run(copy.deepcopy(rec))
    for key in ('amp', 'od', 'conc'):
        t = r[key].time.values
        assert t.min() >= 80 - 1e-9 and t.max() <= 180 + 1e-9
        assert np.isclose(t.min(), 80, atol=0.11) and np.isclose(t.max(), 180, atol=0.11)
    assert r.stim.onset.tolist() == [100, 150]

    job.options['resetTime'] = True
    r = job.run(copy.deepcopy(rec))
    assert r['conc'].time.values[0] == 0
    assert r['conc'].samples.values[0] == 0
    np.testing.assert_allclose(r.stim.onset, [100 - 80, 150 - 80], atol=0.11)

    job.options['preBaseline'] = None
    job.options['postBaseline'] = None
    job.options['resetTime'] = False
    r = job.run(copy.deepcopy(rec))
    assert r['amp'].sizes['time'] == rec['amp'].sizes['time']


def test_trim_baseline_without_stims_warns(rec):
    rec.stim = rec.stim.iloc[0:0]
    with pytest.warns(UserWarning, match='no events'):
        prep.TrimBaseline().run(rec)
