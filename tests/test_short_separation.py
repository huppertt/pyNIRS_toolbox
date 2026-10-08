"""Tests for the short-separation filter, the utilities modules and the connectivity
short-separation correction flag."""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cedalion")

import pyBrainAnalyzIR.dataclasses.dataset as dataset  # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.preproccessing as prep  # noqa: E402
import pyBrainAnalyzIR.testing.simData as simData  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.connectivity import (  # noqa: E402
    hyperscanning, resting_state_connectivity)
from pyBrainAnalyzIR.pipelines.modules.filters import ShortSeparationFilter  # noqa: E402
from pyBrainAnalyzIR.sigproc.short_separation import short_channel_mask  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.utilities import (  # noqa: E402
    DiscardData, RemoveShortSeparationChannels)

pytestmark = pytest.mark.requires_cedalion

LONG = 16
SHORT = 9


@pytest.fixture(scope="module")
def conc_rec():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(0)
        rec, _ = simData.connectivity_shortsep()
        job = prep.resample()
        job.options['Fs'] = 2
        job = prep.intensity_opticaldensity(job)
        job = prep.mbll(job)
        return job.run(rec)


def _conn(rec, correction, **kw):
    job = resting_state_connectivity()
    job.inputName = 'conc'
    job.options['AR'] = 4
    job.options['short_separation_correction'] = correction
    for k, v in kw.items():
        job.options[k] = v
    return job.run(rec)


def test_filter_flattens_short_channels_and_keeps_shape(conc_rec):
    rec = copy.deepcopy(conc_rec)
    job = ShortSeparationFilter()
    job.inputName = 'conc'
    job.outputName = 'conc_ss'
    job.options['remove_short_channels'] = False
    rec = job.run(rec)
    out = rec['conc_ss']
    assert out.dims == conc_rec['conc'].dims
    assert out.sizes == conc_rec['conc'].sizes
    assert out.pint.units == conc_rec['conc'].pint.units
    std = out.pint.dequantify().std('time')
    short = std.channel.str.contains('D1[0-7]|D9$')
    assert float(std.sel(channel=short).max()) < 1e-10 * float(std.sel(channel=~short).min())


def test_filter_removes_short_channels_and_optodes(conc_rec):
    rec = copy.deepcopy(conc_rec)
    job = ShortSeparationFilter()
    job.inputName = 'conc'
    rec = job.run(rec)
    for key in ('amp', 'od', 'conc'):
        assert rec[key].sizes['channel'] == LONG
    assert rec.geo3d.sizes['label'] == 9 + 8
    assert rec.geo2d.sizes['label'] == 9 + 8
    assert 'D9' not in rec.geo3d.label.values


def test_remove_short_separation_channels_module(conc_rec):
    rec = RemoveShortSeparationChannels().run(copy.deepcopy(conc_rec))
    assert all(rec[k].sizes['channel'] == LONG for k in rec.timeseries)
    assert rec.geo3d.sizes['label'] == 17
    # running again on data without short channels is a no-op
    rec = RemoveShortSeparationChannels().run(rec)
    assert rec['conc'].sizes['channel'] == LONG


def test_discard_data(conc_rec):
    rec = copy.deepcopy(conc_rec)
    job = DiscardData()
    job.remove_type = 'od'
    assert job.options['remove_type'] == 'od'
    rec = job.run(rec)
    assert list(rec.timeseries) == ['amp', 'conc']
    job.remove_type = 'last'
    rec = job.run(rec)
    assert list(rec.timeseries) == ['amp']


def test_connectivity_correction_matches_filter_then_connectivity(conc_rec):
    rec_a = _conn(copy.deepcopy(conc_rec), True)
    # The recording is unchanged apart from the connectivity output
    assert list(rec_a.timeseries) == ['amp', 'od', 'conc', 'conn']
    assert rec_a['conc'].sizes['channel'] == LONG + SHORT

    rec_b = ShortSeparationFilter().run(copy.deepcopy(conc_rec))  # 'last' == 'conc'
    rec_b = _conn(rec_b, False)

    a, b = rec_a['conn'].coef, rec_b['conn'].coef
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert set(a.Channel_to) == set(rec_b['conc'].channel.values)


def test_connectivity_without_short_channels_warns(conc_rec):
    rec = RemoveShortSeparationChannels().run(copy.deepcopy(conc_rec))
    with pytest.warns(UserWarning, match="no short-separation"):
        rec_a = _conn(copy.deepcopy(rec), True)
    rec_b = _conn(copy.deepcopy(rec), False)
    pd.testing.assert_frame_equal(rec_a['conn'].coef, rec_b['conn'].coef)


def test_hyperscanning_correction_matches_filter_then_hyperscanning(conc_rec):
    def make():
        dset = dataset.DataSet()
        for _ in range(2):
            dset.import_data(copy.deepcopy(conc_rec))
        dset.add_demographics_by_index(pd.DataFrame({'subject': ['A', 'B'], 'pair': ['d1', 'd1']}))
        return dset

    def hyper(dset, correction):
        job = hyperscanning()
        job.inputName = 'conc'
        job.options['AR'] = 4
        job.options['short_separation_correction'] = correction
        return job.run(dset)

    dset_a = hyper(make(), True)
    assert list(dset_a.dataset[0].timeseries) == ['amp', 'od', 'conc', 'hyperconn']
    dset_b = hyper(ShortSeparationFilter().run(make()), False)
    pd.testing.assert_frame_equal(dset_a.dataset[0]['hyperconn'].coef,
                                  dset_b.dataset[0]['hyperconn'].coef)


def test_utilities_modules_are_discovered_by_the_pipeline_manager():
    pytest.importorskip("PySide6")
    from pyBrainAnalyzIR.vis.pipeline_manager import _discover_modules
    specs = {(s.package_name, s.name) for s in _discover_modules()}
    assert ('utilities', 'Discard Data') in specs
    assert ('utilities', 'Remove Short Separation Channels') in specs
    assert ('filters', 'Short Separation Filter') in specs


def test_warped_shortsep_probe_keeps_all_long_channels():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec, truth = simData.connectivity_shortsep(warped=True, t=np.arange(200) * 0.1)
    assert truth.sizes['channel_to'] == LONG
    assert int(short_channel_mask(rec['amp'], rec.geo3d).sum()) == SHORT
    assert not np.allclose(rec.geo2d.pint.dequantify().values[:, 1], rec.geo2d.pint.dequantify().values[0, 1])
