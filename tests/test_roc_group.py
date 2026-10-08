"""Regression tests for channel ROC analysis at the first and group level.

Covers the bugs found while adding the group-level ROC example to
``examples/03_ROC_Analysis.ipynb``:

* first-level ``Statistics`` labels did not follow the order of the betas;
* ``ChannelROC`` matched p-values to the truth by position and only knew the
  first-level ``'HRF A'`` condition name;
* ``MixedEffects`` shared one random effect across all subjects, applied the
  whitening matrix in the wrong row order and let the drift regressor dominate
  the pooled group variance.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("cedalion")

import pyBrainAnalyzIR.pipelines.modules.glm as glm              # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.mixedeffects as mixed   # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.preproccessing as prep  # noqa: E402
import pyBrainAnalyzIR.testing.channelROC as roc                 # noqa: E402
import pyBrainAnalyzIR.testing.simData as simdata                # noqa: E402

pytestmark = pytest.mark.requires_cedalion


def _first_level(noise_model="ols"):
    job = roc.PipelineList([prep.intensity_opticaldensity, prep.mbll,
                            prep.resample, glm.GLM])
    job.set_all_options({"Fs": 2, "noise_model": noise_model, "ar_order": 8,
                         "verbose": False})
    return job


@pytest.fixture(scope="module")
def first_level_dataset():
    np.random.seed(3)
    dset, truth = simdata.simdataset(num_files=4, snr=10)
    return _first_level("ar_irls").run(dset), truth


# ---------------------------------------------------------------------------
# Statistics labels
# ---------------------------------------------------------------------------

def test_first_level_labels_follow_the_beta_order(simulated_recording):
    rec, _ = simulated_recording
    job = prep.intensity_opticaldensity()
    job = prep.mbll(job)
    job = glm.GLM(job)
    job.set_all_options({"noise_model": "ols", "verbose": False})
    stats = job.run(rec)["stats"]

    table = stats.table().set_index(["Channel", "Type", "Condition"])
    n_types = len(np.unique(stats.betas.type))
    n_conds = len(stats.list_conditions())
    # every (channel, type, condition) label is unique ...
    assert table.index.is_unique
    # ... and the betas are flattened channel-major, then type, then condition
    labels = list(zip(stats.betas.channels.values, stats.betas.type.values,
                      stats.betas.conditions.values))
    block = labels[:n_types * n_conds]
    assert len({ch for ch, _, _ in block}) == 1
    assert [ty for _, ty, _ in block[::n_conds]] == sorted({ty for _, ty, _ in block})
    # the covariance carries the same labels as the betas
    assert list(stats.covariance.channels_rows.values) == list(stats.betas.channels.values)
    assert list(stats.covariance.type_rows.values) == list(stats.betas.type.values)


def test_first_level_labels_match_the_simulated_truth(first_level_dataset):
    """Channels with a simulated response must have larger |t| on average."""
    dset, truth = first_level_dataset
    t_true, t_false = [], []
    for rec in dset.dataset:
        tbl = rec["stats"].table()
        tbl = tbl[tbl["Condition"] == "HRF A"]
        for ch, ty, tval in zip(tbl["Channel"], tbl["Type"], tbl["T-value"]):
            (t_true if bool(truth.sel(channel=ch, chromo=ty)) else t_false).append(abs(tval))
    assert np.mean(t_true) > 2 * np.mean(t_false)


# ---------------------------------------------------------------------------
# ChannelROC condition matching / alignment
# ---------------------------------------------------------------------------

def test_roc_aligns_pvalues_by_label(first_level_dataset):
    dset, truth = first_level_dataset
    stats = dset.dataset[0]["stats"]
    model = roc.ChannelROC()
    pvals = model._condition_pvalues(stats, truth)
    assert pvals.shape == truth.shape

    shuffled = truth.isel(channel=np.random.permutation(truth.sizes["channel"]))
    shuffled = shuffled.transpose("chromo", "channel")
    pvals_shuffled = model._condition_pvalues(stats, shuffled)
    for i, ch in enumerate(shuffled.channel.values):
        for j, ty in enumerate(shuffled.chromo.values):
            k = list(truth.channel.values).index(ch)
            m = list(truth.chromo.values).index(ty)
            assert pvals_shuffled[j, i] == pvals[k, m]


def test_roc_unknown_condition_is_reported(first_level_dataset):
    dset, truth = first_level_dataset
    model = roc.ChannelROC()
    model.condition = "B"
    with pytest.raises(ValueError, match="not found"):
        model._condition_pvalues(dset.dataset[0]["stats"], truth)


# ---------------------------------------------------------------------------
# MixedEffects fixes
# ---------------------------------------------------------------------------

def test_group_model_drops_nuisance_conditions(first_level_dataset):
    dset, _ = first_level_dataset
    out = mixed.MixedEffects().run(copy.deepcopy(dset))
    conds = set(map(str, out["groupstats"].list_conditions()))
    assert conds == {"Condition[HRF A]"}

    job = mixed.MixedEffects()
    job.options["include_nuisance"] = True
    out = job.run(copy.deepcopy(dset))
    assert "Condition[Drift 0]" in set(map(str, out["groupstats"].list_conditions()))


@pytest.mark.parametrize("weighted", [False, True])
def test_group_betas_track_the_subject_average(first_level_dataset, weighted):
    dset, _ = first_level_dataset
    job = mixed.MixedEffects()
    job.options["weighted"] = weighted
    job.options["robust"] = False
    group = job.run(copy.deepcopy(dset))["groupstats"].table()
    group = group.set_index(["Channel", "Type"])["Beta"]

    subj = [r["stats"].table() for r in dset.dataset]
    subj = [t[t["Condition"] == "HRF A"].set_index(["Channel", "Type"])["Beta"] for t in subj]
    mean = sum(subj) / len(subj)
    spread = np.std([s.to_numpy() for s in subj], axis=0).mean()

    err = np.abs(group.loc[mean.index].to_numpy() - mean.to_numpy())
    # unweighted equals the plain mean; weighted is a precision-weighted mean
    tol = 1e-6 + 1e-3 * spread if not weighted else spread
    assert err.mean() < tol


def test_group_dof_is_positive(first_level_dataset):
    dset, _ = first_level_dataset
    group = mixed.MixedEffects().run(copy.deepcopy(dset))["groupstats"]
    assert group.dof == len(dset.dataset) - 1
    assert np.isfinite(group.get_pvalues().to_numpy()).all()


# ---------------------------------------------------------------------------
# group-level ROC
# ---------------------------------------------------------------------------

def test_group_level_roc_with_simdataset():
    np.random.seed(4)
    model = roc.ChannelROC()
    model.data_simulation_function = simdata.simdataset
    model.data_simulation_args = {"num_files": 3, "snr": 5}
    model.pipeline = roc.PipelineList([prep.intensity_opticaldensity, prep.mbll,
                                       prep.resample, glm.GLM, mixed.MixedEffects])
    model.pipeline_args = {"Fs": 2, "noise_model": "ols", "verbose": False}
    model.run(2)

    assert model.iterations == 2
    assert model._stats_name() == "groupstats"
    tp, fp, th = model.results()
    assert len(tp) == 2          # HbO and HbR
    for t, f in zip(tp, fp):
        assert np.all((t >= 0) & (t <= 1)) and np.all((f >= 0) & (f <= 1))
    assert all(np.isfinite(p).all() for p in model._pvals)
