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
import pyBrainAnalyzIR.testing.simEvents as simevents            # noqa: E402

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


# ---------------------------------------------------------------------------
# Simulations with a fixed-effects formula and the truth table
# ---------------------------------------------------------------------------

def test_simdemographics_columns_and_ranges():
    demo = simdata.simdemographics(8, continuous={"age": (20, 60)},
                                   categorical={"group": ["control", "patient"]},
                                   random_seed=1)
    assert len(demo) == 8
    assert demo["age"].between(20, 60).all()
    assert set(demo["group"]) <= {"control", "patient"}


def test_simulated_amplitudes_put_a_response_on_each_condition():
    noise = simdata.ARnoise(random_seed=0)
    nchan = len(noise["amp"].channel)
    pattern = np.arange(nchan) % 2 == 0
    amps = {"A": np.zeros(nchan), "B": pattern * 1.0}
    rec, truth = simdata.Data(noise=noise, stim=simevents.rand_stim_design(ncond=2, random_seed=0),
                              snr=10, amplitudes=amps, random_seed=0)
    truth = np.asarray(truth).reshape(nchan, -1)
    assert truth[:, 0].astype(bool).tolist() == pattern.tolist()

    job = prep.intensity_opticaldensity()
    job = prep.mbll(job)
    job = glm.GLM(job)
    job.set_all_options({"noise_model": "ols", "verbose": False})
    table = job.run(rec)["stats"].table()
    hbo = table[table["Type"] == "HbO"].set_index(["Channel", "Condition"])["T-value"]
    chans = list(map(str, rec["amp"].channel.values))
    active_b = hbo.loc[[(c, "HRF B") for c, on in zip(chans, pattern) if on]].abs()
    quiet_a = hbo.loc[[(c, "HRF A") for c in chans]].abs()
    assert active_b.median() > 3 * quiet_a.median()

    with pytest.raises(ValueError):
        simdata.Data(noise=noise, stim=simevents.rand_stim_design(ncond=1, random_seed=0),
                     amplitudes={"Z": np.ones(nchan)})


@pytest.fixture(scope="module")
def age_simulation():
    rng = np.random.default_rng(4)
    demo = simdata.simdemographics(6, continuous={"age": (20, 60)}, random_seed=rng)
    dset, truth = simdata.simdataset(6, snr=5, demographics=demo, random_seed=rng,
                                     formula="Beta ~ 0 + Condition + Condition:age",
                                     effects={"Condition[HRF A]:age": 0})
    return dset, truth


def test_simdataset_formula_truth_table(age_simulation):
    dset, truth = age_simulation
    assert list(truth.columns[:4]) == ["Channel", "Type", "Condition", "Truth"]
    assert set(truth["Condition"]) == {"Condition[HRF A]", "Condition[HRF A]:age"}
    assert set(truth["Type"]) == {"HbO", "HbR"}
    assert truth.loc[truth["Condition"] == "Condition[HRF A]:age", "Truth"].sum() == 0
    assert truth.loc[truth["Condition"] == "Condition[HRF A]", "Truth"].any()
    assert truth.attrs["formula"] == "Beta ~ 0 + Condition + Condition:age"
    assert len(dset.dataset) == 6
    assert "age" in dset.dataset[0].meta_data


def test_simdataset_formula_terms_match_the_group_model(age_simulation):
    dset, truth = age_simulation
    job = roc.PipelineList([prep.intensity_opticaldensity, prep.mbll, prep.resample,
                            glm.GLM, mixed.MixedEffects])
    job.set_all_options({"Fs": 2, "noise_model": "ols", "verbose": False, "robust": False,
                         "FE_formula": "Beta ~ 0 + Condition + Condition:age"})
    stats = job.run(copy.deepcopy(dset))["groupstats"]
    found = set(map(str, stats.get_pvalues().conditions.values))
    assert set(truth["Condition"]) <= found


def test_simdataset_unknown_effect_term_is_reported():
    demo = simdata.simdemographics(2, continuous={"age": (20, 60)}, random_seed=0)
    with pytest.raises(ValueError, match="unknown terms"):
        simdata.simdataset(2, demographics=demo, formula="Beta ~ 0 + Condition",
                           effects={"Condition[HRF A]:age": 1})


# ---------------------------------------------------------------------------
# ROC scoring rules
# ---------------------------------------------------------------------------

class _StubStats:
    """Minimal statistics object exposing ``get_pvalues()``."""

    def __init__(self, table):
        import xarray as xr
        self._p = xr.DataArray(
            table["P"].to_numpy(), dims="row",
            coords={"channels": ("row", table["Channel"].to_list()),
                    "type": ("row", table["Type"].to_list()),
                    "conditions": ("row", table["Condition"].to_list())})

    def get_pvalues(self):
        return self._p


def _stub_table(conditions, pvalue):
    import pandas as pd
    rows = [(f"S{i}D{i}", ty, cond, pvalue) for cond in conditions
            for i in (1, 2) for ty in ("HbO", "HbR")]
    return pd.DataFrame(rows, columns=["Channel", "Type", "Condition", "P"])


def _stub_truth(conditions):
    import pandas as pd
    rows = [(f"S{i}D{i}", ty, cond, i == 1) for cond in conditions
            for i in (1, 2) for ty in ("HbO", "HbR")]
    return pd.DataFrame(rows, columns=["Channel", "Type", "Condition", "Truth"])


def test_roc_score_rules_for_unsimulated_and_unmodelled_terms():
    model = roc.ChannelROC()
    stats = _StubStats(_stub_table(["Condition[HRF A]", "Condition[HRF A]:age", "Drift 0"], 0.01))
    truth = _stub_truth(["Condition[HRF A]", "Condition[HRF A]:group[T.patient]"])
    scored = model.score(stats, truth)

    # nuisance terms are dropped and the unmodelled group term is ignored
    assert set(scored["Condition"]) == {"Condition[HRF A]", "Condition[HRF A]:age"}
    age = scored[scored["Condition"] == "Condition[HRF A]:age"]
    assert age["Truth"].isna().all() and not age["Simulated"].any()
    main = scored[scored["Condition"] == "Condition[HRF A]"]
    assert main["Simulated"].all() and main["Truth"].sum() == 2


def test_roc_uses_raw_pvalues_when_bh_would_reject_nothing():
    from statsmodels.stats.multitest import multipletests

    condition = "Condition[HRF A]"
    table = _stub_table([condition], .8)
    table["P"] = [.03, .03, .04, .8]
    assert not multipletests(table["P"], method="fdr_bh")[0].any()
    model = roc.ChannelROC()
    scored = model.score(_StubStats(table), _stub_truth([condition]))
    np.testing.assert_array_equal(scored["P"], table["P"])
    model._records.append(scored)
    summary = model.summary().set_index(["Condition", "Type"])
    assert summary.loc[(condition, "HbO"), "FPR (p<0.05)"] == 1.
    assert summary.loc[(condition, "HbR"), "FPR (p<0.05)"] == 0.
    assert summary.loc[(condition, "HbO"), "Sensitivity (p<0.05)"] == 1.
    thresholds, rates = model.type_one_error(condition)["HbO"]
    np.testing.assert_array_equal(thresholds, [.04])
    np.testing.assert_array_equal(rates, [1.])
    _, _, thresholds = model.results(condition)
    assert any(np.isclose(values, .04).any() for values in thresholds)


def test_roc_summary_and_draw_rows():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = roc.ChannelROC()
    model.verbose = False
    rng = np.random.default_rng(0)
    conds = ["Condition[HRF A]", "Condition[HRF A]:age"]
    truth = _stub_truth(["Condition[HRF A]"])
    for _ in range(5):
        table = _stub_table(conds, 0.0)
        table["P"] = rng.uniform(size=len(table))
        table.loc[table["Channel"] == "S1D1", "P"] *= 1e-6
        model._records.append(model.score(_StubStats(table), truth))
        model.iterations += 1

    summary = model.summary()
    assert set(summary["Condition"]) == {roc.ALL_CONDITIONS, *conds}
    age = summary[summary["Condition"] == "Condition[HRF A]:age"]
    assert age["Sensitivity (p<0.05)"].isna().all() and age["FPR (p<0.05)"].notna().all()
    main = summary[summary["Condition"] == "Condition[HRF A]"]
    assert (main["AUC"] > 0.9).all()

    tp, fp, _ = model.results("Condition[HRF A]")
    assert len(tp) == 2
    assert model.type_one_error("Condition[HRF A]:age") is not None

    fig = model.draw()
    assert len(fig.axes) >= 3 * 2   # All + two terms, HbO/HbR (ROC or Type-I panels)
    plt.close(fig)
