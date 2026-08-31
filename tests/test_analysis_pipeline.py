"""End-to-end tests for simulation, first-level GLM, ROC and group models.

These follow the worked examples in ``examples/01_SimulatingData.ipynb``,
``examples/02_PipelineCreation.ipynb``, ``examples/03_ROC_Analysis.ipynb`` and
``examples/GroupAnalysis.ipynb``, but use deliberately small problem sizes so
the suite stays fast enough to run on every push.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cedalion")

import pyBrainAnalyzIR.pipelines.modules.glm as glm            # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.mixedeffects as mixed  # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.preproccessing as prep  # noqa: E402
import pyBrainAnalyzIR.testing.channelROC as roc               # noqa: E402
import pyBrainAnalyzIR.testing.simData as simdata              # noqa: E402
import pyBrainAnalyzIR.testing.simEvents as simevents          # noqa: E402

pytestmark = pytest.mark.requires_cedalion


# ---------------------------------------------------------------------------
# data simulation  (01_SimulatingData)
# ---------------------------------------------------------------------------

def test_ar_noise_has_expected_structure():
    rec = simdata.ARnoise()
    assert "amp" in rec.timeseries
    amp = rec["amp"]
    assert amp.ndim == 3
    assert len(amp.time) > 0
    assert len(amp.channel) > 0


def test_random_stim_design_columns():
    stim = simevents.rand_stim_design()
    for column in ("onset", "duration", "trial_type"):
        assert column in stim.columns
    assert len(stim) > 0


def test_random_stim_design_multiple_conditions():
    stim = simevents.rand_stim_design(ncond=2)
    assert len(set(stim["trial_type"])) == 2


def test_simulated_data_returns_recording_and_truth(simulated_recording):
    rec, truth = simulated_recording
    assert "amp" in rec.timeseries
    # the ground truth marks which channels carry an actual response
    assert truth.data.dtype == bool
    assert truth.data.any(), "simulation produced no active channels"
    assert not truth.data.all(), "simulation produced no inactive channels"


def test_simulated_dataset_has_demographics(simulated_dataset):
    dset = simulated_dataset
    assert len(dset.dataset) == 4
    demo = dset.get_demographics()
    assert "age" in demo.columns
    assert len(demo) == 4


def test_motion_artifact_modifier_changes_the_data():
    np.random.seed(3)
    clean, _ = simdata.Data(snr=25)
    np.random.seed(3)
    noisy, _ = simdata.Data(snr=25, modifiers=[simdata.simMotionArtifact])
    assert not np.allclose(clean["amp"].to_numpy(), noisy["amp"].to_numpy())


# ---------------------------------------------------------------------------
# preprocessing + first level GLM  (02_PipelineCreation)
# ---------------------------------------------------------------------------

def test_preprocessing_pipeline_produces_concentrations(simulated_recording):
    rec, _ = simulated_recording
    job = prep.intensity_opticaldensity()
    job = prep.mbll(job)
    out = job.run(rec)
    assert "od" in out.timeseries
    assert "conc" in out.timeseries


def test_resample_changes_the_sampling_rate(simulated_recording):
    rec, _ = simulated_recording
    job = prep.intensity_opticaldensity()
    job = prep.mbll(job)
    job = prep.resample(job)
    job.options["Fs"] = 2
    out = job.run(rec)

    time = out["conc"].time.to_numpy()
    observed = 1.0 / np.median(np.diff(time))
    assert observed == pytest.approx(2, rel=0.1)


@pytest.fixture()
def first_level_stats(simulated_recording):
    """Run a full first-level GLM once and reuse it across tests."""
    rec, truth = simulated_recording
    job = prep.intensity_opticaldensity()
    job = prep.mbll(job)
    job = prep.resample(job)
    job = glm.GLM(job)
    job.set_all_options({"Fs": 2, "noise_model": "ols"})
    out = job.run(rec)
    return out, truth


def test_glm_produces_statistics(first_level_stats):
    out, _ = first_level_stats
    stats = out["stats"]
    assert stats.betas is not None


def test_glm_pvalues_are_probabilities(first_level_stats):
    out, _ = first_level_stats
    pvals = out["stats"].get_pvalues().to_numpy()
    assert np.all(pvals >= 0) and np.all(pvals <= 1)
    assert np.isfinite(pvals).all()


def test_glm_statistics_table(first_level_stats):
    out, _ = first_level_stats
    table = out["stats"].table()
    for column in ("Channel", "Type", "Condition", "Beta", "P-values"):
        assert column in table.columns
    assert len(table) > 0


def test_glm_recovers_the_simulated_conditions(first_level_stats):
    out, _ = first_level_stats
    conditions = set(np.unique(out["stats"].betas.conditions.to_numpy()))
    assert any("HRF" in str(c) for c in conditions)


def test_glm_ar_irls_runs(simulated_recording):
    """The autoregressive solver is the recommended model for real data."""
    rec, _ = simulated_recording
    job = prep.intensity_opticaldensity()
    job = prep.mbll(job)
    job = prep.resample(job)
    job = glm.GLM(job)
    job.set_all_options({"Fs": 2, "noise_model": "ar_irls", "ar_order": 4})
    out = job.run(rec)
    pvals = out["stats"].get_pvalues().to_numpy()
    assert np.isfinite(pvals).all()


# ---------------------------------------------------------------------------
# ROC analysis  (03_ROC_Analysis)
# ---------------------------------------------------------------------------

def test_pipeline_list_builds_a_chain():
    job = roc.PipelineList([prep.intensity_opticaldensity, prep.mbll, glm.GLM])
    assert job.name == "GLM Model"
    assert job.previous_job.name == "Calculate Modified Beer-Lambert"


def test_channel_roc_defaults():
    model = roc.ChannelROC()
    assert model.iterations == 0
    assert model.pipeline is not None


def test_channel_roc_reset():
    model = roc.ChannelROC()
    model.iterations = 5
    model.reset()
    assert model.iterations == 0


@pytest.fixture()
def roc_model():
    """A tiny two-iteration ROC run (the notebook uses 10)."""
    np.random.seed(5)
    model = roc.ChannelROC()
    model.data_simulation_function = simdata.Data
    model.pipeline = roc.PipelineList([prep.intensity_opticaldensity,
                                       prep.mbll,
                                       prep.resample,
                                       glm.GLM])
    model.data_simulation_args = {"snr": 10}
    model.pipeline_args = {"noise_model": "ols", "Fs": 2}
    model.run(2)
    return model


def test_roc_run_accumulates_iterations(roc_model):
    assert roc_model.iterations == 2


def test_roc_results_are_monotonic_and_bounded(roc_model):
    tp, fp, th = roc_model.results()
    assert len(tp) == len(fp) == len(th)
    for t, f in zip(tp, fp):
        assert np.all((t >= 0) & (t <= 1))
        assert np.all((f >= 0) & (f <= 1))
        # an ROC curve is non-decreasing in both coordinates
        assert np.all(np.diff(t) >= -1e-12)
        assert np.all(np.diff(f) >= -1e-12)


def test_roc_pipeline_args_are_validated():
    model = roc.ChannelROC()
    model.pipeline = roc.PipelineList([prep.intensity_opticaldensity,
                                       prep.mbll,
                                       glm.GLM])
    model.pipeline_args = {"noise_model": "not_a_model"}
    with pytest.raises(ValueError):
        model.run(1)


# ---------------------------------------------------------------------------
# group level / mixed effects  (GroupAnalysis)
# ---------------------------------------------------------------------------

@pytest.fixture()
def group_dataset(simulated_dataset):
    """First-level stats for every file in the dataset."""
    job = prep.intensity_opticaldensity()
    job = prep.mbll(job)
    job = prep.resample(job)
    job = glm.GLM(job)
    job.set_all_options({"Fs": 2, "noise_model": "ols"})
    return job.run(simulated_dataset)


def test_first_level_runs_over_a_dataset(group_dataset):
    for rec in group_dataset.dataset:
        assert rec["stats"] is not None


def test_mixed_effects_produces_group_stats(group_dataset):
    job = mixed.MixedEffects()
    job.options["FE_formula"] = "Beta ~ 0 + Condition"
    out = job.run(group_dataset)

    assert out["groupstats"] is not None
    table = out["groupstats"].table()
    assert len(table) > 0
    for column in ("Channel", "Type", "Condition", "Beta", "P-values"):
        assert column in table.columns


def test_mixed_effects_group_pvalues_are_valid(group_dataset):
    job = mixed.MixedEffects()
    job.options["FE_formula"] = "Beta ~ 0 + Condition"
    out = job.run(group_dataset)

    pvals = out["groupstats"].table()["P-values"].to_numpy()
    finite = pvals[np.isfinite(pvals)]
    assert finite.size > 0
    assert np.all(finite >= 0) and np.all(finite <= 1)


def test_mixed_effects_with_a_demographic_covariate(group_dataset):
    """Reproduces the ``Condition:age`` model from the GroupAnalysis notebook."""
    job = mixed.MixedEffects()
    job.options["FE_formula"] = "Beta ~ 0 + Condition + Condition:age"
    out = job.run(group_dataset)

    conditions = out["groupstats"].table()["Condition"].astype(str)
    assert any("age" in c for c in conditions)


def test_dataset_indexing_and_statistics_roundtrip(group_dataset):
    stats = group_dataset.dataset[0]["stats"]
    assert stats is not None
    assert hasattr(stats, "table")
