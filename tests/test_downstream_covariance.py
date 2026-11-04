"""Contracts for legacy-block ROI/group covariance validation."""
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cedalion")
from pyBrainAnalyzIR.dataclasses.dataset import DataSet  # noqa: E402
from pyBrainAnalyzIR.testing.downstream_covariance import (  # noqa: E402
    SURFACES, inference, paired_subject, floor_task_covariances,
    population_truth, selected_table, summary,
    joint_contrasts, run_first_level, summarize_joint,
)

pytestmark = pytest.mark.requires_cedalion


@pytest.fixture(scope="module")
def surfaces():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return paired_subject(np.random.default_rng(42), 0, 12, "ar_stress")


def test_real_roi_paths_preserve_main_effects_and_propagate_full_covariance(surfaces):
    assert set(surfaces) == set(SURFACES)
    for surface, (records, truths) in surfaces.items():
        old, new = [rec["stats"] for rec in records]
        np.testing.assert_array_equal(old.betas, new.betas)
        assert np.linalg.eigvalsh(new.covariance)[0] > 0
        for term in truths[0]:
            pd.testing.assert_series_equal(truths[0][term], truths[1][term])
        if surface.startswith("roi_stats"):
            for index, local in enumerate((old, new)):
                source = surfaces["channel"][0][index]["stats"]
                w = local.roi_pooling_matrix.values
                np.testing.assert_allclose(local.covariance, w @ source.covariance.values @ w.T)
        else:
            np.testing.assert_array_equal(old.table(), new.table())
        for rec, truth in zip(records, truths):
            scores = inference(rec["stats"], truth["HRF A"], "HRF A")
            shuffled = inference(rec["stats"], truth["HRF A"].sample(frac=1, random_state=5), "HRF A")
            assert scores == shuffled


def test_within_signal_contrast_and_missing_covariance_not_fabricated(surfaces):
    records, truths = surfaces["channel"]
    outputs = []
    for index, rec in enumerate(records):
        stats = rec["stats"].copy()
        stats.ttest(["HRF A-HRF B"])
        outputs.append(stats.table())
        assert np.isfinite(list(inference(stats, truths[index]["HRF A-HRF B"], "HRF A-HRF B").values())).all()
    pd.testing.assert_frame_equal(*outputs)
    stats.covariance.values[0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        inference(stats, truths[0]["HRF A-HRF B"], "HRF A-HRF B")


def test_inverse_variance_roi_truth_uses_condition_specific_pooling_weights(surfaces):
    records, truths = surfaces["roi_stats_iv"]
    source = surfaces["channel"][0][0]["stats"]
    from pyBrainAnalyzIR.testing.downstream_covariance import coefficient_truth
    expected_coefficients = coefficient_truth(source, surfaces["channel"][1][0]["HRF A"])
    for rec, truth in zip(records, truths):
        stats = rec["stats"]
        projected = stats.roi_pooling_matrix.values @ expected_coefficients
        for term in ("HRF A", "HRF B"):
            chosen = stats.betas.conditions.values == term
            np.testing.assert_allclose(truth[term], projected[chosen])
        pd.testing.assert_series_equal(truth["HRF A-HRF B"], truth["HRF A"] - truth["HRF B"])
    assert not np.allclose(truths[0]["HRF B"], .5 * truths[0]["HRF A"], rtol=1e-8, atol=1e-12)


def test_legacy_group_floor_is_study_only_and_preserves_non_task_blocks(surfaces):
    stats = surfaces["channel"][0][0]["stats"].copy()
    indices = np.flatnonzero(stats.betas.conditions.values == "HRF A")
    stats.covariance.values[np.ix_(indices, indices)] = np.ones((8, 8))
    template = surfaces["channel"][0][0]
    from pyBrainAnalyzIR.testing.downstream_covariance import record
    rec = record(stats, template)
    before = stats.covariance.copy()
    assert floor_task_covariances(DataSet([rec])) == 1
    all_task = np.flatnonzero(stats.betas.conditions.values != "Drift 0")
    drift = np.flatnonzero(stats.betas.conditions.values == "Drift 0")
    assert np.linalg.eigvalsh(stats.covariance.values[np.ix_(all_task, all_task)])[0] > 0
    np.testing.assert_array_equal(before.values[np.ix_(drift, drift)],
                                  stats.covariance.values[np.ix_(drift, drift)])


def test_population_truth_uses_selected_group_term_only(surfaces):
    stats = surfaces["roi_stats_equal"][0][1]["stats"].copy()
    stats.betas = stats.betas.assign_coords(conditions=("indices", np.array([
        "Condition[HRF A]:age" if term == "HRF A" else "Condition[HRF B]:age"
        for term in stats.betas.conditions.values
    ])))
    assert len(selected_table(stats, "age")) == 4
    truth = population_truth(stats, "ar_stress", "age")
    assert set(truth) == {0., .25}
    assert population_truth(stats, "ar_global_null", "age").eq(0).all()


def test_summary_averages_subjects_before_bootstrapping_datasets():
    rows = []
    for replicate in range(10):
        for subject in range(3):
            for variant in ["legacy", "repaired"]:
                value = replicate / 20 + .1 * (variant == "repaired")
                rows.append({
                    "regime": "iid", "surface": "roi_stats_equal", "level": "first",
                    "model": "HRF A", "robust": True, "variant": variant,
                    "replicate": replicate, "subject": subject,
                    **{metric: value for metric in (
                        "fdr", "fpr", "power", "sensitivity", "auc", "coverage", "bias", "rmse",
                    )},
                })
    aggregate, entries, paired = summary(rows, 42)
    assert len(aggregate) == 20
    assert all(entry["metrics"]["fdr"]["replicates"] == 10 for entry in entries)
    assert paired[0]["complete_pairs"] == 10
    np.testing.assert_allclose(paired[0]["metrics"]["fdr"]["ci95"], [.1, .1])


def test_all_null_only_summary_retains_undefined_sensitivity_and_auc():
    rows = []
    for replicate in range(3):
        for variant in ("legacy", "repaired"):
            rows.append({
                "regime": "ar_global_null", "surface": "channel", "level": "first",
                "model": "HRF A", "robust": True, "variant": variant,
                "replicate": replicate, "subject": 0, "fdr": .05, "fpr": .05,
                "power": None, "sensitivity": None, "auc": None,
                "coverage": .95, "bias": 0., "rmse": .1,
            })
    aggregate, entries, paired = summary(rows, 42)
    assert len(aggregate) == 6 and len(entries) == 2 and len(paired) == 1
    for entry in entries + paired:
        for metric in ("power", "sensitivity", "auc"):
            assert entry["metrics"][metric]["mean"] is None
            assert entry["metrics"][metric]["replicates"] == 0
        assert entry["metrics"]["fpr"]["replicates"] == 3


def test_exact_null_roi_experiment_has_real_positive_and_negative_roi_truth():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        surfaces = paired_subject(
            np.random.default_rng(42), 6, 12, "correlated_ar", random_effect_sd=0.,
        )
    for surface, (records, truths) in surfaces.items():
        for rec, truth in zip(records, truths):
            score = inference(rec["stats"], truth["HRF A"], "HRF A")
            assert score["null_tests"] > 0
            assert score["alternative_tests"] > 0
            assert score["auc"] is not None
    with pytest.raises(ValueError, match="two independent"):
        run_first_level(replicates=1)


def test_joint_contrasts_use_offdiagonals_and_preserve_missing_variance_errors(surfaces):
    records, truths = surfaces["channel"]
    stats = records[1]["stats"].copy()
    rows = joint_contrasts(stats, truths[1])
    assert rows and {row["model"] for row in rows} == set(truths[1])
    left = np.flatnonzero((stats.betas.conditions.values == "HRF A")
                          & (stats.betas.channels.values == "S1D1"))
    expected_variance = .25 * stats.covariance.values[np.ix_(left, left)].sum()
    row = next(row for row in rows if row["contrast"] == "('S1D1', 'HbO')/('S1D1', 'HbR')/mean"
               and row["model"] == "HRF A")
    assert row["variance"] == pytest.approx(expected_variance)
    stats.covariance.values[:] = np.nan
    with pytest.raises(ValueError, match="joint contrast"):
        joint_contrasts(stats, truths[1])


def test_joint_summary_estimates_variance_across_independent_recordings():
    rows = []
    for replicate in range(10):
        for contrast in ("sum", "difference"):
            rows.append({
                "regime": "iid", "surface": "channel", "variant": "repaired",
                "model": "HRF A", "contrast": contrast, "replicate": replicate,
                "error": replicate / 10, "variance": .1, "coverage": 1., "fpr": 0.,
            })
    entry, = summarize_joint(rows, 42)
    assert entry["replicates"] == 10 and entry["contrasts"] == 2
    assert entry["coverage"]["replicates"] == 10
    assert entry["empirical_to_estimated_variance"] == pytest.approx(np.var(np.arange(10) / 10, ddof=1) / .1)


def test_raw_sensitivity_is_not_bh_power(surfaces, monkeypatch):
    stats = surfaces["channel"][0][1]["stats"]
    table = selected_table(stats, "HRF A").copy()
    table["P-values"] = [.03, .03, .04, .8, .8, .8, .8, .8]
    monkeypatch.setattr(stats, "table", lambda: table)
    truth = pd.Series(
        [1.] * 4 + [0.] * 4, index=pd.MultiIndex.from_frame(table[["Channel", "Type"]]),
    )
    score = inference(stats, truth, "HRF A")
    assert score["sensitivity"] == .75
    assert score["power"] == 0.
