"""Statistics ROI pooling, covariance propagation, and GLM processing order."""
import copy

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("cedalion")
from pyBrainAnalyzIR.dataclasses.dataset import DataSet
from pyBrainAnalyzIR.pipelines.modules import preproccessing as prep
from pyBrainAnalyzIR.pipelines.modules.glm import GLM
from pyBrainAnalyzIR.pipelines.modules.roi import ROIAverage
from pyBrainAnalyzIR.sigproc.roi import roi_average

pytestmark = pytest.mark.requires_cedalion


@pytest.fixture
def analysis_recording(simulated_recording):
    rec, _ = simulated_recording
    rec["amp"] = rec["amp"].isel(channel=slice(0, 4))
    for module, source, target in [
        (prep.resample(), "amp", "amp_rs"),
        (prep.intensity_opticaldensity(), "amp_rs", "od"),
        (prep.mbll(), "od", "conc"),
    ]:
        module.inputName, module.outputName = source, target
        if isinstance(module, prep.resample):
            module.options["Fs"] = 2
        module.run(rec)
    table = pd.DataFrame({
        "source": rec["amp"].source.values[[0, 1, 1, 2, 3]],
        "detector": rec["amp"].detector.values[[0, 1, 1, 2, 3]],
        "name": ["Left", "Left", "Right", "Right", "Right"],
        "weight": [1, 2, 2, 1, 1],
    })
    return rec, table


def _fit(rec, source, target, model="ols"):
    job = GLM()
    job.inputName, job.outputName = source, target
    job.options["noise_model"] = model
    job.run(rec)
    return rec[target]


@pytest.mark.parametrize("weighted", [False, True])
def test_statistics_pooling_matches_full_contrast_algebra(analysis_recording, weighted):
    rec, table = analysis_recording
    stats = _fit(rec, "conc", "stats")
    # Use a dense positive covariance with nonzero cross-condition terms.
    rng = np.random.default_rng(81)
    a = rng.normal(size=stats.covariance.shape)
    stats.covariance = stats.covariance.copy(data=a @ a.T + np.eye(a.shape[0]))
    original = stats.copy()
    pooled = roi_average(stats, table, weighted=weighted)
    projection = pooled.roi_pooling_matrix.values
    np.testing.assert_allclose(pooled.betas, projection @ stats.betas.values)
    np.testing.assert_allclose(pooled.covariance, projection @ stats.covariance.values @ projection.T)
    np.testing.assert_allclose(projection.sum(axis=1), 1)
    assert np.all(np.linalg.eigvalsh(pooled.covariance) >= -1e-10)
    assert pooled.geoROI.region_names == ["Left", "Right"]
    assert pooled.dof == stats.dof
    xr.testing.assert_identical(stats.betas, original.betas)
    xr.testing.assert_identical(stats.covariance, original.covariance)
    for row in range(pooled.betas.size):
        name = pooled.betas.channels.values[row]
        kind = pooled.betas.type.values[row]
        condition = pooled.betas.conditions.values[row]
        candidates = (stats.betas.type.values == kind) & (stats.betas.conditions.values == condition)
        expected = np.zeros(stats.betas.size)
        for member in table[table.name == name].itertuples(index=False):
            selected = candidates & (stats.betas.source.values == member.source) & (
                stats.betas.detector.values == member.detector
            )
            expected[selected] = member.weight
        if weighted:
            expected /= np.diag(stats.covariance)
        expected /= expected.sum()
        np.testing.assert_allclose(projection[row], expected)
    assert np.any(abs(pooled.covariance.values - np.diag(np.diag(pooled.covariance))) > 1e-9)
    # Coefficient order must not affect physical membership matching.
    permutation = rng.permutation(stats.betas.size)
    shuffled = stats.copy()
    shuffled.betas = stats.betas.isel(indices=permutation)
    shuffled.covariance = stats.covariance.isel(rows=permutation, cols=permutation)
    shuffled_pool = roi_average(shuffled, table, weighted=weighted)
    expected = pooled.table().set_index(["Channel", "Type", "Condition"]).sort_index()
    actual = shuffled_pool.table().set_index(["Channel", "Type", "Condition"]).sort_index()
    pd.testing.assert_frame_equal(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("masked", [False, True])
def test_ols_roi_and_glm_order_match_betas_and_covariance(analysis_recording, masked):
    rec, table = analysis_recording
    if masked:
        good = xr.ones_like(rec["conc"].isel(chromo=0, drop=True), dtype=bool)
        good.values[20:25, :] = False
        rec.set_mask("common_bad_samples", good)
    roi_average(rec, table, input_name="conc", output_name="conc_roi", weighted=False)
    before = _fit(rec, "conc_roi", "roi_then_glm")
    channel = _fit(rec, "conc", "channel_glm")
    roi_average(
        rec, table, input_name="channel_glm", output_name="glm_then_roi",
        roi_name="after", weighted=False,
    )
    after = rec["glm_then_roi"]
    np.testing.assert_allclose(before.betas, after.betas, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(before.covariance, after.covariance, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(before.table()["StdErr"], after.table()["StdErr"], rtol=1e-9)
    assert channel.geoROI is None
    assert rec.aux_obj["roi_definitions"]["after"] is after.geoROI


def test_statistics_missing_channels_renormalize_without_nan_contamination(analysis_recording):
    rec, table = analysis_recording
    stats = _fit(rec, "conc", "stats")
    first = stats.betas.channels.values[0]
    missing = stats.betas.channels.values == first
    stats.betas.values[missing] = np.nan
    stats.covariance.values[missing, :] = np.nan
    stats.covariance.values[:, missing] = np.nan
    with pytest.warns(UserWarning, match="Missing channel"):
        pooled = roi_average(stats, table)
    assert np.isfinite(pooled.betas).all()
    assert np.isfinite(pooled.covariance).all()
    left = pooled.betas.channels.values == "Left"
    assert np.all(pooled.roi_pooling_matrix.values[left].sum(axis=1) == 1)
    other = stats.betas.channels.values == rec["amp"].channel.values[1]
    stats.betas.values[other] = np.nan
    stats.covariance.values[other, :] = np.nan
    stats.covariance.values[:, other] = np.nan
    with pytest.warns(UserWarning, match="Missing channel"):
        pooled = roi_average(stats, table)
    assert np.isnan(pooled.betas.values[left]).all()
    assert np.isnan(pooled.covariance.values[left]).all()
    assert np.isfinite(pooled.betas.values[~left]).all()
    assert np.isfinite(pooled.covariance.values[np.ix_(~left, ~left)]).all()


def test_statistics_validation_and_preserving_existing_definitions(analysis_recording):
    rec, table = analysis_recording
    stats = _fit(rec, "conc", "stats")
    legacy = stats.copy()
    legacy.betas = legacy.betas.drop_vars(["source", "detector"])
    current = roi_average(stats, table, weighted=False)
    old = roi_average(legacy, table, weighted=False)
    xr.testing.assert_equal(current.betas, old.betas)
    xr.testing.assert_equal(current.covariance, old.covariance)
    roi_average(rec, table, input_name="stats", output_name="stats_roi", weighted=False)
    with pytest.raises(ValueError, match="already exists"):
        roi_average(rec, table, input_name="stats", output_name="other")
    with pytest.raises(ValueError, match="another ROI"):
        roi_average(current, table)
    with pytest.raises(ValueError, match="used by other"):
        roi_average(
            rec, table, input_name="conc", output_name="new_roi", overwrite=True,
        )
    bad = stats.copy()
    bad.covariance.values[0, 0] = 0
    with pytest.raises(ValueError, match="positive coefficient variance"):
        roi_average(bad, table)
    # A missing off-diagonal covariance is not silently interpreted as zero.
    bad = stats.copy()
    bad.covariance.values[0, 1] = np.nan
    with pytest.warns(UserWarning, match="Nonfinite covariance"):
        result = roi_average(bad, table, weighted=False)
    assert np.isnan(result.covariance).any()
    bad = stats.copy()
    bad.covariance = stats.covariance.isel(rows=slice(1, None))
    with pytest.raises(ValueError, match="matching covariance"):
        roi_average(bad, table)


def test_ols_covariance_matches_multivariate_ols_reference():
    import statsmodels.api as sm
    from pyBrainAnalyzIR.dataclasses.statistics import Statistics

    rng = np.random.default_rng(41)
    design = pd.DataFrame({"constant": np.ones(100), "task": rng.normal(size=100)})
    noise = rng.normal(size=(100, 3)) @ np.array([[1, 0.2, 2], [0.5, 2, 0], [0.1, 0.5, 0.3]])
    y = design.values @ rng.normal(size=(2, 3)) + noise
    results = np.empty((3, 1), dtype=object)
    for channel in range(3):
        results[channel, 0] = sm.OLS(y[:, channel], design).fit()
    model = xr.DataArray(
        results, dims=("channel", "chromo"),
        coords={"channel": ["S1D1", "S2D1", "S2D2"], "chromo": ["HbO"]},
    )
    stats = Statistics(model)
    residual = np.column_stack([result.resid for result in results[:, 0]])
    expected = np.kron(
        residual.T @ residual / (design.shape[0] - design.shape[1]),
        np.linalg.inv(design.values.T @ design.values),
    )
    np.testing.assert_allclose(stats.covariance, expected, rtol=1e-12, atol=1e-14)
    for index, result in enumerate(results[:, 0]):
        np.testing.assert_allclose(
            stats.covariance.values[index * 2:(index + 1) * 2, index * 2:(index + 1) * 2],
            result.cov_params(),
        )


def test_statistics_pipeline_dataset_group_and_drawing(analysis_recording):
    import matplotlib.pyplot as plt
    from pyBrainAnalyzIR.pipelines.modules.mixedeffects import MixedEffects

    rec, table = analysis_recording
    recordings = []
    for i in range(5):
        participant = copy.deepcopy(rec)
        participant.meta_data["subject"] = f"person{i}"
        participant["conc"] = participant["conc"] * (
            1 + np.random.default_rng(i).normal(0, 0.01, participant["conc"].shape)
        )
        _fit(participant, "conc", "stats")
        recordings.append(participant)
    dataset = DataSet(recordings)
    job = ROIAverage()
    job.inputName, job.outputName = "stats", "stats_roi"
    job.options.update(roi_table=table, weighted=False, roi_name="statistics")
    job.run(dataset)
    for recording in recordings:
        assert recording["stats_roi"].geoROI.region_names == ["Left", "Right"]
        assert recording["stats"].geoROI is None
    group = MixedEffects()
    group.inputName = "stats_roi"
    group.options["robust"] = False
    group.run(dataset)
    assert dataset["groupstats"].geoROI.region_names == ["Left", "Right"]
    fig = dataset["groupstats"].draw()
    assert fig.axes[0].get_title() == "ROI membership"
    plt.close(fig)
    dataset["channel_group"] = recordings[0]["stats"].copy()
    job.inputName, job.outputName = "channel_group", "pooled_group"
    job.run(dataset)
    assert dataset["pooled_group"].geoROI.region_names == ["Left", "Right"]


@pytest.mark.parametrize("scope", ["recording", "dataset", "group"])
@pytest.mark.parametrize("weighted", [False, True])
def test_named_roi_module_pools_full_statistics_covariance(analysis_recording, scope, weighted):
    rec, table = analysis_recording
    stats = _fit(rec, "conc", "stats")
    tables = {
        name: rows.drop(columns="name")
        for name, rows in table.groupby("name", sort=False)
    }
    expected = roi_average(stats, table, roi_name="pooled_regions", weighted=weighted)
    data = rec if scope == "recording" else DataSet([rec])
    if scope == "group":
        data["channel_group"] = stats
    module = ROIAverage(roi_tables=tables)
    module.inputName = "channel_group" if scope == "group" else "stats"
    module.outputName = "pooled_regions"
    module.options["weighted"] = weighted
    module.run(data)
    result = data["pooled_regions"] if scope == "group" else rec["pooled_regions"]
    assert result.geoROI.region_names == ["Left", "Right"]
    xr.testing.assert_allclose(result.betas, expected.betas)
    xr.testing.assert_allclose(result.covariance, expected.covariance)
    assert not np.allclose(
        result.covariance.values[:result.betas.size // 2, result.betas.size // 2:], 0,
        rtol=0, atol=1e-30,
    )
    if scope != "group":
        assert "stats_roi_Left" not in rec.timeseries
        assert "stats_roi_Right" not in rec.timeseries


@pytest.mark.parametrize("alias", ["last", "prev", "previous"])
@pytest.mark.parametrize("dataset_scope", [False, True])
def test_roi_then_glm_resolves_previous_output(analysis_recording, alias, dataset_scope):
    rec, table = analysis_recording
    # Keep an unrelated newer key to catch reliance on dictionary insertion order.
    rec["pooled"] = rec["conc"].copy()
    rec["unrelated"] = rec["conc"].copy()
    tables = {name: rows for name, rows in table.groupby("name", sort=False)}
    roi = ROIAverage(roi_tables=tables)
    roi.inputName, roi.outputName = "conc", "pooled"
    roi.options.update(weighted=False, overwrite=True)
    job = GLM(roi)
    job.inputName = alias
    job.options.update(noise_model="ols", verbose=False)
    data = DataSet([rec]) if dataset_scope else rec
    assert job.run(data) is data
    result = rec["stats"]
    assert result.geoROI.region_names == ["Left", "Right"]
    assert set(result.betas.channels.values) == {"Left", "Right"}
    expected = _fit(rec, "pooled", "explicit_stats")
    xr.testing.assert_allclose(result.betas, expected.betas)
    xr.testing.assert_allclose(result.covariance, expected.covariance)
    assert job.inputName == alias


@pytest.mark.requires_qt
@pytest.mark.parametrize("alias", ["last", "prev", "previous"])
def test_pipeline_manager_roi_then_glm_uses_roi_input(analysis_recording, qt_app, alias):
    from pyBrainAnalyzIR.vis.pipeline_manager import PipelineManagerDialog

    rec, table = analysis_recording
    tables = {name: rows for name, rows in table.groupby("name", sort=False)}
    roi = ROIAverage(roi_tables=tables)
    roi.inputName, roi.outputName = "conc", "selected_regions"
    model = GLM()
    model.inputName = alias
    model.options.update(noise_model="ols", verbose=False)
    dialog = PipelineManagerDialog([roi, model], data=rec, roi_tables=tables)
    assert dialog._compute_pipeline_validity() == [True, True]
    pipeline = dialog.build_pipeline()
    pipeline.run(rec)
    assert set(rec["stats"].betas.channels.values) == {"Left", "Right"}
    assert rec["stats"].geoROI.region_names == ["Left", "Right"]
    assert pipeline.inputName == alias


def test_ar_irls_order_is_not_assumed_linear(analysis_recording):
    rec, table = analysis_recording
    roi_average(rec, table, input_name="conc", output_name="conc_roi", weighted=False)
    before = _fit(rec, "conc_roi", "roi_then_glm", "ar_irls")
    channels = _fit(rec, "conc", "channel_glm", "ar_irls")
    after = roi_average(channels, table, weighted=False)
    assert np.isfinite(before.betas).all()
    assert np.isfinite(after.betas).all()
    assert not np.allclose(before.betas, after.betas, rtol=1e-7, atol=1e-12)
    # Each order refits robust weights and AR whitening on different signals,
    # so their relative beta difference has no fixed upper bound.
