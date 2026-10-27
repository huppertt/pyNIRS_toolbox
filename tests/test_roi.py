"""ROI averaging and downstream pipeline integration."""
import copy

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("cedalion")
from pyBrainAnalyzIR.dataclasses.dataset import DataSet  # noqa: E402
from pyBrainAnalyzIR.dataclasses.roi import get_geo_roi, quality_masks  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules import preproccessing as prep  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.filters import bandpass_filter  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.motion_correction import TDDR  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.glm import GLM  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.roi import ROIAverage  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.mixedeffects import MixedEffects  # noqa: E402
from pyBrainAnalyzIR.sigproc.roi import roi_average  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


@pytest.fixture
def roi_table(simulated_recording):
    rec, _ = simulated_recording
    return pd.DataFrame({
        "source": rec["amp"].source.values[:4],
        "detector": rec["amp"].detector.values[:4],
        "weight": [1.0, 2.0, 1.0, 1.0],
        "name": ["Left", "Left", "Right", "Right"],
    })


@pytest.mark.parametrize("weighted", [True, False])
def test_roi_average_exact_formula_and_independent_definitions(
    simulated_recording, roi_table, weighted
):
    rec, _ = simulated_recording
    original = rec["amp"].copy(deep=True)
    geometry = rec.geo3d.copy(deep=True)
    roi_average(rec, roi_table, weighted=weighted)
    raw = original.pint.magnitude[:, :2, :]
    weights = np.array([1.0, 2.0])[:, None]
    if weighted:
        weights = weights / raw.var(axis=0)
    expected = (raw * weights).sum(axis=1) / weights.sum(axis=0)
    np.testing.assert_allclose(rec["amp_roi"].pint.magnitude[:, 0, :], expected)
    assert rec["amp_roi"].pint.units == original.pint.units
    assert rec["amp_roi"].channel.values.tolist() == ["Left", "Right"]
    assert rec["amp"].identical(original)
    assert rec.geo3d.identical(geometry)
    roi_average(
        rec, roi_table, output_name="amp_other", roi_name="other", weighted=False
    )
    assert set(rec.aux_obj["roi_definitions"]) == {"default", "other"}
    assert get_geo_roi(rec, rec["amp_other"]).name == "other"
    with pytest.raises(ValueError, match="already exists"):
        roi_average(rec, roi_table)


def test_roi_average_supports_signed_membership_weights(simulated_recording):
    rec, _ = simulated_recording
    data = rec["amp"].copy(deep=True)
    table = pd.DataFrame({
        "source": data.source.values[:2],
        "detector": data.detector.values[:2],
        "weight": [1.0, -1.0],
        "name": ["Difference", "Difference"],
    })
    roi_average(rec, table, weighted=False)
    expected = (data.pint.magnitude[:, 0, :] - data.pint.magnitude[:, 1, :]) / 2
    np.testing.assert_allclose(rec["amp_roi"].pint.magnitude[:, 0, :], expected)


def test_roi_masks_exclude_bad_channels_and_do_not_touch_channel_masks(
    simulated_recording, roi_table
):
    rec, _ = simulated_recording
    mask = xr.ones_like(rec["amp"].isel(wavelength=0, drop=True), dtype=bool)
    mask.values[10, 0] = False
    mask.values[11, :2] = False
    rec.set_mask("quality", mask)
    roi_average(rec, roi_table, weighted=False)
    np.testing.assert_allclose(
        rec["amp_roi"].pint.magnitude[10, 0], rec["amp"].pint.magnitude[10, 1]
    )
    assert np.isnan(rec["amp_roi"].pint.magnitude[11, 0]).all()
    assert not quality_masks(rec, rec["amp_roi"])["roi_valid"].values[11, 0]
    assert rec.masks["quality"].identical(mask)


def test_roi_pipeline_through_glm(simulated_recording, roi_table):
    rec, _ = simulated_recording
    original = rec["amp"].copy(deep=True)
    channel_mask = xr.ones_like(
        rec["amp"].isel(wavelength=0, drop=True), dtype=bool
    )
    rec.set_mask("channel_quality", channel_mask)
    roi = ROIAverage()
    roi.options["roi_table"] = roi_table
    rec = roi.run(rec)
    steps = [
        (prep.resample(), "amp_roi", "amp_roi_rs"),
        (prep.intensity_opticaldensity(), "amp_roi_rs", "od_roi"),
        (bandpass_filter(), "od_roi", "od_roi_bp"),
        (TDDR(), "od_roi_bp", "od_roi_tddr"),
        (prep.mbll(), "od_roi_tddr", "conc_roi"),
        (GLM(), "conc_roi", "stats_roi"),
    ]
    for step, input_name, output_name in steps:
        step.inputName, step.outputName = input_name, output_name
        if isinstance(step, GLM):
            step.options["noise_model"] = "ols"
        rec = step.run(rec)
    assert rec["stats_roi"].geoROI.region_names == ["Left", "Right"]
    assert set(rec["stats_roi"].table()["Channel"]) == {"Left", "Right"}
    assert np.isfinite(rec["stats_roi"].betas).all()
    assert rec["amp"].identical(original)
    assert rec.masks["channel_quality"].identical(channel_mask)
    assert quality_masks(rec, rec["amp_roi"])["roi_valid"].sizes["time"] == 3000
    assert quality_masks(rec, rec["amp_roi_rs"])["roi_valid"].sizes["time"] < 3000


def test_roi_dataset_pipeline(simulated_recording, roi_table):
    rec, _ = simulated_recording
    dset = DataSet([rec, copy.deepcopy(rec)])
    roi_average(dset, roi_table)
    step = prep.intensity_opticaldensity()
    step.inputName, step.outputName = "amp_roi", "od_roi"
    assert step.run(dset) is dset
    for recording in dset.dataset:
        assert recording["od_roi"].attrs["roi_definition"] == "default"


def test_roi_glm_to_mixed_effects(simulated_recording, roi_table):
    rec, _ = simulated_recording
    recordings = []
    for index in range(5):
        recording = copy.deepcopy(rec)
        recording["amp"] = recording["amp"] * (
            1 + np.random.default_rng(index).normal(
                0, 0.001, recording["amp"].shape
            )
        )
        recording.meta_data["subject"] = f"subject{index}"
        recordings.append(recording)
    dset = DataSet(recordings)
    roi_average(dset, roi_table)
    for module, source, target in [
        (prep.intensity_opticaldensity(), "amp_roi", "od_roi"),
        (prep.mbll(), "od_roi", "conc_roi"),
        (GLM(), "conc_roi", "stats_roi"),
    ]:
        module.inputName, module.outputName = source, target
        if isinstance(module, GLM):
            module.options["noise_model"] = "ols"
        module.run(dset)
    group = MixedEffects()
    group.inputName = "stats_roi"
    group.options["robust"] = False
    group.run(dset)
    assert dset["groupstats"].geoROI.region_names == ["Left", "Right"]
    assert set(dset["groupstats"].table()["Channel"]) == {"Left", "Right"}
    assert np.isfinite(dset["groupstats"].betas).all()
    import matplotlib.pyplot as plt
    figure = dset["groupstats"].draw()
    assert figure.axes[0].get_title() == "ROI membership"
    assert all(not button.active for button in figure._roi_buttons)
    plt.close(figure)


def test_roi_mbll_inverse_and_concentration_average(simulated_recording, roi_table):
    rec, _ = simulated_recording
    prep.mbll(prep.intensity_opticaldensity()).run(rec)
    roi_average(rec, roi_table, input_name="conc", output_name="conc_roi")
    back = prep.conc2od()
    back.inputName, back.outputName = "conc_roi", "od_roi"
    back.run(rec)
    forward = prep.mbll()
    forward.inputName, forward.outputName = "od_roi", "conc_roundtrip"
    forward.run(rec)
    xr.testing.assert_allclose(
        rec["conc_roundtrip"].pint.dequantify().transpose(*rec["conc_roi"].dims),
        rec["conc_roi"].pint.dequantify(),
    )


def test_roi_fully_masked_weighted_channel(simulated_recording, roi_table):
    rec, _ = simulated_recording
    mask = xr.ones_like(rec["amp"].isel(wavelength=0, drop=True), dtype=bool)
    mask.values[:, 0] = False
    rec.set_mask("quality", mask)
    with pytest.warns(UserWarning, match="Entirely masked"):
        roi_average(rec, roi_table)
    np.testing.assert_allclose(
        rec["amp_roi"].pint.magnitude[:, 0], rec["amp"].pint.magnitude[:, 1]
    )


def test_roi_named_plot_and_stats_plot(simulated_recording, roi_table, monkeypatch):
    import matplotlib.pyplot as plt
    from pyBrainAnalyzIR.vis.plot_nirs_inline import plot

    rec, _ = simulated_recording
    roi_average(rec, roi_table)
    monkeypatch.setattr(plt, "show", lambda: None)
    plot(rec, type="amp_roi")
    for step, source, target in [
        (prep.intensity_opticaldensity(), "amp_roi", "od_roi"),
        (prep.mbll(), "od_roi", "conc_roi"),
        (GLM(), "conc_roi", "stats_roi"),
    ]:
        step.inputName, step.outputName = source, target
        if isinstance(step, GLM):
            step.options["noise_model"] = "ols"
        step.run(rec)
    rec["stats_roi"].draw(condnames=["HRF A"])
    assert [tick.get_text() for tick in plt.gcf().axes[1].get_xticklabels()] == [
        "Left", "Right"
    ]
    plt.close("all")


@pytest.mark.parametrize("thresh,full", [("p<0.05", True), ("q<0.05", True), ("q<0.05", False)])
@pytest.mark.parametrize("scalp", [False, True])
def test_roi_statistics_beta_errors_stars_and_fixed_color_keys(
    simulated_recording, thresh, full, scalp
):
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Circle
    from scipy.stats import t
    from statsmodels.stats.multitest import multipletests
    from pyBrainAnalyzIR.dataclasses.statistics import Statistics

    rec, _ = simulated_recording
    if scalp:
        from cedalion.dataclasses.geometry import PointType
        from cedalion import units
        landmarks = xr.DataArray(
            [[0, 100, 0], [-100, 0, 0], [100, 0, 0]], dims=("label", "pos"),
            coords={"label": ["Nz", "LPA", "RPA"],
                    "type": ("label", [PointType.LANDMARK] * 3)},
        ).pint.quantify(units.mm)
        rec.geo3d = xr.concat([rec.geo3d, landmarks], dim="label")
    roi_average(rec, pd.DataFrame({
        "source": rec["amp"].source.values[[0, 1, 0, 2]],
        "detector": rec["amp"].detector.values[[0, 1, 0, 2]],
        "name": ["Left", "Left", "Right", "Right"],
        "weight": [2, 1, 4, 4],
    }))
    stats = Statistics()
    stats.geoROI = get_geo_roi(rec, rec["amp_roi"])
    stats.geo2d, stats.geo3d = rec.geo2d, rec.geo3d
    # Deliberately different from definition order; exercise label-based alignment.
    stats.betas = xr.DataArray(
        [0.1, 4, -4, 2.1, 0.3, -0.1, 0.4, 4], dims="indices",
        coords={
            "channels": ("indices", ["Right", "Left"] * 4),
            "type": ("indices", ["HbO"] * 4 + ["HbR"] * 4),
            "conditions": ("indices", ["A", "A", "B", "B"] * 2),
        },
    )
    errors = np.array([0.5, 1, 0.8, 1, 0.4, 0.2, 0.7, 0.5])
    stats.covariance = xr.DataArray(np.diag(errors**2), dims=("rows", "cols"))
    stats.dof = 100
    fig = stats.draw(thresh=thresh, fdr_correct_full=full)
    assert len(fig.axes) == 7  # one probe, four panels, two color keys
    assert any(isinstance(patch, Circle) for patch in fig.axes[0].patches) == scalp
    probe_lines = fig.axes[0].lines[1:] if scalp else fig.axes[0].lines
    np.testing.assert_allclose(to_rgb(probe_lines[0].get_color()), [0.5, 0, 0.5])
    np.testing.assert_allclose(to_rgb(probe_lines[1].get_color()), [1, 0.5, 0.5])
    np.testing.assert_allclose(to_rgb(probe_lines[2].get_color()), [0, 0, 1])
    table = stats.table()
    significance = 2 * t.cdf(-np.abs(stats.betas.values / errors), stats.dof)
    if thresh.startswith("q"):
        if full:
            significance = multipletests(significance, method="fdr_bh")[1]
        else:
            for condition in ["A", "B"]:
                mask = table["Condition"] == condition
                significance[mask] = multipletests(significance[mask], method="fdr_bh")[1]
    table["expected_significance"] = significance
    for axis, (condition, signal_type) in zip(
        fig.axes[1:5], [("A", "HbO"), ("A", "HbR"), ("B", "HbO"), ("B", "HbR")]
    ):
        selected = table[(table.Condition == condition) & (table.Type == signal_type)]
        selected = selected.set_index("Channel").loc[["Left", "Right"]]
        np.testing.assert_allclose([bar.get_height() for bar in axis.patches], selected.Beta)
        assert [to_rgb(bar.get_facecolor()) for bar in axis.patches] == [to_rgb("red"), to_rgb("blue")]
        segments = axis.containers[0].lines[2][0].get_segments()
        np.testing.assert_allclose([segment[:, 1] for segment in segments], np.column_stack(
            [selected.Beta - selected.StdErr, selected.Beta + selected.StdErr]
        ))
        assert len(axis.texts) == (selected.expected_significance < 0.05).sum()
        assert all(text.get_text() == "*" for text in axis.texts)
        significant = selected[selected.expected_significance < 0.05]
        for text, (name, estimate) in zip(axis.texts, significant.iterrows()):
            endpoint = estimate.Beta + np.sign(estimate.Beta) * estimate.StdErr
            np.testing.assert_allclose(text.xy, (["Left", "Right"].index(name), endpoint))
    fig.canvas.draw()
    for button in fig._roi_buttons:
        assert not button.active
        x, y = button.ax.transAxes.transform((0.5, 0.5))
        for event_name in ("button_press_event", "button_release_event", "motion_notify_event"):
            fig.canvas.callbacks.process(
                event_name, MouseEvent(event_name, fig.canvas, x, y, button=1)
            )
        assert to_rgb(button.ax.get_facecolor()) == to_rgb(button.color)
    assert len(fig.axes[1].patches) == 2
    plt.close(fig)


@pytest.mark.parametrize("dimensions", [
    ("channel", "wavelength", "time"), ("wavelength", "time", "channel")
])
def test_roi_dimension_order(simulated_recording, roi_table, dimensions):
    rec, _ = simulated_recording
    rec["amp"] = rec["amp"].transpose(*dimensions)
    roi_average(rec, roi_table)
    assert rec["amp_roi"].dims == dimensions
    assert quality_masks(rec, rec["amp_roi"])["roi_valid"].dims == ("time", "channel")


@pytest.mark.requires_qt
def test_roi_pipeline_definition_json_roundtrip(roi_table):
    pytest.importorskip("PySide6")
    from pyBrainAnalyzIR.vis.pipeline_manager import pipeline_from_json, pipeline_to_json
    step = ROIAverage()
    step.options["roi_table"] = roi_table
    restored = pipeline_from_json(pipeline_to_json([step]))
    pd.testing.assert_frame_equal(restored[0].options["roi_table"], roi_table)


@pytest.mark.requires_qt
@pytest.mark.parametrize("output_name", [None, "my_regions"])
def test_named_roi_module_tables_selection_and_data_names_roundtrip(simulated_recording, output_name):
    pytest.importorskip("PySide6")
    from pyBrainAnalyzIR.pipelines.modules.roi import ROIAverage
    from pyBrainAnalyzIR.vis.pipeline_manager import pipeline_from_json, pipeline_to_json

    rec, _ = simulated_recording
    tables = {
        "ROI A": pd.DataFrame({
            "source": rec["amp"].source.values[:2],
            "detector": rec["amp"].detector.values[:2],
            "weight": [1.0, -1.0], "name": ["ROI A", "ROI A"],
        })
    }
    module = ROIAverage()
    module.inputName = "conc"
    module.set_roi_selection(tables, ["ROI A"])
    if output_name is not None:
        module.outputName = output_name
    restored = pipeline_from_json(pipeline_to_json([module]))[0]
    assert restored.inputName == "conc"
    assert restored.outputName == (output_name or "conc_roi")
    assert restored.options["roi_names"] == ["ROI A"]
    pd.testing.assert_frame_equal(restored.options["roi_tables"]["ROI A"], tables["ROI A"])
    restored.inputName = "stats"
    restored.set_roi_selection(tables, ["ROI A"])
    assert restored.outputName == (output_name or "stats_roi")


@pytest.mark.parametrize("problem", ["missing_pair", "zero_weights", "duplicate", "empty"])
def test_invalid_roi_definitions(simulated_recording, roi_table, problem):
    rec, _ = simulated_recording
    if problem == "missing_pair":
        roi_table.loc[0, "source"] = "absent"
    elif problem == "zero_weights":
        roi_table.loc[:, "weight"] = 0
    elif problem == "duplicate":
        roi_table = pd.concat([roi_table, roi_table.iloc[:1]])
    else:
        roi_table = roi_table.iloc[:0]
    with pytest.raises(ValueError):
        roi_average(rec, roi_table)
    assert "amp_roi" not in rec.timeseries
