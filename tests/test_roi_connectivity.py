"""ROI inputs, participant-specific definitions, and adjacency-matrix drawing."""
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb
from statsmodels.stats.multitest import multipletests

pytest.importorskip("cedalion")

from pyBrainAnalyzIR.dataclasses.dataset import DataSet
from pyBrainAnalyzIR.math.connectivity import compute_connectivity, compute_hyperscanning
from pyBrainAnalyzIR.pipelines.modules.connectivity import resting_state_connectivity, hyperscanning
from pyBrainAnalyzIR.sigproc.roi import roi_average

pytestmark = pytest.mark.requires_cedalion


def _assert_cell_grid(axis, rows, columns):
    np.testing.assert_array_equal(axis.xaxis.get_minorticklocs(), np.arange(columns + 1) - 0.5)
    np.testing.assert_array_equal(axis.yaxis.get_minorticklocs(), np.arange(rows + 1) - 0.5)
    for tick in axis.xaxis.get_minor_ticks() + axis.yaxis.get_minor_ticks():
        assert tick.gridline.get_visible()
        assert tick.gridline.get_linestyle() == ":"
        assert not tick.tick1line.get_visible()
    assert axis.get_axisbelow() is False


@pytest.fixture
def roi_pair(simulated_recording):
    original, _ = simulated_recording
    recordings = []
    for index in range(2):
        rec = copy.deepcopy(original)
        rec["amp"] = rec["amp"] * (
            1 + np.random.default_rng(index).normal(0, 0.01, rec["amp"].shape)
        )
        rec.meta_data.update(subject=f"person{index}", pair="dyad")
        roi_average(rec, pd.DataFrame({
            "source": rec["amp"].source.values[[0, 1, 0, 2]],
            "detector": rec["amp"].detector.values[[0, 1, 0, 2]],
            "name": ["Left", "Left", "Right", "Right"],
            "weight": [2, 1, 4, 4] if index == 0 else [1, 2, 1, 1],
        }))
        recordings.append(rec)
    return DataSet(recordings)


def _module(cls, ar=0, robust=False):
    job = cls()
    job.inputName = "amp_roi"
    job.options.update(AR=ar, robust=robust, short_separation_correction=False)
    return job


@pytest.mark.parametrize("ar,robust", [(0, False), (2, True)])
def test_resting_roi_correlations_match_core_and_pearson(roi_pair, ar, robust):
    job = _module(resting_state_connectivity, ar, robust)
    assert job.run(roi_pair) is roi_pair
    for rec in roi_pair.dataset:
        conn = rec["conn"]
        expected = compute_connectivity(rec["amp_roi"], AR=ar, robust=robust)
        expected = expected[expected.Type_from == expected.Type_to].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            conn.coef[expected.columns], expected, check_dtype=False,
        )
        assert conn.geoROI["default"].membership.equals(rec.aux_obj["roi_definitions"]["default"].membership)
        assert set(conn.coef.Channel_to) == {"Left", "Right"}
        kind = rec["amp_roi"].wavelength.values[0]
        assert conn.matrix(type_to=kind).shape == (2, 2)
        if not ar and not robust:
            data = rec["amp_roi"].sel(wavelength=kind).pint.magnitude
            np.testing.assert_allclose(
                conn.matrix(type_to=kind), np.corrcoef(data, rowvar=False), atol=1e-12,
            )
        contrast = conn.ttest("rest")
        assert contrast.geoROI["default"].membership.equals(conn.geoROI["default"].membership)


@pytest.mark.parametrize("ar,robust", [(0, False), (2, True)])
@pytest.mark.parametrize("unordered", [False, True])
def test_hyperscanning_keeps_dataset_scope_and_per_person_roi(roi_pair, ar, robust, unordered):
    job = _module(hyperscanning, ar, robust)
    job.options["unordered"] = unordered
    expected = compute_hyperscanning(
        {rec.meta_data["subject"]: rec["amp_roi"] for rec in roi_pair.dataset},
        AR=ar, robust=robust, unordered=unordered,
    )
    expected = expected[
        (expected.Subject_from != expected.Subject_to) & (expected.Type_from == expected.Type_to)
    ].reset_index(drop=True)
    assert job.run(roi_pair) is roi_pair
    conn = roi_pair.dataset[0]["hyperconn"]
    assert conn is roi_pair.dataset[1]["hyperconn"]
    pd.testing.assert_frame_equal(conn.coef[expected.columns], expected, check_dtype=False)
    assert list(conn.geoROI) == ["person0", "person1"]
    assert not conn.geoROI["person0"].membership.equals(conn.geoROI["person1"].membership)
    assert conn.matrix().shape == (4, 4)
    assert "person0:Left" in conn.matrix().index
    assert "person1:Left" in conn.matrix().index


@pytest.mark.parametrize("cls", [resting_state_connectivity, hyperscanning])
def test_roi_connectivity_requires_short_correction_before_averaging(roi_pair, cls):
    job = _module(cls)
    job.options["short_separation_correction"] = True
    with pytest.raises(ValueError, match="before ROI averaging"):
        job.run(roi_pair)
    assert all("amp_roi_ss" not in rec.timeseries for rec in roi_pair.dataset)


def test_hyperscanning_rejects_mixed_roi_channel_inputs(roi_pair):
    roi_pair.dataset[1]["amp_roi"] = roi_pair.dataset[1]["amp"]
    with pytest.raises(ValueError, match="mixes ROI"):
        _module(hyperscanning).run(roi_pair)


@pytest.mark.parametrize("threshold_type,global_fdr", [("p", False), ("q", False), ("q", True)])
def test_roi_adjacency_colors_labels_stars_and_disabled_keys(roi_pair, threshold_type, global_fdr):
    rec = _module(resting_state_connectivity).run(roi_pair).dataset[0]
    conn = rec["conn"]
    off_diagonal = conn.coef.Channel_to != conn.coef.Channel_from
    conn.coef.loc[off_diagonal, "Pvalue"] = [0.001, 0.04, 0.02, 0.2]
    conn.coef.loc[off_diagonal, "Coeff"] = [0.8, -0.4, 0.2, -0.1]
    fig = conn.show(threshold_type=threshold_type, fdr_correct_for_all=global_fdr)
    matrices = [axis for axis in fig.axes if axis.images]
    assert len(matrices) == 2
    probe = fig.axes[0]
    np.testing.assert_allclose(to_rgb(probe.lines[0].get_color()), [0.5, 0, 0.5])
    np.testing.assert_allclose(to_rgb(probe.lines[1].get_color()), [1, 0.5, 0.5])
    assert [button.label.get_text() for button in fig._roi_buttons] == ["Left", "Right"]
    assert all(not button.active for button in fig._roi_buttons)
    table = conn.table()
    kinds = list(pd.unique(table.Type_from))
    expected_sig = table.Pvalue.where(table.Channel_from != table.Channel_to)
    if threshold_type == "q":
        groups = [table] if global_fdr else [table[table.Type_from == kind] for kind in kinds]
        for group in groups:
            indices = group.index[expected_sig.loc[group.index].notna()]
            expected_sig.loc[indices] = multipletests(expected_sig.loc[indices], method="fdr_bh")[1]
    for axis, kind in zip(matrices, kinds):
        _assert_cell_grid(axis, 2, 2)
        np.testing.assert_allclose(axis.images[0].get_array(), conn.matrix(type_to=kind))
        assert axis.images[0].get_clim() == (-1, 1)
        assert [tick.get_text() for tick in axis.get_xticklabels()] == ["Left", "Right"]
        assert [tick.get_text() for tick in axis.get_yticklabels()] == ["Left", "Right"]
        selected = table[table.Type_from == kind]
        assert len(axis.texts) == (expected_sig.loc[selected.index] < 0.05).sum()
        assert all(text.get_text() == "*" for text in axis.texts)
        assert all(text.get_position()[0] != text.get_position()[1] for text in axis.texts)
    fig.canvas.draw()
    plt.close(fig)


def test_hyperscanning_matrix_disambiguates_identical_roi_names(roi_pair):
    conn = _module(hyperscanning).run(roi_pair).dataset[0]["hyperconn"]
    fig = conn.show(thres=1)
    matrices = [axis for axis in fig.axes if axis.images]
    assert len(matrices) == 2
    assert len(fig._roi_buttons) == 4  # A separate map/color key for each participant.
    for axis in matrices:
        _assert_cell_grid(axis, 4, 4)
        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            "person0: Left", "person0: Right", "person1: Left", "person1: Right",
        ]
        values = axis.images[0].get_array()
        assert values.shape == (4, 4)
        assert np.ma.getmaskarray(values)[:2, :2].all()
        assert np.ma.getmaskarray(values)[2:, 2:].all()
        assert np.isfinite(values[:2, 2:]).all()
    probe_axes = [axis for axis in fig.axes if axis.get_title().endswith("ROI membership")]
    assert len(probe_axes) == 2
    assert to_rgb(probe_axes[0].lines[1].get_color()) != to_rgb(probe_axes[1].lines[1].get_color())
    fig.canvas.draw()
    plt.close(fig)


def test_hyperscanning_different_roi_definitions_and_labels(roi_pair):
    for index, rec in enumerate(roi_pair.dataset):
        table = rec.aux_obj["roi_definitions"]["default"].membership.copy()
        if index:
            table["name"] = table["name"].replace({"Left": "Front", "Right": "Back"})
        roi_average(rec, table, output_name="amp_named", roi_name="named")
    job = _module(hyperscanning)
    job.inputName = "amp_named"
    conn = job.run(roi_pair).dataset[0]["hyperconn"]
    matrix = conn.matrix()
    assert matrix.index.tolist() == [
        "person0:Left", "person0:Right", "person1:Front", "person1:Back",
    ]
    fig = conn.show()
    images = [axis for axis in fig.axes if axis.images]
    assert [tick.get_text() for tick in images[0].get_yticklabels()] == [
        "person0: Left", "person0: Right", "person1: Front", "person1: Back",
    ]
    plt.close(fig)
    job.options["unordered"] = True
    with pytest.raises(ValueError, match="requires matching region"):
        job.run(roi_pair)


def test_unordered_roi_hyperscanning_aligns_region_order(roi_pair):
    job = _module(hyperscanning)
    job.options["unordered"] = True
    expected = job.run(roi_pair).dataset[0]["hyperconn"].coef.copy()
    rec = roi_pair.dataset[1]
    rec["amp_roi"] = rec["amp_roi"].isel(channel=[1, 0], wavelength=[1, 0])
    actual = job.run(roi_pair).dataset[0]["hyperconn"].coef
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("cls", [resting_state_connectivity, hyperscanning])
def test_roi_connectivity_concentration_input_and_event_contrasts(roi_pair, cls):
    from pyBrainAnalyzIR.pipelines.modules import preproccessing as prep

    for rec in roi_pair.dataset:
        rec.stim = pd.DataFrame({
            "onset": [30.0, 150.0], "duration": [60.0, 60.0],
            "value": [1.0, 1.0], "trial_type": ["task", "task"],
        })
    for module, source, target in [
        (prep.intensity_opticaldensity(), "amp_roi", "od_roi"),
        (prep.mbll(), "od_roi", "conc_roi"),
    ]:
        module.inputName, module.outputName = source, target
        module.run(roi_pair)
    job = _module(cls)
    job.inputName = "conc_roi"
    job.options["divide_events"] = True
    conn = job.run(roi_pair).dataset[0][job.outputName]
    assert set(conn.conditions) == {"task", "rest"}
    assert set(conn.coef.Type_from) == {"HbO", "HbR"}
    contrasted = conn.ttest("task - rest")
    assert set(contrasted.geoROI) == set(conn.geoROI)
    figure = contrasted.show(threshold_type="q", condnames="task - rest")
    assert len([axis for axis in figure.axes if axis.images]) == 2
    plt.close(figure)


def test_roi_connectivity_cross_type_panels_and_transformed_values(roi_pair):
    conn = _module(resting_state_connectivity).run(roi_pair).dataset[0]["conn"]
    # The module selects same-type connections, so retain all types from the core.
    conn.coef = compute_connectivity(roi_pair.dataset[0]["amp_roi"], AR=0, robust=False)
    conn.coef["geo_from"] = conn.coef["geo_to"] = "default"
    for value_type, column in [("r", "Coeff"), ("z", "Z"), ("t", "Tstat")]:
        fig = conn.show(type=value_type, show_cross_types=True, vrange=[-2, 2])
        axes = [axis for axis in fig.axes if axis.images]
        assert len(axes) == 4
        kinds = list(pd.unique(conn.coef.Type_to))
        for axis, (type_to, type_from) in zip(
            axes, [(to_type, from_type) for from_type in kinds for to_type in kinds]
        ):
            np.testing.assert_allclose(
                axis.images[0].get_array(),
                conn.matrix(type_to=type_to, type_from=type_from, value=column),
            )
            assert axis.images[0].get_clim() == (-2, 2)
        plt.close(fig)
