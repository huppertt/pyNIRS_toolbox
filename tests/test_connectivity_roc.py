"""Simulation and ROC tests for zero-lag connectivity and hyperscanning."""
import numpy as np
import pytest

pytest.importorskip("cedalion")

from pyBrainAnalyzIR.pipelines.modules.connectivity import (  # noqa: E402
    hyperscanning,
    resting_state_connectivity,
)
from pyBrainAnalyzIR.pipelines.modules.nonparametric_statistics import (  # noqa: E402
    Statistics_Bootstrapping,
)
from pyBrainAnalyzIR.pipelines.modules.mixedeffects_connectivity import (  # noqa: E402
    MixedEffectsConnectivity,
)
from pyBrainAnalyzIR.pipelines.modules import preproccessing as prep  # noqa: E402
from pyBrainAnalyzIR.testing.roc.channelROC import PipelineList  # noqa: E402
from pyBrainAnalyzIR.testing.roc.connectivityROC import (  # noqa: E402
    ROC_connectivity,
    ROC_hyperscanning,
)
from pyBrainAnalyzIR.testing.simulations.simData import (  # noqa: E402
    simdataset_connectivity,
    simdataset_hyperscanning,
)
from pyBrainAnalyzIR.dataclasses.dataset import DataSet  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


def _roc_model(model, module, simulation_args):
    model.data_simulation_args = {
        **simulation_args,
        "t": np.arange(600) * 0.1,
        "sigma": 0,
        "random_seed": 12,
    }
    model.pipeline = PipelineList([
        prep.intensity_opticaldensity,
        prep.mbll,
        prep.resample,
        module,
    ])
    model.pipeline_args = {
        "Fs": 2,
        "AR": 0,
        "robust": False,
        "short_separation_correction": True,
    }
    return model


def test_simdataset_connectivity_returns_labeled_group_truth():
    dataset, truth = simdataset_connectivity(
        num_files=2, t=np.arange(100) * 0.1, sigma=0, random_seed=2
    )

    assert len(dataset.dataset) == 2
    assert truth.dims == ("channel_to", "channel_from")
    np.testing.assert_array_equal(truth.values, truth.values.T)
    assert dataset.get_demographics().subject.tolist() == ["sub-001", "sub-002"]


def test_simdataset_hyperscanning_returns_paired_edge_truth():
    dataset, truth = simdataset_hyperscanning(
        num_dyads=2, t=np.arange(100) * 0.1, sigma=0, random_seed=3
    )

    assert len(dataset.dataset) == 4
    assert dataset.get_demographics().pair.nunique() == 2
    assert set(truth.Type_to) >= {"HbO", "HbR"}
    assert truth.Truth.all()
    assert truth.Pair.nunique() == 2


def test_hyperscanning_effect_size_is_standardized_log_intensity():
    kwargs = {
        "num_dyads": 1,
        "t": np.arange(600) * 0.1,
        "sigma": 0,
        "random_seed": 31,
    }
    baseline, _ = simdataset_hyperscanning(effect_size=0, **kwargs)
    injected, _ = simdataset_hyperscanning(effect_size=0.2, **kwargs)

    baseline_amp = np.asarray(baseline.dataset[0]["amp"].pint.magnitude)
    injected_amp = np.asarray(injected.dataset[0]["amp"].pint.magnitude)
    baseline_log_sd = np.log(baseline_amp[:, 0, :]).std(axis=0, ddof=1)
    injected_log_sd = np.log(injected_amp[:, 0, :] / baseline_amp[:, 0, :])

    np.testing.assert_allclose(
        injected_log_sd.std(axis=0, ddof=1),
        0.2 * baseline_log_sd,
        rtol=0.03,
    )


@pytest.mark.parametrize(
    "model,module,args,truth_column",
    [
        (ROC_connectivity, resting_state_connectivity, {"num_files": 2}, "Truth"),
        (ROC_hyperscanning, hyperscanning, {"num_dyads": 1}, "Truth"),
    ],
)
def test_connectivity_roc_scores_labeled_edges(model, module, args, truth_column):
    roc = _roc_model(model(), module, args).run(1)
    table = roc.roc_table()

    assert len(table) > 0
    if model is ROC_hyperscanning:
        edge_columns = [
            "Pair", "Channel_to", "Type_to", "Channel_from", "Type_from",
            "Condition", "Iteration",
        ]
        assert not table.duplicated(edge_columns).any()
    assert table[truth_column].any()
    assert table.P.between(0, 1).all()
    assert len(roc.results()) > 0
    assert not roc.summary().empty
    type1_curve = roc.type1_error_curve([0.01, 0.05, 0.1])
    assert set(type1_curve["Nominal alpha"]) == {0.01, 0.05, 0.1}
    assert type1_curve["Empirical type-I error"].between(0, 1).all()
    assert (type1_curve["Null tests"] > 0).all()
    assert roc.draw_type1_error() is not None


@pytest.mark.parametrize(
    "module,simulation,arguments",
    [
        (
            resting_state_connectivity,
            simdataset_connectivity,
            {"num_files": 1},
        ),
        (
            hyperscanning,
            simdataset_hyperscanning,
            {"num_dyads": 1},
        ),
    ],
)
def test_bootstrap_supports_connectivity_and_hyperscanning(
    module, simulation, arguments
):
    data, _ = simulation(
        **arguments,
        t=np.arange(900) * 0.1,
        sigma=0,
        random_seed=81,
    )
    pipeline = PipelineList([
        prep.intensity_opticaldensity,
        prep.mbll,
        prep.resample,
        module,
    ])
    pipeline.set_all_options({
        "Fs": 2,
        "AR": 0,
        "robust": False,
        "short_separation_correction": True,
        "grouping_variable": "pair",
    })
    bootstrap = Statistics_Bootstrapping(pipeline)
    bootstrap.options["iterations"] = 19
    bootstrap.options["random_seed"] = 82

    result = bootstrap.run(DataSet(data.dataset))
    for recording in result.dataset:
        estimates = recording.aux_obj["bootstrap"]
        assert estimates.iterations == 19
        assert estimates.shift_window_seconds == 30.
        assert estimates.analysis in {"connectivity", "hyperscanning"}
        assert estimates.table.EmpiricalP.between(0, 1).all()
        assert estimates.table.PowerAtObservedEffect.between(0, 1).all()


@pytest.mark.parametrize(
    "simulation,module,arguments,input_name,robust",
    [
        (
            simdataset_connectivity,
            resting_state_connectivity,
            {"num_files": 4},
            "conn",
            False,
        ),
        (
            simdataset_hyperscanning,
            hyperscanning,
            {"num_dyads": 3},
            "hyperconn",
            True,
        ),
    ],
)
def test_mixed_effects_connectivity_fits_group_edges(
    simulation, module, arguments, input_name, robust
):
    data, _ = simulation(
        **arguments,
        t=np.arange(600) * 0.1,
        sigma=0,
        random_seed=93,
    )
    pipeline = PipelineList([
        prep.intensity_opticaldensity,
        prep.mbll,
        prep.resample,
        module,
    ])
    pipeline.set_all_options({
        "Fs": 2,
        "AR": 0,
        "robust": False,
        "short_separation_correction": True,
        "grouping_variable": "pair",
        "verbose": False,
    })
    data = pipeline.run(data)

    group_model = MixedEffectsConnectivity()
    group_model.options["robust"] = robust
    result = group_model.run(data)["groupconn"]

    assert not result.coef.empty
    assert result.coef.Pvalue.between(0, 1).all()
    assert result.fit_data.hyperscanning == (input_name == "hyperconn")
    assert result.fit_data.grouping_column == (
        "pair" if input_name == "hyperconn" else "subject"
    )
    assert {"geo_to", "geo_from"} <= set(result.coef.columns)
    assert {"Beta", "FixedFit", "ConditionalFit", "RobustWeight"} <= set(
        result.fit_data.observations.columns
    )
    if input_name == "hyperconn":
        assert len(result.fit_data.participant_roles) == 3
        assert {"participant-1", "participant-2"} == set(result.geo2d)
        assert set(result.coef.geo_to) <= set(result.geo2d)
        assert set(result.coef.geo_from) <= set(result.geo2d)


def test_hyperscanning_preserves_explicit_participant_order():
    data, _ = simdataset_hyperscanning(
        num_dyads=2,
        t=np.arange(240) * 0.1,
        sigma=0,
        warped=True,
        random_seed=94,
    )
    demographics = data.get_demographics()
    demographics["role"] = demographics["subject"].str.rsplit("-", n=1).str[-1]
    data.add_demographics_by_index(demographics)

    pipeline = PipelineList([
        prep.intensity_opticaldensity,
        prep.mbll,
        prep.resample,
        hyperscanning,
    ])
    pipeline.set_all_options({
        "Fs": 2,
        "AR": 0,
        "robust": False,
        "short_separation_correction": False,
        "grouping_variable": "pair",
        "order_variable": "role",
        "symmetric": False,
        "verbose": False,
    })
    result = pipeline.run(data)
    assert not result.dataset[0]["hyperconn"].coef.empty
