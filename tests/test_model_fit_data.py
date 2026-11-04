"""Original-space first- and second-level fit retention."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("cedalion")
import cedalion.dataclasses as cdc  # noqa: E402
from cedalion.models.glm.design_matrix import DesignMatrix  # noqa: E402

from pyBrainAnalyzIR.dataclasses.dataset import DataSet  # noqa: E402
from pyBrainAnalyzIR.dataclasses.statistics import Statistics  # noqa: E402
from pyBrainAnalyzIR.dataclasses.model_fit import GroupModelFit  # noqa: E402
from pyBrainAnalyzIR.math.glm.solve import fit  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.mixedeffects import MixedEffects  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


@pytest.mark.parametrize("cross_covariance", [0., 1.5, -1.5])
def test_full_model_interval_includes_intercept_slope_covariance(cross_covariance):
    frame = pd.DataFrame({
        "Channel": ["S1D1"] * 3, "Type": ["HbO"] * 3, "Condition": ["A"] * 3,
        "age": [7., 10., 13.], "StdErr": [.2] * 3,
        "FixedFit": [4., 10., 16.], "ConditionalFit": [4.1, 10.1, 16.1],
        "Beta": [4.3, 10.3, 16.3],
    })
    covariance = np.array([[4., cross_covariance], [cross_covariance, 1.]])
    model = GroupModelFit(
        observations=frame, coefficients=pd.DataFrame(
            [[10., 2.]], columns=["Condition[A]", "Condition[A]:age"],
            index=pd.MultiIndex.from_tuples([("HbO", "S1D1")]),
        ), covariance=covariance, formula="Beta ~ 0 + Condition + Condition:age",
        centers={"age": 10.}, demographics=frame[["age"]],
    )
    points, curve, _ = model.partial_effect("S1D1", "HbO", "Condition[A]:age")
    z = curve.age.to_numpy() - 10.
    np.testing.assert_allclose(curve.Fit, 2 * z)
    np.testing.assert_allclose(curve.FitSE ** 2, z ** 2)
    np.testing.assert_allclose(curve.TotalFitSE ** 2, 4 + 2 * z * cross_covariance + z ** 2)
    np.testing.assert_allclose(points.Beta, [-5.7, .3, 6.3])
    np.testing.assert_allclose(points.StdErr, .2)
    if cross_covariance == 0:
        assert np.all(curve.TotalFitSE > curve.FitSE)
    else:
        assert np.any(curve.TotalFitSE < curve.FitSE)
    _, intercept, axis = model.partial_effect("S1D1", "HbO", "Condition[A]")
    assert axis is None
    np.testing.assert_allclose(intercept.FitSE, 2.)
    np.testing.assert_allclose(intercept.TotalFitSE, 2.)
    model.covariance[0, 0] = -100.
    with pytest.raises(ValueError, match="Full fixed-effects fit has invalid"):
        model.partial_effect("S1D1", "HbO", "Condition[A]:age")


@pytest.mark.parametrize("extra", [
    "age", "I(age ** 2)", "age:C(gender)", 'Q("age")', "center(age)", "C(age)",
])
def test_partial_effects_remove_other_terms_including_dc_and_preserve_residuals(extra):
    import patsy

    demo = pd.DataFrame({"age": [20., 30., 40., 50.], "gender": ["F", "M", "F", "M"]})
    frame = pd.DataFrame([
        {"Channel": "S1D1", "Type": "HbO", "Condition": task, "fileIdx": subject,
         "age": row.age, "gender": row.gender, "StdErr": .25}
        for subject, row in enumerate(demo.itertuples()) for task in ("HRF A", "HRF B")
    ])
    formula = f"Beta ~ 0 + Condition + Condition:{extra} + Condition:C(gender)"
    training = frame.copy()
    training["age"] -= 35.
    design = patsy.dmatrix(formula.split("~")[1], training, return_type="dataframe")
    coefficients = np.arange(10., 10. + design.shape[1])
    fixed = design.to_numpy() @ coefficients
    residual = np.arange(len(frame)) * .1
    random = np.repeat([.4, -.1, .2, -.3], 2)
    frame["FixedFit"] = fixed
    frame["ConditionalFit"] = fixed + random
    frame["Beta"] = fixed + random + residual
    retained = GroupModelFit(
        observations=frame, coefficients=pd.DataFrame(
            [coefficients], columns=design.columns,
            index=pd.MultiIndex.from_tuples([("HbO", "S1D1")], names=["Type", "Channel"]),
        ), covariance=np.diag(np.arange(1., design.shape[1] + 1)),
        formula=formula, centers={"age": 35.}, demographics=demo,
    )
    before = frame.copy(deep=True)
    for column, coefficient in enumerate(design.columns):
        points, curve, axis = retained.partial_effect("S1D1", "HbO", coefficient)
        mask = frame.Condition == ("HRF A" if "HRF A" in coefficient else "HRF B")
        contribution = design.loc[mask].iloc[:, column].to_numpy() * coefficients[column]
        np.testing.assert_allclose(points.FixedFit, contribution)
        np.testing.assert_allclose(points.ConditionalFit, contribution + random[mask])
        np.testing.assert_allclose(points.Beta, contribution + random[mask] + residual[mask])
        np.testing.assert_array_equal(points.StdErr, frame.loc[mask, "StdErr"])
        if coefficient == "Condition[HRF A]":
            assert axis is None
            np.testing.assert_allclose(curve.Fit, coefficients[column])
            np.testing.assert_allclose(curve.FitSE, np.sqrt(column + 1))
        elif coefficient == "Condition[HRF A]:age":
            assert axis == "age"
            np.testing.assert_allclose(curve.Fit, coefficients[column] * (curve.age - 35.))
            np.testing.assert_allclose(curve.FitSE, abs(curve.age - 35.) * np.sqrt(column + 1))
        elif "I(age ** 2)" in coefficient:
            assert axis == "age"
            np.testing.assert_allclose(curve.Fit, coefficients[column] * (curve.age - 35.) ** 2)
        elif "age:" in coefficient:
            assert axis == "age × gender (design value)"
            np.testing.assert_allclose(curve.Fit, coefficients[column] * curve[axis])
        elif 'Q("age")' in coefficient or "center(age)" in coefficient:
            assert axis == "age"
            np.testing.assert_allclose(curve.Fit, coefficients[column] * (curve.age - 35.))
        elif "C(age)" in coefficient:
            assert axis == "age"
            assert isinstance(points.age.dtype, pd.CategoricalDtype)
            assert len(curve) == 4
            for value, fit_value in zip(curve.age, curve.Fit):
                np.testing.assert_allclose(fit_value, points.loc[points.age == value, "FixedFit"])
        centered_grid = curve.copy()
        centered_grid["age"] -= 35.
        prediction_design = np.asarray(patsy.build_design_matrices([design.design_info], centered_grid)[0])
        expected_variance = np.sum(prediction_design ** 2 * np.arange(1., design.shape[1] + 1), axis=1)
        np.testing.assert_allclose(curve.TotalFitSE ** 2, expected_variance)
        restored = pickle.loads(pickle.dumps(retained))
        pd.testing.assert_frame_equal(restored.partial_effect("S1D1", "HbO", coefficient)[0], points)
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match="display contrasts are unavailable"):
        retained.partial_effect("S1D1", "HbO", "HRF A-HRF B")


def small_model(seed=13):
    rng = np.random.default_rng(seed)
    time = np.arange(96) / 2
    design = np.column_stack([np.ones(len(time)), np.sin(time / 4), np.cos(time / 7)])
    values = np.empty((len(time), 2, 2))
    for channel in range(2):
        for chromo in range(2):
            coefficients = [1, 0.1 * seed + channel, -0.05 * seed + chromo]
            values[:, channel, chromo] = design @ coefficients + rng.normal(0, 0.3, len(time))
    ts = xr.DataArray(
        values, dims=("time", "channel", "chromo"),
        coords={"time": time, "channel": ["S1D1", "S2D2"], "chromo": ["HbO", "HbR"]},
        attrs={"units": "uM"},
    ).pint.quantify()
    common = xr.DataArray(
        np.repeat(design[:, :, None], 2, axis=2),
        dims=("time", "regressor", "chromo"),
        coords={"time": time, "regressor": ["Drift 0", "HRF A", "HRF B"], "chromo": ts.chromo},
    )
    return ts, DesignMatrix(common=common)


@pytest.mark.parametrize("noise_model", ["ols", "ar_irls"])
def test_original_time_course_and_mask_retention(noise_model):
    ts, design = small_model()
    mask = xr.ones_like(ts.isel(chromo=0, drop=True), dtype=bool).pint.dequantify()
    mask.values[10:15, 0] = False
    results = fit(ts, design, noise_model=noise_model, ar_order=4, quality_mask=mask, verbose=False)
    stats = Statistics(results)
    data = stats.fit_data.data
    xr.testing.assert_equal(data.observed, ts.pint.dequantify())
    np.testing.assert_array_equal(data.time, ts.time)
    assert not data.valid.sel(channel="S1D1").values[10:15].any()
    for channel in ts.channel.values:
        for chromo in ts.chromo.values:
            result = results.sel(channel=channel, chromo=chromo).item()
            local = data.sel(channel=channel, chromo=chromo)
            expected = design.common.sel(chromo=chromo).values @ result.params
            np.testing.assert_allclose(local.fitted, expected)
            valid = local.valid.values
            observed = local.observed.values[valid]
            r2 = 1 - np.sum((observed - expected[valid]) ** 2) / np.sum((observed - observed.mean()) ** 2)
            assert float(local.r_squared) == pytest.approx(r2)
            assert np.isfinite(local.f_statistic)
            assert 0 <= local.f_pvalue <= 1
            if noise_model == "ols":
                assert float(local.f_statistic) == pytest.approx(result.fvalue)
                assert float(local.f_pvalue) == pytest.approx(result.f_pvalue)
    restored = pickle.loads(pickle.dumps(stats))
    xr.testing.assert_identical(restored.fit_data.data, data)
    stats.remove_condition("Drift 0")
    stats.ttest(["HRF A-HRF B"])
    xr.testing.assert_identical(stats.fit_data.data, data)


def test_completely_masked_channel_has_no_fabricated_diagnostics():
    ts, design = small_model()
    mask = xr.ones_like(ts.isel(chromo=0, drop=True), dtype=bool).pint.dequantify()
    mask.loc[dict(channel="S1D1")] = False
    stats = Statistics(fit(ts, design, quality_mask=mask, verbose=False))
    local = stats.fit_data.data.sel(channel="S1D1")
    assert not local.valid.any()
    assert np.isnan(local.r_squared).all()
    assert np.isnan(local.f_statistic).all()
    table = stats.table()
    assert table.loc[table.Channel == "S2D2", "Q-values"].notna().all()
    assert table.loc[table.Channel == "S1D1", "Q-values"].isna().all()
    np.testing.assert_allclose(stats.get_qvalues(), table["Q-values"], equal_nan=True)


def make_dataset():
    recordings = []
    for index in range(8):
        ts, design = small_model(index + 1)
        stats = Statistics(fit(ts, design, verbose=False))
        rec = cdc.Recording()
        rec["stats"] = stats
        rec.meta_data.update(subject=f"sub-{index}", age=float(20 + 4 * index),
                             gender="F" if index % 2 else "M")
        recordings.append(rec)
    return DataSet(recordings)


@pytest.fixture
def small_dataset():
    return make_dataset()


@pytest.mark.parametrize("scope", ["shared", "channel"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("robust", [False, True])
def test_group_predictions_are_original_not_whitened(small_dataset, monkeypatch, weighted, centered, robust, scope):
    import pyBrainAnalyzIR.math.mixed_effects as mixed_math

    captured = {}
    original_fit = mixed_math.fitlme

    def capture(x, y, z, **kwargs):
        result = original_fit(x, y, z, **kwargs)
        captured.update(x=x, y=y, z=z, beta=result[0], random=result[1], weights=result[4])
        return result

    monkeypatch.setattr(mixed_math, "fitlme", capture)
    job = MixedEffects()
    job.options["FE_formula"] = "Beta ~ 0 + Condition + Condition:age"
    job.options["robust"] = robust
    job.options["weighted"] = weighted
    job.options["center_variables"] = centered
    job.options["RE_formula"] = "subject:cond:channel:chromo" if scope == "channel" else "subject"
    stats = job.run(small_dataset)["groupstats"]
    retained = stats.fit_data
    frame = retained.observations
    np.testing.assert_array_equal(frame.WhitenedRobustWeight, captured["weights"].ravel())
    assert np.all(frame.WeightStdErr > 0)
    if not robust and weighted:
        np.testing.assert_allclose(frame.WeightStdErr, frame.StdErr)
    if not robust and not weighted:
        np.testing.assert_array_equal(frame.WeightStdErr, 1.)
    for row in frame.itertuples():
        original = small_dataset.dataset[int(row.fileIdx)]["stats"].table()
        source = original[
            (original.Channel == row.Channel)
            & (original.Type == row.Type)
            & (original.Condition == row.Condition)
        ].iloc[0]
        assert row.Beta == source.Beta
        assert row.StdErr == source.StdErr
        assert row.age == 20 + 4 * row.fileIdx
        beta = retained.coefficients.loc[(row.Type, row.Channel)]
        centered_age = row.age - (34 if centered else 0)
        expected = beta[f"Condition[{row.Condition}]"] + centered_age * beta[f"Condition[{row.Condition}]:age"]
        assert row.FixedFit == pytest.approx(expected)
        signal = retained.coefficients.index.get_loc((row.Type, row.Channel))
        random_index = (signal * len(small_dataset.dataset) * 2 + int(row.fileIdx) * 2
                        if scope == "channel" else int(row.fileIdx))
        if scope == "channel" and row.Condition == "HRF B":
            random_index += 1
        individual = captured["random"].ravel()[random_index]
        assert row.ConditionalFit == pytest.approx(expected + individual)
    if weighted:
        assert not np.allclose(frame["FixedFit"], (captured["x"] @ captured["beta"]).ravel())
    else:
        np.testing.assert_allclose(
            frame["ConditionalFit"],
            (captured["x"] @ captured["beta"] + captured["z"] @ captured["random"]).ravel(),
        )
    curve = retained.population_curve("S1D1", "HbO", "HRF A", "age")
    assert curve.age.min() == 20
    assert curve.age.max() == 48
    coefficients = retained.coefficients.loc[("HbO", "S1D1")]
    expected = coefficients["Condition[HRF A]"] + (
        curve.age - (34 if centered else 0)
    ) * coefficients["Condition[HRF A]:age"]
    np.testing.assert_allclose(curve.Fit, expected)
    assert np.isfinite(curve.FitSE).all()
    restored = pickle.loads(pickle.dumps(stats))
    pd.testing.assert_frame_equal(
        restored.fit_data.population_curve("S1D1", "HbO", "HRF A", "age"), curve
    )


def test_group_categorical_curve_and_full_fit_survive_contrasts(small_dataset):
    job = MixedEffects()
    job.options["FE_formula"] = "Beta ~ 0 + Condition + Condition:gender"
    job.options["robust"] = False
    stats = job.run(small_dataset)["groupstats"]
    retained = stats.fit_data
    curve = retained.population_curve("S1D1", "HbO", "HRF A", "gender")
    beta = retained.coefficients.loc[("HbO", "S1D1")]
    expected = [
        beta["Condition[HRF A]"] + (beta["Condition[HRF A]:gender[T.M]"] if gender == "M" else 0)
        for gender in curve.gender
    ]
    np.testing.assert_allclose(curve.Fit, expected)
    stats.keep_condition("Condition[HRF A]")
    pd.testing.assert_frame_equal(
        retained.population_curve("S1D1", "HbO", "HRF A", "gender"), curve
    )


@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("covariate", ["age", "gender"])
def test_adjustment_preserves_residuals_and_uses_reference_design(small_dataset, centered, covariate):
    job = MixedEffects()
    job.options["FE_formula"] = "Beta ~ 0 + Condition * age * gender"
    job.options["center_variables"] = centered
    job.options["robust"] = False
    retained = job.run(small_dataset)["groupstats"].fit_data
    raw = retained.points("S1D1", "HbO", "HRF A")
    adjusted = retained.adjusted_points("S1D1", "HbO", "HRF A", covariate)
    # Evaluate an independent Patsy design at each subject's selected x value.
    import patsy
    training = retained.observations.copy()
    reference = raw.copy()
    for name in retained.covariates:
        if name != covariate:
            values = retained.demographics[name]
            reference[name] = values.mean() if pd.api.types.is_numeric_dtype(values) else values.mode().iloc[0]
    for name, center in retained.centers.items():
        training[name] -= center
        reference[name] -= center
    design = patsy.dmatrix(retained.formula.split("~")[1], training)
    x = patsy.build_design_matrices([design.design_info], reference)[0]
    expected = np.asarray(x) @ retained.coefficients.loc[("HbO", "S1D1")].to_numpy()
    np.testing.assert_allclose(adjusted.FixedFit, expected)
    np.testing.assert_allclose(adjusted.Beta, raw.Beta - raw.FixedFit + expected)
    np.testing.assert_allclose(
        adjusted.Beta - adjusted.ConditionalFit, raw.Beta - raw.ConditionalFit,
    )
    np.testing.assert_allclose(
        adjusted.ConditionalFit - adjusted.FixedFit, raw.ConditionalFit - raw.FixedFit,
    )
    np.testing.assert_array_equal(adjusted.StdErr, raw.StdErr)
    pd.testing.assert_frame_equal(retained.points("S1D1", "HbO", "HRF A"), raw)


@pytest.mark.parametrize("formula, expected", [
    ("Beta ~ 0 + Condition + Condition:age", ["age"]),
    ("Beta ~ Condition * age * C(gender)", ["age", "gender"]),
    ("Beta ~ Condition + np.log(age)", ["age"]),
    ("Beta ~ Condition + Q('gender')", ["gender"]),
    ("Beta ~ Condition", []),
])
def test_only_fitted_demographics_are_available(small_dataset, formula, expected):
    job = MixedEffects()
    job.options["robust"] = False
    retained = job.run(small_dataset)["groupstats"].fit_data
    retained.formula = formula
    assert set(retained.fitted_covariates) == set(expected)
