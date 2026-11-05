"""Reject invalid covariance precision rather than silently flooring eigenvalues."""
import copy

import numpy as np
import pandas as pd
import patsy
import pytest

pytest.importorskip("cedalion")
from pyBrainAnalyzIR.pipelines.modules.mixedeffects import (  # noqa: E402
    MixedEffects, _covariance_whitener, _random_effects_design, _weight_implied_se,
)
from tests.test_model_fit_data import make_dataset  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


def test_weight_implied_uncertainty_uses_squared_row_multipliers_and_full_covariance():
    inverse = np.array([[2., 0.], [1., 3.]])
    weights = np.array([.5, .25])
    transform = np.diag(weights) @ np.linalg.inv(inverse)
    expected = np.linalg.inv(transform.T @ transform)
    np.testing.assert_allclose(_weight_implied_se(inverse, weights) ** 2, np.diag(expected))
    np.testing.assert_allclose(_weight_implied_se(inverse, [1., 1.]), [2., np.sqrt(10)])
    np.testing.assert_allclose(_weight_implied_se(None, weights), [2., 4.])
    assert np.isinf(_weight_implied_se(inverse, [0., 1.])).all()
    np.testing.assert_array_equal(_weight_implied_se(inverse, [1., 0.]), [2., np.inf])
    np.testing.assert_array_equal(_weight_implied_se(None, [1., 0.]), [1., np.inf])
    for invalid in ([np.nan, 1.], [-.1, 1.], [1.1, 1.]):
        with pytest.raises(ValueError, match="robust multipliers"):
            _weight_implied_se(inverse, invalid)


@pytest.mark.parametrize("mode", ["full", "diagonal", "unweighted"])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("method", ["cholesky", "svd"])
def test_retained_weight_uncertainty_matches_final_whitened_precision(monkeypatch, mode, controlled, method):
    from scipy.linalg import block_diag
    import pyBrainAnalyzIR.math.mixed_effects as mixed_math
    data = make_dataset()
    original = mixed_math.fitlme
    captured = {}

    def capture(x, y, z, **kwargs):
        result = original(x, y, z, **kwargs)
        # Controlled final multipliers isolate the retention/back-transformation
        # from the robust algorithm's data-dependent downweighting.
        weights = np.linspace(.15, 1., len(y)).reshape(-1, 1) if controlled else result[4]
        captured["weights"] = weights.ravel()
        return (*result[:4], weights)

    monkeypatch.setattr(mixed_math, "fitlme", capture)
    job = MixedEffects()
    job.options["robust"] = not controlled
    job.options["weighted"] = mode != "unweighted"
    job.options["whitening_method"] = method
    if mode != "unweighted":
        job.options["covariance_weighting"] = mode
    stats = job.run(data)["groupstats"]
    frame = stats.fit_data.observations
    np.testing.assert_array_equal(frame.WhitenedRobustWeight, captured["weights"])
    if mode == "unweighted":
        expected = 1 / captured["weights"]
    else:
        blocks, tables = [], []
        for index, rec in enumerate(data.dataset):
            source = rec["stats"]
            keep = np.flatnonzero(source.betas.conditions.values != "Drift 0")
            covariance = source.covariance.values[np.ix_(keep, keep)]
            blocks.append(_covariance_whitener(covariance, mode, index, method))
            tables.append(source.table().query("Condition != 'Drift 0'"))
        order = pd.concat(tables, ignore_index=True).sort_values(["Type", "Channel"], kind="stable").index
        whitening = block_diag(*blocks)[np.ix_(order, order)]
        transform = captured["weights"][:, None] * whitening
        if np.all(captured["weights"] > 0):
            covariance = np.linalg.inv(transform.T @ transform)
            expected = np.sqrt(np.diag(covariance))
        else:
            inverse = np.linalg.inv(whitening)
            positive = captured["weights"] > 0
            expected = np.sqrt(np.sum((inverse[:, positive] / captured["weights"][positive]) ** 2, axis=1))
            expected[np.any(inverse[:, ~positive] != 0, axis=1)] = np.inf
        assert np.all(expected >= frame.StdErr - 1e-12)
    np.testing.assert_allclose(frame.WeightStdErr, expected)
    effective_method = method if mode == "full" else "not_applicable"
    assert stats.covariance.attrs["whitening_method"] == effective_method
    assert frame.attrs["whitening_method"] == effective_method
    import pickle
    restored = pickle.loads(pickle.dumps(stats.fit_data))
    pd.testing.assert_frame_equal(restored.observations, frame)
    points, _, _ = stats.fit_data.partial_effect("S1D1", "HbO", "Condition[HRF A]")
    subset = frame.query("Channel == 'S1D1' and Type == 'HbO' and Condition == 'HRF A'")
    np.testing.assert_array_equal(points.WeightStdErr, subset.WeightStdErr)


def test_whitening_default_and_invalid_method_are_explicit():
    job = MixedEffects()
    assert job.options["whitening_method"] == "cholesky"
    cov = np.array([[4., 1.], [1., 2.]])
    np.testing.assert_array_equal(
        _covariance_whitener(cov, "full", 0), _covariance_whitener(cov, "full", 0, "cholesky"),
    )
    with pytest.raises(ValueError, match="Unknown covariance whitening method"):
        _covariance_whitener(cov, "full", 0, "invalid")


def test_svd_pipeline_whitens_all_designs_and_response_with_the_same_basis(monkeypatch):
    from scipy.linalg import block_diag
    import pyBrainAnalyzIR.math.mixed_effects as mixed_math
    data = make_dataset()
    captured = []
    original = mixed_math.fitlme

    def capture(x, y, z, **kwargs):
        captured.append((x.copy(), y.copy(), z.copy()))
        return original(x, y, z, **kwargs)

    monkeypatch.setattr(mixed_math, "fitlme", capture)
    job = MixedEffects()
    job.options["robust"] = False
    job.options["weighted"] = False
    job.run(copy.deepcopy(data))
    job.options["weighted"] = True
    job.options["whitening_method"] = "svd"
    result = job.run(copy.deepcopy(data))["groupstats"]
    tables, blocks = [], []
    for index, rec in enumerate(data.dataset):
        stats = rec["stats"]
        keep = np.flatnonzero(stats.betas.conditions.values != "Drift 0")
        blocks.append(_covariance_whitener(stats.covariance.values[np.ix_(keep, keep)], "full", index, "svd"))
        tables.append(stats.table().query("Condition != 'Drift 0'"))
    order = pd.concat(tables, ignore_index=True).sort_values(["Type", "Channel"], kind="stable").index
    w = block_diag(*blocks)[np.ix_(order, order)]
    for unweighted, whitened in zip(*captured):
        np.testing.assert_allclose(whitened, w @ unweighted, atol=1e-12)
    assert result.covariance.attrs["random_effects_coefficients"] == len(data.dataset)


def test_nonrobust_bases_match_with_the_same_variance_ratio(monkeypatch):
    import pyBrainAnalyzIR.math.mixed_effects as mixed_math
    monkeypatch.setattr(mixed_math, "solveForTheta", lambda *args: -4.)
    data = make_dataset()
    job = MixedEffects()
    job.options["robust"] = False
    cholesky = job.run(copy.deepcopy(data))["groupstats"]
    job.options["whitening_method"] = "svd"
    svd = job.run(copy.deepcopy(data))["groupstats"]
    np.testing.assert_allclose(svd.betas, cholesky.betas, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(svd.covariance, cholesky.covariance, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(svd.fit_data.observations.FixedFit, cholesky.fit_data.observations.FixedFit)
    np.testing.assert_allclose(
        svd.fit_data.observations.ConditionalFit, cholesky.fit_data.observations.ConditionalFit,
    )


@pytest.mark.parametrize("method", ["cholesky", "svd"])
def test_full_whitener_is_inverse_covariance_precision(method):
    cov = np.array([[4., 1.], [1., 2.]])
    whitening = _covariance_whitener(cov, "full", 0, method)
    np.testing.assert_allclose(whitening @ cov @ whitening.T, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(whitening.T @ whitening, np.linalg.inv(cov))
    if method == "svd":
        u, s, _ = np.linalg.svd(cov)
        np.testing.assert_allclose(whitening, np.diag(1 / np.sqrt(s)) @ u.T)


@pytest.mark.parametrize("covariance", [
    [[1., 1.2], [1.2, 1.]],  # materially indefinite
    [[1., 1.], [1., 1.]],  # singular
    [[1., 0.], [0., 1e-18]],  # numerically singular
])
@pytest.mark.parametrize("method", ["cholesky", "svd"])
def test_invalid_full_covariance_is_rejected_with_explicit_alternatives(covariance, method):
    with pytest.raises(ValueError, match="recording 3.*minimum eigenvalue.*covariance_weighting='diagonal'"):
        _covariance_whitener(covariance, "full", 3, method)
    diagonal = _covariance_whitener(covariance, "diagonal", 3)
    np.testing.assert_allclose(np.diag(diagonal), 1 / np.sqrt(np.diag(covariance)))


@pytest.mark.parametrize("mode", ["full", "diagonal"])
@pytest.mark.parametrize("variance", [0, -1, np.nan, np.inf])
def test_invalid_variances_are_not_assigned_precision(mode, variance):
    with pytest.raises(ValueError, match="finite, positive"):
        _covariance_whitener([[variance, 0], [0, 1]], mode, 0)


def test_diagonal_option_matches_explicitly_diagonalized_input():
    data = make_dataset()
    stats = data.dataset[0]["stats"]
    covariance = stats.covariance.values
    covariance[0, 1] = covariance[1, 0] = 2 * np.sqrt(covariance[0, 0] * covariance[1, 1])
    job = MixedEffects()
    job.options["include_nuisance"] = True
    job.options["robust"] = False
    with pytest.raises(ValueError, match="not numerically positive definite"):
        job.run(copy.deepcopy(data))
    job.options["covariance_weighting"] = "diagonal"
    result = job.run(copy.deepcopy(data))["groupstats"]
    for rec in data.dataset:
        cov = rec["stats"].covariance
        rec["stats"].covariance = cov.copy(data=np.diag(np.diag(cov.values)))
    job.options["covariance_weighting"] = "full"
    expected = job.run(data)["groupstats"]
    np.testing.assert_allclose(result.betas, expected.betas)
    np.testing.assert_allclose(result.covariance, expected.covariance)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("formula,terms", [("subject", 1), ("subject:cond", 2)])
@pytest.mark.parametrize("method", ["cholesky", "svd"])
def test_shared_subject_design_precedes_whitening_and_links_recordings(monkeypatch, weighted, formula, terms, method):
    import pyBrainAnalyzIR.math.mixed_effects as mixed_math
    data = make_dataset()
    for index, rec in enumerate(data.dataset):
        rec.meta_data["subject"] = index // 2
    captured = {}
    original = mixed_math.fitlme

    def capture(x, y, z, **kwargs):
        captured.update(z=z.copy())
        return original(x, y, z, **kwargs)

    monkeypatch.setattr(mixed_math, "fitlme", capture)
    job = MixedEffects()
    job.options["weighted"] = weighted
    job.options["whitening_method"] = method
    job.options["robust"] = False
    job.options["RE_formula"] = formula
    stats = job.run(data)["groupstats"]
    frame = stats.fit_data.observations
    assert stats.covariance.attrs["random_effects_groups"] == 4
    assert stats.covariance.attrs["random_effects_coefficients"] == 4 * terms
    assert stats.dof == 4 * 2 - 2
    assert "subject" not in stats.fit_data.centers
    expected = np.zeros((len(frame), 4 * terms))
    for index, row in enumerate(frame.itertuples()):
        column = int(row.subject) * terms + int(terms == 2 and row.Condition == "HRF B")
        expected[index, column] = 1
    if weighted:
        file_order = pd.concat([
            rec["stats"].table().query("Condition != 'Drift 0'").assign(fileIdx=index)
            for index, rec in enumerate(data.dataset)
        ], ignore_index=True)
        from scipy.linalg import block_diag
        whitening = block_diag(*[
            _covariance_whitener(rec["stats"].covariance.values[np.ix_(
                np.flatnonzero(rec["stats"].betas.conditions.values != "Drift 0"),
                np.flatnonzero(rec["stats"].betas.conditions.values != "Drift 0"),
            )], "full", index, method)
            for index, rec in enumerate(data.dataset)
        ])
        order = file_order.sort_values(["Type", "Channel"], kind="stable").index.to_numpy()
        expected = whitening[np.ix_(order, order)] @ expected
    np.testing.assert_allclose(captured["z"], expected)


def test_subject_grouping_errors_and_explicit_recording_fallback():
    data = make_dataset()
    data.dataset[0].meta_data["subject"] = None
    job = MixedEffects()
    job.options["robust"] = False
    with pytest.raises(ValueError, match="missing or empty IDs"):
        job.run(copy.deepcopy(data))
    job.options["group_by"] = "missing_subject_column"
    with pytest.raises(ValueError, match="missing from demographics"):
        job.run(copy.deepcopy(data))
    for rec in data.dataset:
        rec.meta_data.pop("subject")
    job.options["group_by"] = "auto"
    with pytest.warns(UserWarning, match="No subject or subjectID"):
        stats = job.run(copy.deepcopy(data))["groupstats"]
    assert stats.covariance.attrs["random_effects_group_by"] == "fileIdx"
    job.options["group_by"] = "fileIdx"
    explicit = job.run(data)["groupstats"]
    np.testing.assert_array_equal(stats.betas, explicit.betas)


def test_custom_subject_column_and_subject_id_alias():
    data = make_dataset()
    for rec in data.dataset:
        rec.meta_data["subjectID"] = rec.meta_data.pop("subject")
    job = MixedEffects()
    job.options["robust"] = False
    stats = job.run(copy.deepcopy(data))["groupstats"]
    assert stats.covariance.attrs["random_effects_group_by"] == "subjectID"
    for rec in data.dataset:
        rec.meta_data["participant"] = rec.meta_data.pop("subjectID")
    job.options["group_by"] = "participant"
    custom = job.run(data)["groupstats"]
    np.testing.assert_allclose(custom.betas, stats.betas)
    np.testing.assert_allclose(custom.covariance, stats.covariance)


def test_single_condition_default_is_not_saturated():
    data = make_dataset()
    for rec in data.dataset:
        rec["stats"] = rec["stats"].copy()
        rec["stats"].keep_condition("HRF A")
    job = MixedEffects()
    stats = job.run(data)["groupstats"]
    assert stats.covariance.attrs["random_effects_coefficients"] == 8
    assert len(stats.fit_data.observations) == 8 * 2 * 2
    assert not stats.covariance.attrs["conditional_residual_collapse"]
    assert np.isfinite(stats.covariance.values).all()
    assert np.linalg.eigvalsh(stats.covariance.values)[0] > 0


def test_shared_intercept_reduces_random_design_rank_not_measurement_whitening(monkeypatch):
    import pyBrainAnalyzIR.math.mixed_effects as mixed_math
    data = make_dataset()
    captured = []
    original = mixed_math.fitlme

    def capture(x, y, z, **kwargs):
        captured.append((x.copy(), y.copy(), z.copy()))
        return original(x, y, z, **kwargs)

    monkeypatch.setattr(mixed_math, "fitlme", capture)
    job = MixedEffects()
    job.options["robust"] = False
    job.run(copy.deepcopy(data))
    job.options["RE_formula"] = "subject:cond:channel:chromo"
    job.options["group_by"] = "fileIdx"
    job.run(copy.deepcopy(data))
    shared, legacy = captured[0], captured[1]
    np.testing.assert_array_equal(shared[0], legacy[0])
    np.testing.assert_array_equal(shared[1], legacy[1])
    assert shared[2].shape[1] == np.linalg.matrix_rank(shared[2]) == 8
    assert legacy[2].shape[1] == np.linalg.matrix_rank(legacy[2]) == 8 * 2 * 2 * 2


@pytest.fixture
def random_frame():
    return pd.DataFrame([
        {"subject": subject, "cond": cond, "channel": channel, "chromo": chromo,
         "site": subject // 2}
        for channel in ("S1D1", "S2D2", "S3D3")
        for chromo in ("HbO", "HbR")
        for subject in range(4)
        for cond in ("A", "B")
    ])


@pytest.mark.parametrize("formula,columns", [
    ("subject", 4), ("~subject", 4), ("subject:cond", 8),
    ("subject:cond:channel", 24), ("subject:cond:channel:chromo", 48),
    ("site", 2), ("subject + site", 6), ("subject + subject:cond", 12),
])
def test_random_terms_split_only_explicit_factors(random_frame, formula, columns):
    design, terms = _random_effects_design(formula, random_frame)
    assert design.shape == (len(random_frame), columns)
    assert sum(terms.values()) == columns
    expected_terms = patsy.ModelDesc.from_formula(formula).rhs_termlist
    start = 0
    for term in expected_terms:
        names = [factor.name() for factor in term.factors]
        if not names:
            continue
        width = terms[term.name()]
        block = design[:, start:start + width]
        start += width
        np.testing.assert_array_equal(block.sum(axis=1), np.ones(len(random_frame)))
        for _, indices in random_frame.groupby(names, sort=False).groups.items():
            indices = list(indices)
            np.testing.assert_array_equal(block[indices], np.tile(block[indices[0]], (len(indices), 1)))


@pytest.mark.parametrize("formula,message", [
    ("subject:unknown", "Unknown random-effects factor"),
    ("~1", "must name a grouping factor"),
    ("Beta ~ subject", "must have no response"),
])
def test_invalid_random_formulas_fail_explicitly(random_frame, formula, message):
    with pytest.raises(ValueError, match=message):
        _random_effects_design(formula, random_frame)


def test_missing_values_in_other_random_terms_are_rejected(random_frame):
    random_frame.loc[0, "site"] = np.nan
    with pytest.raises(ValueError, match="missing or empty factor"):
        _random_effects_design("subject + site", random_frame)
