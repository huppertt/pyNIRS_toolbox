"""Robust AR coefficient estimation and its production option plumbing."""
import pickle
import warnings

import numpy as np
import pytest
from scipy.signal import lfilter

pytest.importorskip("cedalion")
from pyBrainAnalyzIR.dataclasses.statistics import Statistics  # noqa: E402
from pyBrainAnalyzIR.math.glm import ar_irls as estimator  # noqa: E402
from pyBrainAnalyzIR.math.glm.solve import _fit_ar_irls, fit  # noqa: E402
from pyBrainAnalyzIR.pipelines.modules.glm import GLM  # noqa: E402
from pyBrainAnalyzIR.testing.helpers.raw_null_calibration import simulate, traced_fit  # noqa: E402
from tests.test_model_fit_data import small_model  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


@pytest.mark.parametrize("max_jobs", [1, 2])
def test_solver_progress_counts_completed_fits_on_calling_thread(max_jobs):
    import threading

    ts, design = small_model()
    updates = []
    owner = threading.get_ident()

    def progress(completed, total):
        assert threading.get_ident() == owner
        updates.append((completed, total))

    result = fit(ts, design, max_jobs=max_jobs, verbose=False, progress_callback=progress)
    total = ts.sizes["channel"] * ts.sizes["chromo"]
    assert updates == [(index, total) for index in range(total + 1)]
    reference = fit(ts, design, max_jobs=1, verbose=False)
    for actual, expected in zip(result.values.ravel(), reference.values.ravel()):
        np.testing.assert_array_equal(actual.params, expected.params)


@pytest.mark.parametrize("coefficients", [[.65], [.5, .2]])
def test_robust_ar_recovers_correlation_with_impulses_in_target_and_lags(coefficients):
    rng = np.random.default_rng(42)
    clean = lfilter([1], np.r_[1, -np.array(coefficients)], rng.normal(size=2100))[100:]
    contaminated = clean.copy()
    contaminated[50::80] += 15 * np.where(np.arange(len(contaminated[50::80])) % 2, 1, -1)
    order = len(coefficients)
    legacy = estimator._ar_fit(contaminated, order)
    robust = estimator._ar_fit(contaminated, order, robust=True)
    np.testing.assert_allclose(robust[1:], coefficients, atol=.06)
    assert np.linalg.norm(robust[1:] - coefficients) < np.linalg.norm(legacy[1:] - coefficients) / 3
    np.testing.assert_allclose(
        estimator._ar_fit(contaminated * 100, order, robust=True), robust * np.r_[100, np.ones(order)],
        rtol=1e-8, atol=1e-10,
    )


def test_ols_mode_retains_exact_legacy_ar_regression_and_order():
    x, signals = simulate(np.random.default_rng(42))
    values = signals["both"][:, 0]
    n = len(values)
    design = np.column_stack([
        np.ones(2 * n),
        np.vstack([estimator._lag_matrix(values, 8), estimator._lag_matrix(values[::-1], 8)]),
    ])
    expected = estimator._stepwise_bic(design, np.r_[values, values[::-1]])
    np.testing.assert_array_equal(estimator._ar_fit(values, 8), expected)
    robust = estimator._ar_fit(values, 8, robust=True)
    assert len(robust) == len(expected)
    result, residual, whitening, _ = traced_fit(values, x, 8, ar_estimator="robust")
    assert result.ar_estimator == "robust"
    np.testing.assert_array_equal(result.ar_filter, whitening)
    np.testing.assert_allclose(result.resid, result.weights * residual, atol=1e-12)
    assert _fit_ar_irls(values, x, 8).ar_estimator == "robust"
    assert _fit_ar_irls(values, x, 8, ar_estimator="ols").ar_estimator == "ols"


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("method", ["robust", "ols"])
def test_solver_retains_masks_method_and_positive_covariance(masked, method):
    import xarray as xr
    ts, design = small_model()
    mask = xr.ones_like(ts.isel(chromo=0, drop=True), dtype=bool).pint.dequantify()
    if masked:
        mask.values[10:20, 0] = False
    results = fit(ts, design, noise_model="ar_irls", ar_order=4, max_jobs=2,
                  quality_mask=mask, ar_estimator=method)
    for result in results.values.ravel():
        assert result.ar_estimator == method
        assert result.ar_order <= 4
    if masked:
        assert np.isnan(results.values[0, 0].weighted_residual[10:20]).all()
    stats = Statistics(results)
    assert stats.covariance.attrs["ar_estimator"] == [method]
    assert np.linalg.eigvalsh(stats.covariance)[0] > 0
    restored = pickle.loads(pickle.dumps(stats))
    assert restored.covariance.attrs["ar_estimator"] == [method]


def test_pipeline_forwards_ar_order_and_estimator(monkeypatch, simulated_recording):
    import pyBrainAnalyzIR.pipelines.modules.glm as module
    import pyBrainAnalyzIR.pipelines.modules.preproccessing as prep
    recording, _ = simulated_recording
    pipeline = prep.mbll(prep.intensity_opticaldensity())
    recording = pipeline.run(recording)
    original = module.glm.fit
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module.glm, "fit", capture)
    job = GLM()
    job.options["ar_order"] = 2
    job.options["ar_estimator"] = "ols"
    job.options["verbose"] = False
    output = job.run(recording)
    assert calls[0]["ar_order"] == 2
    assert calls[0]["ar_estimator"] == "ols"
    assert calls[0]["progress_callback"] is None
    assert output["stats"].covariance.attrs["ar_estimator"] == ["ols"]
    assert GLM().options["ar_estimator"] == "robust"


def test_standalone_module_keeps_terminal_progress_and_loads_older_instances(
    simulated_recording, capsys,
):
    from pyBrainAnalyzIR.pipelines.modules import preproccessing as prep

    recording, _ = simulated_recording
    prep.mbll(prep.intensity_opticaldensity()).run(recording)
    job = GLM()
    job.options.update(noise_model="ols", verbose=True)
    del job.progress_callback  # Pipelines saved before the callback was introduced.
    job.run(recording)
    assert "stats" in recording.timeseries
    assert "%" in capsys.readouterr().err


def test_invalid_method_and_nonfinite_robust_estimate_raise(monkeypatch):
    ts, design = small_model()
    with pytest.raises(ValueError, match="ar_estimator"):
        fit(ts, design, ar_estimator="unknown")
    x, signals = simulate(np.random.default_rng(42))
    with pytest.raises(ValueError, match="ar_estimator"):
        estimator.ar_irls_GLM(signals["none"][:, 0], x, ar_estimator="unknown")
    monkeypatch.setattr(estimator, "_robust_fit", lambda *args, **kwargs: (np.array([np.nan]),))
    with pytest.raises(ValueError, match="nonfinite"):
        estimator._ar_fit(signals["none"][:, 0], 1, robust=True)


def test_slow_robust_ar_fit_converges_with_larger_budget(monkeypatch):
    from pyBrainAnalyzIR.testing.simulations import simData
    from pyBrainAnalyzIR.pipelines.modules import preproccessing as prep

    recording, _ = simData.Data(snr=5, random_seed=0)
    recording = prep.mbll(prep.intensity_opticaldensity()).run(recording)
    original = estimator._robust_fit
    slow = []
    budgets = []

    def capture(design, target, *args, **kwargs):
        if "max_iterations" in kwargs:
            budgets.append(kwargs["max_iterations"])
            if not slow:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    original(design, target, max_iterations=50)
                if any("Bisquare regression" in str(w.message) for w in caught):
                    slow.append((design.copy(), target.copy()))
        return original(design, target, *args, **kwargs)

    monkeypatch.setattr(estimator, "_robust_fit", capture)
    job = GLM()
    job.options["verbose"] = False
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Bisquare regression")
        job.run(recording)
    assert slow and budgets and set(budgets) == {1000}
    design, target = slow[0]
    with pytest.warns(RuntimeWarning, match="reached 50 iterations"):
        original(design, target)
    converged = original(design, target, max_iterations=1000)
    reference = original(design, target, max_iterations=2000)
    for actual, expected in zip(converged, reference):
        np.testing.assert_array_equal(actual, expected)


def test_bisquare_still_warns_when_iteration_budget_is_exhausted():
    rng = np.random.default_rng(42)
    design = np.column_stack([np.ones(100), rng.normal(size=100)])
    target = design @ [1., 2.] + rng.normal(size=100)
    target[:5] += 20
    with pytest.warns(RuntimeWarning, match="reached 1 iterations"):
        result = estimator._robust_fit(design, target, max_iterations=1)
    assert np.isfinite(result[0]).all()
    default = estimator._robust_fit(design, target)
    legacy = estimator._robust_fit(design, target, max_iterations=50)
    for actual, expected in zip(default, legacy):
        np.testing.assert_array_equal(actual, expected)
