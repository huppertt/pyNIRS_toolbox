"""Paired ablations and actual FDR, rather than raw-p false-positive rates."""
import warnings

import numpy as np
import pytest

pytest.importorskip("cedalion")
from pyBrainAnalyzIR.math.glm.solve import fit  # noqa: E402
from pyBrainAnalyzIR.testing.covariance_fdr import (  # noqa: E402
    covariance_variants, score_covariance, score_family, summarize_rows,
)
from tests.test_model_fit_data import small_model  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


def test_fdr_is_false_discovery_proportion_not_raw_false_positive_rate():
    score = score_family([.001, .02, .04, .9], [True, False, False, False])
    assert score["discoveries"] == 2
    assert score["false_discoveries"] == 1
    assert score["fdr"] == .5
    assert score["fpr"] == pytest.approx(2 / 3)
    assert score["power"] == 1
    assert score_family([.8, .9], [False, False])["fdr"] == 0
    assert score_family([.001, .9], [False, False])["fdr"] == 1


@pytest.mark.parametrize("noise_model", ["ols", "ar_irls"])
def test_ablation_isolates_marginal_change_and_off_diagonal_projection(noise_model):
    ts, design = small_model()
    results = fit(ts, design, noise_model=noise_model, max_jobs=1, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        stats, variants, _ = covariance_variants(results)
    np.testing.assert_array_equal(np.diag(variants["legacy_raw"]), np.diag(variants["legacy_blocks_psd"]))
    np.testing.assert_array_equal(np.diag(variants["marginals_only"]), np.diag(variants["individual_blocks_psd"]))
    np.testing.assert_array_equal(variants["legacy_blocks_psd"], variants["current_repaired"])
    import xarray as xr
    truth = xr.DataArray([[True, False], [False, True]], dims=["channel", "chromo"],
                         coords={"channel": ts.channel, "chromo": ts.chromo})
    scores = {name: score_covariance(stats, matrix, truth) for name, matrix in variants.items()}
    assert scores["legacy_raw"] == scores["legacy_blocks_psd"]
    assert scores["marginals_only"] == scores["individual_blocks_psd"]
    assert scores["legacy_raw"] == scores["current_repaired"]
    if noise_model == "ols":
        for value in variants.values():
            np.testing.assert_array_equal(value, stats.covariance)


def test_paired_fdr_interval_uses_replicates_and_preserves_pairing():
    rows = []
    for replicate in range(10):
        for variant in ["legacy_raw", "marginals_only", "individual_blocks_psd",
                        "legacy_blocks_psd", "current_repaired"]:
            shift = .1 if variant in ("marginals_only", "current_repaired") else 0
            value = replicate / 20 + shift
            rows.append({
                "snr": 1., "noise_model": "ar_irls", "family": "task_all_types",
                "variant": variant, "replicate": replicate,
                **{name: value for name in [
                    "fdr", "fpr", "power", "auc", "discoveries", "any_false_discovery",
                ]},
            })
    summary, paired = summarize_rows(rows, 42)
    assert len(summary) == 5
    current = next(item for item in paired if item["variant_minus_legacy"] == "current_repaired")
    assert current["metrics"]["fdr"]["mean"] == pytest.approx(.1)
    np.testing.assert_allclose(current["metrics"]["fdr"]["ci95"], [.1, .1])
