"""Paired example-03 ROC/FDR ablation; production estimators are not modified.

Run ``python -m pyBrainAnalyzIR.testing.covariance_fdr --output results.json``.
All variants share simulated data, GLM coefficients, degrees of freedom, and BH
families. The raw/marginal-only variants are diagnostic, potentially indefinite
matrices; they must not be used as full-covariance group inputs.
"""
import argparse
import contextlib
import copy
import io
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t
from statsmodels.stats.multitest import multipletests
import cedalion.models.glm as cedalion_glm

from pyBrainAnalyzIR.dataclasses.statistics import Statistics, _ar_irls_joint_covariance
from pyBrainAnalyzIR.math.glm.covariance import regularize_joint_covariance
from pyBrainAnalyzIR.math.glm.solve import fit
from pyBrainAnalyzIR.pipelines.modules import glm, preproccessing as prep
from pyBrainAnalyzIR.testing import simData
from pyBrainAnalyzIR.testing.channelROC import PipelineList
from pyBrainAnalyzIR.testing.covariance_validation import bootstrap_interval


VARIANTS = ("legacy_raw", "marginals_only", "individual_blocks_psd", "legacy_blocks_psd", "current_repaired")
METRICS = ("fdr", "fpr", "power", "auc", "discoveries", "any_false_discovery")


def covariance_variants(results):
    """Isolate block replacement from off-block projection/shrinkage."""
    stats = Statistics(results)
    if not results.attrs.get("description", "").startswith("AR_IRLS"):
        # OLS is deliberately unaffected by this AR-only regularization.
        return stats, {name: stats.covariance.values.copy() for name in VARIANTS}, {}
    fits = list(results.values.ravel())
    raw = _ar_irls_joint_covariance(fits)
    blocks = [np.asarray(result.cov_params()) for result in fits]
    marginal = raw.copy()
    sizes = [len(block) for block in blocks]
    offsets = np.cumsum([0] + sizes)
    legacy_blocks = []
    for index, block in enumerate(blocks):
        section = slice(offsets[index], offsets[index + 1])
        legacy_blocks.append(raw[section, section].copy())
        marginal[section, section] = block
    legacy_psd, legacy_info = regularize_joint_covariance(raw, legacy_blocks)
    individual_psd, _ = regularize_joint_covariance(raw, blocks)
    current = stats.covariance.values.copy()
    for name, matrix in (("current", current), ("legacy_psd", legacy_psd)):
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} has unavailable uncertainty; cannot score as a valid paired fit.")
    np.testing.assert_array_equal(np.diag(marginal), np.diag(individual_psd))
    np.testing.assert_array_equal(np.diag(raw), np.diag(legacy_psd))
    np.testing.assert_array_equal(current, legacy_psd)
    task = np.flatnonzero(stats.betas.conditions.values == "HRF A")
    diagnostics = {
        "current": copy.deepcopy(stats.covariance.attrs),
        "legacy_blocks_psd": legacy_info,
        "legacy_min_eigenvalue": float(np.linalg.eigvalsh((raw + raw.T) / 2)[0]),
        "median_task_variance_ratio_current_to_legacy": float(np.median(
            np.diag(current)[task] / np.diag(raw)[task],
        )),
        "min_task_variance_ratio_current_to_legacy": float(np.min(
            np.diag(current)[task] / np.diag(raw)[task],
        )),
        "max_task_variance_ratio_current_to_legacy": float(np.max(
            np.diag(current)[task] / np.diag(raw)[task],
        )),
    }
    return stats, dict(zip(VARIANTS, (raw, marginal, individual_psd, legacy_psd, current))), diagnostics


def score_family(pvalues, positive, qvalues=None):
    """Realized FDP=V/max(R,1); its replicate average estimates FDR."""
    pvalues = np.asarray(pvalues)
    positive = np.asarray(positive, dtype=bool)
    if not len(pvalues) or not np.isfinite(pvalues).all() or pvalues.shape != positive.shape:
        raise ValueError("A nonempty finite, truth-aligned test family is required.")
    qvalues = multipletests(pvalues, method="fdr_bh")[1] if qvalues is None else np.asarray(qvalues)
    reject = qvalues < .05
    false = int((reject & ~positive).sum())
    discoveries = int(reject.sum())
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    auc = (
        (rankdata(-pvalues)[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
        if n_positive and n_negative else None
    )
    return {
        "fdr": false / max(discoveries, 1),
        "fpr": float(np.mean(pvalues[~positive] < .05)) if n_negative else None,
        "power": float(np.mean(reject[positive])) if n_positive else None,
        "auc": float(auc) if auc is not None else None,
        "discoveries": discoveries,
        "any_false_discovery": float(false > 0),
        "false_discoveries": false, "null_tests": n_negative, "alternative_tests": n_positive,
    }


def score_covariance(stats, covariance, truth):
    values = np.asarray(stats.betas)
    variances = np.diag(covariance)
    if not np.isfinite(variances).all() or (variances <= 0).any():
        raise ValueError("Scalar tests require positive finite variances.")
    p = 2 * t.cdf(-np.abs(values / np.sqrt(variances)), stats.dof)
    task = np.flatnonzero(stats.betas.conditions.values == "HRF A")
    labels = pd.MultiIndex.from_arrays([
        stats.betas.channels.values[task], stats.betas.type.values[task],
    ])
    truth = truth.transpose("channel", "chromo")
    truth_series = pd.Series(
        truth.values.ravel().astype(bool),
        index=pd.MultiIndex.from_product([truth.channel.values, truth.chromo.values]),
    )
    aligned = truth_series.reindex(labels)
    if aligned.isna().any():
        raise ValueError("Simulation truth does not cover every scored task coefficient.")
    positive = aligned.to_numpy(dtype=bool)
    output = {"task_all_types": score_family(p[task], positive)}
    for signal in np.unique(stats.betas.type.values[task]):
        mask = stats.betas.type.values[task] == signal
        output[f"task_{signal}"] = score_family(p[task][mask], positive[mask])
    # Matches Statistics.table's global BH family, then scores task entries.
    # This is task-subset FDP, NOT FDR over the global family including drifts.
    global_q = multipletests(p, method="fdr_bh")[1]
    output["toolbox_global_q_task_subset"] = score_family(p[task], positive, global_q[task])
    return output


def summarize_rows(rows, seed):
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    keys = ["snr", "noise_model", "family"]
    summary, paired = [], []
    for key, group in frame.groupby(keys):
        for variant, selected in group.groupby("variant"):
            summary.append({
                **dict(zip(keys, key)), "variant": variant,
                "metrics": {
                    metric: bootstrap_interval(
                        selected[metric].to_numpy(dtype=float), rng,
                        bounded=metric != "discoveries",
                    ) for metric in METRICS
                },
            })
        for variant in VARIANTS[1:]:
            comparison = {**dict(zip(keys, key)), "variant_minus_legacy": variant, "metrics": {}}
            for metric in METRICS:
                wide = group.pivot(index="replicate", columns="variant", values=metric)
                delta = (wide[variant] - wide["legacy_raw"]).to_numpy(dtype=float)
                comparison["metrics"][metric] = bootstrap_interval(delta, rng)
            paired.append(comparison)
    return summary, paired


def run(replicates=100, snrs=(0., 1., 10.), fs=4., seed=20261029):
    if replicates < 2:
        raise ValueError("At least two independent simulation replicates are required.")
    rows, diagnostics, curves = [], [], []
    warning_counts = Counter()
    preprocessing = PipelineList([prep.intensity_opticaldensity, prep.mbll, prep.resample])
    preprocessing.set_all_options({"Fs": fs})
    options = glm.GLM().options
    thresholds = np.unique(np.r_[0., np.logspace(-10, -1, 120), np.linspace(.1, 1., 91)])
    for scenario, snr in enumerate(snrs):
        for replicate in range(replicates):
            rng = np.random.default_rng(np.random.SeedSequence([seed, scenario, replicate]))
            with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("always")
                rec, truth = simData.Data(snr=snr, random_seed=rng)
                # With snr=0 no coefficient received a response, even though Data's
                # channel assignment mask still marks half the channels.
                if snr == 0:
                    truth = truth * False
                rec = preprocessing.run(rec)
                ts = rec["conc"]
                design = (
                    cedalion_glm.design_matrix.hrf_regressors(ts, rec.stim, options["basis_function"])
                    & cedalion_glm.design_matrix.drift_regressors(ts, drift_order=options["drift_order"])
                )
                for noise_model in ("ols", "ar_irls"):
                    results = fit(ts, design, noise_model=noise_model, max_jobs=1, verbose=False)
                    stats, matrices, info = covariance_variants(results)
                    diagnostics.append({"snr": snr, "replicate": replicate,
                                        "noise_model": noise_model, **info})
                    for variant, matrix in matrices.items():
                        scored = score_covariance(stats, matrix, truth)
                        for family, metrics in scored.items():
                            rows.append({
                                "snr": snr, "replicate": replicate, "noise_model": noise_model,
                                "variant": variant, "family": family, **metrics,
                            })
                        task = np.flatnonzero(stats.betas.conditions.values == "HRF A")
                        p = 2 * t.cdf(
                            -np.abs(np.asarray(stats.betas)[task] / np.sqrt(np.diag(matrix)[task])),
                            stats.dof,
                        )
                        positive = np.array([
                            bool(truth.sel(channel=ch, chromo=ty))
                            for ch, ty in zip(stats.betas.channels.values[task], stats.betas.type.values[task])
                        ])
                        curves.append({
                            "snr": snr, "replicate": replicate, "noise_model": noise_model,
                            "variant": variant,
                            "fpr": [float(np.mean(p[~positive] <= level)) for level in thresholds],
                            "tpr": [float(np.mean(p[positive] <= level)) if positive.any() else None
                                    for level in thresholds],
                        })
            warning_counts.update(str(item.message) for item in caught)
            print(f"SNR {snr:g}: {replicate + 1}/{replicates}", flush=True)
    summary, paired = summarize_rows(rows, seed + 1)
    return {
        "configuration": {
            "replicates_per_snr": replicates, "snrs": list(snrs), "Fs": fs, "seed": seed,
            "ar_order": 30, "alpha": .05,
            "ar_estimator": "robust",
            "definition": "FDR = mean over replicates of V/max(R,1) at BH q<0.05; FPR uses raw p<0.05.",
            "pairing": "Same data, coefficients, DOF, BH families; only covariance changes.",
            "current_marginal_source": "legacy_ar_irls_joint_diagonal_blocks",
            "primary_family": "task_all_types: HRF A across all channels and both chromophores.",
            "toolbox_family_note": "toolbox_global_q_task_subset is task-subset FDP after BH over "
                                   "all coefficients including drifts; not global-family FDR.",
        },
        "summary": summary, "paired_differences": paired, "raw_metrics": rows,
        "covariance_diagnostics": diagnostics, "warning_counts": dict(warning_counts),
        "roc_thresholds": thresholds.tolist(), "roc_replicates": curves,
    }


def plot_results(result, path):
    """Plot paired raw-p ROC and actual BH false-discovery proportions."""
    import matplotlib.pyplot as plt
    labels = {
        "legacy_raw": "Legacy",
        "individual_blocks_psd": "Rejected individual blocks",
        "current_repaired": "PSD, legacy marginals (current)",
    }
    colors = {"legacy_raw": "tab:blue", "individual_blocks_psd": "tab:red",
              "current_repaired": "tab:green"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    nonnull = [snr for snr in result["configuration"]["snrs"] if snr > 0]
    for axis, snr in zip(axes[0, :2], nonnull):
        for variant, label in labels.items():
            selected = [
                row for row in result["roc_replicates"]
                if row["snr"] == snr and row["noise_model"] == "ar_irls" and row["variant"] == variant
            ]
            axis.plot(np.mean([row["fpr"] for row in selected], axis=0),
                      np.mean([row["tpr"] for row in selected], axis=0),
                      label=label, color=colors[variant])
        axis.plot([0, 1], [0, 1], "--", color=".7")
        axis.set(xlabel="Raw-p false-positive rate", ylabel="Sensitivity",
                 title=f"AR-IRLS ROC, SNR={snr:g}", xlim=(0, 1), ylim=(0, 1))
        axis.legend(fontsize=8)
    ratios = [row["median_task_variance_ratio_current_to_legacy"]
              for row in result["covariance_diagnostics"] if row["noise_model"] == "ar_irls"]
    axes[0, 2].hist(ratios, bins=25, color="tab:red")
    axes[0, 2].axvline(1, color="black", linestyle="--")
    axes[0, 2].set(title="Marginal task variance change", xlabel="Current / legacy variance",
                   ylabel="Recordings")
    for axis, snr in zip(axes[1], result["configuration"]["snrs"]):
        entries = [
            row for row in result["summary"]
            if row["snr"] == snr and row["family"] == "task_all_types"
            and ((row["noise_model"] == "ar_irls" and row["variant"] in labels)
                 or (row["noise_model"] == "ols" and row["variant"] == "current_repaired"))
        ]
        ticklabels = []
        for index, row in enumerate(entries):
            estimate = row["metrics"]["fdr"]
            mean = estimate["mean"]
            low, high = estimate["ci95"]
            ols = row["noise_model"] == "ols"
            axis.bar(index, mean, color=".5" if ols else colors[row["variant"]])
            axis.errorbar(index, mean, yerr=[[max(0, mean - low)], [max(0, high - mean)]],
                          color="black", capsize=3)
            ticklabels.append("OLS control" if ols else labels[row["variant"]])
        axis.axhline(.05, color="black", linestyle="--", label="Nominal q=0.05")
        axis.set(title=f"Actual FDR at BH q<0.05, SNR={snr:g}", ylabel="Mean V / max(R, 1)",
                 xticks=range(len(entries)), xticklabels=ticklabels, ylim=(0, 1.05))
        axis.tick_params(axis="x", labelrotation=25)
    fig.suptitle("Example 03: paired recordings, fixed coefficients/DOF, task-wide BH family")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--snrs", type=float, nargs="+", default=[0., 1., 10.])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.replicates, args.snrs)
    # Pandas converts missing AUC/power and NumPy scalars into valid JSON values.
    result = json.loads(pd.Series(result).to_json(double_precision=15))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(result["raw_metrics"]).to_csv(args.output.with_suffix(".csv"), index=False)
    plot_results(result, args.output.with_suffix(".png"))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
