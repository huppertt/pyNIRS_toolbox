"""Paired marginal-block-preserving covariance validation for ROI and groups.

The legacy full-covariance eigenvalue floor is reproduced ONLY for the study.
All production fits retain strict covariance validation. Raw ROI covariances
are scored without inventing positive variances; invalid cases are reported.
Both covariance variants use the default robust AR estimator; ``legacy`` here
means unrepaired covariance assembly, not OLS AR coefficient estimation.
"""
import argparse
import copy
import json
import warnings
from collections import Counter
from pathlib import Path

import cedalion.dataclasses as cdc
import numpy as np
import pandas as pd
from scipy.stats import t

from pyBrainAnalyzIR.dataclasses.dataset import DataSet
from pyBrainAnalyzIR.dataclasses.roi import quality_masks
from pyBrainAnalyzIR.dataclasses.statistics import Statistics, _ar_irls_joint_covariance
from pyBrainAnalyzIR.math.glm.solve import fit, _recording_quality_mask
from pyBrainAnalyzIR.pipelines.modules.mixedeffects import MixedEffects
from pyBrainAnalyzIR.sigproc.roi import roi_average
from pyBrainAnalyzIR.testing.helpers.covariance_fdr import score_family
from pyBrainAnalyzIR.testing.helpers.covariance_validation import (
    REGIMES, MODELS, bootstrap_interval, simulate_subject,
)


SURFACES = ("channel", "roi_stats_equal", "roi_stats_iv",
            "roi_timecourse_equal", "roi_timecourse_iv")
VARIANTS = ("legacy_full_floor", "legacy_diagonal", "repaired_full", "repaired_diagonal")
METRICS = ("fdr", "fpr", "power", "sensitivity", "auc", "coverage", "bias", "rmse")


def geometry():
    points = [[index * 40, 0, 0] for index in range(4)]
    points += [[index * 40 + 25, 0, 0] for index in range(4)]
    return cdc.build_labeled_points(
        np.array(points, dtype=float), crs="pos", units="mm",
        labels=[f"S{i + 1}" for i in range(4)] + [f"D{i + 1}" for i in range(4)],
        types=[cdc.PointType.SOURCE] * 4 + [cdc.PointType.DETECTOR] * 4,
    )


def roi_table():
    return pd.DataFrame({
        "source": [f"S{i + 1}" for i in range(4)],
        "detector": [f"D{i + 1}" for i in range(4)],
        "name": ["Active", "Active", "Null", "Null"],
        "weight": [1., 2., 1., 2.],
    })


def selected_table(stats, model):
    table = stats.table()
    if model == "age":
        return table[table.Condition.str.contains("HRF A", regex=False)
                     & table.Condition.str.contains("age", regex=False)]
    if model == "categorical":
        return table[table.Condition.str.contains("HRF A", regex=False)
                     & table.Condition.str.contains("[T.M]", regex=False)]
    if model == "mean":
        return table[table.Condition == "Condition[HRF A]"]
    return table[table.Condition == model]


def inference(stats, expected, model):
    table = selected_table(stats, model).set_index(["Channel", "Type"])
    target = expected.reindex(table.index).to_numpy()
    p = table["P-values"].to_numpy()
    estimates = table.Beta.to_numpy()
    se = table.StdErr.to_numpy()
    if table.empty or not np.isfinite(np.column_stack([p, estimates, se, target])).all() or (se <= 0).any():
        raise ValueError("Invalid or truth-misaligned scalar inference; no rates were imputed.")
    score = score_family(p, np.abs(target) > 1e-12)
    radius = t.ppf(.975, stats.dof) * se
    score.update(
        sensitivity=float(np.mean(p[np.abs(target) > 1e-12] < .05))
        if np.any(np.abs(target) > 1e-12) else None,
        coverage=float(np.mean(np.abs(estimates - target) <= radius)),
        bias=float(np.mean(estimates - target)),
        rmse=float(np.sqrt(np.mean((estimates - target) ** 2))),
    )
    return score


def joint_contrasts(stats, truths):
    """Fixed cross-signal sum/difference tests using the full stored covariance.

    Unlike PSD checks, their empirical sampling variance and coverage test the
    statistical accuracy of off-diagonal entries. No variance floors are used.
    """
    selected = np.flatnonzero(stats.betas.conditions.values == "HRF A")
    labels = [(str(stats.betas.channels.values[index]), str(stats.betas.type.values[index]))
              for index in selected]
    condition_indices = {
        term: {label: index for index, label in enumerate(zip(
            map(str, stats.betas.channels.values), map(str, stats.betas.type.values),
        )) if stats.betas.conditions.values[index] == term}
        for term in ("HRF A", "HRF B")
    }
    rows = []
    for left, left_label in enumerate(labels):  # pylint: disable=too-many-nested-blocks
        for right_label in labels[left + 1:]:
            if left_label[0] != right_label[0] and left_label[1] != right_label[1]:
                continue
            for term in truths:
                for sign, name in ((1, "mean"), (-1, "difference")):
                    weights = np.zeros(len(stats.betas))
                    for label, weight in ((left_label, .5), (right_label, sign * .5)):
                        weights[condition_indices["HRF A"][label]] = weight
                        if term == "HRF B":
                            weights[condition_indices["HRF A"][label]] = 0.
                            weights[condition_indices["HRF B"][label]] = weight
                        elif term == "HRF A-HRF B":
                            weights[condition_indices["HRF B"][label]] = -weight
                    active = np.flatnonzero(weights)
                    variance = float(weights[active] @ stats.covariance.values[np.ix_(active, active)]
                                     @ weights[active])
                    estimate = float(weights @ stats.betas.values)
                    expected = .5 * (truths[term].loc[left_label] + sign * truths[term].loc[right_label])
                    if not np.isfinite([variance, estimate, expected]).all() or variance <= 0:
                        raise ValueError(f"Invalid joint contrast variance for {left_label}, {right_label}, {term}.")
                    se = np.sqrt(variance)
                    rows.append({
                        "model": term, "contrast": f"{left_label}/{right_label}/{name}",
                        "error": estimate - expected, "variance": variance,
                        "coverage": float(abs(estimate - expected) <= t.ppf(.975, stats.dof) * se),
                        "fpr": float(2 * t.sf(abs(estimate / se), stats.dof) < .05)
                        if abs(expected) <= 1e-12 else None,
                    })
    return rows


def summarize_joint(rows, seed):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    rng = np.random.default_rng(seed)
    output = []
    for key, group in frame.groupby(["regime", "surface", "variant", "model"]):
        rates = group.groupby("replicate")[["coverage", "fpr"]].mean()
        per_contrast = group.groupby("contrast").agg(
            empirical_variance=("error", "var"), estimated_variance=("variance", "mean"),
        )
        output.append({
            **dict(zip(("regime", "surface", "variant", "model"), key)),
            "replicates": int(group.replicate.nunique()), "contrasts": len(per_contrast),
            "coverage": bootstrap_interval(rates.coverage, rng, bounded=True),
            "raw_fpr": bootstrap_interval(rates.fpr, rng, bounded=True),
            "empirical_to_estimated_variance": float(
                per_contrast.empirical_variance.mean() / per_contrast.estimated_variance.mean(),
            ),
            "mean_bias": float(group.error.mean()),
        })
    return output


def record(stats, template):
    rec = cdc.Recording()
    rec["stats"] = stats
    rec.meta_data.update(template.meta_data)
    rec.geo3d = template.geo3d
    return rec


def coefficient_truth(stats, truth):
    return np.array([
        1. if condition == "Drift 0" else float(truth.loc[(channel, kind)]) * (
            .5 if condition == "HRF B" else 1.
        )
        for channel, kind, condition in zip(
            stats.betas.channels.values, stats.betas.type.values, stats.betas.conditions.values,
        )
    ])


def projected_truth(stats, original_truth, condition="HRF A"):
    target = stats.roi_pooling_matrix.values @ original_truth
    task = stats.betas.conditions.values == condition
    return pd.Series(target[task], index=pd.MultiIndex.from_arrays([
        stats.betas.channels.values[task], stats.betas.type.values[task],
    ], names=["Channel", "Type"]))


def task_truths(a, b):
    return {"HRF A": a, "HRF B": b, "HRF A-HRF B": a - b}


def paired_subject(rng, subject, subjects, regime, *, random_effect_sd=.06):
    records, truth, ts, design, mask = simulate_subject(
        rng, subject, subjects, regime, two_conditions=True, return_inputs=True,
        random_effect_sd=random_effect_sd,
    )
    sources = [f"S{i + 1}" for i in range(4)]
    detectors = [f"D{i + 1}" for i in range(4)]
    geo = geometry()
    for rec in records:
        rec.geo3d = geo
        stats = rec["stats"]
        stats.geo3d = geo
        stats.betas = stats.betas.assign_coords(
            source=("indices", np.repeat(sources, 6)),
            detector=("indices", np.repeat(detectors, 6)),
        )
    legacy, repaired = [rec["stats"] for rec in records]
    np.testing.assert_array_equal(legacy.table(), repaired.table())
    surfaces = {"channel": (records, [task_truths(truth, .5 * truth)] * 2)}
    original_truth = coefficient_truth(legacy, truth)
    table = roi_table()
    for weighted, suffix in ((False, "equal"), (True, "iv")):
        pools = [roi_average(stats, table, weighted=weighted) for stats in (legacy, repaired)]
        np.testing.assert_array_equal(pools[0].betas, pools[1].betas)
        surfaces[f"roi_stats_{suffix}"] = (
            [record(stats, records[0]) for stats in pools],
            [task_truths(projected_truth(stats, original_truth),
                         projected_truth(stats, original_truth, "HRF B")) for stats in pools],
        )
        # Common per-ROI masks keep the noiseless pooled task representable by
        # the same GLM. With changing membership, there is no single true beta.
        rec = cdc.Recording()
        rec.geo3d = geo
        rec["conc"] = ts.assign_coords(source=("channel", sources), detector=("channel", detectors))
        if mask is not None:
            common_mask = mask.copy()
            for indices in ([0, 1], [2, 3]):
                common_mask.values[:, indices] = mask.values[:, indices].all(axis=1)[:, None]
            rec.set_mask("simulation", common_mask)
        roi_average(rec, table, input_name="conc", output_name="roi", weighted=weighted)
        regional = rec["roi"]
        roi_mask = _recording_quality_mask(regional, quality_masks(rec, regional))
        results = fit(regional, design, noise_model="ar_irls", ar_order=8,
                      quality_mask=roi_mask, max_jobs=1, verbose=False)
        new = Statistics(results)
        new.geo3d, new.geoROI = geo, rec.aux_obj["roi_definitions"]["default"]
        old = copy.deepcopy(new)
        old.covariance.values[:] = _ar_irls_joint_covariance(list(results.values.ravel()))
        np.testing.assert_array_equal(old.table(), new.table())
        weights = new.geoROI.weights
        actual = []
        labels = []
        for region in regional.channel.values:
            for kind in regional.chromo.values:
                w = weights.sel(channel=region)
                if "chromo" in w.dims:
                    w = w.sel(chromo=kind)
                actual.append(sum(float(weight) * truth.loc[(channel, kind)]
                                  for channel, weight in zip(w.source_channel.values, w.values)))
                labels.append((region, kind))
        regional_truth = pd.Series(actual, index=pd.MultiIndex.from_tuples(
            labels, names=["Channel", "Type"],
        ))
        surfaces[f"roi_timecourse_{suffix}"] = (
            [record(stats, records[0]) for stats in (old, new)],
            [task_truths(regional_truth, .5 * regional_truth)] * 2,
        )
    return surfaces


def floor_task_covariances(dataset):
    """Controlled baseline reproduction, after ROI pooling and nuisance removal."""
    count = 0
    for rec in dataset.dataset:
        stats = rec["stats"]
        indices = np.flatnonzero(np.char.startswith(stats.betas.conditions.values.astype(str), "HRF "))
        block = stats.covariance.values[np.ix_(indices, indices)]
        eigenvalues, vectors = np.linalg.eigh((block + block.T) / 2)
        threshold = 1e-10 * np.max(np.abs(eigenvalues))
        if eigenvalues[0] <= threshold:
            count += 1
            stats.covariance.values[np.ix_(indices, indices)] = (
                vectors * np.maximum(eigenvalues, threshold)
            ) @ vectors.T
    return count


def population_truth(stats, regime, model):
    table = selected_table(stats, model)
    amplitude = {"mean": .3, "age": .25, "categorical": .3}[model]
    expected = []
    for channel in table.Channel:
        active = channel in ("S1D1", "S2D2", "Active")
        expected.append(amplitude if active and regime != "ar_global_null" else 0.)
    return pd.Series(expected, index=pd.MultiIndex.from_frame(table[["Channel", "Type"]]))


def summary(rows, seed):
    frame = pd.DataFrame(rows)
    frame[list(METRICS)] = frame[list(METRICS)].astype(float)
    keys = ["regime", "surface", "level", "model", "robust", "variant"]
    aggregate = frame.groupby(keys + ["replicate"], as_index=False).mean(numeric_only=True)
    rng = np.random.default_rng(seed)
    summaries, pairs = [], []
    for values, group in aggregate.groupby(keys):
        summaries.append({
            **dict(zip(keys, values)),
            "metrics": {metric: bootstrap_interval(
                group[metric], rng, bounded=metric not in ("bias", "rmse"),
            ) for metric in METRICS},
        })
    for values, group in aggregate.groupby(keys[:-1]):
        wide = {metric: group.pivot(index="replicate", columns="variant", values=metric)
                for metric in METRICS}
        comparisons = (
            ("legacy", "repaired"), ("legacy_full_floor", "repaired_full"),
            ("legacy_diagonal", "repaired_diagonal"),
        )
        for before, after in comparisons:
            if before not in wide["fdr"] or after not in wide["fdr"]:
                continue
            pairs.append({
                **dict(zip(keys[:-1], values)), "before": before, "after": after,
                "metrics": {metric: bootstrap_interval(
                    (wide[metric][after] - wide[metric][before]).to_numpy(dtype=float), rng,
                ) for metric in METRICS},
                "complete_pairs": int((wide["fdr"][before].notna() & wide["fdr"][after].notna()).sum()),
            })
    return aggregate.to_dict("records"), summaries, pairs


def run(replicates=100, subjects=12, regimes=REGIMES, seed=20261030):  # pylint: disable=too-many-nested-blocks
    if replicates < 2 or subjects < 6 or subjects % 2:
        raise ValueError("Use >=2 replicates and an even subject count >=6.")
    rows, diagnostics, failures = [], [], []
    warning_counts = Counter()
    fits_attempted = 0
    for regime_index, regime in enumerate(regimes):
        for iteration in range(replicates):
            rng = np.random.default_rng(np.random.SeedSequence([seed, regime_index, iteration]))
            datasets = {surface: [[], []] for surface in SURFACES}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for subject in range(subjects):
                    surfaces = paired_subject(rng, subject, subjects, regime)
                    for surface, (records, truths) in surfaces.items():
                        for index, variant in enumerate(("legacy", "repaired")):
                            stats = records[index]["stats"]
                            datasets[surface][index].append(records[index])
                            if index:
                                diagnostics.append({
                                    "regime": regime, "replicate": iteration, "subject": subject,
                                    "surface": surface, **stats.covariance.attrs,
                                })
                            for term in ("HRF A", "HRF B", "HRF A-HRF B"):
                                local = stats.copy()
                                if "-" in term:
                                    local.ttest([term])
                                try:
                                    score = inference(local, truths[index][term], term)
                                except ValueError as exc:
                                    failures.append({
                                        "regime": regime, "replicate": iteration, "subject": subject,
                                        "surface": surface, "level": "first", "model": term,
                                        "variant": variant, "error": str(exc),
                                    })
                                    continue
                                rows.append({
                                    "regime": regime, "replicate": iteration, "subject": subject,
                                    "surface": surface, "level": "first", "model": term,
                                    "robust": True, "variant": variant, **score,
                                })
                for surface, paired in datasets.items():  # pylint: disable=too-many-nested-blocks
                    # The group experiment targets task A; the existing ROC
                    # notebook separately exercises simultaneous A/B group fits.
                    for recordings in paired:
                        for rec in recordings:
                            rec["stats"] = rec["stats"].copy()
                            rec["stats"].keep_condition("HRF A")
                    legacy, repaired = [DataSet(records) for records in paired]
                    floored = copy.deepcopy(legacy)
                    floor_count = floor_task_covariances(floored)
                    diagnostics.append({"regime": regime, "replicate": iteration,
                                        "surface": surface, "legacy_group_blocks_floored": floor_count})
                    for model, formula in MODELS.items():
                        for robust in (False, True):
                            for variant in VARIANTS:
                                data = (
                                    floored if variant == "legacy_full_floor" else
                                    legacy if variant == "legacy_diagonal" else repaired
                                )
                                job = MixedEffects()
                                job.options["FE_formula"] = formula
                                job.options["robust"] = robust
                                job.options["covariance_weighting"] = (
                                    "diagonal" if variant.endswith("diagonal") else "full"
                                )
                                fits_attempted += 1
                                try:
                                    group = job.run(copy.deepcopy(data))["groupstats"]
                                    score = inference(group, population_truth(group, regime, model), model)
                                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                                    failures.append({
                                        "regime": regime, "replicate": iteration, "surface": surface,
                                        "level": "group", "model": model, "robust": robust,
                                        "variant": variant, "error": f"{type(exc).__name__}: {exc}",
                                    })
                                    continue
                                rows.append({
                                    "regime": regime, "replicate": iteration, "subject": -1,
                                    "surface": surface, "level": "group", "model": model,
                                    "robust": robust, "variant": variant, **score,
                                    "conditional_scale_collapse": float(
                                        group.covariance.attrs["conditional_residual_collapse"],
                                    ),
                                })
                                if surface == "channel":
                                    for weighted, suffix in ((False, "equal"), (True, "iv")):
                                        # Group effects are shared by the members of each ROI;
                                        # their population target is unchanged by normalized weights.
                                        regional = roi_average(group, roi_table(), weighted=weighted)
                                        region_score = inference(
                                            regional, population_truth(regional, regime, model), model,
                                        )
                                        rows.append({
                                            "regime": regime, "replicate": iteration, "subject": -1,
                                            "surface": f"group_then_roi_{suffix}", "level": "group",
                                            "model": model, "robust": robust, "variant": variant,
                                            **region_score,
                                        })
            warning_counts.update(str(item.message) for item in caught)
            print(f"{regime}: {iteration + 1}/{replicates}; "
                  f"group attempts={fits_attempted}, failures={len(failures)}", flush=True)
    aggregate, summaries, pairs = summary(rows, seed + 1)
    return {
        "configuration": {
            "replicates_per_regime": replicates, "subjects": subjects, "regimes": list(regimes),
            "seed": seed, "conditions": 2, "ar_order": 8, "alpha": .05,
            "ar_estimator": "robust",
            "group_random_effects": {
                "formula": "subject", "group_by": "auto",
            },
            "roc_metrics": "fpr and sensitivity use raw p<.05; auc ranks raw p. "
                           "fdr and power are supplementary BH metrics, not ROC calibration.",
            "group_condition": "HRF A from simultaneous two-condition first-level fits",
            "marginal_source": "legacy_ar_irls_joint_diagonal_blocks",
            "family": "BH over selected task, within-signal contrast, or group term across signals.",
            "first_level_fdr": "Mean subject FDP within replicate, then mean across independent datasets.",
            "timecourse_mask_policy": "Intersection within each ROI so pooled noiseless signal has a "
                                      "single true coefficient; channel-fit pooling retains unequal masks.",
            "legacy_floor_policy": "Only for legacy full group input: floor retained task eigenvalues "
                                   "to 1e-10*maxabs, after ROI pooling. Raw first-level ROI not floored.",
            "group_fits_attempted": fits_attempted,
            "failure_policy": "Failures listed explicitly; complete-pair counts reported; no imputed rates.",
        },
        "summary": summaries, "paired_differences": pairs, "replicate_metrics": aggregate,
        "raw_metrics": rows, "covariance_diagnostics": diagnostics,
        "failures": failures, "warning_counts": dict(warning_counts),
    }


def run_first_level(replicates=100, regimes=REGIMES, seed=20261031):
    """ROI discrimination with exact-null subject effects, not population nulls.

    The group study has random subject effects in a population-null ROI, making
    its individual betas non-null. This separate experiment removes random
    effects so both exact-null and active ROIs exist for first-level ROC.
    """
    if replicates < 2:
        raise ValueError("Use at least two independent recordings.")
    rows, diagnostics, failures, joint_rows = [], [], [], []
    warning_counts = Counter()
    for regime_index, regime in enumerate(regimes):
        for iteration in range(replicates):
            rng = np.random.default_rng(np.random.SeedSequence([seed, regime_index, iteration]))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                surfaces = paired_subject(
                    rng, int(rng.integers(12)), 12, regime, random_effect_sd=0.,
                )
                for surface, (records, truths) in surfaces.items():
                    diagnostics.append({
                        "regime": regime, "replicate": iteration, "surface": surface,
                        **records[1]["stats"].covariance.attrs,
                    })
                    for index, variant in enumerate(("legacy", "repaired")):
                        try:
                            contrasts = joint_contrasts(records[index]["stats"], truths[index])
                        except ValueError as exc:
                            failures.append({
                                "regime": regime, "replicate": iteration, "surface": surface,
                                "level": "joint_contrast", "variant": variant, "error": str(exc),
                            })
                        else:
                            joint_rows.extend({
                                "regime": regime, "replicate": iteration, "surface": surface,
                                "variant": variant, **entry,
                            } for entry in contrasts)
                        for term in ("HRF A", "HRF B", "HRF A-HRF B"):
                            local = records[index]["stats"].copy()
                            if "-" in term:
                                local.ttest([term])
                            try:
                                score = inference(local, truths[index][term], term)
                            except ValueError as exc:
                                failures.append({
                                    "regime": regime, "replicate": iteration, "surface": surface,
                                    "model": term, "variant": variant, "error": str(exc),
                                })
                                continue
                            rows.append({
                                "regime": regime, "replicate": iteration, "subject": 0,
                                "surface": surface, "level": "first", "model": term,
                                "robust": True, "variant": variant, **score,
                            })
            warning_counts.update(str(item.message) for item in caught)
            print(f"{regime}: {iteration + 1}/{replicates}; failures={len(failures)}", flush=True)
    aggregate, summaries, pairs = summary(rows, seed + 1)
    return {
        "configuration": {
            "replicates_per_regime": replicates, "regimes": list(regimes), "seed": seed,
            "random_effect_sd": 0., "conditions": 2, "ar_order": 8, "alpha": .05,
            "ar_estimator": "robust",
            "roc_metrics": "fpr and sensitivity use raw p<.05; auc ranks raw p. "
                           "fdr and power are supplementary BH metrics, not ROC calibration.",
            "scope": "Independent first-level recordings with exact-null and active ROIs.",
            "marginal_source": "legacy_ar_irls_joint_diagonal_blocks",
        },
        "summary": summaries, "paired_differences": pairs, "replicate_metrics": aggregate,
        "raw_metrics": rows, "covariance_diagnostics": diagnostics,
        "failures": failures, "warning_counts": dict(warning_counts),
        "joint_contrast_summary": summarize_joint(joint_rows, seed + 2),
        "joint_contrast_metrics": joint_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--subjects", type=int, default=12)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=REGIMES)
    parser.add_argument("--first-level-only", action="store_true",
                        help="Remove random subject effects for exact-null first-level ROI ROC.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        run_first_level(args.replicates, args.regimes) if args.first_level_only
        else run(args.replicates, args.subjects, args.regimes)
    )
    result = json.loads(pd.Series(result).to_json(double_precision=15))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(result["replicate_metrics"]).to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"Saved {args.output}; {len(result['failures'])} inference/fitting failures.", flush=True)


if __name__ == "__main__":
    main()
