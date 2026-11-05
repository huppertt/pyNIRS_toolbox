"""Paired Monte Carlo validation of the interim AR-IRLS covariance repair.

Run with ``python -m pyBrainAnalyzIR.testing.helpers.covariance_validation --help``.
The legacy eigenvalue floor is reproduced ONLY here, never in production fits.
Intervals resample independent simulation replicates, not correlated channels.
"""
from __future__ import annotations

import argparse
import copy
import json
import warnings
from collections import Counter
from pathlib import Path

import cedalion.dataclasses as cdc
import numpy as np
import pandas as pd
import xarray as xr
from cedalion.models.glm.design_matrix import DesignMatrix
from scipy.signal import fftconvolve, lfilter
from scipy.stats import gamma, rankdata, t

from pyBrainAnalyzIR.dataclasses.dataset import DataSet
from pyBrainAnalyzIR.dataclasses.statistics import Statistics, _ar_irls_joint_covariance
from pyBrainAnalyzIR.math.glm.solve import fit
from pyBrainAnalyzIR.pipelines.modules.mixedeffects import MixedEffects


REGIMES = ("iid", "correlated_ar", "ar_outliers", "ar_global_null", "ar_stress")
VARIANTS = ("legacy_floor", "repaired_full", "repaired_diagonal")
MODELS = {
    "mean": "Beta ~ 0 + Condition",
    "age": "Beta ~ 0 + Condition + Condition:age",
    "categorical": "Beta ~ 0 + Condition + Condition:C(group)",
}


def simulate_subject(rng, subject, subjects, regime, *, two_conditions=False, return_inputs=False,
                     random_effect_sd=.06):
    """Concentration-space event GLM with known subject-level coefficients."""
    n = 120 if regime == "ar_stress" else 360
    time = np.arange(n) / 2
    events = np.zeros(n)
    for onset in [12, 38, 67, 94, 122, 151]:
        events[2 * onset:2 * onset + 4] = 1
    htime = np.arange(0, 30, .5)
    hrf = gamma.pdf(htime, 6) - gamma.pdf(htime, 16) / 6
    task = fftconvolve(events, hrf)[:n]
    task /= task.max()
    design = np.column_stack([np.ones(n), task])
    task_b = np.roll(task, 24)
    if two_conditions:
        design = np.column_stack([design, task_b])
    age = (subject - (subjects - 1) / 2) / subjects * 2
    male = subject % 2
    # Remove the small finite-sample age/group correlation exactly.
    age -= (male - .5) * 2 / subjects
    active = np.array([True] * 4 + [False] * 4)
    if regime == "ar_global_null":
        active[:] = False
    beta = active * (.3 + .25 * age + .3 * (male - .5))
    random_effect = rng.normal(0, random_effect_sd, 8) * np.array([1] * 6 + [0] * 2)
    if regime != "ar_global_null":
        beta += random_effect
    innovations = rng.normal(size=(n + 100, 8))
    if regime != "iid":
        common = rng.normal(size=(n + 100, 1))
        correlation = .98 if regime == "ar_stress" else .35
        innovations = np.sqrt(1 - correlation) * innovations + np.sqrt(correlation) * common
        noise = lfilter([1], [1, -.65], innovations, axis=0)[100:]
        noise *= np.sqrt(1 - .65 ** 2)
    else:
        noise = innovations[100:]
    noise *= .25 * np.linspace(.6, 1.4, 8)
    if regime in ("ar_outliers", "ar_global_null", "ar_stress"):
        # Channel-specific impulses and short motion shifts stress unequal IRLS weights.
        for channel in range(8):
            indices = rng.choice(n, 5, replace=False)
            noise[indices, channel] += rng.normal(0, 2, 5)
            start = int(rng.integers(30, n - 20))
            noise[start:start + 8, channel] += rng.normal(0, .8)
    channels = [f"S{i + 1}D{i + 1}" for i in range(4)]
    chromo = ["HbO", "HbR"]
    values = (1 + task[:, None] * beta + noise).reshape(n, 4, 2)
    if two_conditions:
        values += (.5 * task_b[:, None] * beta).reshape(n, 4, 2)
    ts = xr.DataArray(
        values, dims=("time", "channel", "chromo"),
        coords={"time": time, "channel": channels, "chromo": chromo},
        attrs={"units": "uM"},
    ).pint.quantify()
    common = xr.DataArray(
        np.repeat(design[:, :, None], 2, axis=2),
        dims=("time", "regressor", "chromo"),
        coords={"time": time, "regressor": ["Drift 0", "HRF A", "HRF B"] if two_conditions
                else ["Drift 0", "HRF A"], "chromo": chromo},
    )
    quality_mask = None
    if regime == "ar_stress":
        quality_mask = xr.DataArray(
            np.ones((n, 4), dtype=bool), dims=("time", "channel"),
            coords={"time": time, "channel": channels},
        )
        for channel in range(4):
            start = 25 + channel * 10
            quality_mask.values[start:start + 20, channel] = False
    results = fit(ts, DesignMatrix(common=common), noise_model="ar_irls",
                  ar_order=8, quality_mask=quality_mask, verbose=False)
    repaired = Statistics(results)
    legacy = copy.deepcopy(repaired)
    legacy.covariance.values[:] = _ar_irls_joint_covariance(list(results.values.ravel()))
    records = []
    for stats in (legacy, repaired):
        rec = cdc.Recording()
        rec["stats"] = stats
        rec.meta_data.update(subject=f"sub-{subject}", age=age, group="M" if male else "F")
        records.append(rec)
    truth = pd.Series(beta, index=pd.MultiIndex.from_product(
        [channels, chromo], names=["Channel", "Type"],
    ))
    if return_inputs:
        return records, truth, ts, DesignMatrix(common=common), quality_mask
    return records, truth


def metrics(stats, truth, selector):
    table = stats.table()
    table = table.loc[selector(table)].set_index(["Channel", "Type"])
    expected = truth.reindex(table.index).to_numpy()
    p = table["P-values"].to_numpy()
    estimates = table["Beta"].to_numpy()
    se = table["StdErr"].to_numpy()
    if not np.isfinite(np.column_stack([p, estimates, se, expected])).all():
        raise ValueError("Nonfinite inference encountered in validation.")
    positive = np.abs(expected) > 1e-12
    negative = ~positive
    score = -p
    n_positive, n_negative = positive.sum(), negative.sum()
    auc = (
        (rankdata(score)[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
        if n_positive and n_negative else np.nan
    )
    radius = t.ppf(.975, stats.dof) * se
    return {
        "auc": float(auc),
        "false_positive_rate": float(np.mean(p[negative] < .05)) if n_negative else np.nan,
        "power": float(np.mean(p[positive] < .05)) if n_positive else np.nan,
        "coverage": float(np.mean(np.abs(estimates - expected) <= radius)),
        "bias": float(np.mean(estimates - expected)),
        "rmse": float(np.sqrt(np.mean((estimates - expected) ** 2))),
        "null_tests": int(n_negative), "alternative_tests": int(n_positive),
    }


def legacy_floor_for_study(dataset):
    """Reproduce the removed group whitening floor on the retained task block."""
    count = 0
    for rec in dataset.dataset:
        stats = rec["stats"]
        indices = np.flatnonzero(stats.betas.conditions.values == "HRF A")
        block = stats.covariance.values[np.ix_(indices, indices)]
        eigenvalues, vectors = np.linalg.eigh((block + block.T) / 2)
        threshold = 1e-10 * np.max(np.abs(eigenvalues))
        count += int(eigenvalues[0] <= threshold)
        stats.covariance.values[np.ix_(indices, indices)] = (
            vectors * np.maximum(eigenvalues, threshold)
        ) @ vectors.T
    return count


def bootstrap_interval(values, rng, bounded=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"mean": None, "ci95": [None, None], "replicates": 0}
    # A percentile bootstrap cannot express uncertainty when every replicate is
    # zero/one. Exact binomial event bounds conservatively bound a [0,1] mean:
    # E(rate) <= P(rate > 0), and analogously for 1-rate.
    if bounded and (np.all(values == 0) or np.all(values == 1)):
        boundary = float(.025 ** (1 / len(values)))
        interval = [0., 1 - boundary] if values[0] == 0 else [boundary, 1.]
        return {"mean": float(values.mean()), "ci95": interval,
                "replicates": len(values), "ci_method": "exact_replicate_boundary_bound"}
    draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()),
            "ci95": list(map(float, np.quantile(draws, [.025, .975]))),
            "replicates": len(values), "ci_method": "replicate_bootstrap"}


def summarize(rows, seed):
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    output = []
    metric_names = ("auc", "false_positive_rate", "power", "coverage", "bias", "rmse")
    keys = ["regime", "level", "model", "robust", "variant"]
    for key, group in frame.groupby(keys, dropna=False):
        entry = dict(zip(keys, key))
        entry["metrics"] = {
            name: bootstrap_interval(
                group[name], rng, bounded=name in ("auc", "false_positive_rate", "power", "coverage"),
            ) for name in metric_names
        }
        output.append(entry)
    paired = []
    for key, group in frame.groupby(keys[:-1], dropna=False):
        for name in metric_names:
            wide = group.pivot(index="replicate", columns="variant", values=name)
            if "legacy_floor" in wide and "repaired_full" in wide:
                paired.append({
                    **dict(zip(keys[:-1], key)), "metric": name,
                    "repaired_minus_legacy": bootstrap_interval(
                        (wide["repaired_full"] - wide["legacy_floor"]).to_numpy(), rng,
                    ),
                })
    return output, paired


def calibration_flags(summary):
    """Flag nominal calibration outside replicate-bootstrap intervals.

    These are exploratory checks across many test families, not multiplicity-
    corrected confirmatory tests or evidence of calibration when no flag occurs.
    """
    flags = []
    for entry in summary:
        for metric, nominal in (("false_positive_rate", .05), ("coverage", .95)):
            interval = entry["metrics"][metric]["ci95"]
            if interval[0] is None or interval[1] is None:
                continue
            if not interval[0] <= nominal <= interval[1]:
                flags.append({
                    **{key: entry[key] for key in ("regime", "level", "model", "robust", "variant")},
                    "metric": metric, "nominal": nominal,
                    **entry["metrics"][metric],
                })
    return flags


def run_study(replicates=100, subjects=12, seed=20260901, regimes=REGIMES):
    if replicates < 2 or subjects < 6 or subjects % 2:
        raise ValueError("Use at least two replicates and an even subject count >= 6.")
    rows, diagnostics, failures = [], [], []
    warning_counts = Counter()
    for regime_index, regime in enumerate(regimes):
        if regime not in REGIMES:
            raise ValueError(f"Unknown noise regime: {regime}")
        for replicate in range(replicates):
            rng = np.random.default_rng(np.random.SeedSequence([seed, regime_index, replicate]))
            legacy_records, repaired_records = [], []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for subject in range(subjects):
                    simulation = simulate_subject(rng, subject, subjects, regime)
                    records, truth = simulation[:2]
                    legacy_records.append(records[0])
                    repaired_records.append(records[1])
                    info = records[1]["stats"].covariance.attrs
                    diagnostics.append({"regime": regime, "replicate": replicate,
                                        "subject": subject, **info})
                    for variant, rec in zip(("legacy_floor", "repaired_full"), records):
                        rows.append({
                            "regime": regime, "replicate": replicate, "level": "first",
                            "model": "task", "robust": True, "variant": variant,
                            "subject": subject,
                            **metrics(rec["stats"], truth, lambda table: table.Condition == "HRF A"),
                        })
                datasets = {
                    "legacy_floor": DataSet(legacy_records),
                    "repaired_full": DataSet(repaired_records),
                    "repaired_diagonal": DataSet(repaired_records),
                }
                floored = legacy_floor_for_study(datasets["legacy_floor"])
                diagnostics[-1]["legacy_task_blocks_floored_in_replicate"] = floored
                for model, formula in MODELS.items():
                    for robust in (False, True):
                        for variant, dataset in datasets.items():
                            job = MixedEffects()
                            job.options["FE_formula"] = formula
                            job.options["robust"] = robust
                            job.options["covariance_weighting"] = (
                                "diagonal" if variant == "repaired_diagonal" else "full"
                            )
                            try:
                                stats = job.run(copy.deepcopy(dataset))["groupstats"]
                            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                                failures.append({
                                    "regime": regime, "replicate": replicate, "model": model,
                                    "robust": robust, "variant": variant,
                                    "error": f"{type(exc).__name__}: {exc}",
                                })
                                continue
                            target = {"mean": .3, "age": .25, "categorical": .3}[model]
                            expected = pd.Series(
                                [0.] * 8 if regime == "ar_global_null" else [target] * 4 + [0.] * 4,
                                index=truth.index,
                            )

                            def selector(table, model=model):
                                if model == "age":
                                    return table.Condition.str.contains("age", regex=False)
                                if model == "categorical":
                                    return table.Condition.str.contains("[T.M]", regex=False)
                                return np.ones(len(table), dtype=bool)
                            result = metrics(stats, expected, selector)
                            null_tests = 8 if regime == "ar_global_null" else 4
                            if result["null_tests"] != null_tests or result["alternative_tests"] != 8 - null_tests:
                                raise AssertionError("Group truth/term alignment failed.")
                            rows.append({
                                "regime": regime, "replicate": replicate, "level": "group",
                                "model": model, "robust": robust, "variant": variant,
                                "subject": -1, **result,
                            })
            warning_counts.update(str(item.message) for item in caught)
            print(f"{regime}: {replicate + 1}/{replicates}", flush=True)
    # Average within replicate before estimating intervals or paired differences.
    frame = pd.DataFrame(rows)
    keys = ["regime", "replicate", "level", "model", "robust", "variant"]
    aggregate = frame.groupby(keys, as_index=False).mean(numeric_only=True).to_dict("records")
    summary, paired = summarize(aggregate, seed + 1)
    return {
        "configuration": {"replicates_per_regime": replicates, "subjects": subjects,
                          "seed": seed, "regimes": list(regimes), "alpha": .05,
                          "ar_estimator": "robust",
                          "group_random_effects": {"formula": "subject", "group_by": "auto"},
                          "bootstrap_replicates": 2000,
                          "interval_policy": "Replicate percentile bootstrap; conservative exact "
                                             "replicate-event bounds for all-zero/all-one rates.",
                          "seed_policy": "SeedSequence([seed, index_in_selected_regimes, replicate])",
                          "scope": "concentration-space synthetic event GLMs; not clinical certification"},
        "summary": summary, "paired_differences": paired, "failures": failures,
        "calibration_flags": calibration_flags(summary),
        "warning_counts": dict(warning_counts), "covariance_diagnostics": diagnostics,
        "replicate_metrics": aggregate, "raw_metrics": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--subjects", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=REGIMES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.replicates, args.subjects, args.seed, args.regimes)
    # NaN AUCs are not meaningful when a test family has only one truth class.
    cleaned = json.loads(pd.Series(result).to_json())
    args.output.write_text(json.dumps(cleaned, indent=2) + "\n")
    pd.DataFrame(result["replicate_metrics"]).to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"Saved {args.output}; {len(result['failures'])} group fitting failures.", flush=True)
    if result["calibration_flags"]:
        print(f"WARNING: {len(result['calibration_flags'])} exploratory calibration checks "
              "exclude nominal FPR/coverage. Numerical validity is not inferential validity.",
              flush=True)


if __name__ == "__main__":
    main()
