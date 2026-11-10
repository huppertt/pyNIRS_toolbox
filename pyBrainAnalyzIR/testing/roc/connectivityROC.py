"""ROC analysis for resting-state and hyperscanning connectivity pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt

import pyBrainAnalyzIR.testing.simulations.simData as simdata
from pyBrainAnalyzIR.pipelines.modules import preproccessing as prep
from pyBrainAnalyzIR.pipelines.modules.connectivity import (
    hyperscanning,
    resting_state_connectivity,
)
from pyBrainAnalyzIR.testing.roc.channelROC import PipelineList


_EDGE_COLUMNS = [
    "Pair", "Subject_to", "Channel_to", "Type_to",
    "Subject_from", "Channel_from", "Type_from", "Condition",
]


def _connection_table(data, truth, hyperscan):
    """Align connection p-values to edge truth using labels, never row position."""
    from pyBrainAnalyzIR.dataclasses.dataset import DataSet

    recordings = data.dataset if isinstance(data, DataSet) else [data]
    demographics = data.get_demographics() if hyperscan else None
    result_rows = []
    seen_pairs = set()
    for record_index, recording in enumerate(recordings):
        connectivity = recording["hyperconn" if hyperscan else "conn"]
        if hyperscan:
            pair = str(demographics.iloc[record_index]["pair"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            truth_rows = truth[truth.Pair.astype(str) == pair]
        else:
            pair = str(recording.meta_data.get("subject", record_index))
            truth_rows = None

        coefficients = connectivity.coef.copy()
        if hyperscan:
            cross = coefficients.Subject_to.astype(str) != coefficients.Subject_from.astype(str)
            coefficients = coefficients.loc[cross].copy()
            coefficients = coefficients.loc[
                coefficients.apply(
                    lambda row: (str(row.Subject_to), str(row.Channel_to), str(row.Type_to))
                    > (str(row.Subject_from), str(row.Channel_from), str(row.Type_from)),
                    axis=1,
                )
            ]
        else:
            coefficients = coefficients.loc[
                (coefficients.Type_to.astype(str) == coefficients.Type_from.astype(str))
                & (coefficients.Channel_to.astype(str) > coefficients.Channel_from.astype(str))
            ].copy()

        for row in coefficients.itertuples(index=False):
            entry = {
                "Pair": pair,
                "Subject_to": str(getattr(row, "Subject_to", "")),
                "Channel_to": str(row.Channel_to),
                "Type_to": str(row.Type_to),
                "Subject_from": str(getattr(row, "Subject_from", "")),
                "Channel_from": str(row.Channel_from),
                "Type_from": str(row.Type_from),
                "Condition": str(row.Condition),
                "Coefficient": float(row.Coeff),
                "P": float(row.Pvalue),
            }
            if hyperscan:
                key = {column: entry[column] for column in _EDGE_COLUMNS}
                matched = truth_rows
                for column, value in key.items():
                    matched = matched[matched[column].astype(str) == value]
                entry["Truth"] = bool(matched.Truth.iloc[0]) if len(matched) else False
            else:
                truth_channels = set(map(str, truth.channel_to.values))
                true = (
                    truth.sel(
                        channel_to=entry["Channel_to"],
                        channel_from=entry["Channel_from"],
                    ).item() != 0
                    if {
                        entry["Channel_to"], entry["Channel_from"]
                    }.issubset(truth_channels)
                    else False
                )
                entry["Truth"] = bool(true)
            result_rows.append(entry)
    return pd.DataFrame(result_rows)


class _ConnectivityROC:
    """Shared simulation, scoring, and summary implementation."""

    is_hyperscanning = False

    def __init__(self):
        self.iterations = 0
        self.data_simulation_function = (
            simdata.simdataset_hyperscanning
            if self.is_hyperscanning else simdata.simdataset_connectivity
        )
        self.data_simulation_args = {}
        self.pipeline = PipelineList([
            prep.intensity_opticaldensity,
            prep.mbll,
            prep.resample,
            hyperscanning if self.is_hyperscanning else resting_state_connectivity,
        ])
        self.pipeline_args = {
            "Fs": 2,
            "AR": 18,
            "robust": True,
            "verbose": False,
        }
        self.verbose = False
        self._records = []

    def run(self, num_iterations):
        """Run ``num_iterations`` independent simulations through the pipeline."""
        self.pipeline.set_all_options(self.pipeline_args)
        for _ in range(num_iterations):
            if self.verbose:
                print(f"Iteration {self.iterations}")
            data, truth = self.data_simulation_function(**self.data_simulation_args)
            analyzed = self.pipeline.run(data)
            scored = _connection_table(analyzed, truth, self.is_hyperscanning)
            scored["Iteration"] = self.iterations
            self._records.append(scored)
            self.iterations += 1
        return self

    def reset(self):
        self.iterations = 0
        self._records = []
        return self

    def roc_table(self):
        """All scored edge tests, one row per edge and simulation iteration."""
        if not self._records:
            return pd.DataFrame(columns=[
                *_EDGE_COLUMNS, "Coefficient", "P", "Truth", "Iteration",
            ])
        return pd.concat(self._records, ignore_index=True)

    def summary(self, alpha=0.05):
        """Sensitivity, false-positive rate, and AUC over simulated edges."""
        rows = []
        for (signal_type, condition), group in self.roc_table().groupby(
            ["Type_to", "Condition"], sort=True
        ):
            pvalues = group.P.to_numpy(dtype=float)
            truth = group.Truth.to_numpy(dtype=bool)
            positives = truth.sum()
            negatives = len(truth) - positives
            auc = np.nan
            if positives and negatives:
                ranks = rankdata(-pvalues)
                auc = float(
                    (ranks[truth].sum() - positives * (positives + 1) / 2)
                    / (positives * negatives)
                )
            rows.append({
                "Type": signal_type,
                "Condition": condition,
                "True edges": int(positives),
                "Null edges": int(negatives),
                f"Sensitivity (p<{alpha})": (
                    float(np.mean(pvalues[truth] < alpha)) if positives else np.nan
                ),
                f"FPR (p<{alpha})": (
                    float(np.mean(pvalues[~truth] < alpha)) if negatives else np.nan
                ),
                "AUC": auc,
            })
        return pd.DataFrame(rows)

    def results(self):
        """Return ROC point arrays by signal type and connectivity condition."""
        curves = {}
        for key, group in self.roc_table().groupby(
            ["Type_to", "Condition"], sort=True
        ):
            truth = group.Truth.to_numpy(dtype=bool)
            pvalues = group.P.to_numpy(dtype=float)
            order = np.argsort(pvalues, kind="stable")
            ordered_truth = truth[order]
            true_count = max(int(truth.sum()), 1)
            false_count = max(int((~truth).sum()), 1)
            curves[key] = (
                np.r_[0., np.cumsum(ordered_truth) / true_count],
                np.r_[0., np.cumsum(~ordered_truth) / false_count],
                np.r_[1., pvalues[order]],
            )
        return curves

    def draw(self):
        """Plot edge-level ROC curves and return the figure."""
        curves = self.results()
        figure, axis = plt.subplots(figsize=(6, 5))
        for (signal_type, condition), (tpr, fpr, _) in curves.items():
            axis.plot(fpr, tpr, label=f"{signal_type}, {condition}")
        axis.plot([0, 1], [0, 1], ":", color="0.5")
        axis.set(
            xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate",
            ylabel="Sensitivity", title="Connectivity ROC",
        )
        axis.legend()
        figure.tight_layout()
        return figure

    def type1_error_curve(self, alpha_levels=None):
        """Return empirical null-edge rejection rates across nominal alpha."""
        if alpha_levels is None:
            alpha_levels = np.linspace(0.0, 0.2, 101)
        alpha_levels = np.asarray(alpha_levels, dtype=float)
        if (
            alpha_levels.ndim != 1
            or not len(alpha_levels)
            or not np.isfinite(alpha_levels).all()
            or np.any((alpha_levels < 0) | (alpha_levels > 1))
        ):
            raise ValueError("alpha_levels must be a nonempty sequence in [0, 1].")

        table = self.roc_table()
        rows = []
        for (signal_type, condition), group in table.groupby(
            ["Type_to", "Condition"], sort=True
        ):
            null_pvalues = group.loc[~group.Truth, "P"].to_numpy(dtype=float)
            for alpha in alpha_levels:
                rows.append({
                    "Type": signal_type,
                    "Condition": condition,
                    "Nominal alpha": alpha,
                    "Empirical type-I error": (
                        float(np.mean(null_pvalues <= alpha))
                        if len(null_pvalues) else np.nan
                    ),
                    "Null tests": len(null_pvalues),
                })
        return pd.DataFrame(rows)

    def draw_type1_error(self, alpha_levels=None):
        """Plot empirical type-I error against nominal alpha."""
        curve = self.type1_error_curve(alpha_levels)
        figure, axis = plt.subplots(figsize=(6, 5))
        for (signal_type, condition), group in curve.groupby(
            ["Type", "Condition"], sort=True
        ):
            axis.plot(
                group["Nominal alpha"],
                group["Empirical type-I error"],
                label=f"{signal_type}, {condition}",
            )
        upper = float(curve["Nominal alpha"].max()) if not curve.empty else 0.2
        axis.plot([0, upper], [0, upper], ":", color="0.5", label="Nominal")
        axis.set(
            xlim=(0, upper), ylim=(0, upper),
            xlabel="Nominal alpha", ylabel="Empirical type-I error",
            title="Type-I error calibration",
        )
        axis.legend()
        figure.tight_layout()
        return figure


class ROC_connectivity(_ConnectivityROC):
    """ROC analysis for group-level repeated-recording connectivity simulations."""


class ROC_hyperscanning(_ConnectivityROC):
    """ROC analysis for between-participant connections in simulated dyads."""

    is_hyperscanning = True
