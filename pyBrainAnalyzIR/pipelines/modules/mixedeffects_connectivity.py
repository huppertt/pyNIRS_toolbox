"""Group-level mixed-effects models for resting and hyperscanning.

Connectivity estimation is performed at the individual or dyad level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from pyBrainAnalyzIR.dataclasses.connectivity import Connectivity
from pyBrainAnalyzIR.dataclasses.model_fit import ConnectivityGroupFit
from pyBrainAnalyzIR.dataclasses.options_variables import (
    BooleanOption,
    OptionsDict,
    StringOption,
)
from pyBrainAnalyzIR.math.mixed_effects import fitlme
from pyBrainAnalyzIR.pipelines.modules.mixedeffects import (
    _random_effects_design,
)
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module


_EDGE_KEYS = (
    "Subject_to",
    "Channel_to",
    "Type_to",
    "Subject_from",
    "Channel_from",
    "Type_from",
    "Condition",
)


def _input_connectivity(dataset, input_name):
    if input_name != "auto":
        names = [input_name]
    else:
        names = ["hyperconn", "conn"]
    for name in names:
        if dataset.dataset and all(
            name in recording.timeseries for recording in dataset.dataset
        ):
            return name
    raise ValueError(
        "Mixed-effects connectivity needs a first-level 'conn' or 'hyperconn' "
        "result on every recording. Run the corresponding connectivity "
        "module first."
    )


def _pair_observations(dataset, input_name, demographics):
    hyperscanning = input_name == "hyperconn"
    records = []
    seen_pairs = set()
    role_lookup = {}

    for index, recording in enumerate(dataset.dataset):
        connectivity = recording[input_name]
        demo = demographics.iloc[index]
        if hyperscanning:
            if "pair" not in demographics:
                raise ValueError(
                    "Hyperscanning mixed effects requires a 'pair' column "
                    "to identify independent dyads."
                )
            group = str(demo["pair"])
            if group in seen_pairs:
                continue
            seen_pairs.add(group)
            members = demographics.index[
                demographics["pair"].astype(str) == group
            ]
            if len(members) != 2:
                raise ValueError(
                    f"Hyperscanning group {group!r} must contain exactly "
                    "two recordings."
                )
            member_ids = demographics.loc[members, "subject"].astype(str)
            if (
                member_ids.str.strip().eq("").any()
                or member_ids.nunique() != 2
            ):
                raise ValueError(
                    f"Hyperscanning group {group!r} must have two distinct "
                    "subject IDs."
                )
            roles = {
                str(
                    demographics.loc[member, "subject"]
                ): f"participant-{role + 1}"
                for role, member in enumerate(members)
            }
            role_lookup[group] = roles
        else:
            group = str(demo.get("_group", demo.get("subject", index)))
            roles = {}

        table = connectivity.coef.copy()
        if table.empty:
            continue
        table["group"] = group
        table["fileIdx"] = index
        for column in demographics.columns:
            table[column] = demo[column]
        if hyperscanning:
            for endpoint in ("to", "from"):
                name_column = f"Subject_{endpoint}"
                geo_column = f"geo_{endpoint}"
                table[name_column] = table[name_column].astype(str).map(roles)
                if table[name_column].isna().any():
                    raise ValueError(
                        "Connectivity output refers to an unknown member of "
                        f"dyad {group!r}."
                    )
                if geo_column in table:
                    table[geo_column] = table[name_column]
        else:
            table["Subject_to"] = "subject"
            table["Subject_from"] = "subject"
            table = table.loc[
                (
                    table["Type_to"].astype(str)
                    == table["Type_from"].astype(str)
                )
                & (
                    table["Channel_to"].astype(str)
                    > table["Channel_from"].astype(str)
                )
            ].copy()
        if hyperscanning:
            table = table.loc[
                table.apply(
                    lambda row: (
                        str(row.Subject_to),
                        str(row.Channel_to),
                        str(row.Type_to),
                    )
                    > (
                        str(row.Subject_from),
                        str(row.Channel_from),
                        str(row.Type_from),
                    ),
                    axis=1,
                )
            ].copy()
        records.append(table)

    if not records:
        raise ValueError(
            "The first-level connectivity results contain no "
            "connection estimates."
        )
    return pd.concat(records, ignore_index=True), hyperscanning, role_lookup


class MixedEffectsConnectivity(cedalion_module):
    """Fit edge-wise group mixed-effects models to first-level connectivity.

    Connectivity coefficients are Fisher-Z transformed before group fitting.
    No time-course whitening is performed; ``robust`` controls the mixed-model
    iteratively reweighted estimator. Hyperscanning results are deduplicated to
    one observation per dyad and endpoint participant IDs are normalized to
    their within-dyad order.
    """

    is_statistics_output = True

    def __init__(self, previous_job=None):
        self.name = "Mixed Effects Connectivity"
        self.advanced_module = False
        self.inputName = "auto"
        self.outputName = "groupconn"
        self.description = (
            "Fit mixed-effects models to first-level connectivity"
        )
        self.previous_job = previous_job
        self.options = OptionsDict(
            {
                "FE_formula": StringOption(
                    "Beta ~ 0 + Condition",
                    description="Fixed-effects formula",
                    help=(
                        "Patsy formula for group effects; Beta is the "
                        "Fisher-Z connectivity."
                    ),
                ),
                "RE_formula": StringOption(
                    "subject",
                    description="Random-effects formula",
                    help=(
                        "Random-effects grouping terms, e.g. subject or "
                        "subject:Condition."
                    ),
                ),
                "group_by": StringOption(
                    "auto",
                    description="Independent-unit grouping column",
                    help=(
                        "auto uses subject for resting connectivity and "
                        "pair for hyperscanning."
                    ),
                ),
                "center_variables": BooleanOption(
                    True,
                    description="Mean-center numeric demographics",
                    help=(
                        "Center numeric fixed-effect covariates before "
                        "fitting."
                    ),
                ),
                "robust": BooleanOption(
                    True,
                    description="Use robust mixed-effects fitting",
                    help=(
                        "Down-weight outlying subject/dyad estimates. "
                        "No temporal prewhitening is used."
                    ),
                ),
            }
        )

    def _cite(self):
        return None

    def _runlocal(self, dataset):
        input_name = _input_connectivity(dataset, self.inputName)
        demographics = dataset.get_demographics().copy(deep=True)
        if input_name == "hyperconn" and "subject" not in demographics:
            raise ValueError(
                "Hyperscanning mixed effects requires subject IDs in "
                "demographics."
            )
        if "subject" not in demographics:
            demographics["subject"] = [
                f"recording-{i}" for i in range(len(dataset.dataset))
            ]
        demographics["fileIdx"] = demographics.index

        hyperscanning = input_name == "hyperconn"
        group_by = self.options["group_by"]
        if group_by == "auto":
            group_by = (
                "pair"
                if hyperscanning
                else next(
                    (
                        name
                        for name in ("subject", "subjectID")
                        if name in demographics
                    ),
                    "fileIdx",
                )
            )
        if group_by not in demographics:
            raise ValueError(
                f"Grouping column {group_by!r} is missing from demographics."
            )
        if demographics[group_by].isna().any():
            raise ValueError(
                f"Grouping column {group_by!r} contains missing IDs."
            )
        demographics["_group"] = demographics[group_by].astype(str)
        centers = {}
        if self.options["center_variables"]:
            excluded = {"fileIdx", "_group", group_by, "subject", "subjectID"}
            for column in demographics.select_dtypes(include="number"):
                if column not in excluded:
                    centers[column] = float(demographics[column].mean())
                    demographics[column] = (
                        demographics[column] - centers[column]
                    )

        observations, hyperscanning, roles = _pair_observations(
            dataset,
            input_name,
            demographics,
        )
        observations["Beta"] = np.arctanh(
            np.clip(
                observations["Coeff"].astype(float), -np.tanh(6), np.tanh(6)
            )
        )
        observations["subject"] = observations["group"]
        observations["cond"] = observations["Condition"].astype(str)
        observations["channel"] = observations["Channel_to"].astype(str)
        observations["chromo"] = observations["Type_to"].astype(str)
        observations["fileIdx"] = observations["fileIdx"].astype(str)
        if not np.isfinite(observations["Beta"]).all():
            raise ValueError(
                "First-level connectivity contains non-finite "
                "Fisher-Z estimates."
            )

        fixed_formula = self.options["FE_formula"]
        random_formula = self.options["RE_formula"]
        output_rows = []
        fit_rows = []
        covariance = {}
        edge_keys = [column for column in _EDGE_KEYS if column in observations]
        edge_keys.remove("Condition")
        for key_values, edge_data in observations.groupby(
            edge_keys, sort=False, dropna=False
        ):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            edge = dict(zip(edge_keys, key_values))
            design_data = edge_data.copy()
            design_data["Condition"] = design_data["Condition"].astype(str)
            _, fixed_design = patsy.dmatrices(
                fixed_formula,
                design_data,
                NA_action="raise",
                return_type="dataframe",
            )
            fixed = fixed_design.to_numpy(dtype=float)
            if np.linalg.matrix_rank(fixed) != fixed.shape[1]:
                raise ValueError(
                    "Fixed-effects design is rank deficient for "
                    f"connection {edge}."
                )
            random, random_terms = _random_effects_design(
                random_formula, design_data
            )
            beta, random_beta, cov_beta, _, weights = fitlme(
                fixed,
                design_data["Beta"].to_numpy(dtype=float),
                random,
                robust_flag=self.options["robust"],
            )
            beta = np.asarray(beta, dtype=float).reshape(-1)
            cov_beta = np.asarray(cov_beta, dtype=float).reshape(
                fixed.shape[1], fixed.shape[1]
            )
            if not np.isfinite(beta).all() or not np.isfinite(cov_beta).all():
                raise RuntimeError(
                    f"Mixed-effects fit failed for connection {edge}."
                )
            independent_units = design_data[group_by].nunique()
            n_conditions = design_data["Condition"].nunique()
            dof = independent_units * n_conditions - np.linalg.matrix_rank(
                fixed
            )
            if dof <= 0:
                raise ValueError(
                    f"Mixed-effects fit for connection {edge} has no "
                    "residual degrees of freedom."
                )

            fixed_fit = fixed @ beta
            random_beta = np.asarray(random_beta, dtype=float).reshape(-1)
            conditional_fit = fixed_fit + random @ random_beta
            connection_id = tuple(edge.items())
            covariance[connection_id] = cov_beta.copy()
            fit_rows.append(
                design_data.assign(
                    FixedFit=fixed_fit,
                    ConditionalFit=conditional_fit,
                    RobustWeight=np.asarray(weights, dtype=float).reshape(-1),
                )
            )
            for column_index, effect in enumerate(fixed_design.columns):
                se = float(
                    np.sqrt(max(cov_beta[column_index, column_index], 0.0))
                )
                statistic = beta[column_index] / se if se > 0 else np.nan
                pvalue = (
                    float(2 * stats.t.sf(abs(statistic), dof))
                    if np.isfinite(statistic)
                    else np.nan
                )
                row = dict(edge)
                row.update(
                    {
                        "Condition": str(effect),
                        "Coeff": float(np.tanh(beta[column_index])),
                        "Pvalue": pvalue,
                        "dfe": float(dof),
                        "ZstdErr": se,
                        "Robust": bool(self.options["robust"]),
                        "RandomEffects": str(random_terms),
                    }
                )
                output_rows.append(row)

        result = Connectivity()
        result.coef = pd.DataFrame(output_rows)
        result.description = "Mixed-effects connectivity: " + fixed_formula
        first_recording = dataset.dataset[0]
        result.geo2d = first_recording[input_name].geo2d.copy()
        result.geo3d = first_recording[input_name].geo3d.copy()
        result.geoROI = first_recording[input_name].geoROI.copy()
        result.geo2d_rotation_data = first_recording[
            input_name
        ].geo2d_rotation_data.copy()
        result.geo3d_rotation_data = first_recording[
            input_name
        ].geo3d_rotation_data.copy()
        result.meta_data = first_recording[input_name].meta_data.copy()
        result.head_model = first_recording[input_name].head_model
        result.fit_data = ConnectivityGroupFit(
            observations=pd.concat(fit_rows, ignore_index=True),
            coefficients=result.coef.copy(),
            covariance=covariance,
            formula=fixed_formula,
            random_effects_formula=random_formula,
            grouping_column=group_by,
            centers=centers,
            hyperscanning=hyperscanning,
            participant_roles=roles,
        )
        dataset[self.outputName] = result
        return dataset
