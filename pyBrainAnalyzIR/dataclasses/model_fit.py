"""Original-space model data retained for interactive fit inspection."""
from __future__ import annotations

from dataclasses import dataclass
import ast

import numpy as np
import pandas as pd
import patsy
import xarray as xr
from scipy.stats import f


def _factor_uses_name(factor, name: str) -> bool:
    for node in ast.walk(ast.parse(factor.name(), mode="eval")):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        is_quoted_name = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Q"
            and bool(node.args)
        )
        if (
            is_quoted_name
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == name
        ):
            return True
    return False


def _fit_metrics(observed, fitted, design, result):
    """Descriptive original-space R² and a covariance-based joint Wald F-test."""
    metrics = dict.fromkeys(
        ("r_squared", "f_statistic", "f_pvalue", "df_num", "df_den"), np.nan
    )
    if not observed.size or not np.isfinite(result.params).all():
        return metrics
    residual = observed - fitted
    total = np.sum((observed - observed.mean()) ** 2)
    if total > 0:
        metrics["r_squared"] = 1 - np.sum(residual ** 2) / total

    # Test all nonconstant regressors, leaving an intercept unrestricted.
    tested = np.ptp(design, axis=0) > 0
    covariance = np.asarray(result.cov_params())[np.ix_(tested, tested)]
    coefficients = np.asarray(result.params)[tested]
    if not coefficients.size or not np.isfinite(covariance).all():
        return metrics
    rank = np.linalg.matrix_rank(covariance)
    dof = getattr(result, "_pybrain_satterthwaite_dof", np.nan)
    if not np.isfinite(dof):
        dof = getattr(result, "df_resid", result.nobs - np.linalg.matrix_rank(design))
    if rank and dof > 0:
        statistic = float(coefficients @ np.linalg.pinv(covariance) @ coefficients / rank)
        if statistic >= 0:
            metrics.update(
                f_statistic=statistic, f_pvalue=float(f.sf(statistic, rank, dof)),
                df_num=float(rank), df_den=float(dof),
            )
    return metrics


@dataclass
class FirstLevelFit:
    """Time courses, masks and diagnostics; predictions use the unwhitened design.

    R² is descriptive in original units. The joint Wald F uses the estimator's
    coefficient covariance and residual/effective degrees of freedom, not a
    naive independent-error F-test on autocorrelated original residuals.
    """

    data: xr.Dataset
    units: str

    @classmethod
    def from_results(cls, ts, design_matrix, results, quality_mask=None):
        signal_dim = results.dims[1]
        observed = ts.pint.dequantify().transpose("time", "channel", signal_dim).copy(deep=True)
        fitted = xr.full_like(observed, np.nan, dtype=float)
        valid = xr.zeros_like(observed, dtype=bool)
        template = observed.isel(time=0, drop=True)
        metrics = {
            name: xr.full_like(template, np.nan, dtype=float)
            for name in ("r_squared", "f_statistic", "f_pvalue", "df_num", "df_den")
        }
        for chromo, channels, design in design_matrix.iter_computational_groups(observed):
            x = np.asarray(design.values, dtype=float)
            for channel in channels:
                selection = {"channel": channel, signal_dim: chromo}
                result = results.sel(selection).item()
                y = observed.sel(selection).values
                keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
                if quality_mask is not None:
                    keep &= quality_mask.sel(channel=channel).values.astype(bool)
                prediction = x @ np.asarray(result.params)
                fitted.loc[selection] = prediction
                valid.loc[selection] = keep
                values = _fit_metrics(y[keep], prediction[keep], x[keep], result)
                for name, value in values.items():
                    metrics[name].loc[selection] = value
        data = xr.Dataset(
            dict(observed=observed, fitted=fitted, valid=valid, **metrics)
        )
        if signal_dim != "chromo":
            data = data.rename({signal_dim: "chromo"})
        return cls(data=data, units=str(observed.attrs.get("units", "")))


@dataclass
class GroupModelFit:
    """Subject estimates and fitted values before covariance whitening.

    ``observations`` retains raw demographics, first-level Beta/StdErr, and
    both FixedFit and ConditionalFit (= X beta + Z b). Covariance and fixed
    coefficients are kept independently of later display contrasts.
    """

    observations: pd.DataFrame
    coefficients: pd.DataFrame
    covariance: np.ndarray
    formula: str
    centers: dict[str, float]
    demographics: pd.DataFrame

    @property
    def covariates(self) -> list[str]:
        return [str(column) for column in self.demographics if column != "fileIdx"]

    @property
    def fitted_covariates(self) -> list[str]:
        return [name for name in self.covariates if self.uses_covariate(name)]

    def uses_covariate(self, name: str) -> bool:
        """Check fixed-effect formula factors, including Patsy's quoted names."""
        description = patsy.ModelDesc.from_formula(self.formula)
        return any(_factor_uses_name(factor, name)
                   for term in description.rhs_termlist for factor in term.factors)


    def _training_design(self) -> pd.DataFrame:
        training = self.observations.copy()
        for name, offset in self.centers.items():
            training[name] = training[name] - offset
        design = patsy.dmatrix(self.formula.split("~", 1)[1], training, return_type="dataframe")
        if list(design.columns) != list(self.coefficients.columns):
            raise ValueError("The reconstructed fixed-effects design differs from the fitted model.")
        return design

    def points(self, channel: str, signal_type: str, condition: str) -> pd.DataFrame:
        frame = self.observations
        selected = frame[
            (frame["Channel"] == channel)
            & (frame["Type"] == signal_type)
            & (frame["Condition"] == condition)
        ]
        if selected.empty:
            raise ValueError(f"No original observations for {channel}, {signal_type}, {condition}.")
        return selected.copy()

    def partial_effect(
        self, channel: str, signal_type: str, coefficient: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
        """Isolate one displayed coefficient, subtracting every other fixed term.

        This includes removing the task intercept for a slope display. Original
        residuals, random effects and first-level measurement SEs are retained;
        uncertainty in the fitted subtraction is not added to those SEs.
        FitSE describes the selected coefficient; TotalFitSE describes the full
        fixed-effects mean, translated to the partial-fit scale with the fitted
        nuisance offset held fixed. Neither is a subject prediction SE.
        """
        if coefficient not in self.coefficients:
            raise ValueError(f"{coefficient!r} is not an original fitted coefficient; "
                             "partial plots for display contrasts are unavailable.")
        design = self._training_design()
        column = design.columns.get_loc(coefficient)
        term = next(
            term for term in design.design_info.terms
            if design.design_info.term_name_slices[term.name()].start
            <= column < design.design_info.term_name_slices[term.name()].stop
        )
        variables = [name for name in self.fitted_covariates
                     if any(_factor_uses_name(factor, name) for factor in term.factors)]
        frame = self.observations
        selected = (frame.Channel == channel) & (frame.Type == signal_type)
        if any(_factor_uses_name(factor, "Condition") for factor in term.factors):
            # Use the actual coding, not names: task labels can overlap or use
            # custom contrasts. Keep zero-valued subjects within an active task.
            active = design.iloc[:, column] != 0
            tasks = pd.unique(frame.loc[active, "Condition"])
            if len(tasks) == 1:
                selected &= frame.Condition == tasks[0]
            elif len(tasks) > 1:
                variables.insert(0, "Condition")
        points = frame.loc[selected].copy()
        if points.empty:
            raise ValueError(f"No original observations for {channel}, {signal_type}, {coefficient}.")
        x = design.loc[selected].to_numpy()
        beta = self.coefficients.loc[(signal_type, channel)].to_numpy()
        contribution = x[:, column] * beta[column]
        other = x @ beta - contribution
        points["Beta"] = points.Beta - other
        points["ConditionalFit"] = points.ConditionalFit - other
        points["FixedFit"] = contribution
        reference = points.copy()
        for name in self.covariates:
            if name not in variables:
                values = self.demographics[name]
                numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
                if any(info.type == "categorical" and _factor_uses_name(factor, name)
                       for factor, info in design.design_info.factor_infos.items()):
                    numeric = False
                reference[name] = values.mean() if numeric else values.mode().iloc[0]
        if len(variables) == 1:
            axis = variables[0]
            values = points[axis]
            categorical = any(
                design.design_info.factor_infos[factor].type == "categorical"
                and _factor_uses_name(factor, axis) for factor in term.factors
            )
            numeric = (
                not categorical and pd.api.types.is_numeric_dtype(values)
                and not pd.api.types.is_bool_dtype(values)
            )
            levels = np.linspace(values.min(), values.max(), 100) if numeric else pd.unique(values)
            grid = pd.DataFrame([reference.iloc[0].copy() for _ in levels])
            grid[axis] = levels
            if categorical and pd.api.types.is_numeric_dtype(values):
                points[axis] = pd.Categorical(values, categories=pd.unique(values))
        elif variables:
            axis = " × ".join(variables) + " (design value)"
            points[axis] = x[:, column]
            reference[axis] = x[:, column]
            grid = reference.sort_values(axis, kind="stable").copy()
        else:
            axis = None
            grid = reference.iloc[[0]].copy()
        centered = grid.copy()
        for name, center in self.centers.items():
            centered[name] = centered[name] - center
        curve_design = np.asarray(patsy.build_design_matrices([design.design_info], centered)[0])
        curve_x = curve_design[:, column]
        position = self.coefficients.index.get_loc((signal_type, channel)) * len(beta)
        block = slice(position, position + len(beta))
        covariance = self.covariance[block, block]
        variance = covariance[column, column]
        if not np.isfinite(variance) or variance < 0:
            raise ValueError("Selected coefficient has invalid population variance.")
        total_variance = np.einsum("ij,jk,ik->i", curve_design, covariance, curve_design)
        tolerance = 1e-12 * np.einsum("ij,jk,ik->i", abs(curve_design), abs(covariance), abs(curve_design))
        if not np.isfinite(total_variance).all() or np.any(total_variance < -tolerance):
            raise ValueError("Full fixed-effects fit has invalid population variance.")
        grid["Fit"] = curve_x * beta[column]
        grid["FitSE"] = np.abs(curve_x) * np.sqrt(variance)
        grid["TotalFitSE"] = np.sqrt(np.maximum(total_variance, 0))
        return points, grid, axis

    def _reference_points(self, points: pd.DataFrame, covariate: str | None) -> pd.DataFrame:
        if covariate is not None and covariate not in self.covariates:
            raise ValueError(f"Unknown demographic covariate: {covariate}.")
        reference = points.copy()
        for name in self.covariates:
            if name == covariate:
                continue
            values = self.demographics[name]
            numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
            reference[name] = values.mean() if numeric else values.mode().iloc[0]
        return reference

    def adjusted_points(
        self, channel: str, signal_type: str, condition: str, covariate: str,
    ) -> pd.DataFrame:
        """Shift betas and conditional fits to the curve's reference covariates.

        Residuals and random-effect contributions are preserved. StdErr remains
        first-level measurement uncertainty conditional on this fitted shift;
        uncertainty in the estimated adjustment is not propagated.
        """
        points = self.points(channel, signal_type, condition)
        reference = self._reference_points(points, covariate)
        prediction = self._population_prediction(channel, signal_type, reference)
        shift = prediction["Fit"].to_numpy() - points["FixedFit"].to_numpy()
        points["Beta"] = points["Beta"] + shift
        points["ConditionalFit"] = points["ConditionalFit"] + shift
        points["FixedFit"] = prediction["Fit"].to_numpy()
        return points

    def population_curve(
        self, channel: str, signal_type: str, condition: str, covariate: str | None,
    ) -> pd.DataFrame:
        """Fixed-effect curve with other demographics held at mean/modal values.

        Rebuild Patsy's training design before evaluating new rows, preserving
        categorical coding and stateful transforms without storing unpicklable
        DesignInfo objects. Centering is applied once, using the training means.
        Returned SE describes uncertainty of the population mean, not prediction
        uncertainty for a new subject.
        """
        points = self.points(channel, signal_type, condition)
        row = self._reference_points(points, covariate).iloc[0].copy()
        if covariate is None:
            grid = pd.DataFrame([row])
        elif covariate not in self.covariates:
            raise ValueError(f"Unknown demographic covariate: {covariate}.")
        else:
            values = points[covariate].dropna()
            if values.empty:
                raise ValueError(f"No observed values for covariate {covariate}.")
            if (
                pd.api.types.is_numeric_dtype(values)
                and not pd.api.types.is_bool_dtype(values)
                and values.nunique() > 1
            ):
                levels = np.linspace(values.min(), values.max(), 100)
            else:
                levels = values.unique()
            grid = pd.DataFrame([row.copy() for _ in levels])
            grid[covariate] = levels
        return self._population_prediction(channel, signal_type, grid)

    def _population_prediction(
        self, channel: str, signal_type: str, grid: pd.DataFrame,
    ) -> pd.DataFrame:
        original_grid = grid.copy()
        grid = grid.copy()
        for name, center in self.centers.items():
            grid[name] = grid[name] - center

        design = self._training_design()
        x = np.asarray(patsy.build_design_matrices([design.design_info], grid)[0])
        position = self.coefficients.index.get_loc((signal_type, channel))
        count = self.coefficients.shape[1]
        block = slice(position * count, (position + 1) * count)
        covariance = self.covariance[block, block]
        original_grid["Fit"] = x @ self.coefficients.iloc[position].to_numpy()
        variance = np.einsum("ij,jk,ik->i", x, covariance, x)
        original_grid["FitSE"] = np.sqrt(np.maximum(variance, 0))
        return original_grid


@dataclass
class ConnectivityGroupFit:
    """First-level edge observations and fitted group means for connectivity."""

    observations: pd.DataFrame
    coefficients: pd.DataFrame
    covariance: dict[tuple, np.ndarray]
    formula: str
    random_effects_formula: str
    grouping_column: str
    centers: dict[str, float]
    hyperscanning: bool
    participant_roles: dict[str, dict[str, str]]
