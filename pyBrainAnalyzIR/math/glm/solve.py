"""Local GLM solver, following Cedalion's result shape and estimator choices."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api
from scipy.linalg import toeplitz
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

import cedalion.typing as cdt
import cedalion.xrutils as xrutils

from cedalion.models.glm.design_matrix import DesignMatrix

from .ar_irls import ar_irls_GLM


class _MissingFitResult:
    """NaN-valued fit result for a channel with no observations."""

    def __init__(self, columns, n_time):
        self.params = pd.Series(np.nan, index=columns, dtype=float)
        self.bse = self.params.copy()
        self.resid = np.full(n_time, np.nan)
        self.nobs = 0
        self._pybrain_full_resid = self.resid.copy()
        self._pybrain_valid_count = 0

    def cov_params(self):
        return pd.DataFrame(np.nan, index=self.params.index, columns=self.params.index)


def _channel_fit(y, x, noise_model="ols", ar_order=30, valid_mask=None):
    """Fit one channel, preserving Cedalion behavior when no observations are missing."""
    y_values = np.asarray(y, dtype=float).reshape(-1)
    x_values = np.asarray(x, dtype=float)
    valid = np.isfinite(y_values) & np.isfinite(x_values).all(axis=1)
    if valid_mask is not None:
        requested = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if requested.size != y_values.size:
            raise ValueError("The quality mask length must match the time series.")
        valid &= requested

    has_missing = not valid.all()
    if not has_missing:
        if noise_model == "ols":
            return statsmodels.api.OLS(y, x).fit()
        if noise_model == "rls":
            return statsmodels.api.RecursiveLS(y, x).fit()
        if noise_model == "wls":
            return statsmodels.api.WLS(y, x).fit()
        if noise_model == "ar_irls":
            return _fit_ar_irls(y, x, ar_order)
        if noise_model == "gls":
            ols_resid = np.asarray(statsmodels.api.OLS(y, x).fit().resid)
            resid_fit = statsmodels.api.OLS(
                ols_resid[1:],
                statsmodels.api.add_constant(ols_resid[:-1]),
            ).fit()
            rho = resid_fit.params[1]
            order = toeplitz(range(len(ols_resid)))
            return statsmodels.api.GLS(y, x, sigma=rho**order).fit()
        if noise_model == "glsar":
            return statsmodels.api.GLSAR(y, x, ar_order).iterative_fit(4)
        raise NotImplementedError(f"Unsupported noise model: {noise_model}")

    if not valid.any():
        return _MissingFitResult(
            getattr(x, "columns", range(x_values.shape[1])), y_values.size
        )

    y_masked = y_values.copy()
    y_masked[~valid] = np.nan
    x_frame = pd.DataFrame(x_values, columns=getattr(x, "columns", None))

    if noise_model == "ols":
        result = statsmodels.api.OLS(y_masked, x_frame, missing="drop").fit()
    elif noise_model == "wls":
        result = statsmodels.api.WLS(y_masked, x_frame, missing="drop").fit()
    elif noise_model == "rls":
        safe_x = x_values.copy()
        safe_x[~np.isfinite(safe_x)] = 0
        result = statsmodels.api.RecursiveLS(y_masked, safe_x).fit()
    elif noise_model == "gls":
        initial = statsmodels.api.OLS(
            y_masked, x_frame, missing="drop"
        ).fit()
        ols_resid = np.asarray(initial.resid)
        if ols_resid.size < 3:
            raise ValueError("GLS requires at least three valid residual samples.")
        resid_fit = statsmodels.api.OLS(
            ols_resid[1:],
            statsmodels.api.add_constant(ols_resid[:-1]),
        ).fit()
        rho = resid_fit.params[1]
        order = toeplitz(range(y_values.size))
        result = statsmodels.api.GLS(
            y_masked, x_frame, sigma=rho**order, missing="drop"
        ).fit()
    elif noise_model == "glsar":
        result = statsmodels.api.GLSAR(
            y_masked, x_frame, ar_order, missing="drop"
        ).iterative_fit(4)
    elif noise_model == "ar_irls":
        result = _fit_ar_irls(
            y_values, x_frame, ar_order, valid_mask=valid
        )
    else:
        raise NotImplementedError(f"Unsupported noise model: {noise_model}")

    result._pybrain_full_resid = _full_length_residual(result, valid, y_values.size)
    result._pybrain_valid_count = int(valid.sum())
    return result


def _fit_ar_irls(y, x, ar_order, valid_mask=None):
    """Fit AR-IRLS and retain its robust-weight Satterthwaite degrees of freedom."""
    result = ar_irls_GLM(y, x, pmax=ar_order, valid_mask=valid_mask)
    result._pybrain_satterthwaite_dof = _satterthwaite_dof(
        result.model.exog,
        result.weights,
    )
    return result


def _satterthwaite_dof(design, weights):
    """Estimate effective residual degrees of freedom for a weighted fit.

    This evaluates the same trace ratio used by MATLAB nirs-toolbox AR-IRLS,
    using small design-matrix products instead of constructing an n-by-n matrix.
    """
    design = np.asarray(design, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if (
        design.ndim != 2
        or design.shape[0] != weights.size
        or not np.isfinite(design).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        return np.nan

    weighted_design = weights[:, None] * design
    gram_inverse = np.linalg.pinv(weighted_design.T @ weighted_design)
    rank = np.linalg.matrix_rank(weighted_design)

    def weighted_trace(power):
        moment = design.T @ (
            (weights ** (power + 2))[:, None] * design
        )
        return float(np.trace(gram_inverse @ moment))

    trace_wk = weighted_trace(1)
    trace_w2k = weighted_trace(2)
    trace_w3k = weighted_trace(3)
    middle = design.T @ ((weights**3)[:, None] * design)
    trace_wkwk = float(np.trace(gram_inverse @ middle @ gram_inverse @ middle))

    trace_h2 = float(np.sum(weights**2) - 2 * trace_wk + rank)
    trace_h4 = float(
        np.sum(weights**4)
        - 4 * trace_w3k
        + 4 * trace_w2k
        + 2 * trace_wkwk
        - 4 * trace_wk
        + rank
    )
    if not np.isfinite(trace_h2) or not np.isfinite(trace_h4) or trace_h4 <= 0:
        return np.nan
    return trace_h2**2 / trace_h4


def _full_length_residual(result, valid, length):
    residual = np.asarray(result.resid, dtype=float).reshape(-1)
    full = np.full(length, np.nan)
    if residual.size == length:
        full[:] = residual
        full[~valid] = np.nan
    elif residual.size == int(valid.sum()):
        full[valid] = residual
    else:
        raise ValueError(
            "The regression result residuals cannot be aligned to the input time axis."
        )
    return full


def _recording_quality_mask(ts, masks):
    """Combine clean-sample masks and collapse subtype dimensions conservatively."""
    combined = None
    for name, mask in masks.items():
        if not isinstance(mask, xr.DataArray):
            raise TypeError(f"Recording mask {name!r} must be an xarray DataArray.")
        extra_dims = set(mask.dims) - set(ts.dims)
        if extra_dims:
            raise ValueError(
                f"Recording mask {name!r} has dimensions {sorted(extra_dims)} "
                "that are not present in the GLM input."
            )
        if "time" in mask.dims and "time" in ts.dims:
            if "time" not in mask.coords or "time" not in ts.coords:
                if mask.sizes["time"] == ts.sizes["time"]:
                    mask = mask.assign_coords(time=ts.time)
                else:
                    raise ValueError(
                        f"Recording mask {name!r} has no time coordinates and "
                        "does not match the GLM input length."
                    )
            elif not mask.time.equals(ts.time):
                try:
                    mask = mask.reindex(time=ts.time, method="nearest")
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Recording mask {name!r} cannot be aligned to GLM time."
                    ) from exc

        try:
            aligned_mask, _ = xr.align(mask, ts, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError(
                f"Recording mask {name!r} coordinates do not match the GLM input."
            ) from exc

        if not np.issubdtype(aligned_mask.dtype, np.bool_):
            aligned_mask = aligned_mask.fillna(True).astype(bool)
        expanded = aligned_mask.broadcast_like(ts).transpose(*ts.dims)
        subtype_dims = [dim for dim in expanded.dims if dim not in ("time", "channel")]
        if subtype_dims:
            expanded = expanded.all(dim=subtype_dims)
        expanded = expanded.transpose("time", "channel")
        combined = expanded if combined is None else combined & expanded
    return combined


def fit(
    ts: cdt.NDTimeSeries,
    design_matrix: DesignMatrix,
    noise_model: str = "ols",
    ar_order: int = 30,
    max_jobs: int = -1,
    verbose: bool = False,
    quality_mask: xr.DataArray | None = None,
):
    """Fit channel-wise GLMs to a time series and design matrix.

    If ``quality_mask`` is provided, bad samples are excluded independently
    for each channel. Without missing samples, estimator behavior matches
    Cedalion's GLM solver.

    Direct callers: :class:`pyBrainAnalyzIR.pipelines.modules.glm.GLM` and
    callers in ``tests/test_glm_quality_masks.py`` and ``tests/test_glm_basis.py``.
    Direct dependencies: ``DesignMatrix.iter_computational_groups`` and
    :func:`_channel_fit`.
    Tests: ``tests/test_glm_quality_masks.py`` and ``tests/test_glm_basis.py``.
    """
    ts = ts.pint.dequantify()
    dim3_name = xrutils.other_dim(design_matrix.common, "time", "regressor")

    reg_results = xr.DataArray(
        np.empty((ts.sizes["channel"], ts.sizes[dim3_name]), dtype=object),
        dims=("channel", dim3_name),
        coords=xrutils.coords_from_other(ts.isel(time=0), dims=("channel", dim3_name)),
    )

    for dim3, group_channels, group_design_matrix in design_matrix.iter_computational_groups(ts):
        group_y = ts.sel({"channel": group_channels, dim3_name: dim3}).transpose(
            "time", "channel"
        )
        x = pd.DataFrame(
            group_design_matrix.values,
            columns=group_design_matrix.regressor.values,
        )

        channel_masks = {}
        if quality_mask is not None:
            for channel in group_y.channel.values:
                channel_masks[channel] = quality_mask.sel(channel=channel).to_numpy()

        args_list = [
            (
                group_y.loc[:, channel],
                x,
                noise_model,
                ar_order,
                channel_masks.get(channel),
            )
            for channel in group_y.channel.values
        ]
        if max_jobs == 1:
            batch_results = (
                _channel_fit(*args)
                for args in tqdm(args_list, disable=not verbose)
            )
        else:
            with parallel_config(backend="threading", n_jobs=max_jobs):
                batch_results = tqdm(
                    Parallel(return_as="generator")(
                        delayed(_channel_fit)(*args) for args in args_list
                    ),
                    total=len(args_list),
                    disable=not verbose,
                )

        for channel, result in zip(group_y.channel.values, batch_results):
            reg_results.loc[channel, dim3] = result

    descriptions = {
        "ols": "OLS model via statsmodels.regression",
        "rls": "Recursive LS model via statsmodels.regression",
        "gls": "Generalized LS model via statsmodels.regression",
        "glsar": "Generalized LS AR-model via statsmodels.regression",
        "ar_irls": "AR_IRLS",
        "wls": "Weighted LS model via statsmodels.regression",
    }
    reg_results.attrs["description"] = descriptions.get(noise_model, "")
    return reg_results


def predict(
    ts: cdt.NDTimeSeries,
    thetas: xr.DataArray,
    design_matrix: DesignMatrix,
) -> cdt.NDTimeSeries:
    """Predict a time series using a design matrix and fitted coefficients.

    Direct callers: ``tests/test_glm_quality_masks.py``.
    Direct dependencies: ``DesignMatrix.iter_computational_groups`` and
    ``xarray.dot``.
    Tests: ``tests/test_glm_quality_masks.py``.
    """
    dim3_name = xrutils.other_dim(design_matrix.common, "time", "regressor")
    prediction = defaultdict(list)

    for dim3, group_channels, group_design_matrix in design_matrix.iter_computational_groups(ts):
        coefficients = thetas.sel({"channel": group_channels, dim3_name: [dim3]})
        prediction[dim3].append(
            xr.dot(group_design_matrix, coefficients, dim="regressor")
        )

    prediction = [xr.concat(values, dim="channel") for values in prediction.values()]
    return xr.concat(prediction, dim=dim3_name)
