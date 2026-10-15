"""AR-IRLS GLM fitting with time-grid-preserving missing-data handling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.signal
import statsmodels.api as sm

import cedalion.math.ar_model


def _interpolate_missing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Interpolate missing samples for whitening only; excluded samples stay excluded."""
    indices = np.arange(values.size)
    observed = indices[valid]
    samples = values[valid]
    if observed.size == 0:
        raise ValueError("AR-IRLS cannot fit a channel with no valid observations.")
    if observed.size >= 4:
        from scipy.interpolate import CubicSpline

        return CubicSpline(observed, samples, extrapolate=True)(indices)

    filled = np.interp(indices, observed, samples)
    if observed.size > 1:
        left = indices < observed[0]
        right = indices > observed[-1]
        left_slope = (samples[1] - samples[0]) / (observed[1] - observed[0])
        right_slope = (samples[-1] - samples[-2]) / (observed[-1] - observed[-2])
        filled[left] = samples[0] + (indices[left] - observed[0]) * left_slope
        filled[right] = samples[-1] + (indices[right] - observed[-1]) * right_slope
    return filled


def ar_irls_GLM(y, x, pmax=40, M=None, valid_mask=None):
    """Fit the Cedalion AR-IRLS model, excluding invalid observations.

    Missing samples are interpolated only to construct the continuous AR whitening
    filter. The robust regression is fit only at observed, finite samples.
    """
    if M is None:
        M = sm.robust.norms.HuberT()

    y_values = np.asarray(y, dtype=float).reshape(-1)
    x_values = np.asarray(x, dtype=float)
    if x_values.ndim != 2 or x_values.shape[0] != y_values.size:
        raise ValueError("AR-IRLS requires a 2D design matrix aligned with the data.")

    valid = np.isfinite(y_values) & np.isfinite(x_values).all(axis=1)
    if valid_mask is not None:
        requested = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if requested.size != y_values.size:
            raise ValueError("The quality mask length must match the time series.")
        valid &= requested
    if not valid.any():
        raise ValueError("AR-IRLS cannot fit a channel with no valid observations.")

    columns = getattr(x, "columns", None)
    index = getattr(x, "index", None)
    x_full = pd.DataFrame(x_values, columns=columns, index=index)
    y_fit = pd.Series(y_values[valid]).reset_index(drop=True)
    x_fit = x_full.loc[valid].reset_index(drop=True)

    params = sm.RLM(y_fit, x_fit, M=M).fit()
    residual = y_fit - x_fit @ params.params

    for _ in range(4):
        residual_values = np.full(y_values.size, np.nan)
        residual_values[valid] = np.asarray(residual)
        residual_filled = _interpolate_missing(residual_values, valid)
        arcoef = cedalion.math.ar_model.bic_arfit(
            pd.Series(residual_filled), pmax=pmax
        )
        whitening_filter = np.hstack([1, -arcoef.params[1:]])

        if valid.all():
            y_work = y_fit.copy()
            x_work = x_fit.copy()
        else:
            y_filled = _interpolate_missing(y_values, valid)
            y_work = pd.Series(y_filled)
            x_work = x_full.reset_index(drop=True)

        y_offset = y_work[0]
        whitened_y = pd.Series(
            scipy.signal.lfilter(whitening_filter, 1, y_work - y_offset)
        ) + y_offset

        whitened_x = np.zeros(x_work.shape)
        x_array = x_work.to_numpy()
        for column_index in range(x_array.shape[1]):
            x_offset = x_work.iloc[0, column_index]
            whitened_x[:, column_index] = scipy.signal.lfilter(
                whitening_filter,
                1,
                x_array[:, column_index] - x_offset,
            ) + x_offset
        whitened_x = pd.DataFrame(whitened_x, columns=x_work.columns)

        if valid.all():
            params = sm.RLM(whitened_y, whitened_x, M=M).fit()
            residual = y_fit - x_fit @ params.params
        else:
            params = sm.RLM(
                whitened_y[valid].reset_index(drop=True),
                whitened_x.loc[valid].reset_index(drop=True),
                M=M,
            ).fit()
            residual = y_values[valid] - x_values[valid] @ np.asarray(params.params)

    if not valid.all():
        full_residual = np.full(y_values.size, np.nan)
        full_residual[valid] = np.asarray(residual)
        params._pybrain_full_resid = full_residual
        params._pybrain_valid_count = int(valid.sum())
    return params
