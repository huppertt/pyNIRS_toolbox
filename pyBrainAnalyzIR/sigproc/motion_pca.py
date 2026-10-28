"""PCA motion correction with guards for edge cases in Cedalion's implementation.

Adapted from ``cedalion.sigproc.motion.pca`` and ``pca_recurse`` (Cedalion v26.5.1,
MIT License, Copyright 2024-2025 The Cedalion Developers), which follow Homer3
``hmrR_MotionCorrectPCA.m``.  Cedalion itself is left unmodified; this copy adds:

* data without motion artefacts is returned unchanged (Cedalion fails reshaping an
  empty array);
* an empty leading motion segment is skipped (Cedalion raises ``IndexError``);
* zero-variance measurements in the motion samples (e.g. a one-sample segment) are
  not divided by zero when z-scoring (Cedalion produces NaNs and ``svd`` fails).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import svd

import cedalion.xrutils as xrutils
from cedalion.sigproc.quality import TAINTED, id_motion, id_motion_refine


def motion_correct_pca(ts, t_inc, n_sv=0.97):
    """Remove the top principal components of the motion samples of ``ts``.

    Args:
        ts: time series with ``channel``, ``wavelength`` and ``time`` dimensions.
        t_inc: boolean time mask; ``False`` marks motion-tainted samples.
        n_sv: number of components to remove, or the fraction of variance if below 1.

    Returns:
        ``(ts_cleaned, n_sv, svs)`` as in Cedalion; ``(ts, 0, [])`` without motion.
    """
    if not np.any(t_inc.values == TAINTED):
        return ts, 0, np.array([])

    y, _ = xrutils.apply_mask(ts, t_inc == TAINTED, "drop", "none")
    y = y.stack(measurement=["channel", "wavelength"]).sortby("wavelength").pint.dequantify()
    std = y.std("time")
    std = std.where(std > 0, 1.0)
    y_zscore = (y - y.mean("time")) / std

    ts_stacked = (
        ts.stack(measurement=["channel", "wavelength"]).sortby("wavelength").pint.dequantify()
    )

    yo = y_zscore.copy()
    V, St, _ = svd(np.dot(y_zscore.T, y_zscore))
    svs = St / np.sum(St)
    svsc = np.cumsum(svs)
    if 0 < n_sv < 1:
        n_sv = np.where(~(svsc < n_sv))[0][0]

    ev = np.zeros((svs.shape[0], 1))
    ev[:n_sv] = 1
    ev = np.diag(np.squeeze(ev, axis=1))
    yc = yo - np.dot(np.dot(yo, V), np.dot(ev, V.T))
    yc = (yc * std) + y.mean("time")

    lstMs = np.where(np.diff(t_inc.values.astype(int)) == -1)[0]
    lstMf = np.where(np.diff(t_inc.values.astype(int)) == 1)[0]
    if len(lstMs) == 0:
        lstMs = np.asarray([0])
    if len(lstMf) == 0:
        lstMf = np.asarray([len(t_inc) - 1])
    if lstMs[0] > lstMf[0]:
        lstMs = np.insert(lstMs, 0, 0)
    if lstMs[-1] > lstMf[-1]:
        lstMf = np.append(lstMf, len(t_inc) - 1)

    lstMb = np.cumsum(lstMf - lstMs) - 1

    yc_ts = yc.values
    tmp_cleaned = ts_stacked.copy().values
    tmp_uncleaned = ts_stacked.copy().values

    for jj in range(tmp_cleaned.shape[1]):
        lst = np.arange(lstMs[0], lstMf[0])
        if len(lst) == 0:
            pass
        elif lstMs[0] > 0:
            tmp_cleaned[lst, jj] = yc_ts[: lstMb[0] + 1, jj] - yc_ts[0, jj] + tmp_cleaned[lst[0], jj]
        else:
            tmp_cleaned[lst, jj] = (
                yc_ts[: lstMb[0] + 1, jj] - yc_ts[lstMb[0], jj] + tmp_cleaned[lst[-1], jj]
            )

        for kk in range(len(lstMf) - 1):
            lst = np.arange(lstMf[kk] - 1, lstMs[kk + 1] + 1)
            tmp_cleaned[lst, jj] = (
                tmp_uncleaned[lst, jj] - tmp_uncleaned[lst[0], jj] + tmp_cleaned[lst[0], jj]
            )
            lst = np.arange(lstMs[kk + 1], lstMf[kk + 1])
            tmp_cleaned[lst, jj] = (
                yc_ts[lstMb[kk] + 1: lstMb[kk + 1] + 1, jj]
                - yc_ts[lstMb[kk] + 1, jj]
                + tmp_cleaned[lst[0], jj]
            )

        if lstMf[-1] < len(tmp_uncleaned) - 1:
            lst = np.arange(lstMf[-1] - 1, len(tmp_uncleaned))
            tmp_cleaned[lst, jj] = (
                tmp_uncleaned[lst, jj] - tmp_uncleaned[lst[0], jj] + tmp_cleaned[lst[0], jj]
            )

    ts_cleaned = ts_stacked.copy()
    ts_cleaned.values = tmp_cleaned
    ts_cleaned = ts_cleaned.unstack("measurement").pint.quantify()
    ts_cleaned = ts_cleaned.transpose("channel", "wavelength", "time")
    ts_cleaned = ts_cleaned.assign_coords({"source": ts.source})
    return ts_cleaned, n_sv, svs


def motion_correct_pca_recurse(
    ts, t_motion=0.5, t_mask=1, stdev_thresh=20, amp_thresh=5, n_sv=0.97, max_iter=5,
):
    """Repeat motion detection and :func:`motion_correct_pca` until clean.

    Returns:
        ``(ts_cleaned, svs, n_sv, t_inc)`` as in Cedalion's ``pca_recurse``.
    """
    def detect(data):
        t_inc = id_motion_refine(id_motion(data, t_motion, t_mask, stdev_thresh, amp_thresh),
                                 "all")[0]
        t_inc.values = np.hstack([t_inc.values[0], t_inc.values[:-1]])
        return t_inc

    t_inc = detect(ts)
    ts_cleaned, n_sv_ret, svs = ts.copy(), -1, np.ndarray([])
    for _ in range(int(max_iter)):
        if not np.any(~t_inc.values):
            break
        ts_cleaned, n_sv_ret, svs = motion_correct_pca(ts_cleaned, t_inc, n_sv=n_sv)
        t_inc = detect(ts_cleaned)
    return ts_cleaned, svs, n_sv_ret, t_inc
