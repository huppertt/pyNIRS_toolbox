"""AR innovations model (port of nirs.math.innovations)."""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from cedalion.math.ar_model import bic_arfit


def innovations(Y, pmax):
    """Remove serial correlation from each column of Y with a BIC-selected AR model.

    Args:
        Y: array (time, n_signals).
        pmax: maximum AR model order.

    Returns:
        yfilt: whitened (innovations) signals, same shape as Y.
        f: list with one whitening filter per column, ``[1, -a1, ..., -a_pmax]``
           (zero-padded to length pmax + 1).
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    pmax = int(min(Y.shape[0], round(pmax)))

    yfilt = np.full(Y.shape, np.nan)
    f = []
    for i in range(Y.shape[1]):
        y = Y[:, i] - Y[:, i].mean()
        coefs = np.asarray(bic_arfit(pd.Series(y), pmax=pmax).params)[1:]
        fi = np.zeros(pmax + 1)
        fi[0] = 1
        fi[1:len(coefs) + 1] = -coefs
        f.append(fi)
        yfilt[:, i] = lfilter(fi, [1], y)
    return yfilt, f
