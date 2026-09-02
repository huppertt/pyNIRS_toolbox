import numpy as np
from scipy import stats

def corrcoef(d, verbose=False, mask=None):
    """
    Robust correlation coefficient.

    Python translation of the MATLAB corrcoef.m function.

    Parameters
    ----------
    d : array_like, shape (n_samples, n_variables)
        Input data matrix. Columns represent variables/signals.
    verbose : bool, optional
        Print progress information.
    mask : array_like, optional
        Boolean mask indicating valid observations. If None, all
        observations are initially considered valid.

    Returns
    -------
    r : ndarray, shape (n_variables, n_variables)
        Robust correlation coefficient matrix.
    p : ndarray, shape (n_variables, n_variables)
        Two-sided p-value matrix.

    Notes
    -----
    The implementation follows the original MATLAB algorithm:

        1. Remove NaNs using a validity mask.
        2. Median-center each column.
        3. Robustly scale each column.
        4. Perform robust pairwise regression.
        5. Combine reciprocal regression coefficients using
           r = sign(r + r.T) * sqrt(r * r.T).
        6. Calculate t statistics and p-values.
    """

    d = np.asarray(d, dtype=float)

    if d.ndim != 2:
        raise ValueError("d must be a 2-dimensional array")

    n_samples, n_variables = d.shape

    # MATLAB:
    # if(nargin<3)
    #     mask=ones(size(d));
    # end
    if mask is None:
        mask = np.ones_like(d, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != d.shape:
            raise ValueError("mask must have the same shape as d")

    # MATLAB:
    # mask = mask & ~isnan(d);
    # d(isnan(d)) = 0;

    nan_mask = np.isnan(d)
    mask = mask & ~nan_mask

    d = d.copy()
    d[nan_mask] = 0.0

    # MATLAB:
    # d=bsxfun(@minus,d,median(d,1));

    med = np.median(d, axis=0)
    d = d - med

    # MATLAB:
    # d=bsxfun(@rdivide,d,1.4826*mad(d,1,1));
    #
    # MATLAB mad(...,1,...) = mean absolute deviation
    mad_mean = np.mean(np.abs(d), axis=0)

    scale = 1.4826 * mad_mean

    # Avoid division by zero for constant columns
    scale[scale == 0] = 1.0

    d = d / scale

    # All pairwise combinations
    r = np.ones((n_variables, n_variables), dtype=float)

    n_pairs = n_variables ** 2

    if verbose:
        print("Progress")
        print("  0 %", end="", flush=True)

    cnt = 1

    for i in range(n_variables):
        for j in range(n_variables):

            if verbose:
                percent = round(100 * cnt / n_pairs)
                print(
                    "\r{:3d} %".format(percent),
                    end="",
                    flush=True
                )

            # MATLAB:
            # lst=find(mask(:,i) & mask(:,j));

            valid = mask[:, i] & mask[:, j]

            x = d[valid, j]
            y = d[valid, i]

            # MATLAB:
            # r(i,j)=regress(d(lst,i),d(lst,j));
            #
            # This is a regression WITHOUT an intercept:
            #
            # y = beta*x
            #
            # The returned regression coefficient is beta.

            r[i, j] = _robust_regress(y, x)

            cnt += 1

    if verbose:
        print("\rcompleted   ")

    # MATLAB:
    #
    # % Section 2.3 Eqn 6
    # % r = sqrt(b1*b2)
    #
    # r=sign(r+r').*abs(sqrt(r.*r'));

    with np.errstate(invalid="ignore"):
        r = np.sign(r + r.T) * np.abs(np.sqrt(r * r.T))

    # MATLAB:
    # r(abs(r)>1) = fix(r(abs(r)>1));

    too_large = np.abs(r) > 1
    r[too_large] = np.fix(r[too_large])

    # MATLAB:
    # n=sum(mask(:,1));

    n = np.sum(mask[:, 0])

    # MATLAB:
    # Tstat = r .* sqrt((n-2) ./ (1 - r.^2));

    with np.errstate(divide="ignore", invalid="ignore"):
        Tstat = r * np.sqrt((n - 2) / (1 - r ** 2))

    # MATLAB:
    # p = 2*nirs.math.tpvalue(-abs(Tstat),n-2);

    # Equivalent two-sided Student's t probability.
    p = 2 * stats.t.cdf(-np.abs(Tstat), df=n - 2)

    # MATLAB:
    # p=p+eye(size(p));

    p = p + np.eye(n_variables)

    return r, p


def _robust_regress(y, x, max_iter=100, tol=1e-6):
    """
    Robust regression corresponding to the regression used by
    the original MATLAB implementation.

    Fits:

        y = beta * x

    without an intercept using iterative robust weighting.

    The weighting function follows the wfun() function included
    in the original MATLAB file.
    """

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.nan

    # Initial least-squares estimate, no intercept
    denominator = np.sum(x * x)

    if denominator == 0:
        return 0.0

    beta = np.sum(x * y) / denominator

    for _ in range(max_iter):

        residual = y - beta * x

        # MATLAB wfun:
        #
        # s = mad(r, 0) / 0.6745;
        #
        # MATLAB mad(r,0) is median absolute deviation.
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))

        s = mad / 0.6745

        # Degenerate residual distribution
        if s == 0 or not np.isfinite(s):
            break

        # MATLAB:
        #
        # r = r/s/4.685;
        #
        u = residual / s / 4.685

        # MATLAB:
        #
        # w = (1 - r.^2) .* (r < 1 & r > -1);

        w = (1 - u ** 2) * ((u < 1) & (u > -1))

        denominator = np.sum(w * x * x)

        if denominator == 0:
            break

        beta_new = np.sum(w * x * y) / denominator

        if np.abs(beta_new - beta) < tol * (1 + np.abs(beta)):
            beta = beta_new
            break

        beta = beta_new

    return beta