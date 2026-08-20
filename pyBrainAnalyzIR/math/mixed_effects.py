import numpy as np
from scipy import sparse
from scipy.linalg import cholesky, qr, inv, pinv
from scipy.optimize import minimize, fmin
import warnings

def make_scalar(x):
    if(x.__class__==np.ndarray):
        if(x.shape==np.array([0]).shape):
            return x.item()
    elif(x.__class__==list):
        return x[0]
    else:
        return x

def fitlme(X, Y, Z, robust_flag=False, zero_theta=False, verbose=False):
    """
    Robust linear mixed-effects model fitting
    [beta,bHat,covb,LL,w] = fitlme(X,Y,Z,robust_flag,zero_theta,verbose)
    
    TODO: Add support for covariance patterns other than isotropic
    
    Parameters:
    -----------
    X : array-like
        Fixed effects design matrix
    Y : array-like
        Response variable(s)
    Z : array-like
        Random effects design matrix
    robust_flag : bool, optional
        Whether to use robust fitting (default: False)
    zero_theta : bool, optional
        Whether to force theta to zero (default: False)
    verbose : bool, optional
        Whether to print iteration information (default: False)
        
    Returns:
    --------
    beta : ndarray
        Fixed effects coefficients
    bHat : ndarray
        Random effects estimates
    covb : ndarray
        Covariance matrix of fixed effects
    LL : ndarray
        Log-likelihood
    w : ndarray
        Weights
    """
    
    nT = Y.shape[0]
    nY = Y.shape[1] if Y.ndim > 1 else 1
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    nX = X.shape[1]
    nZ = Z.shape[1]
    
    # Handle all-NaN variables
    bad_vars = (np.all(np.isnan(Y), axis=0) | 
                np.all(Y == Y[0:1, :], axis=0))
    
    if np.any(bad_vars):
        beta = np.full((nX, nY), np.nan)
        bHat = np.full((nZ, nY), np.nan)
        covb = np.full((nX, nX, nY), np.nan)
        LL = np.full((1, nY), np.nan)
        w = np.full((nT, nY), np.nan)
        
        result = fitlme(X, Y[:, ~bad_vars], Z, robust_flag, zero_theta, verbose)
        beta[:, ~bad_vars] = result[0]
        bHat[:, ~bad_vars] = result[1]
        covb[:, :, ~bad_vars] = result[2]
        LL[0, ~bad_vars] = result[3]
        w[:, ~bad_vars] = result[4]
        
        return beta, bHat, covb, LL, w
    
    # Handle bad time points
    bad_times = (np.any(np.isnan(X), axis=1) | 
                 np.any(np.isnan(Y), axis=1) | 
                 np.any(np.isnan(Z), axis=1))
    
    if np.any(bad_times):
        X = X[~bad_times, :]
        Y = Y[~bad_times, :]
        Z = Z[~bad_times, :]
    
    nT0 = nT
    nT = X.shape[0]
    beta = np.full((nX, nY), np.nan)
    bHat = np.full((nZ, nY), np.nan)
    covb = np.full((nX, nX, nY), np.nan)
    LL = np.full((1, nY), np.nan)
    w = np.full((nT, nY), np.nan)
    
    if X.size == 0 or Y.size == 0:
        return beta, bHat, covb, LL, w
    
    # Run separately on each unique predictee
    if nY > 1:
        _, uinds, indsu = np.unique(Y.T, axis=0, return_index=True, return_inverse=True)
        
        for i in range(len(uinds)):
            ind = uinds[i]
            out = np.where(indsu == i)[0]
            ubeta, ubHat, ucovb, uLL, uw = fitlme(X, Y[:, ind:ind+1], Z, robust_flag, zero_theta, verbose)
            
            beta[:, out] = np.tile(ubeta, (1, len(out)))
            bHat[:, out] = np.tile(ubHat, (1, len(out)))
            covb[:, :, out] = np.tile(ucovb, (1, 1, len(out)))
            LL[0, out] = uLL
            w[~bad_times, out] = np.tile(uw, (1, len(out)))
        
        return beta, bHat, covb, LL, w
    
    # Ensure no sparse matrices
    if sparse.issparse(X):
        X = X.toarray()
    if sparse.issparse(Y):
        Y = Y.toarray()
    if sparse.issparse(Z):
        Z = Z.toarray()
    
    # Compute theta
    if zero_theta:
        theta = 0
    else:
        theta = solveForTheta(X, Y, Z)
    
    # Solve initial model
    LL_val, beta_val, bHat_val, covb_val, sigma2 = solveLME(X, Y, Z, theta)
    
    # Robust loop
    if robust_flag:
        iter_count = 1
        tune = 4.685
        D = 100 * np.sqrt(np.finfo(X.dtype).eps)
        
        # Adjust by leverage to account for prior weight differences in design matrix
        if X.shape[0] > 5e4:
            iX = inv(X.T @ X)
            lev = np.zeros((X.shape[0], 1))
            for i in range(X.shape[0]):
                lev[i, 0] = X[i:i+1, :] @ iX @ X[i:i+1, :].T
        else:
            lev = np.diag(X @ pinv(X)).reshape(-1, 1)
        
        adj = 1.0 / np.sqrt(1 - np.minimum(0.9999, lev))
        xrank = np.linalg.matrix_rank(X)
        num_params = max(1, xrank)
        
        while iter_count < 10:
            # Calculate residuals and weights from previous iteration
            resid = (Y - X @ beta_val - Z @ bHat_val) * adj
            resid_s,_ = studentizeResiduals(resid, num_params)
            w_iter = bisquare(resid_s, tune)
            w_mat = sparse.diags(w_iter.ravel(), 0)
            beta0 = beta_val.copy()
            
            # Bail out if weights are bad
            if len(np.unique(w_mat @ Y)) < 2 or np.any(np.isnan(w_iter)) or np.any(~np.isfinite(w_iter)):
                beta_val[:] = np.nan
                bHat_val[:] = np.nan
                covb_val[:] = np.nan
                LL_val = np.nan
                return beta, bHat, covb, LL, w
            
            # Re-estimate using new weights
            if not zero_theta:
                theta = solveForTheta(w_mat @ X, w_mat @ Y, w_mat @ Z, theta)  # Get optimal theta
            
            _, beta_val, bHat_val = solveLME(w_mat @ X, w_mat @ Y, w_mat @ Z, theta)[:3]  # Solve model
            
            if verbose:
                print(f'Robust fit iteration {iter_count} : {np.max(np.abs(beta_val - beta0))}')
            
            # Terminate if estimated coefficients have converged
            if not np.any(np.abs(beta_val - beta0) > D * np.maximum(np.abs(beta_val), np.abs(beta0))):
                break
            
            iter_count += 1
        
        # Calculate sigma
        sigma = robustSigma(X, Y, Z, adj, num_params, tune, beta_val, bHat_val)
        if not robust_flag:
            sigma = max(sigma, np.sqrt((sigma2 * xrank**2 + sigma_robust**2 * nT) / (xrank**2 + nT)))
        
        # Calculate covariance betas
        theta = make_scalar(theta)
        Lambda = np.sqrt(np.exp(theta)) * sparse.eye(nZ)  # Isotropic covariance pattern
        Iq = sparse.diags(np.ones(nZ), 0, shape=(nZ, nZ))
        R, _, S = chol_permuted(Lambda.T @ sparse.csr_matrix(Z.T @ Z) @ Lambda + Iq)
        Q1 = ((X.T @ Z @ Lambda) @ S) @ inv(R.toarray())
        R1R1t = X.T @ X - Q1 @ Q1.T
        R1 = cholSafe(R1R1t, 'lower')
        
        if xrank < X.shape[1]:
            warnings.warn('low rank X. Using pseudo-inverse')
            invR1 = pinv(R1)
        else:
            invR1 = inv(R1[:xrank, :xrank])
        
        covb_val = sigma**2 * (invR1.T @ invR1)
        w0 = np.diag(w_mat.toarray())
        w_result = np.zeros((nT0, 1))
        w_result[~bad_times] = w0.reshape(-1, 1)
    else:
        w_result = np.ones((nT0, 1))
        covb_val = covb_val
    
    beta[:, 0] = beta_val.ravel()
    bHat[:, 0] = bHat_val.ravel()
    covb[:, :, 0] = covb_val
    LL[0, 0] = LL_val
    w[:, 0] = w_result.ravel()
    
    return beta, bHat, covb, LL, w


def solveForTheta(X, y, Z, theta0=None):
    """
    Use unconstrained nonlinear optimization to find theta that minimizes log-likelihood
    """
    if theta0 is None:
        theta0 = 0
    
    if Z.size == 0:
        return 0
    
    # Find a valid initial value for theta
    max_iter = 100
    iter_count = 0
    while not np.isfinite(calcLogLikelihood(X, y, Z, theta0)):
        theta0 = theta0 / 2
        iter_count += 1
        assert iter_count < max_iter, 'Could not find valid initial value of theta for optimization'
    
#    warnings.filterwarnings('ignore', category=np.linalg.LinAlgWarning)
    
    # Try to use optimization methods
    try:
        from scipy.optimize import minimize
        result = minimize(lambda x: calcLogLikelihood(X, y, Z, x), 
                         theta0, 
                         method='BFGS', 
                         options={'disp': False})
        theta = result.x
    except:
        result = fmin(lambda x: calcLogLikelihood(X, y, Z, x), 
                     theta0, 
                     maxfun=1000, 
                     disp=False)
        theta = result
    
#    warnings.filterwarnings('default', category=np.linalg.LinAlgWarning)
    
    return theta


def calcLogLikelihood(X, y, Z, theta=0):
    """
    Calculate log-likelihood associated with a given value of theta
    """
    LL = solveLME(X, y, Z, theta)[0]
    return np.nanmean(-LL)


def solveLME(X, Y, Z, theta=0, weights=None):
    """
    Solve the linear mixed effects model
    
    Returns:
    --------
    PLogLik : float
        Penalized log-likelihood
    beta : ndarray
        Fixed effects coefficients
    bHat : ndarray
        Random effects estimates
    covb : ndarray
        Covariance matrix of fixed effects
    sigma2 : float
        Error variance
    """
    if weights is None:
        weights = sparse.eye(X.shape[0])
    if Z is None or Z.size == 0:
        Z = np.array([]).reshape(X.shape[0], 0)
    
    nT, nY = Y.shape if Y.ndim > 1 else (Y.shape[0], 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    nX = X.shape[1]
    nZ = Z.shape[1] if Z.size > 0 else 0
    
    X0 = X.copy()
    if sparse.issparse(weights):
        X = weights @ X
        Y = weights @ Y
        Z = weights @ Z
    else:
        X = weights @ X
        Y = weights @ Y
        Z = weights @ Z
    
    theta=make_scalar(theta)

    Lambda = makeLFromTheta(theta, nZ, 'isotropic')
    
    Lambda = np.sqrt(np.exp(theta)) * sparse.eye(nZ)  # Isotropic covariance pattern
    Iq = sparse.diags(np.ones(nZ), 0, shape=(nZ, nZ))
    
    sZ = sparse.csr_matrix(Z)
    Lambda = sparse.csr_matrix(Lambda)
    
    a = Lambda.T @ (sZ.T @ sZ) @ Lambda + Iq
    R, _, S = chol_permuted(a)
    Q1 = ((X.T @ sZ @ Lambda) @ S) @ inv(R.toarray())
    R1R1t = X.T @ X - Q1 @ Q1.T
    R1 = cholSafe(R1R1t, 'lower')
    
    # Parameter estimates
    cDeltab = inv(R.toarray()).T @ (S.T @ ((Lambda.T @ Z.T @ Y)))
    cbeta = inv(R1) @ (X.T @ Y - Q1 @ cDeltab)
    beta = inv(R1.T) @ cbeta
    Deltab = S @ (inv(R.toarray()) @ (cDeltab - Q1.T @ beta))
    bHat = Lambda @ Deltab
    
    # Estimate error
    resid = (Y - X @ beta - Z @ bHat)
    r2 = np.sum(Deltab**2) + np.sum(resid**2)
    
    # Calculate log-likelihood
    PLogLik = (-nT/2) * (1 + np.log(2*np.pi*r2/nT)) - logDet(R.toarray())
    
    # Calculate coefficient covariance
    dfe = max(nT - nX, 0)
    if Z.size == 0:
        sigma2 = r2 / dfe
    else:
        sigma2 = r2 / nT
    
    xr = np.linalg.matrix_rank(X)
    if xr < R1.shape[0]:
        invR1 = pinv(R1)
    else:
        invR1 = inv(R1[:xr, :xr])
    
    covb = sigma2 * (invR1.T @ invR1)
    
    return PLogLik, beta, bHat, covb, sigma2


def logDet(M):
    """
    Safely compute the log of the determinant
    Matlab's det uses LU which is inaccurate for large matrices, and the
    determinant can easily overflow a double. Here we use QR and avoid overflowing
    by moving log to the inside, since log(a*b) = log(a)+log(b)
    """
    _, R = qr(M, mode='economic')
    d = np.sum(np.log(np.diag(R)))
    return d


def cholSafe(d, mode='upper'):
    """
    Cholesky decomposition that won't error if unstable
    """
    delta = np.finfo(d.dtype).eps
    I = np.eye(d.shape[0])
    
    p = 1
    iter_count = 1
    max_iter = 1000
    
    while p != 0:
        try:
            if mode == 'lower':
                R = cholesky(d + delta * I, lower=True)
            else:
                R = cholesky(d + delta * I, lower=False)
            p = 0
        except np.linalg.LinAlgError:
            p = 1
        delta = 2 * delta
        iter_count += 1
        assert iter_count < max_iter, 'Could not perform cholesky factorization'
    
    return R


def chol_permuted(A):
    """
    Cholesky decomposition with permutation (emulating MATLAB's chol with 3 outputs)
    """
    if sparse.issparse(A):
        A_dense = A.toarray()
    else:
        A_dense = A
    
    # Try standard Cholesky first
    try:
        R = cholesky(A_dense, lower=False)
        p = 0
        S = np.eye(A_dense.shape[0])
        return sparse.csr_matrix(R), p, S
    except np.linalg.LinAlgError:
        # If that fails, add regularization
        delta = np.finfo(A_dense.dtype).eps
        I = np.eye(A_dense.shape[0])
        R = cholSafe(A_dense + delta * I, mode='upper')
        p = 0
        S = np.eye(A_dense.shape[0])
        return sparse.csr_matrix(R), p, S


def robustSigma(X, Y, Z, adj, num_params, tune, beta, bHat):
    """
    Get robust estimate of sigma
    """
    nT, nX = X.shape
    resid = (Y - X @ beta - Z @ bHat) * adj
    resid_s, s = studentizeResiduals(resid, num_params)
    
    r = resid_s / tune
    r1 = resid_s / tune - 1e-4
    r2 = resid_s / tune + 1e-4
    w = bisquare(r, 1)
    w1 = bisquare(r1, 1)
    w2 = bisquare(r2, 1)
    dw = (r2 * w2**2 - r1 * w1**2) / 2e-4
    
    a = np.mean(dw)
    h = (1.0 / adj)**2
    b = np.sum(h * (r * w**2)**2) / (nT - nX)
    K = 1 + (nX / nT) * (1 - a) / a
    s = K * np.sqrt(b) * s * tune / a
    
    return s


def bisquare(resid, tune=None):
    """
    Bisquare weighting function
    """
    if tune is None:
        tune = 4.685
    
    # don't need to divide by s because it is already studentized
    # s = mad(resid, 0) / 0.6745
    r = resid / tune
    w = (1 - r**2) * ((r < 1) & (r > -1))
    
    return w


def studentizeResiduals(resid, num_params):
    """
    Studentize residuals
    """
    sa_resid = np.sort(np.abs(resid.ravel()))
    sigma = np.nanmedian(sa_resid[num_params-1:]) / 0.6745
    resid_s = resid / sigma
    
    return resid_s, sigma


def makeLFromTheta(theta, nZ, type_pattern):
    """
    Create Lambda matrix from theta
    """
    theta=make_scalar(theta)
    Lambda = np.sqrt(np.exp(theta)) * sparse.eye(nZ)  # Isotropic covariance pattern
    
    # switch(type)
    #     case 'isotropic'
    #         Lambda = sqrt(exp(theta)) * speye(nZ); % Isotropic covariance pattern
    # end
    
    # 
    # % (2) Get slme.Psi, set current theta and set sigma = 1.
    #            Psi = slme.Psi;
    #            Psi = setUnconstrainedParameters(Psi,theta);
    #            Psi = setSigma(Psi,1);
    # 
    #            % (3) Get lower triangular Cholesky factor of D matrix in Psi.
    #            % Lambda has size q by q where q = size(Z,2).
    #            Lambda = getLowerTriangularCholeskyFactor(Psi);
    
    return Lambda