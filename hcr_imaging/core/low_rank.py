from sklearn.decomposition import NMF
import time, logging
import numpy as np
from scipy import optimize as opt

log = logging.getLogger(__name__)

################################################################################

def stable_rank(M, n=1):
    """The ratio between squared Frobenius norm and the squared spectral norm of a matrix."""
    return np.linalg.norm(M, ord='fro')**2 / (np.linalg.svd(M, compute_uv=False)[:n]**2).sum()

################################################################################

def nmf(X, n, init='nndsvd', random_state=0, **kws):
    '''Non-negative mean factorization using sklearn'''
    model = NMF(n_components=n, init=init, random_state=random_state, **kws)
    if n == 0:
        return np.empty((X.shape[0], 0)), np.empty((0, X.shape[1])), (X*X).sum()
    return model.fit_transform(X), model.components_, model.reconstruction_err_

################################################################################

def unit_nmf(X, n, **kws):
    if n == 0:
        return np.empty((X.shape[0], 0)), np.empty((0,)), np.empty((X.shape[1], 0)), (X*X).sum()
    W, H, err = nmf(X, n, **kws)
    H = H.T
    w, h = 1 / W.sum(0), 1 / H.sum(0)
    W *= w
    H *= h
    n = 1 / (w * h)
    order = np.argsort(n)[::-1]
    return W[:, order], n[order], H[:, order], err

################################################################################

def marginal_nmf(X, m, n, **kws):
    '''Return something like the singular values'''
    out = [nmf(X, i, **kws)[2] for i in range(m, n)]
    return np.diff(out)

################################################################################

def low_rank_objective(n, method='nmf', *args, **kwargs):
    if method == 'nmf':
        return lambda M: nmf(M, n, *args, **kwargs)[2]
    elif method == 'svd':
        return lambda M: np.linalg.norm(np.linalg.svd(M, compute_uv=False)[n:], *args, **kwargs)
    else:
        raise ValueError('Unknown low rank method method {}'.format(repr(method)))

################################################################################

def solve_nv(M, objective, bounds=None, tol=1e-6, xtol=1e-6, iters=5000, method=None, disp=False, **kwargs):
    """Solve for M(0, 0) element that minimizes the effective rank"""
    method = 'Brent' if bounds is None else 'Bounded'

    def obj(x, M=M.copy()):
        M[0, 0] = x
        return objective(M)

    cpu, clock = time.process_time(), time.time()

    res = opt.minimize_scalar(obj, M[0, 0], bounds=bounds, method=method,
        options=dict(disp=disp, maxiter=iters, xatol=xtol))

    cpu, clock = time.process_time() - cpu, time.time() - clock

    return res.x, dict(cpu=cpu, clock=clock, value=res.fun, success=res.success)

################################################################################

def solve_constraints():
    """get from mathematica"""
    pass

################################################################################

def total_least_squares(A, B):
    '''
    Total least squares using SVD
    A: [n, m]
    B: [n, o]
    returns: [m, o]
    '''
    n = B.shape[1]
    U = np.linalg.svd(np.hstack([A, B]))[2]
    return np.linalg.solve(U[-n:, -n:], -U[-n:, :-n]).T

################################################################################

def eigensystem(A, relative=False):
    '''
    Returns the eigenvalues and eigenvectors sorted in terms of decreasing magnitude
    The eigenvectors are returned as M such that M[i] is the ith eigenvector
    '''
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(np.abs(eigvals))[::-1]
    if relative:
        eigvals /= eigvals.sum()
    return eigvals[order], eigvecs.T[order]

################################################################################

