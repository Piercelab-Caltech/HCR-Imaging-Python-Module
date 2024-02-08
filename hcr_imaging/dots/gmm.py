'''
Utilities and algorithms for Gaussian mixture optimization
'''

import numpy as np, functools, logging
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.optimize import minimize as _minimize
from ..cpp import gmm as _gmm
import time

DEFAULT_TRUNCATE = 8
log = logging.getLogger(__name__)

################################################################################

def gmm(image, m, grid, pairs, gradient, gaussian=None, bounds=None):
    '''
    Invoke Gaussian mixture model objective and optionally gradient from C++
    Assumes image and pairs are already in Fortran ordering
    '''
    gaussian = {} if gaussian is None else gaussian
    gradient = np.zeros(m.shape, dtype=image.dtype, order='F') if gradient else None
    weights = np.zeros(m.shape[2], dtype=image.dtype)
    bounds = (0, np.inf) if bounds is None else bounds
    pairs = np.asarray(pairs, dtype=np.ulonglong)
    m = np.asarray(m, dtype=image.dtype, order='F')
    grid = tuple(np.asarray(g, dtype=image.dtype) for g in grid)
    value = _gmm(image, m, grid, pairs, weights=weights, gradient=gradient,
        min=bounds[0], max=bounds[1], warm_start=False, tolerance=1e-8,
        iters=10000, truncate=gaussian.get('truncate', 5), regularize=0)
    return value, weights, gradient

################################################################################

def flat_view(x):
    out = x.ravel('A')
    assert out.base is x
    return out

################################################################################

def minimize(value_grad, x0, bounds=None, **kws):
    x, g = [np.zeros_like(x0, dtype=np.float64) for _ in range(2)]
    x[:] = x0
    xf, gf = flat_view(x), flat_view(g)
    def fun(y):
        xf[:] = y
        value, grad = value_grad(x)
        g[:] = grad
        return value, gf

    if bounds is not None:
        lo, up = [y.T for y in bounds.T]
        assert np.all(lo <= up)
        assert np.all(x0 <= up)
        assert np.all(x0 >= lo)
        order = 'F' if x.flags.fortran else 'C'
        bounds = np.array([flat_view(np.array(b, order=order)) for b in (lo, up)]).T
        assert np.all(flat_view(x) >= bounds[:, 0])
        assert np.all(flat_view(x) <= bounds[:, 1])
        bounds = bounds.astype(object)

    info = dict(_minimize(fun, flat_view(x), jac=True, bounds=bounds, **kws))
    xf[:] = info.pop('x')
    return x, info

################################################################################

def proximal_pairs(M, S, truncate):
    pts = M.T / S.max(1)
    locs = np.array(tuple(cKDTree(pts).query_pairs(2 * truncate))).T
    return np.asfortranarray(locs if len(locs) else np.ndarray([2, 0]), dtype=np.ulonglong)

################################################################################

def calculate_gmm(I, M, S, *, dtype, box, truncate=DEFAULT_TRUNCATE, gaussian=None, bounds=None):
    '''Optimized weights of a GMM model'''
    grid = tuple((b * np.arange(n)).astype(dtype) for b, n in zip(box, I.shape))

    MS = np.zeros((2, M.shape[1], M.shape[0]), dtype=dtype, order='F')
    MS[0].T[:] = M
    MS[1].T[:] = S

    locs = proximal_pairs(*MS, truncate)
    I = np.asfortranarray(I, dtype=dtype)

    value, weights, _ = gmm(I, MS, grid, locs, gradient=False, gaussian=gaussian, bounds=bounds)
    norm = np.linalg.norm(flat_view(I))
    assert norm > 0
    return value * norm**-2, weights

################################################################################

def optimize_gmm(I, M, S, *, optimize, sigma_bounds, box, dtype, options=None,
               truncate=DEFAULT_TRUNCATE, method='L-BFGS-B', gaussian=None, bounds=None):
    '''
    Optimized centers, widths, and weights of a GMM model
    optimize: 'center' 'sphere' or 'ellipsoid'
    '''
    optimize = dict(center=0, sphere=1, ellipsoid=2)[optimize]

    n, nd = M.shape
    log.info('Number of points: {}'.format(n))
    log.info('Number of position DOF: {}'.format(nd))

    I = np.asfortranarray(I, dtype=dtype)
    grid = tuple((b * np.arange(n)).astype(dtype) for b, n in zip(box, I.shape))
    norm = np.linalg.norm(flat_view(I))**-2
    weights = np.zeros(len(M), dtype=dtype)

    MS = np.zeros((2, nd, n), dtype=dtype, order='F')
    MS[0].T[:] = M
    MS[1].T[:] = S

    if sigma_bounds is not None:
        MS[1] = np.clip(abs(MS[1]), *sigma_bounds)

    def set_MS(x):
        MS[0] = x[:nd]
        if optimize:
            MS[1] = x[nd:]

    def value_grad(x):
        assert np.all(np.isfinite(x))
        set_MS(x)
        locs = proximal_pairs(*MS, truncate)
        value, weights[:], grad = gmm(I, MS, grid, locs, gradient=True, gaussian=gaussian, bounds=bounds)

        if optimize == 0:
            grad = grad[0]
        if optimize == 1:
            grad = grad.reshape((2 * nd, n), order='F')
            grad[nd] = grad[nd:].sum(0)
            grad = grad[:nd+1]
        if optimize == 2:
            grad = grad.reshape((2 * nd, n), order='F')
        
        if not np.isfinite(value):
            raise FloatingPointError('Objective {} is non-finite'.format(value))
        if not np.all(np.isfinite(grad)):
            raise FloatingPointError('Gradient is non-finite')

        value *= norm
        grad *= norm

        if value < -1:
            log.warning('GMM objective {} is below -1'.format(value))
        return value, grad

    if optimize == 0:
        x = MS[0]
    if optimize == 1:
        x = np.concatenate([MS[0], [MS[1].mean(0)]], axis=0)
    if optimize == 2:
        x = MS.reshape(2 * nd, n)

    # bounds on sigma
    bds = np.zeros(x.shape + (2,), dtype=np.float64, order='F')
    # if sigma_bounds is None:
    bds[nd:, :, 0] = 0
    bds[nd:, :, 1] = np.inf
    # else:
        # bds[nd:, :, :] = sigma_bounds

    for d, g in enumerate(grid):
        bds[d, :, 0] = g.min()
        bds[d, :, 1] = g.max()

    # print(x[:nd].min(axis=1))
    # print(x[:nd].max(axis=1))
    # print([(g.min(), g.max()) for g in grid])
    # bds = None
    # print('GRID', bds.shape, x.shape, bds)
    x = np.asfortranarray(x)

    log.info('Total DOF: {}'.format(x.shape))
    #raise ValueError(x, bds)

    x, info = minimize(value_grad, x, method=method, options=options or {'disp': False}, bounds=bds)
    set_MS(x)

    return dict(centers=MS[0].T, sigmas=MS[1].T, weights=weights, info=info)

################################################################################