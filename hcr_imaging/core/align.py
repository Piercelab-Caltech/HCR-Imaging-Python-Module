'''
Experimental utilities to align two different images based on their contained dots
'''

import numpy as np
from autograd import numpy as anp, value_and_grad
from scipy.optimize import minimize
from scipy.spatial import cKDTree as Tree
import logging

log = logging.getLogger(__name__)

def align_dots(centers, weights, sigmas, ball, method='BFGS', guess=None, **kwargs):
    '''
    Align dots between multiple channels
    centers: Gaussian centers (# channels, # points, # dimensions)
    sigmas:  Gaussian standard deviations (same shape as centers)
    weights: Gaussian weights (# channels, # points)
    ball:    Maximum distance between Gaussians to consider
    '''
    n, nd = len(centers), centers[0].shape[1]
    assert n == len(weights)
    log.info('Optimizing dot alignment (n={}, nd={})'.format(n, nd))

    if np.ndim(sigmas) == 0:
        sigmas = [np.full_like(c, sigmas) for c in centers]

    if guess is None:
        x0 = np.zeros([n-1, nd], dtype=centers[0].dtype)
    else:
        x0 = np.diff(guess, axis=0).astype(centers[0].dtype)

    # weights = [w / w.sum() for w in weights]

    trees = [Tree(c) for c in centers]
    pairs = np.full([len(trees)]*2, None)
    for j, u in enumerate(trees):
        for i, t in zip(range(j), trees):
            pairs[i, j] = np.array([(x, y) for x, v in enumerate(t.query_ball_tree(u, ball)) for y in v])

    best = sum(np.linalg.norm(w)**2 for w in weights)
    # best is not really right but it does roughly nondimensionalize the objective
    history = []
    def objective(x):
        '''Objective taking shifts[1:] as input'''
        out = 0
        x = x.cumsum(axis=0)
        for j, (sj, cj, wj) in enumerate(zip(sigmas, centers, weights)):
            for i, si, ci, wi in zip(range(j), sigmas, centers, weights):
                I, J = pairs[i, j].T
                res = (cj[J] - ci[I]) + ((x[j-1] - x[i-1]) if i else x[j-1]) # ((x[j-1] - x[i-1]) if i else x[j-1])
                inv_s = (si[I]**2 + sj[J]**2) ** -0.5
                scales = wi[I] * wj[J] * np.prod(inv_s, axis=1)
                out = out + anp.dot(scales, anp.exp(-0.5 * anp.square(res * inv_s).sum(1)))
        out = out * (-1 * (2 * np.pi)**(-0.5 * nd) / best)
        history.append(getattr(out, '_value', out))
        return out

    fgrad = value_and_grad(lambda x: objective(x.reshape(n-1, nd)))

    info = dict(minimize(fgrad, x0.ravel(), jac=True, method='BFGS', **kwargs))
    x = np.zeros([n, nd], info['x'].dtype)
    x[1:] = info.pop('x').reshape(n-1, nd).cumsum(axis=0)
    x -= x.mean(0)
    info['history'] = np.array(history)
    return x, info
