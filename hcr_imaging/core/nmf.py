
import warnings
from itertools import permutations
from scipy.optimize import minimize
import sys
import networkx as nx
from scipy.stats import multivariate_normal
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree as Tree
import numpy as np
from autograd import value_and_grad, numpy as anp
from .low_rank import *

################################################################################

def rotate(M, U, w, V):
    A = U @ np.diag(w) @ M
    B = V @ np.linalg.inv(M.T)
    a = A.sum(0)
    b = B.sum(0)
    A /= a
    B /= b
    n = a * b
    return A, n, B

################################################################################

def rotate_components(H, U, w, V):
    def objective(x):
        assert not np.any(np.isnan(x))
        M = anp.reshape(x, (int(len(x) ** 0.5),) * 2)
        n = anp.diagonal(anp.linalg.inv(M) @ anp.diag(w) @ M)
        A = U @ M
        B = V @ anp.linalg.inv(M.T)
        if A.min() < 0 or B.min() < 0:
            return 1e10  # np.inf
        return anp.linalg.norm(n[0] * anp.outer(A[1:, 0], B[1:, 0]) - H[1:, 1:], 'fro')

    starts = [np.eye(3)[list(p)].reshape(-1) for p in permutations(tuple(range(3)))]
    obj = value_and_grad(objective)

    def opt(s, jac=False):
        #     print(s)
        if not jac:
            out = minimize(objective, s, method='powell')
    #         print(out)
            return out
        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")
            return minimize(value_and_grad(objective), s, jac=True, method='BFGS')

    res = min([opt(s) for s in starts], key=lambda r: r.fun)
    A, n, B = rotate(res.x.reshape(3, 3), U, w, V)
    return A, n, B, res.x.reshape(3, 3), res.fun

################################################################################

def optimize00(H, tol, iters=100):
    H = H.copy()
    x = H[0, 0]
    for i in range(iters):
        print('H[0, 0] %10.6f' % H[0, 0], end='\r')
        U, w, V, _ = unit_nmf(H, 3, beta_loss='kullback-leibler', solver='mu', init='nndsvda')
        A, n, B, M, _ = rotate_components(H, U, w, V)
        x = n[0] * A[0, 0] * B[0, 0]
        diff = x - H[0, 0]
        H[0, 0] = max(0, x)
        if abs(diff) < tol:
            return x
    print('H[0, 0]', H[0, 0])
    return x
    # raise ValueError('Fixed point did not converge: {} vs {}'.format(diff, tol))

################################################################################

def split_array(A, lengths):
    for l in lengths:
        yield A[:l]
        A = A[l:]
    assert len(A) == 0

################################################################################

def gaussian_matching(x, y, sigma, radius):
    nd = np.shape(x)[1]
    tx, ty = Tree(x), Tree(y)
    pairs = np.array([(i, j) for i, v in enumerate(tx.query_ball_tree(ty, radius)) for j in v])
    if not len(pairs):
        return np.zeros([2, 0], dtype=np.uint32)

    A = lil_matrix((len(x) + len(y),) * 2)
    isigma = np.ones(nd) / (np.sqrt(2) * sigma)
    for i, j in pairs:
        A[i, len(x) + j] = A[len(x) + j, i] = np.exp(-0.5 * np.linalg.norm(isigma * (x[i] - y[j]))**2)
    edges = np.array(list(map(sorted, nx.max_weight_matching(nx.from_scipy_sparse_matrix(A))))).T.astype(np.uint32)
    assert edges.ndim == 2
    edges[1] -= len(x)
    dist = multivariate_normal(mean=np.zeros(nd), cov=np.diag((np.zeros(nd) + np.sqrt(2) * sigma)**2))
    return edges, dist.pdf(x[edges[0]] - y[edges[1]])

################################################################################

def make_bins(data, weights, n, attempts=100):
    i = np.argsort(data)
    spaces = n + 1
    for _ in range(attempts):
        out = np.unique(np.interp(np.linspace(0, 1, spaces), weights[i].cumsum() / weights.sum(), data[i]))
        if len(out) == n+1:
            break
        else:
            spaces += 1
    bump = 1e-10 * (out[-1] - out[0])
    out[:-1] -= bump
    out[-1] += bump
    return out

################################################################################

def nonzero_bins(data, weights, n, attempts=100):
    '''Bin nonzero data as normal, but put zero data as their own bin'''
    nz = data.nonzero()
    data, weights = data[nz], weights[nz]
    bins = make_bins(data, weights, n, attempts)
    bins[0] = data.min() / 2
    return np.concatenate([[-1], bins])

################################################################################

def nmf_error(A, n, norm='fro'):
    if n == 0:
        return 1
    W, H, _ = nmf(A, n)  # _ is same if it's fro
    return np.linalg.norm(W @ H - A, norm) / np.linalg.norm(A, norm)

################################################################################
