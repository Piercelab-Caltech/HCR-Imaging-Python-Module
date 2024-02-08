'''
Utilities and algorithms for dot colocalization
'''

from scipy.spatial import cKDTree as Tree
from scipy.spatial.distance import euclidean
import numpy as np

def remove_close_pairs(centers, weights, radii, agglomerate=True, **kws):
    '''
    Return the indices of the centers which have the largest weight in their neighborhood of given radius
    '''
    if not np.any(radii):
        return [], centers.copy(), weights.copy()
    if agglomerate:
        tree = Tree(centers / radii, **kws)
        w = np.array(weights, dtype=np.float64)
        c = w[:, None] * np.array(centers, dtype=np.float64) # weighted centers
        index = np.arange(len(weights))
        roots = []
        for i in range(len(w)):
            if not w[i] > 0:
                continue
            matches = tree.query_ball_point(c[i]/w[i]/radii, 1.0) # find matching points
            parent = index[matches[np.argmax(w[matches])]] # find largest matching point
            if i == parent: # if self is largest, it is a root
                roots.append(i)
            else: # add this weight to the parent
                index[i] = parent
                w[parent] += w[i]
                c[parent] += c[i]
        roots = np.array(roots, dtype=np.uint32)
        w = w[roots]
        return roots, c[roots] / w[:, None], w
    else:
        p, q = Tree(centers, **kws).query_pairs(radii, output_type='ndarray').T
        least = np.where(weights[p] < weights[q], p, q)
        keep = np.setdiff1d(np.arange(len(weights)), least)
        return [], centers[keep], weights[keep]


def pair_array(data, dtype=np.int64):
    '''Return a list of pairs as an array'''
    data = tuple(data)
    return np.array(data) if data else np.zeros([0, 2], dtype=dtype)


def distance(x, y):
    '''Return Euclidean distance between two vectors'''
    return np.linalg.norm(x - y, axis=-1)


def min_query(tree1, tree2, radius):
    '''Returns nearest neighbor within radius in tree 2 to every point in tree 1 as an int64(N, 2) list of i j pairs'''
    out = tree1.query_ball_tree(tree2, radius)
    return pair_array((i, min(o, key=lambda j: euclidean(x, tree2.data[j]))) for i, (x, o) in enumerate(zip(tree1.data, out)) if o)


def query(tree1, tree2, radius):
    '''Tests mutual nearest neighbors below a given radius, return int64(N, 2) list of i j pairs'''
    xy = min_query(tree1, tree2, radius)
    yx = min_query(tree2, tree1, radius)[:, (1, 0)]
    return pair_array(i for i in xy if i in yx)

def residuals(X1, X2, radius):
    pairs = query(Tree(X1), Tree(X2), radius)
    return X2[pairs[:, 1]] - X1[pairs[:, 0]], pairs

def colocalize(X1, X2, thresholds):
    '''
    X1, X2: (N,3) data points
    thresholds: dict of keys xy, z, r
    Returns list of (i, j) pairs that are colocalized
    '''
    if thresholds.get('r') is not None:
        return query(Tree(X1), Tree(X2), thresholds['r']) #pylint: disable=E1102
    p = query(Tree(X1[:, 1:]), Tree(X2[:, 1:]), thresholds['xy']) #pylint: disable=E1102
    if thresholds.get('z') is None:
        return p
    return pair_array(i for i in p if abs(X1[i[0], 0] - X2[i[1], 0]) < thresholds['z'])


def partitions(X, W, thresholds):
    '''Returns X and W for matched points and unmatched points...'''
    ij = colocalize(*X, thresholds).T
    cx = np.mean([x[p] for x, p in zip(X, ij)], axis=0)
    cw = np.mean([w[p] for w, p in zip(W, ij)], axis=0)
    U = [np.setdiff1d(np.arange(len(x)), p) for x, p in zip(X, ij)]
    # print(np.shape(X), np.shape(W), len(U), np.shape(U))
    unmatched = [(x[u], w[u]) for x, w, u in zip(X, W, U)]
    if np.ndim(cx) == 0:
        cx = np.zeros((0,) + np.shape(X[0])[1:], dtype=cx.dtype)
    if np.ndim(cw) == 0:
        cw = np.zeros((0,), dtype=cw.dtype)
    return [(cx, cw)] + unmatched


def threshold(threshold, weights, *args):
    '''
    Thresholds weights below a given fraction of their maxima
    Also prunes any other arrays in the same manner as weights
    '''
    i = np.where(weights >= threshold * weights.max())
    return [x[i] for x in (weights,) + args]


def agglomerate_pairs(x, y, factor=1.1, start=1e-3):
    '''Greedily pair points using an increasing threshold radius'''
    idx = [list(range(len(z))) for z in (x, y)]
    pairs = []
    distance = start * np.sqrt(np.var(np.concatenate([x, y])))
    while all(idx):
        xi, yi = x[idx[0]], y[idx[1]]
        tx, ty = Tree(xi), Tree(yi)
        tmp = {}
        for i, pts in enumerate(tx.query_ball_tree(ty, distance)):
            if pts:
                j = pts[np.argmin(np.linalg.norm(yi[pts] - xi[i]))]
                tmp.setdefault(j, []).append(i)

        rms = []
        for j, pts in tmp.items():
            i = pts[np.argmin(np.linalg.norm(yi[j] - xi[pts]))]
            rms.append((idx[0][i], idx[1][j]))

        for i, j in rms:
            idx[0].remove(i)
            idx[1].remove(j)
            pairs.append((i, j))

        if not tmp:
            distance *= factor

    return np.array(pairs, dtype=np.uint32) if pairs else np.empty((0, 2), dtype=np.uint32)
