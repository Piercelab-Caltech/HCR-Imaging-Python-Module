from ..core.image import *
from .coloc import *
from .gmm import optimize_gmm, calculate_gmm
import numpy as np, pandas as pd
from scipy.spatial import cKDTree as Tree
from multiprocessing import cpu_count
from functools import wraps
from concurrent.futures import ThreadPoolExecutor as Executor

# Minimum distance in terms of sigma that peaks have to be to have 2 sepaarate maxima in DOH
DOH_MINIMUM_FACTOR = 2 * np.sqrt((5 - np.sqrt(10)) /3)
# Minimum distance in terms of sigma that peaks have to be to have 2 separate maxima
MAX_MINIMUM_FACTOR = 2.0

################################################################################

def interactive_colocalize(thresholds, weights, centers, coloc, fig, update):
    ''''Interactive colocalize inside notebook'''
    def analyze(t1, t2, xy, z):
        nonlocal thresholds
        thresholds = [t1, t2]
        W, X = threshold((t1, t2), weights, centers)
        parts = partitions(X, W, xy=xy, z=z)
        c = len(parts[0][0])
        a, b = len(X[0]), len(X[1])
        return parts, [c / (a + b - c), c / a, c / b, a, b, c]

    parts, counts = analyze(*thresholds, **coloc)
    circs = [circles(fig, i[0], radius=2*np.sqrt(i[1]), color=c) for i, c in zip(parts, hexes(*'yrg'))]

    fields = 'J CA CB A B AB'.split()
    source = Source(dict(Description=fields, Value=counts))
    columns = [TableColumn(field=i, title=i) for i in ('Description', 'Value')]
    table = DataTable(source=source, columns=columns, width=200, height=200)

    def draw(t1, t2, xy, z):
        parts, counts = analyze(t1, t2, xy, z)
        for c, p in zip(circs, parts):
            c.data_source.data = dict(radius=2*np.sqrt(p[1]), x=p[0][:, 2], y=p[0][:, 1])
        source.data = dict(Description=fields, Value=counts)
        update()

    return draw, table

################################################################################

def parallelize(items, function):
    '''
    Run a function in parallel over the given items
    '''
    with Executor(cpu_count()) as ex:
        return list(ex.map(function, items))

def parallelize_first(function):
    '''
    If the first argument is a tuple, parallelize a function over the first argument
    Otherwise, call the function normally
    '''
    @wraps(function)
    def fun(first, *args, **kws):
        if isinstance(first, tuple):
            return parallelize(first, lambda x: function(x, *args, **kws))
        return function(first, *args, **kws)
    return fun

################################################################################

@parallelize_first
def detect_dots_v1(image, lo_pass, hi_pass, sigma0, euclidean, max_dots, thresh, min_ratio):
    '''
    An old dot detection algorithm. Unsupported.
    '''
    min_separation = min_ratio * DOH_MINIMUM_FACTOR * sigma0
    out = candidate_dots(image.astype(np.float32), low=lo_pass, high=hi_pass, sigma=sigma0, min_separation=min_separation, euclidean=True)
    out.update(optimize_gmm(out['blurred'].data, out['pixel_centers'][:max_dots], (sigma0**2 + lo_pass**2)**0.5, sigma_bounds=(lo_pass, hi_pass),
        box=box(out['blurred']), optimize='center', method='L-BFGS-B', options={'ftol': 1e-8, 'disp': False}), dtype=np.float32)

    keep = remove_close_pairs(out['centers'], out['weights'], radii=min_separation)
    out['weights'], out['centers'] = threshold(thresh, out['weights'][keep], out['centers'][keep])
    return out

################################################################################

@parallelize_first
def detect_dots(image, max_dots, hi_pass=0.5, sigma=0.2, hi_pass_factor=1, hi_pass_threshold=None, *, 
    limit_radii=None, agglomerate=False, max_maxima=int(1e6), allow_edges=True, optimize='center', lo_pass=0, 
    gaussian=None, weight_sigma=None, min_ratio=1, dtype=np.float32):
    '''
    Detect dots in a given image.
    If first argument is a tuple, evaluate in parallel for each item in the tuple

    - `max_dots` (int): maximum number of dots to consider
    - `lo_pass` (float/array): lengthscale less than the dot width (in microns)
    - `hi_pass` (float/array): lengthscale greater than the dot width (in microns)
    - `sigma` (float/array): expected dot width (in microns)
    - `hi_pass_factor` (float): remove features larger than the high pass lengthscale by a factor of this number
    - `hi_pass_threshold` (float): after removing high pass features, zero any pixels below this threshold if given
    - `limit_radii` (float/array): don't detect any candidate dots that are closer than this radius or radii
    - `min_ratio` (float): factor of separation between dots, use only if limit_radii is not specified
    - `max_maxima` (int): number of local maxima in the determinant of Hessian to consider (typically the default is fine)
    - `agglomerate` (bool): if dots are combined due to being too close, should their weights also be combined?
    - `allow_edges` (bool): allow maxima to be detected on the edges of the image
    - `weight_sigma` (float/array): expected dot width (in microns) -- used for Gaussian optimization, defaults to the same as `sigma`
    - `optimize` (str): optimization type (center, ellipsoid, or sphere)
    - `gaussian` (dict): additional Gaussian mixture optimization options
    '''
    if weight_sigma is None:
        weight_sigma = sigma
    if limit_radii is None:
        limit_radii = min_ratio * DOH_MINIMUM_FACTOR * sigma
    else:
        assert min_ratio == 1, 'Specifying min_ratio has no effect if limit_radii is specified'
    # Run convolutions
    out = candidate_dots(image.astype(dtype), low=lo_pass, high=hi_pass, sigma=sigma, 
        min_separation=limit_radii, euclidean=True, hi_pass_factor=hi_pass_factor, 
        allow_edges=allow_edges, hi_pass_threshold=hi_pass_threshold, max_maxima=max_maxima)
    # Remove candidate dots that are too close. Agglomeration doesn't make sense here
    _, out['pixel_centers'], out['maxima'] = remove_close_pairs(
        out['pixel_centers'], out['maxima'], radii=limit_radii, agglomerate=False)
    # Solve GMM without position modification
    out['solve_value'], out['solve_weights'] = calculate_gmm(out['blurred'], out['pixel_centers'],
        np.full_like(out['pixel_centers'], weight_sigma), box=box(image), dtype=dtype, gaussian=gaussian)
    # Only take top dots
    order = np.argsort(out['solve_weights'])[::-1]
    out['pixel_centers'] = out['pixel_centers'][order]
    out['maxima'] = out['maxima'][order]
    # Optimize GMM positions
    options = {'gtol': 1e-8, 'disp': False}#, 'ftol': 0e-20}
    out.update(optimize_gmm(out['blurred'].data, out['pixel_centers'][:max_dots], sigma,
        box=box(out['blurred']), method='L-BFGS-B', options=options,
        optimize=optimize, gaussian=gaussian, sigma_bounds=None, dtype=dtype))
    out['full_centers'], out['full_weights'] = c, w = out['centers'], out['weights']
    # Remove zero dots
    c, w = c[w > 0], w[w > 0]
    # Remove dots that became too close
    _, c, w = remove_close_pairs(c, w, radii=limit_radii, agglomerate=agglomerate)
    # Threshold results
    # if maxima_threshold is not None:
    #     w, c = threshold(maxima_threshold, w, c)
    out['centers'], out['weights'] = c, w

    out['dots'] = pd.DataFrame(np.concatenate([out['weights'][:, None], out['centers']], axis=1), 
        columns=['weight'] + list(image.dims))
    return out

################################################################################
