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
def detect_dots_v1(image, sigma_sharp, sigma_smooth, sigma0, euclidean, n_candidates, thresh, k_min):
    '''
    An old dot detection algorithm. Unsupported.
    '''
    min_separation = k_min * DOH_MINIMUM_FACTOR * sigma0
    out = candidate_dots(image.astype(np.float32), low=sigma_sharp, high=sigma_smooth, sigma=sigma0, min_separation=min_separation, euclidean=True)
    out.update(optimize_gmm(out['blurred'].data, out['pixel_centers'][:n_candidates], (sigma0**2 + sigma_sharp**2)**0.5, sigma_bounds=(sigma_sharp, sigma_smooth),
        box=box(out['blurred']), optimize='center', method='L-BFGS-B', options={'ftol': 1e-8, 'disp': False}), dtype=np.float32)

    keep = remove_close_pairs(out['centers'], out['weights'], radii=min_separation)
    out['weights'], out['centers'] = threshold(thresh, out['weights'][keep], out['centers'][keep])
    return out

################################################################################

@parallelize_first
def detect_dots(image, n_candidates, sigma_smooth=0.5, sigma=0.2, smooth_factor=1, smooth_threshold=None, *, 
    r_min=None, agglomerate=False, n_maxima=int(1e6), allow_edges=True, optimize='center', sigma_sharp=0, 
    gaussian=None, weight_sigma=None, k_min=2, dtype=np.float32):
    '''
    Detect dots in a given image. 
    If first argument is a tuple, evaluate in parallel for each item in the tuple.

    - `sigma` (float/array): approximate expected dot radius (in microns)
    - `sigma_smooth` (float/array): lengthscale greater than the dot width (in microns)
    - `sigma_sharp` (float/array): lengthscale less than the dot width, will result in a preliminary blur if nonzero (in microns)
    - `n_candidates` (int): loose upper bound on the number of dots
    - `k_min` (float): factor of separation between dots, used only if r_min is not specified
    - `r_min` (float/array): minimum separation between dots in the same channel (in microns)
    - `n_maxima` (int): upper bound on the number of local maxima in the determinant of Hessian considered

    Typically, the default values of `n_maxima=1000000`, `sigma_sharp=0`, `k_min=2`, and `r_min=0` will work well.
    For `n_candidates`, a rough number may be chosen, so long as it is obviously higher than the true number of dots.
    However, a larger number results in a longer computation time.
    A number of less commonly supplied options are also specifiable for advanced users:

    - `smooth_factor` (float): remove features larger than the high pass lengthscale by a factor of this number
    - `smooth_threshold` (float): after removing high pass features, zero any pixels below this threshold if given
    - `agglomerate` (bool): if dots are combined due to being too close, should their weights also be combined?
    - `allow_edges` (bool): allow maxima to be detected on the edges of the image
    - `weight_sigma` (float/array): expected dot width (in microns) -- used for Gaussian optimization, defaults to the same as `sigma`
    - `optimize` (str): optimization type (center, ellipsoid, or sphere)
    - `gaussian` (dict): additional Gaussian mixture optimization options
    '''
    if weight_sigma is None:
        weight_sigma = sigma
    if r_min is None:
        r_min = k_min * DOH_MINIMUM_FACTOR * sigma
    else:
        assert k_min == 1, 'Specifying k_min has no effect if r_min is specified'
    # Run convolutions
    out = candidate_dots(image.astype(dtype), low=sigma_sharp, high=sigma_smooth, sigma=sigma, 
        min_separation=r_min, euclidean=True, smooth_factor=smooth_factor, 
        allow_edges=allow_edges, smooth_threshold=smooth_threshold, n_maxima=n_maxima)
    # Remove candidate dots that are too close. Agglomeration doesn't make sense here
    _, out['pixel_centers'], out['maxima'] = remove_close_pairs(
        out['pixel_centers'], out['maxima'], radii=r_min, agglomerate=False)
    # Solve GMM without position modification
    out['solve_value'], out['solve_weights'] = calculate_gmm(out['blurred'], out['pixel_centers'],
        np.full_like(out['pixel_centers'], weight_sigma), box=box(image), dtype=dtype, gaussian=gaussian)
    # Only take top dots
    order = np.argsort(out['solve_weights'])[::-1]
    out['pixel_centers'] = out['pixel_centers'][order]
    out['maxima'] = out['maxima'][order]
    # Optimize GMM positions
    options = {'gtol': 1e-8, 'disp': False}#, 'ftol': 0e-20}
    out.update(optimize_gmm(out['blurred'].data, out['pixel_centers'][:n_candidates], sigma,
        box=box(out['blurred']), method='L-BFGS-B', options=options,
        optimize=optimize, gaussian=gaussian, sigma_bounds=None, dtype=dtype))
    out['full_centers'], out['full_weights'] = c, w = out['centers'], out['weights']
    # Remove zero dots
    c, w = c[w > 0], w[w > 0]
    # Remove dots that became too close
    _, c, w = remove_close_pairs(c, w, radii=r_min, agglomerate=agglomerate)
    # Threshold results
    # if maxima_threshold is not None:
    #     w, c = threshold(maxima_threshold, w, c)
    out['centers'], out['weights'] = c, w

    out['dots'] = pd.DataFrame(np.concatenate([out['weights'][:, None], out['centers']], axis=1), 
        columns=['weight'] + list(image.dims))
    return out

################################################################################
