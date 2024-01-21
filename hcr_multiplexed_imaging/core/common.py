'''
Common functionality
'''

import functools, multiprocessing, pickle, codecs, itertools, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np, xarray as xa, scipy.sparse as sp
from scipy import stats

################################################################################

ABBREVIATIONS = {
    'x': 'x',
    'y': 'y',
    'z': 'z',
    'xyz': 'r',
    'wavelength': 'l', # emission
    'fluorophore': 'f',
    'experiment': 'q',
    'excitation': 'e',
    'component': 'c'
}

# this number times sigma gives the requested parameter
SIGMA_EQUIVALENTS = {
    'fwhm': 2 * np.sqrt(2 * np.log(2)),
    'hwhm': 1 * np.sqrt(2 * np.log(2)),
}

################################################################################

def hist(axes, X, histtype='step', fraction=True, *args, **kwargs):
    '''
    matplotlib histogram with changed default histogram type
    `fraction` (bool): plot fractions rather than absolute counts
    '''
    X = np.asarray(X).reshape(-1)
    if fraction:
        weights = np.ones(len(X)) / len(X)
        return axes.hist(X, histtype=histtype, *args, weights=weights, **kwargs)
    else:
        return axes.hist(X, histtype=histtype, *args, **kwargs)

################################################################################

def zoom(I, slices):
    '''zoom an image using its coordinate values (inclusive on both sides)'''
    ranges = []
    for d in I.dims:
        if d in slices:
            c = getattr(I, d)
            min, max = slices[d]
            idx = np.where(np.logical_and(c >= min, c <= max))[0]
            if len(idx):
                ranges.append(slice(idx[0], idx[-1] + 1))
            elif len(I):
                ranges.append(slice(0, 0))
            else:
                ranges.append(slice(None))
        else:
            ranges.append(slice(None))
    return I[tuple(ranges)]

################################################################################

def histogram_percentile(percentiles, x, hist):
    return np.interp(list(percentiles), np.cumsum(hist) / np.sum(hist), x)

################################################################################

def stable_set(iterable):
    if hasattr(iterable, 'dtype'):
        return np.unique(iterable)
    x = []
    for i in iterable:
        if i not in x:
            x.append(i)
    return tuple(x)

################################################################################

def log_bins(x, n):
    '''List of n logarithmically spaced bins that cover the given data, first and last bins rounded to log10'''
    i = np.floor(np.log10(np.min(x)))
    j = np.ceil(np.log10(np.max(x)))
    return 10.0 ** np.linspace(i, j, n+1)

################################################################################

def timed(name, f, *args, **kws):
    start = time.time()
    out = f(*args, **kws)
    print('{} = {:.3} seconds'.format(name, time.time() - start))
    return out

################################################################################

def dimension_chunks(array, kws):
    for x in array.chunk(kws).chunks:
        if len(x) == 1:
            yield (slice(None),)
        else:
            sums = np.cumsum(x)
            yield (slice(0, x[0]),) + tuple(slice(a, b) for a, b in zip(sums, sums[1:]))

def chunk_indices(array, kws):
    return itertools.product(*dimension_chunks(array, kws))

################################################################################

FLOAT_TYPES = [np.float32, np.float64, getattr(np, 'float128', None)]

################################################################################

def sums_to_1(x):
    s = as_float(x.dtype)(x.sum())
    return (s if s == 0 else np.rec(s)) * x

################################################################################

def as_float(dtype, types=FLOAT_TYPES):
    '''Return float type of at least the same number of bits as the given dtype'''
    try:
        bits = np.finfo(dtype).bits
    except ValueError:
        bits = np.iinfo(dtype).bits
    return next(t for t in FLOAT_TYPES if t is not None and np.finfo(t).bits >= bits)

################################################################################

def _apply_norm(x, p):
    norm = np.linalg.norm(x.reshape(-1), ord=p)
    if norm != 0:
        x /= norm

def normed(p, A, *, axis=None):
    A = np.array(A, dtype=as_float(A.dtype))
    if axis is None:
        _apply_norm(p, A)
    else:
        for a in A.transpose(axis, *(i for i in range(A.ndim) if i != axis)):
            _apply_norm(a, p)
    return A

################################################################################

def merge_maps(*mappings):
    out = {}
    for m in mappings:
        out.update(m)
    return out

def pop(mapping, *keys, cls=dict):
    try:
        mapping = mapping.items()
    except AttributeError:
        pass
    return cls(kv for kv in mapping if kv[0] not in keys)

def remove(items, *values, cls=None):
    if cls is None:
        try:
            return type(items)(v for v in items if v not in values)
        except TypeError:
            pass
    cls = tuple if cls is None else cls
    return type(items)(v for v in items if v not in values)

################################################################################

def is_computed(x):
    return getattr(x, 'chunks', None) is None

################################################################################

def apply(array, function, *args, **kwargs):
    '''Apply a function or method name to an array and copy over its attributes'''
    if callable(function):
        out = function(array, *args, **kwargs)
    else:
        out = getattr(array, function)(*args, **kwargs)
    out.attrs.update(array.attrs)
    return out

################################################################################

def sparse_mapping(mapping, shape, dtype=None):
    out = sp.lil_matrix(tuple(shape), dtype=dtype)
    for (i, j), v in mapping.items():
        out[i, j] = v
    return out.tocsc()

################################################################################

def collapse_first(A, n):
    return A.reshape(-1, *A.shape[n:])

def reshape_square(X):
    n2 = np.prod(np.shape(X))
    n = int(n2 ** 0.5)
    assert n * n == n2, 'Number of array elements is not a square'
    return np.reshape(X, (n, n))

################################################################################

def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

################################################################################

def dimensions(image):
    out = dict(image.coords)
    for d in image.dims:
        if d not in out:
            out[d] = np.arange(a.sizes[d])
    return out

################################################################################

def inclusive_bounds(image):
    '''voxel dimensions * number of pixels in each dimension'''
    # return tuple(image.coords.get(d, ) for d in image.dims)
    out = []
    for s, d in zip(image.shape, image.dims):
        if s == 0:
            out.append((0, 0))
        elif d in image.coords:
            out.append((image.coords[d][0], (image.coords[d][-1] - image.coords[d][0]) * s / (s-1)))
        else:
            out.append((0, image.sizes[d]))
    return np.array(out, dtype=np.float64)

################################################################################

def as_scalar(value):
    '''Replacement for numpy.asscalar'''
    return np.asarray(value).item()

################################################################################

def box(image, default=1, tol=1e-4):
    '''voxel dimensions'''
    out = []
    for d in image.dims:
        x = image.coords.get(d, [])
        if np.ndim(x) == 1 and len(x) > 1 and is_number(x[0]) and (len(x) == 2 or abs(np.diff(np.diff(x))).max() < tol * (np.max(x) - np.min(x))):
            out.append(abs(as_scalar(x[1] - x[0])))
        else:
            out.append(default)
    return np.array(out) if is_number(default) else out

################################################################################

def dump64(obj):
    return codecs.encode(pickle.dumps(obj), 'base64').decode()

def load64(s):
    return pickle.loads(codecs.decode(s.encode(), 'base64'))

################################################################################

def pool(workers=multiprocessing.cpu_count()):
    return ThreadPoolExecutor(workers)


def _use_pool(f, *args, pool=multiprocessing.cpu_count(), **kwargs):
    if isinstance(pool, int):
        with ThreadPoolExecutor(pool) as p:
            return f(*args, pool=p, **kwargs)
    return f(*args, pool=pool, **kwargs)

def use_pool(f):
    return functools.wraps(f)(functools.partial(_use_pool, f))

################################################################################

def assert_unique(values):
    values = set(values)
    # assert len(values) == 1
    return next(iter(values))

################################################################################

class Star:
    def __init__(self, fun, args, kwargs, unpack):
        self.fun, self.args, self.kwargs, self.unpack = fun, args, kwargs, unpack

    def __call__(self, x):
        if self.unpack:
            return self.fun(*x, *self.args, **self.kwargs)
        else:
            return self.fun(x, *self.args, **self.kwargs)


def starmap(pool, fun, iters, *args, unpack=None, lazy=False, **kwargs):
    unpack = isinstance(iters, zip) if unpack is None else unpack
    f = Star(fun, args, kwargs, unpack)
    if pool is None:
        out = map(f, iters)
    elif pool == 'thread':
        with ThreadPoolExecutor() as ex:
            out = ex.map(f, iters)
    elif pool == 'process':
        with ProcessPoolExecutor() as ex:
            out = ex.map(f, iters)
    else:
        out = pool.map(f, iters)
    return out if lazy else list(out)
from scipy import stats

################################################################################

def rsquared(i, j):
    x, y = [i.data.ravel(), j.data.ravel()]
    slope, interecept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2

def rsquared_matrix(*img):
    out = np.zeros([len(img)] * 2)
    for i in range(len(img)):
        out[i, i] = 1
        for j in range(i+1, len(img)):
            out[i, j] = out[j, i] = rsquared(img[i], img[j])
    return out

################################################################################

def max_z(img):
    '''Max z projection'''
    return img.max('z') if 'z' in img.dims else img