import copy, logging, numexpr, numpy as np
from skimage import io as skio, img_as_float
from tifffile import TiffFile as Tiff
from scipy import ndimage
from scipy.spatial import cKDTree as Tree
from xarray import DataArray
from .common import use_pool, apply, box
from .arrays import gather
from .files import output_file, FILES

log = logging.getLogger(__name__)

################################################################################

def crop(I, **slices):
    '''Crop an image using its coordinate values (inclusive on both sides)'''
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

def convert_image(I, dtype):
    '''
    Convert images between integer and floating point types
    Normalization is applied such that integers map to floats in [0, 1]
    Normalization is applied such that floats map to integers in [0, imax]
    '''
    assert I.dtype in (np.float32, np.float64, np.uint16, np.uint16)
    assert dtype in (np.float32, np.float64, np.uint16, np.uint16)
    x = I.dtype
    if x in (np.float32, np.float64):
        if dtype == np.uint8:
            I = I * 255
        if dtype == np.uint16:
            I = I * 65535
        I = I.astype(dtype)
    elif x == np.uint16:
        I = I.astype(dtype)
        if dtype in (np.float32, np.float64):
            I *= 1.0 / 65535
    elif x == np.uint8:
        I = I.astype(dtype)
        if dtype in (np.float32, np.float64):
            I *= 1.0 / 255
    return I

def new_image(data, dims, coords, **kwargs):
    '''Wrapper of DataArray giving some more flexible defaults'''
    dims = tuple(dims)
    crds = {d: coords[d] for d in dims if d in coords}
    return DataArray(data, crds, dims, attrs=kwargs)


def project(operation, x, axis, **kws):
    '''
    Apply a reduction operation to all axes except the one(s) specified
    operation: 'mean', 'max', 'min', etc
    '''
    dims = list(getattr(x, 'dims', range(x.ndim)))
    if axis in dims:
        axis = (axis,)
    for a in axis:
        try:
            dims.remove(a % x.ndim if isinstance(a, int) else a)
        except ValueError:
            raise KeyError('Axis {} not in dims {}'.format(a, dims))
    return getattr(x, operation)(tuple(dims), **kws)

def clip(image, bounds):
    o = copy.deepcopy(image)
    if np.isscalar(bounds):
        np.clip(image.data / bounds, 0, 1, out=o.data)
    else:
        np.clip((image.data - bounds[0]) / (bounds[1] - bounds[0]), 0, 1, out=o.data)
    return o

def rgba32(image, clip=None, scale=255):
    if clip is None:
        I = np.clip(image.data, 0, scale)
    elif np.isscalar(clip):
        I = np.clip(image.data * (scale / clip), 0, scale)
    else:
        I = np.clip((image.data - clip[0]) * (scale / (clip[1] - clip[0])), 0, scale)
    I = I.astype(np.uint8)
    dims = list(image.dims)
    dims.remove('c')
    try:
        I = I.view(dtype=np.uint32).squeeze(-1)
    except ValueError:
        raise ValueError('Image shape {} is not interpretable as RGBA data'.format(I.shape))
    return new_image(I, dims, image.coords)

def info(image):
    return 'Image({}, {})'.format(list(image.shape), list(box(image)))


def open_tiff(fn, box=None, reverse=True, dims='zyx', **kwargs):
    '''Read image from a file'''
    if hasattr(fn, 'exists'):
        assert fn.exists()
        fn = str(fn)
    if box is None:
        with Tiff(fn) as t:
            tags = {k: v.value for k, v in t.pages[0].tags.items()}
        try:
            descr = dict(i.split('=') for i in tags['ImageDescription'].split('\n'))
            x, y = [j/i for i, j in (tags[s + 'Resolution'] for s in 'XY')]
            z = float(descr['spacing'])
            box = dict(z=z, y=y, x=x)
        except KeyError:
            box = dict(z=1, y=1, x=1)
    I = skio.imread(fn)
    I = np.fliplr(I).copy() if reverse else I
    dims = dims[:len(I.shape)]
    coords = {d: np.arange(s) * box[d] for d, s in zip(dims, I.shape) if d in box}
    return DataArray(I, dims=tuple(dims), coords=coords, attrs=kwargs)

FILES['.lsm'] = FILES['.tiff'] = FILES['.tif'] = (open_tiff, None)

################################################################################

def upscale(I, target):
    '''interpolate the log of the image, whose values should be nonnegative'''
    factor = np.round(np.maximum(1, box(I) / target)).astype('int')
    if np.all(np.array(1) == factor): return I
    log.info('Upscaling image of resolution {} by factor {}'.format(box(I), factor))
    return new_image(ndimage.interpolation.zoom(I.data, factor), I.dims, box(I) / factor)
    # logarithmic interpolation, didn't work well:
    #I = ndimage.interpolation.zoom(np.log(I + np.finfo(I.dtype).tiny), factor)
    #np.exp(I, out=I)
    #return I

################################################################################

def color_stack(*channels, fill=None, dim='k'):
    '''Stack images as color channels in the same image, probably RGB or RGBA'''
    if fill is not None:
        channels = channels + tuple(fill)[len(channels):]
    ref = next(filter(np.ndim, channels))
    o = new_image(np.ndarray(ref.shape + (len(channels),), dtype=ref.dtype), ref.dims +  (dim,), ref.coords)
    for i, c in enumerate(channels):
        o.data[((slice(None),) * np.ndim(ref)) + (i,)] = c
    return o

################################################################################

def boxed(sigma, I):
    '''Translate array of real distances into array of pixel distances'''
    return np.asarray(sigma) / np.asarray(box(I))

################################################################################

def lo_pass_blur(I, sigma, truncate=5.0, mode='reflect'):
    '''
    Gaussian blur of the image
    I: the image
    sigma: the standard deviation of the blur in real distance
    truncate: the unitless convolution truncation factor
    '''
    o = copy.deepcopy(I)
    log.info('Running low pass blur with pixel sigma={}'.format(boxed(sigma, I)))
    if np.any(sigma):
        ndimage.filters.gaussian_filter(I, sigma=boxed(sigma, I), truncate=truncate, output=o.data, mode=mode)
    return o

################################################################################

def mean_filter(I, lengths, mode='reflect'):
    o = copy.deepcopy(I)
    lengths = [1 + 2 * lengths.get(d, 0) for d in o.dims]
    ndimage.filters.uniform_filter(I, lengths, output=o.data, mode=mode)
    return o

################################################################################

def hi_pass_blur(I, sigma, truncate=5.0, factor=1.0, threshold=None):
    '''Subtract a blurred image from an unblurred image, threshold all resultant negative values to 0'''
    o = copy.deepcopy(I)
    log.info('Running high pass blur with pixel sigma={} and factor {}'.format(boxed(sigma, I), factor))
    if not np.any(sigma) or not factor:
        return o
    o[:] = ndimage.filters.gaussian_filter(I.data, sigma=boxed(sigma, I), truncate=truncate)
    o *= -factor
    o += I
    if threshold is not None:
        log.info('Thresholding image to threshold={}'.format(threshold))
        np.maximum(o.data, threshold, o.data)
    else:
        log.info('Not thresholding image')
    return o

################################################################################

def mdet_of_hessian(I, sigmas, factors=None, truncate=5.0, map=map):
    '''
    Minus determinant of the Hessian of the blurred image
    Done for 2D and 3D
    '''
    if factors is None:
        factors = [1] * len(sigmas)
    assert I.ndim in (2, 3)

    log.info('Starting DoH convolutions with pixel sigmas {}, factors {}, and truncation scale {}'.format(boxed(sigmas, I), factors, truncate))

    def apply(factor, sigma, order):
        out = ndimage.filters.gaussian_filter(I, sigma=boxed(sigma, I), order=order, truncate=truncate)
        if factor != 1: 
            out *= factor
        return out

    if I.ndim == 3:
        orders = [2,0,0], [0,2,0], [0,0,2], [1,1,0], [1,0,1], [0,1,1]
    if I.ndim == 2:
        orders = [2,0], [0,2], [1,1]

    r = [apply(f, s, o) for f, s in zip(factors, sigmas) for o in orders]
    for i in reversed(range(len(orders), len(r))):
        r[i % len(orders)] += r.pop()

    if I.ndim == 3:
        xx, yy, zz, xy, xz, yz = r
        numexpr.evaluate('xx*yy*zz + 2*xy*xz*yz - xx*yz**2 - yy*xz**2 - zz*xy**2', optimization='aggressive', out=xx)
    if I.ndim == 2:
        xx, yy, xy = r
        numexpr.evaluate('xx*yy - xy*xy', optimization='aggressive', out=xx)

    xx *= -(np.prod(box(I)) ** -2) # negate and make derivatives per um rather than per pixel
    log.info('Finished DOH convolutions')
    return new_image(xx, I.dims, I.coords)

################################################################################

def doh_weight(values, *sigma, ndim=None):
    '''
    Correct DOH values to be proportional to a Gaussian integrated intensity
    Should work for any dimension
    '''
    sigma2 = sum(s * s for s in sigma)
    if ndim is not None:
        sigma2 = sigma2 + np.zeros(ndim)
    n = len(sigma2)
    k = (2.0 * np.pi)**(0.5 * n) * np.prod(sigma2)**(1 - 0.5 / n)
    return numexpr.evaluate('k * values**(1.0 / n)')

################################################################################

def max_detect(I, *, radius, allow_edges: bool, euclidean: bool, n_maxima: int):
    '''
    use a cubic filter if not euclidean else do a distance search
    radius and box can be scalars or arrays, usually just box is an array
    n_maxima: only consider the highest # of maxima
    '''
    # Find all local maxima if Euclidean else set up the box
    size = 3 if euclidean else np.array(2 * np.round(radius / box(I)) + 1).astype('int')
    A = I.data
    log.info('Running maximum filter with size={} and allow_edges={}'.format(size, allow_edges))
    # A = A + 1e-30 * (A.max() - A.min()) * np.random.standard_normal(A.shape).astype(A.dtype)
    if allow_edges:
        mask = np.where(A == ndimage.filters.maximum_filter(A, size=size, mode='reflect'))
    else:
        log.info('Padding edges with maximum value (not infinity)')
        mask = np.where(A == ndimage.filters.maximum_filter(A, size=size, mode='constant', cval=A.max()))
    values = A[mask]
    log.info('Found {} total maxima'.format(len(values)))
    # mi
    order = np.argsort(values)[::-1][:n_maxima]
    values, mask = values[order], [m[order] for m in mask]

    log.info('Got {} maxima out of {} pixels'.format(len(values), np.prod(A.shape)))
    if euclidean:
        log.info('Running Euclidean distance search on {} points'.format(len(values)))
        pairs = Tree(np.asarray(mask).T * (box(I) / radius)).query_pairs(1.0, output_type='ndarray').T
        order = np.setdiff1d(np.arange(len(values)), np.where(np.less(*values[pairs]), *pairs))
        order = order[np.argsort(values[order])[::-1]]
    else:
        order = np.argsort(values)[::-1]
    log.info('Done max detect')
    return values[order], np.array([m[order] for m in mask]).T

################################################################################

def candidate_dots(I, *, low, high, sigma, min_separation, euclidean, allow_edges, n_maxima, smooth_factor=1, smooth_threshold=None):
    '''returns image, hessian image, centers in um, maxima values'''
    assert I.dtype in (np.float32, np.float64, float)
    log.info('Step 1: low pass blur')
    if low is not None:
        I = lo_pass_blur(I, low)
    log.info('Step 2: high pass blur')
    if high is not None:
        I = hi_pass_blur(I, high, factor=smooth_factor, threshold=smooth_threshold)
    log.info('Step 3: DoH convolution')
    H = mdet_of_hessian(I, sigmas=[sigma])
    log.info('Step 4: maximum detection')
    V, M = max_detect(H, radius=min_separation, allow_edges=allow_edges, euclidean=euclidean, n_maxima=n_maxima)
    log.info('Step 5: determine weights')
    V = doh_weight(V, sigma, sigma, ndim=np.ndim(I))
    return dict(pixel_centers=M * box(I), blurred=I, hessian=H, maxima=V)

################################################################################

def synthetic_deltas(n, shape, resolution, dtype=np.float32, dims=None):
    '''
    Make n delta functions in an array at random non-integer points
    The sum of the resultant array is exactly n.
    To make a diffraction limited image, call lo_pass_blur() on the output of this function.
    '''
    if dims is None:
        dims = 'zxy' if len(shape) == 3 else 'xy'
    resolution = np.asarray(resolution)
    out = DataArray(np.zeros(shape, dtype=dtype), dims=tuple(dims), coords={k: p * np.arange(n) for k, n, p in zip(dims, shape, resolution)}, attrs=dict(kind='synthetic'))
    spots = np.random.random(size=(n, len(shape))) * (np.asarray(shape) - 1)
    for s in spots:
        fracs = np.ones((2,) * len(shape), dtype=dtype)
        floor = np.floor(s).astype(int)
        for i, f in enumerate(s - floor): # linear interpolation
            fracs[tuple(0 if j == i else slice(None) for j in range(len(shape)))] *= 1 - f
            fracs[tuple(1 if j == i else slice(None) for j in range(len(shape)))] *= f
        out[tuple(slice(i, i+2) for i in floor)] += fracs
    out.attrs['dots'] = (spots + 0.5) * resolution
    return out

################################################################################

@use_pool
def shift_image(img, shift, *, low=None, high=None, pool):
    assert img.dims[0] not in 'xyz'
    # assert img.dtype in (np.float32, np.float64, float)

    def run(i):
        if low is not None:
            img.data[i] = lo_pass_blur(img[i], low)
        if high is not None:
            img.data[i] = hi_pass_blur(img[i], high, threshold=0)
        img.data[i] = ndimage.shift(img.data[i], shift / box(img[i]), order=1, mode='constant')

    tuple(pool.map(run, range(img.shape[0])))

################################################################################

def calculate_mask(images, factor, method='mean'):
    mask = None
    for img in images:
        update = np.greater_equal(img, factor * getattr(np, method)(img))
        if mask is None:
            mask = update
        else:
            mask |= update
    return mask

################################################################################

def calculate_mask(images, factor, method='mean', compare='less'):
    '''Return mask which is true where ALL of images are <compare> than factor * <method>(array)'''
    compare, method = (f if callable(f) else getattr(np, f) for f in (compare, method))
    mask, tmp = None, None
    for img in images:
        if mask is None:
            mask = compare(img, factor * method(img))
        elif tmp is None:
            tmp = compare(img, factor * method(img))
            mask &= tmp
        else:
            compare(img, factor * method(img), out=getattr(tmp, 'values', tmp))
            mask &= tmp
    return mask

################################################################################

@use_pool
def background(arrays, pool, axes='xyz'):
    out = [pool.submit(lambda l: apply(l.compute(), 'mean', tuple(axes)), l) for l in arrays]
    out = [v.result() for v in out]
    return gather(out, 'el', [v.laser for v in out])