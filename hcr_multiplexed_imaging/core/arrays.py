'''
Utilities to deal with xarray.DataArray objects
'''

import numpy as np, xarray as xa
from .common import remove

################################################################################

def array(x, dims, coords=None, attrs=None, reshape=None, name=None):
    '''
    reshape may be given as a letter ('A', 'F', 'C')
    '''
    dims = tuple(dims)
    coords = {} if coords is None else {k: v for k, v in coords.items() if k in dims}
    if reshape:
        x = np.asarray(x).reshape(*[len(coords[d]) if d in coords else -1 for d in dims], order=reshape)
    return xa.DataArray(x, dims=dims, coords=coords, attrs=attrs, name=name)

################################################################################

def stack(x, dims, new=None, axis=0, coord=None):
    '''Concatenate arrays which may be different shapes on an existing axis'''
    dims = tuple(dims)
    new = ''.join(dims) if new is None else new
    attrs, out = {}, []
    keys = set(k for a in x for k in a.attrs)
    for i, a in enumerate(x):
        for k in keys:
            attrs.setdefault(k, []).append(a.attrs.get(k))
        for d in reversed(dims):
            a = a.expand_dims(d) if d not in a.dims else a
        if coord is not None:
            a.coords[dims[0]] = [coord[i]]
        a = a.stack({new: dims})
        if axis is not None:
            axes = list(remove(a.dims, new))
            axes.insert(axis, new)
            a = a.transpose(*axes)
        # xarray has issues when an attribute is a list when concatenating
        a.attrs = {k: tuple(v) if isinstance(v, list) else v for k, v in a.attrs.items()}
        out.append(a)
    try:
        out = xa.concat(out, new)
    except Exception:
        raise ValueError('Could not concatenate arrays on {} \n'.format(new) + '\n'.join(map(str, out)))
    for k, v in attrs.items():
        out.attrs.setdefault(k, v) # some are already handled by xarray defaults
    return out

################################################################################

def values(x):
    '''Cast to a numpy.ndarray'''
    return x.values if hasattr(x, 'values') else np.asarray(x)

################################################################################

def norm2(x, axis=None, dtype=None):
    if dtype is None:
        dtype = as_float(x.dtype)
    if axis is None:
        x = np.asarray(x).reshape(-1)
        n = 1 + (len(x) >> 12)
        bound = n * (len(x) // n)
        t = x[bound:].astype(dtype)
        out = t @ t
        t = np.ndarray(len(x) // n, dtype=dtype)
        for s in np.split(x[:bound], n):
            t[:] = s
            out += t @ t
        return out
    if isinstance(axis, int):
        x = np.transpose(x, [axis] + [i for i in range(np.ndim(x)) if i != axis])
    else:
        x = x.transpose(axis, *remove(x.dims, axis))
    out = np.zeros(x.shape[1:], dtype=dtype)
    t = np.empty(x.shape[1:], dtype=dtype)
    for y in x:
        t[:] = y
        t *= t
        out += t
    return out

################################################################################

def test_norm2():
    t = xa.DataArray(np.random.random([500,600]), dims=('x', 'y'))
    assert abs(norm2(t) - np.linalg.norm(t.values.ravel())**2) < 1e-8
    assert np.all(abs(np.linalg.norm(t.values, axis=0)**2 - norm2(t, axis='x')).max() < 1e-8)

################################################################################

def move_axis(x, axis, position=None):
    if axis is None:
        return x
    dims = list(remove(x.dims, axis))
    dims.insert(position, axis)
    return x.transpose(*dims)

################################################################################

def rstack(x, axis=0, dims='xyz', new='r'):
    return move_axis(x.stack({new: tuple(d for d in x.dims if d in dims)}), new, axis)

################################################################################

def spatial_covariance(x, weights, chunks={}):
    assert is_computed(x) and is_computed(weights)
    assert x.dims[:len(weights.dims)] == weights.dims, (x.dims, weights.dims)
    out = 0
    for i in chunk_indices(weights, chunks):
        xi = x[i].astype(weights.dtype)
        yi = xi.reset_index('el').rename(el='el2', l='l2', e='e2').set_index({'el2': ('e2', 'l2')})
        out += xa.dot(xi, weights[i] * yi, dims=tuple('xyz'))
    return out

################################################################################

def gather(x, dim, coord=None, attrs=0, dtype=None):
    '''Gather DataArray objects along a new dimension'''
    dims = (dim,) + (x[0].dims if len(x) else ())
    coords = dict(x[0].coords) if len(x) else {}
    if coord is not None:
        coords[dim] = list(coord) if isinstance(coord, tuple) else coord
    if isinstance(attrs, int) and attrs < len(x):
        attrs = x[attrs].attrs
    return xa.DataArray(np.array([y.values for y in x], dtype=dtype), dims=dims, coords=coords, attrs=attrs)

################################################################################
