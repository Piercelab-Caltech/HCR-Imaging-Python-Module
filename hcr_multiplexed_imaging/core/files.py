import pathlib, datetime, tempfile, shutil, pandas as pd, numpy as np, xarray as xa, functools, hashlib
from .common import dump64, load64, box

################################################################################

def file_md5(file_obj):
    '''md5 value of a file object'''
    hash_md5 = hashlib.md5()
    for chunk in iter(lambda: file_obj.read(4096), b""):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()

################################################################################

def auto_attrs(x):
    '''Automatic attributes adding during file save'''
    return {
        'time_created': datetime.datetime.utcnow().astimezone().isoformat(),
        'resolution': list(map(float, box(x, 0.0))),
        'shape': list(map(int, x.shape)),
        'dimensions': list(map(str, x.dims)),
        'gigabytes': x.nbytes / 1e9
    }

################################################################################

def save_netcdf(x, path, create=True):
    '''
    exactly equivalent to calling to_netcdf, but pickles attrs and fixes up multi indices
    '''
    path = pathlib.Path(path)
    if create:
        path.absolute().parent.mkdir(parents=True, exist_ok=True)
    x = x.assign_attrs(auto_attrs(x))
    multis = [(k, tuple(v.names)) for k, v in x.indexes.items() if isinstance(v, pd.MultiIndex)]
    for k, v in reversed(multis):
        x = x.reset_index(k)

    x.attrs['multi_indices'] = multis
    attrs = dump64(x.attrs)
    x.attrs.clear()
    x.attrs['pickle64'] = attrs
    if path.exists():
        path.unlink()
    return x.to_netcdf(path, mode='w')

def load_netcdf(path, cache=False, local=False, **kws):
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    s = str(path)

    x = xa.open_dataarray(s, cache=cache, **kws)

    if 'pickle64' in x.attrs:
        x.attrs.update(load64(x.attrs.pop('pickle64')))

    if 'multi_indices' in x.attrs:
        multis = x.attrs.pop('multi_indices')
        for k, v in multis:
            x = x.set_index({k: v})

    x.attrs['path'] = path.absolute()
    x.attrs['file_name'] = str(path.absolute())
    return x

################################################################################

def load_csv(path, indices=('kind', 'l'), **kws):
    '''Read Alexa data from a folder of CSVs'''
    indices = tuple(indices)
    df = pd.read_csv(path, **kws)
    x = [df[k] for k in df.keys()]
    return xa.DataArray(x[1:], {indices[0]: [k.lower() for k in df.keys()][1:], indices[1]: x[0]}, indices)

def save_csv(x, path):
    raise NotImplementedError

################################################################################

FILES = {
    '.nc': (load_netcdf, save_netcdf),
    '.npy': (np.load, lambda x, f, *args, **kws: np.save(f, x, *args, **kws))
}

def load(path, *args, **kws):
    '''Load a file using a function deduced from its suffix'''
    path = pathlib.Path(path)
    try:
        out = FILES[path.suffix][0](path, *args, **kws)
        if isinstance(out, xa.DataArray):
            out.attrs['path'] = path
            out.name = path.stem
        return out
    except Exception as e:
        raise ValueError('Failed to load file ' + str(path)) from e


def save(x, path, *args, **kws):
    '''Save a file using a function deduced from its suffix'''
    path = pathlib.Path(path)
    return FILES[path.suffix][1](x, path, *args, **kws)

################################################################################

_DIRTY_OUTPUT = False

class dirty_output:
    def __init__(self, dirty):
        global _DIRTY_OUTPUT
        self.value = _DIRTY_OUTPUT
        _DIRTY_OUTPUT = dirty

    def __enter__(self):
        return _DIRTY_OUTPUT

    def __exit__(self, type, value, traceback):
        global _DIRTY_OUTPUT
        _DIRTY_OUTPUT = self.value

################################################################################

def augment_path(path, add, suffix=None):
    if suffix is None:
        suffix = path.suffix
    return path.with_name(path.name.replace(path.suffix, '') + add + suffix)

def output(path, *args, write=None, multi=False):
    if write is None and _DIRTY_OUTPUT:
        write = True
    path = pathlib.Path(path)
    if args:
        paths = [augment_path(path, a) for a in args]
    else:
        paths = [path]
    multi = multi or len(paths) > 1

    if write is None and all(p.exists() for p in paths):
        out = tuple(load(p) for p in paths) if multi else load(paths[0])
        return lambda f: out

    def closure(function, *args, **kwargs):
        if write is None or write:
            out = function(*args, **kwargs)
            [save(x, p) for x, p in zip(out, paths)] if multi else save(out, paths[0])
            out = tuple(load(p) for p in paths) if multi else load(paths[0])
        return out
    return closure

################################################################################

def output_file(function):
    '''
    write (None: write file if not present, False: never write a file, True always write a file)
    '''
    @functools.wraps(function)
    def wrap(names, *args, write=None, **kwargs):
        single = isinstance(names, (str, pathlib.PurePath))
        if single:
            names = (names,)
        names = [pathlib.Path(n) for n in names]
        out = []
        if write is None and all(n.exists() for n in names):
            for n in names:
                out.append(load(n))
            return out[0] if single else tuple(out)
        out = function(*args, **kwargs)
        if single:
            out = [out]
        if write is None or write:
            for o, n in zip(out, names):
                save(o, n)
        return out[0] if single else tuple(out)
    return wrap
