import threading, pathlib, numpy as np
from lxml import etree
from xarray import DataArray
from czifile import CziFile
import dask
import dask.array as da

from .common import assert_unique
from .files import FILES

################################################################################

# these seem to be the actually used dimensions in our Zeiss data
_SQUEEZE = [np.array(x, dtype=bool) for x in [
    [0, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
]]

def squeezed(A, squeeze):
    try:
        return A.squeeze(squeeze)
    except ValueError:
        raise ValueError('Cannot squeeze dimensions {} of array shape {}'.format(squeeze, A.shape))

def czi_squeeze(shape):
    shape = np.asarray(shape)
    for s in _SQUEEZE:
        if len(s) == len(shape) and np.prod(shape[s]) == np.prod(shape):
            out = tuple(np.where(np.logical_not(s))[0])
            return out
    raise ValueError('No squeeze found', shape)

################################################################################

# def load_czi_data(fn, z=None, squeeze=None):
#     '''Load CZI file, requires czifile package'''
#     with DaskCziFile(str(fn)) as czi:
#         squeeze = czi_squeeze(czi.shape) if squeeze is None else squeeze
#         data = squeezed(czi.asarray(), squeeze).copy()
#         # if z is None:
#         # else:
#         #     data = czi_slice(czi, z, squeeze)
#         return data, czi.metadata

def czi_array(data, scalings: dict, wavelengths, dims=None, **kws):
    '''
    Make xarray of lambda, z, x, y pixel data
    scalings: dimensions of pixels
    kws: extra attributes
    wavelengths: wavelength array (#wavelengths, 2) for start and end of bin
    '''
    assert wavelengths.shape[1] == 2
    if dims is None:
        dims = 'l' + 'zxy'[-(data.ndim-1):]
    dims = tuple(dims)
    crds = {k: np.arange(data.shape[dims.index(k)]).astype(np.float64) * v for k, v in scalings.items()}
    if 'l' in dims and data.shape[dims.index('l')] == wavelengths.shape[0]:
        crds.setdefault('l', wavelengths.mean(1))
        crds['l0'] = ('l', wavelengths[:, 0])
        crds['l1'] = ('l', wavelengths[:, 1])
    return DataArray(data, crds, dims, attrs=kws)

################################################################################

class DaskCziFile(CziFile):
    def __init__(self, fn, tags={}, dims=None, squeeze=None):
        super().__init__(str(fn))
        xml = self.metadata()
        if isinstance(xml, str):
            xml = etree.fromstring(xml)
        self.tags = parse_metadata(xml)
        self.tags['file_name'] = str(fn)
        self.tags['path'] = pathlib.Path(fn)
        self.tags['xml'] = etree.tostring(xml).decode()
        self.tags.update(tags)
        self.squeeze = squeeze or czi_squeeze(self.shape)
        self.dims = dims
        self.thread_lock = threading.Lock()

    def _mask_shape(self, shape):
        for s in self.squeeze:
            assert shape[s] == (0, 1) or shape[s] == 1
        return [x for i, x in enumerate(shape) if i not in self.squeeze]

    def _squeeze_index(self, entry):
        o = [(i-j, i-j+k) for i, j, k in zip(entry.start, self.start, entry.shape)]
        return self._mask_shape(o)

    def _read_subblock(self, subblock, resize=True, order=0):
        with self.thread_lock:
            tile = subblock.data(resize=resize, order=order)
            return squeezed(tile, self.squeeze)#.copy()

    def array(self):
        idx = tuple(map(self._squeeze_index, self.filtered_subblock_directory))

        pos = tuple(map(sorted, map(set, tuple(zip(*idx)))))
        A = np.full(tuple(map(len, pos)), None, dtype=object)
        for i, t in zip(idx, self.filtered_subblock_directory):
            r = dask.delayed(self._read_subblock)(t.data_segment())
            block = da.from_delayed(r, dtype=t.dtype, shape=self._mask_shape(t.shape))
            A[tuple(p.index(j) for p, j in zip(pos, i))] = block
        data = da.block(A.tolist())
        assert np.prod(self.shape) == np.prod(data.shape)
        return czi_array(data, **self.tags, dims=self.dims)

################################################################################

def open_czi(fn, *, squeeze=None, dims=None, **kws):
    '''
    fn: file path
    z: int or None, gets only that z slice
    squeeze: throw out these length-1 dimensions
    kws: extra keywords passed to czi_array
    '''
    fn = pathlib.Path(fn).expanduser()
    if not fn.exists():
        raise FileNotFoundError(str(fn))
    try:
        with CziFile(str(fn)):
            pass
    except ValueError:
        raise ValueError('{} is not a CZI file'.format(fn))
    return DaskCziFile(fn, squeeze=squeeze, dims=dims, tags=kws).array()

def save_czi(x, fn):
    raise NotImplementedError('CZIs cannot be saved yet')

FILES['.czi'] = (open_czi, save_czi)

################################################################################

def _mask_shape(shape, squeeze):
    '''Return reduced shape of an array after it has been squeezed according to squeeze'''
    return [x for i, x in enumerate(shape) if i not in squeeze]

# def _squeeze_index(entry, start, squeeze):
#     o = [(i-j, i-j+k) for i, j, k in zip(entry.start, start, entry.shape)]
#     return _mask_shape(o, squeeze)

# def _read_subblock(sub, squeeze, resize=True, order=0):
#     '''Read a subblock and squeeze it according to squeeze'''
#     return squeezed(sub.data_segment().data(resize=resize, order=order), squeeze)

def czi_shape(fn, squeeze=None):
    with CziFile(str(fn)) as czi:
        if squeeze is None:
            squeeze = czi_squeeze(czi.shape)
        return tuple(_mask_shape(czi.shape, squeeze))

################################################################################

def dump_xml(x):
    '''Print XML node'''
    print(etree.tostring(x).decode())


def find_xml(x, *values):
    '''Recursive finding child in XML node'''
    for v in values:
        y = x.find(v)
        if y is None:
            raise KeyError(v, etree.tostring(x).decode())
        x = y
    return x

def find_float(x, *values):
    try:
        return float(find_xml(x, *values).text)
    except KeyError:
        return 0.0 # float('nan')

def beam_splitters(meta, exclude=('None', 'Mirror', 'Plate')):
    bs = [find_xml(b, 'Filter').text for b in find_xml(meta, 'Metadata', 'Experiment', 'ExperimentBlocks', 'AcquisitionBlock', 'MultiTrackSetup', 'TrackSetup', 'BeamSplitters')]
    return sorted(b for b in bs if b not in exclude)

def parse_metadata(meta):
    '''Get image dimensions, detection wavelengths, excitation wavelengths, beam splitters from XML metadata'''
    if isinstance(meta, str):
        meta = etree.fromstring(meta)

    ch = find_xml(meta[0], 'Information', 'Image', 'Dimensions', 'Channels')
    reals = lambda a: list(map(float, a))
    o = {}
    o['pinhole'] = find_float(ch[0], 'PinholeSizeAiry')
    o['gain'] = assert_unique(reals(find_float(c, 'DetectorSettings', 'Gain') for c in ch))
    o['pixel_time'] = assert_unique(reals(find_float(c, 'LaserScanInfo', 'PixelTime') for c in ch))
    o['averaging'] = int(assert_unique(reals(find_float(c, 'LaserScanInfo', 'Averaging') for c in ch)))
    o['mode'] = {
        'SpectralImaging': 'spectral',
        'LaserScanningConfocalMicroscopy': 'bandpass'
    }[find_xml(ch[0], 'AcquisitionMode').text]

    try:
        o['wavelengths'] = np.array([find_xml(c, 'DetectionWavelength', 'Ranges').text.split('-') for c in ch], dtype=np.float64)
    except KeyError:
        o['wavelengths'] = np.zeros((len(ch), 2), dtype=np.float64)

    lasers = [assert_unique([round(find_float(c, 'LightSourcesSettings', 'LightSourceSettings', 'Wavelength'), 3) for c in ch])]
    powers = [assert_unique(reals(1-find_float(c, 'LightSourcesSettings', 'LightSourceSettings', 'Attenuation') for c in ch))]
    assert len(lasers) == len(powers)
    o['lasers'] = {str(int(l)): round(p, 8) for l, p in zip(lasers, powers) if l != 0}

    try:
        o['beam_splitters'] = beam_splitters(meta)
    except KeyError:
        pass

    try:
        items = find_xml(meta[0], 'Scaling', 'Items')
        o['scalings'] = {i.attrib['Id'].lower() : 1e6 * find_float(i, 'Value') for i in items}
    except KeyError:
        setup = find_xml(meta[0], 'Experiment', 'ExperimentBlocks', 'AcquisitionBlock', 'AcquisitionModeSetup')
        o['scalings'] = {t.lower(): 1e6 * find_float(setup, 'Scaling' + t) for t in 'XYZ'} # convert to microns
    o['channel_names'] = [c.attrib['Name'] for c in find_xml(meta, 'Metadata', 'Information', 'Image', 'Dimensions', 'Channels')]
    return o
