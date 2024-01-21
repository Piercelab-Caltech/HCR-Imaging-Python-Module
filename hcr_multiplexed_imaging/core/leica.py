import numpy as np
import read_lif
from lxml import etree
from .files import FILES
from .arrays import array

################################################################################

def open_sdm(s):
    '''Get unmixing coefficients as array (component, sequence)'''
    e = etree.parse(str(s)).getroot()
    return np.array([[float(seq.text) for seq in dye] for dye in e])

FILES['.sdm'] = (open_sdm, None)

################################################################################

def open_lif(s, **kws):
    series = read_lif.Reader(s, **kws).getSeries()
    return {s.root.getAttribute('Name'): s for s in series}

FILES['.lif'] = (open_lif, None)

################################################################################

def open_sdm(s):
    e = etree.parse(str(s)).getroot()
    return np.array([[float(seq.text) for seq in dye] for dye in e[-1]]).T

FILES['.sdm'] = (open_sdm, None)

################################################################################

def get_wavelengths(d):
    return int(d.getAttribute('Channel')), [float(d.getAttribute(k)) for k in ('TargetWaveLengthBegin', 'TargetWaveLengthEnd')]

def get_settings(c):
    try:
        wavelengths = dict(get_wavelengths(m) for m in c.getElementsByTagName('Spectro')[0].getElementsByTagName('MultiBand'))
        channels = [d for d in c.getElementsByTagName('DetectorList')[0].getElementsByTagName('Detector') 
                    if int(d.getAttribute('IsActive'))]
        return {c.getAttribute('Name'): wavelengths[int(c.getAttribute('Channel'))] for c in channels}
    except IndexError:
        return {}

def find_setting(c, name):
    return next(k.getElementsByTagName('Value')[0].childNodes[0].data for k in c.childNodes 
             if k.getElementsByTagName('Key')[0].childNodes[0].data == name)

def channel_info(c, settings):
    try:
        index = int(find_setting(c, 'SequentialSettingIndex'))+1
        name = find_setting(c, 'DetectorName')
        return dict(index=index, wavelengths=settings[index][name])
    except (StopIteration, KeyError):
        return {}

def get_metadata(img):
    attachments = img.root.childNodes[0].childNodes[0].getElementsByTagName('Attachment')
    hardware = next(a for a in attachments if a.getAttribute('Name') == 'HardwareSetting')
    sequential = hardware.getElementsByTagName('LDM_Block_Sequential')[0]
    confocal = sequential.getElementsByTagName('ATLConfocalSettingDefinition')
    settings = [get_settings(c) for c in confocal]    
    return {
        'settings': settings,
        'channel_info': [channel_info(c, settings) for c in img.getChannels()]
    }

################################################################################

def open_lif_series(s, dims=None, dtype=None, name=None, metadata=get_metadata):
    x = sorted((int(d.getAttribute('DimID')), int(d.getAttribute("NumberOfElements")), float(d.getAttribute("Length"))) for d in s.getDimensions())
    if dims is None:
        dims = 'xyl' if len(x) == 2 else 'xylz'
    x.insert(dims.index('l'), (dims.index('l'), len(s.getChannels()), 0))
    # x = x + [(len(x), len(s.getChannels()), 0)]
    indices, shape, extents = zip(*x)
    if dtype is None:
        size = s.getMemorySize() // np.prod(shape)
        dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[size]
    nbytes = int(np.prod(shape) * dtype().itemsize // np.ubyte().itemsize)
    assert nbytes == s.getMemorySize(), (nbytes, s.getMemorySize())
    s.f.seek(s._Serie__offset)
    data = np.fromfile(s.f, dtype=np.ubyte, count=nbytes).view(dtype).reshape(shape, order='F')
    crds = [np.linspace(0, m * 1e6, n) for m, n in zip(extents, shape)]
    name = name or s.root.getAttribute('Name')
    data = data[tuple(slice(None) if d != 'y' else slice(None, None, -1) for d in dims)]
    return array(data, dims=dims, coords=dict(zip(dims, crds)), name=name, attrs=metadata(s))

################################################################################
