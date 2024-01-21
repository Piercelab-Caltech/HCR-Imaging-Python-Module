'''
Utilities to convert wavelengths to/from colors and hexes to/from RGB values
'''

import numpy as np, xarray as xa

################################################################################

HEXES = dict(r='#ff3300', y='#ffff00', g='#33cc33', b='#0066ff')

def hexes(*names):
    '''Look up the hexes for common colors'''
    return [HEXES.get(n.lower(), n) for n in names]

################################################################################

def hex_from_rgb(r, g, b):
    assert all(x >= 0 and x <= 1 for x in (r, g, b))
    return '#%02x%02x%02x' % tuple(int(x * 255) for x in (r, g, b))

################################################################################

def rgb_from_nm(nm, alpha=None):
    '''
    Converts wavelength between 380 and 750 nm to RGB float32 values in [0, 1]

    Based on http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    which was based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''
    nm = np.clip(nm, 380, 750)
    R, G, B = 0, 0, 0
    if nm >= 380 and nm <= 440:
        B = 0.3 + 0.7 * (nm - 380) / (440 - 380)
        R = B * (440 - nm) / (440 - 380)
    elif nm >= 440 and nm <= 490:
        G = (nm - 440) / (490 - 440)
        B = 1.0
    elif nm >= 490 and nm <= 510:
        G = 1.0
        B = (510 - nm) / (510 - 490)
    elif nm >= 510 and nm <= 580:
        R = (nm - 510) / (580 - 510)
        G = 1.0
    elif nm >= 580 and nm <= 645:
        R = 1.0
        G = (645 - nm) / (645 - 580)
    elif nm >= 645 and nm <= 750:
        R = 0.3 + 0.7 * (750 - nm) / (750 - 645)

    return np.array([R, G, B] if alpha is None else [R, G, B, alpha], dtype=np.float32)

################################################################################

def rgba_from_nm(nm, dim='k', coord='rgba', alpha=None, minmax=None):
    '''Return array of RGB values for the given wavelengths'''
    nm = np.asarray(nm)
    x = nm if minmax is None else 380 + (750 - 380) * (nm - minmax[0]) / (minmax[1] - minmax[0])
    coord = list(coord)[:3] if alpha is None else list(coord)
    return xa.DataArray([rgb_from_nm(l, alpha=alpha) for l in x], dims=('l', dim), coords={'l': nm, dim: coord})

################################################################################

# def rgba_from_nm(x, minmax=None):
#     rgb = rgbs_from_nm(x.l, minmax=minmax)
#     x = xa.dot(x, rgb)
# def rgb_space(n, dim='k', coord=['r', 'g', 'b']):
#     '''n evenly spaced rgb values'''
#     return xa.DataArray([rgb_from_nm(l) for l in np.linspace(380, 750, n)], dims=('l', dim), coords={'l': nm, dim: coord})

################################################################################

def test_colors(axes):
    for x in rgba_from_nm(np.linspace(414, 691, 100), alpha=1).T:
        k = as_scalar(x.k)
        axes.plot(x.l, x, color=k if k in 'rgb' else 'k')
