from bokeh.models import Range1d, ColumnDataSource, Slider, CustomJS, Select
from bokeh.models.tools import BoxSelectTool, TapTool
from bokeh.layouts import row, column
from bokeh.io import output_file, output_notebook, push_notebook
from bokeh.io import save as save_bokeh, show as show_bokeh
from bokeh.resources import Resources

import pkg_resources, functools, pathlib, numpy as np, types, fractions
from .common import inclusive_bounds
from .files import FILES

###############################################################################

def image_tap(figure):
    t = TapTool()
    figure.add_tools()

###############################################################################

def javascript(name, path=None, cache={}):
    '''Fetch javascript text from a JS file'''
    if path is None:
        path = pkg_resources.resource_filename('hcr_multiplexed_imaging', 'templates/callbacks.js')
    js = cache.get(path)
    if js is None:
        js = pathlib.Path(path).read_text().split('//callback=')
        js = cache[path] = dict(i.split(maxsplit=1) for i in js[1:])
    return js[name]

################################################################################

@functools.wraps(show_bokeh)
def disp(fig, *args, notebook_handle=True, **kws):
    '''Display figure in a notebook'''
    if isinstance(fig, types.FunctionType):
        fig = fig()
    show_bokeh(fig, *args, notebook_handle=notebook_handle, **kws)

################################################################################

class CustomResources(Resources):
    def __init__(self, custom_css, *args, **kws):
        super().__init__(*args, **kws)
        self.custom_css = custom_css

    @property
    def css_raw(self):
        return super().css_raw + self.custom_css

def save_html(element, path, title='Image', background=None, resources='cdn'):
    '''
    Must have no Python callbacks
    '''
    if background is None:
        css = []
    else:
        c = {'white': '#ffffff', 'black': '#000000'}.get(background, background)
        css = ['.bk-root {\nbackground-color: %s;\nborder-color: %s;\n}' % (c, c)]
    save_bokeh(element, str(path), resources=CustomResources(css, resources), title=str(title))

FILES.setdefault('.html', [None, None])[1] = save_html

################################################################################

def interact(function):
    '''Show a plot allowing bokeh Python callbacks'''
    def callback():
        with open('blah.txt', 'a') as f:
            print('blah', file=f)
    def modify(doc):
        doc.add_periodic_callback(callback, period_milliseconds=5000)
        doc.add_root(function())
    return show_bokeh(modify, notebook_handle=True)

################################################################################

def maximum_slider(I, n=100):
    m = float(max(map(np.max, I)))
    return Slider(title='Maximum intensity', value=m, start=m/n, end=m, step=m/n)

def image_choice(n, names=None):
    names = list(map(str, range(n) if names is None else names))
    assert len(names) == n
    return Select(title='Image', options=names, value=names[0])

################################################################################

def rgba32(x, alpha=255):
    '''Pack uint8 type array into a uint32 array of RGBA values'''
    x = x.astype(np.uint8)
    if x.shape[-1] == 3:
        assert alpha is not None, 'alpha channel is missing'
        x = np.append(x, np.full(x.shape[:-1] + (1,), alpha, dtype=np.uint8), axis=-1)
    return np.ascontiguousarray(x).view(dtype=np.uint32).squeeze(-1)

################################################################################

def box_select(fig, callback):
    box = ColumnDataSource({'box': 4 * [0]})
    select = fig.select(dict(type=BoxSelectTool))[0]
    select.callback = CustomJS(args=dict(source=box), code='''
        var g = cb_data.geometry;
        source.data = {'box': [g.x0, g.x1, g.y0, g.y1]};
        source.change.emit();
        ''')

    box.on_change('data', lambda attr, old, new: callback(np.array(new['box']).reshape(2, 2)))
    return box

################################################################################

def set_bounds(p, I, bounds=None, set_axes=True):
    if bounds is None:
        bounds = inclusive_bounds(I)
    dw = bounds[1][1] - bounds[1][0] 
    dh = bounds[0][1] - bounds[0][0]
    if set_axes:
        p.x_range, p.y_range = (Range1d(*bounds[i]) for i in (1, 0))
        frac = fractions.Fraction.from_float(dh / dw).limit_denominator(2000)
        # Upscale so the image width is about the same as the specified width
        scale = max(round(p.width / frac.denominator), 1)
        p.frame_width = scale * frac.denominator
        p.frame_height = scale * frac.numerator
    return dict(x=bounds[1][0], y=bounds[0][0], dw=dw, dh=dh)

################################################################################

def zslider(I):
    b, e = sorted(map(float, I.z[[0, -1]]))
    return Slider(title='Current Z position', value=b, start=b, end=e, step=(e - b) / (len(I.z) - 1))

################################################################################

def show_line_slider(p, x, lines):
    lines = np.asarray(lines)
    all_lines = ColumnDataSource({'lines': lines})
    line = ColumnDataSource({'x': np.asarray(x), 'y': lines[0]})
    p.line('x', 'y', color='black', source=line)
    slider = Slider(value=0, start=0, end=lines.shape[0]-1, step=1)
    slider.js_on_change('value', CustomJS(args=dict(line=line, all_lines=all_lines, slider=slider), code=javascript('line_slider')))
    return slider

################################################################################

def show_zxyk(p, I, dtype=np.uint16, bounds=None, set_axes=True, callback=None, names=None):
    '''
    Show an image which has dimensions z, x, y, k; k means rgba (should be 3 or 4 long in that dim)
    I: a list of images of with dimensions [z, x, y, k]
    '''
    choice = image_choice(len(I), names)
    slider = maximum_slider(I)
    I = [i.transpose(*'zyxk') for i in I]
    zslice = zslider(I[0])

    bounds = set_bounds(p, I[0][0], bounds, set_axes)
    imax = np.iinfo(dtype).max
    scale = float(255 / imax * slider.value)
    zxy = [(i.values * (imax / slider.value)).astype(dtype) for i in I]
    xy = rgba32(zxy[0][0] * (255 / imax)) # packed as uint32

    print('Image MB: %.3f' % sum(u.nbytes/1e6 for u in zxy))

    source = ColumnDataSource({'xy': [xy], 'zxy': [zxy]})

    for x in (slider, zslice, choice):
        x.js_on_change('value', CustomJS(args=dict(source=source, scale=scale, zslice=zslice, slider=slider, choice=choice), code=javascript('show_zxyk')))

    renderer = p.image_rgba(image='xy', **bounds, source=source)

    if callback is not None:
        box_select(p, callback)

    return [choice, slider, zslice]

################################################################################

def show_zxy(p, I, dtype=np.uint16, bounds=None, set_axes=True, callback=None, names=None):
    '''
    Show an image which has dimensions z, x, y
    I: a list of images, each with dimensions [z, x, y]
    '''
    choice = image_choice(len(I), names)
    slider = maximum_slider(I)
    I = [i.transpose(*'zyx') for i in I]
    zslice = zslider(I[0])

    bounds = set_bounds(p, I[0][0], bounds, set_axes)
    dmax = np.iinfo(dtype).max
    itype, imax = np.float32, 1
    scale = float(imax / dmax * slider.value)
    zxy = [(i.values * (dmax / slider.value)).astype(dtype) for i in I] # packed as dtype
    xy = (zxy[0][0] * (imax / dmax)).astype(itype) # packed as float32
    source = ColumnDataSource({'zxy': [zxy], 'xy': [xy]})

    for x in (slider, zslice, choice):
        x.js_on_change('value', CustomJS(args=dict(source=source, scale=scale, zslice=zslice, slider=slider, choice=choice), code=javascript('show_zxy')))

    renderer = p.image(image='xy', **bounds, source=source, palette='Greys256')

    if callback is not None:
        box_select(p, callback)

    return [choice, slider, zslice]

################################################################################

def show_channels(p, I, dtype=np.uint16, bounds=None, set_axes=True, callback=None, names=None):
    return (show_xy if I[0].ndim == 2 else show_zxy)(p, I, dtype, bounds, set_axes, callback, names)

################################################################################

def show_xy(p, I, dtype=np.uint16, bounds=None, set_axes=True, callback=None, names=None):
    '''
    Show an image which has dimensions x, y
    I: a list of images, each with dimensions [x, y]
    '''
    choice = image_choice(len(I), names)
    slider = maximum_slider(I)
    I = [i.transpose(*'yx') for i in I]
    bounds = set_bounds(p, I[0], bounds, set_axes)
    itype, imax = np.float32, 1
    dmax = np.iinfo(dtype).max
    scale = float(imax / dmax * slider.value)
    xy0 = [(i.values * (dmax / slider.value)).astype(dtype) for i in I] # packed as dtype
    xy = (xy0[0] * (imax / dmax)).astype(itype) # packed as float32
    source = ColumnDataSource({'xy0': [xy0], 'xy': [xy]})

    for x in (slider, choice):
        x.js_on_change('value', CustomJS(args=dict(source=source, slider=slider, choice=choice, scale=scale), code=javascript('show_xy')))

    renderer = p.image(image='xy', **bounds, source=source, palette='Greys256')

    if callback is not None:
        box_select(p, callback)

    return [choice, slider]

################################################################################

def show_xyk(p, I, dtype=np.uint16, bounds=None, set_axes=True, callback=None, names=None):
    '''
    Show an image which has dimensions x, y, k; k means rgba (should be 3 or 4 long in that dim)
    I: a list of images, each with dimensions [x, y, k]
    '''
    choice = image_choice(len(I), names)
    slider = maximum_slider(I)
    I = [i.transpose(*'yxk') for i in I]
    bounds = set_bounds(p, I[0], bounds, set_axes)

    imax = np.iinfo(dtype).max
    scale = float(255 / imax * slider.value)
    xyk = [(i.values * (imax / slider.value)).astype(dtype) for i in I] # packed as dtype
    xy = rgba32(xyk[0] * (255 / imax)) # packed as uint8 in uint32

    print('Image MB: %.3f' % sum(u.nbytes/1e6 for u in xyk))

    source = ColumnDataSource({'xyk': [xyk], 'xy': [xy]})

    for x in (slider, choice):
        x.js_on_change('value', CustomJS(args=dict(source=source, slider=slider, choice=choice, scale=scale), code=javascript('show_xyk')))

    renderer = p.image_rgba(image='xy', **bounds, source=source)

    if callback is not None:
        box_select(p, callback)

    return [choice, slider]

################################################################################

