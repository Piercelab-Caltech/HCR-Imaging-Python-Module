import bokeh
from bokeh.models import Range1d, HoverTool, Legend
from bokeh.models.sources import ColumnDataSource as Source
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.io import export_png, export_svgs
import bokeh.plotting
from bokeh.layouts import row, column

from .colors import hexes
from .common import inclusive_bounds
from .image import rgba32, color_stack
import tempfile, os, logging, pathlib, numpy as np

log = logging.getLogger(__name__)

################################################################################

TOOLS = ['pan','box_zoom', 'wheel_zoom', 'reset', 'tap', 'hover', 'box_select', 'save']

def figure(style=None, title='Figure', tools=TOOLS, shape=(450, 300), font='normal', 
           scales=('linear', 'linear'), labels='xy', toolbar_location='above', 
           toolbar_autohide=False, **kws):
    '''
    Make Bokeh figure with some convenience defaults and custom settings
    style: 'light', 'dark', 'black': presets for the styling
    shape: xy shape of figure
    toolbar_autohide: whether to autohide the toolbar, which can be annoying
    kws: extra keywords for `bokeh.plotting.figure`
    '''
    shape = (shape, shape) if isinstance(shape, int) else shape
    p = bokeh.plotting.figure(tools=','.join(tools), title=str(title), x_axis_type=scales[0], y_axis_type=scales[1],
        width=shape[0], height=shape[1], toolbar_location=toolbar_location, **kws)
    p.xaxis.axis_label, p.yaxis.axis_label = labels
    for ax in (p.xaxis, p.yaxis):
        ax.axis_label_text_font_style = font
    if style == 'dark' or style == 'black':
        p.title.text_color = 'white'
        fill = 'black'
        line = 'white' if style == 'dark' else 'black'
        p.background_fill_color = fill
        p.border_fill_color = fill
        p.outline_line_color = fill
        for ax in (p.xaxis, p.yaxis):
            ax.axis_line_color = line
            ax.major_label_text_color = line
            ax.major_tick_line_color = line
            ax.minor_tick_line_color = line
    p.toolbar.autohide = toolbar_autohide
    return p

################################################################################

def minimal_figure(shape=(450, 300), scales=('linear', 'linear'), **kws):
    '''
    Make Bokeh figure with no axes or plot labels
    kws: extra keywords for `bokeh.plotting.figure`
    '''
    shape = (shape, shape) if isinstance(shape, int) else shape
    p = bokeh.plotting.figure(tools='', title='', x_axis_type=scales[0], y_axis_type=scales[1],
        width=shape[0], height=shape[1], toolbar_location=None, **kws)
    p.xaxis.axis_label, p.yaxis.axis_label = '', ''
    p.axis.visible = False
    p.outline_line_color = None
    p.grid.visible = False
    return p

################################################################################

def line(fig, x, y, width=2, color='black', mute=0.2, legend=None, **kws):
    return fig.line(x, y, line_width=width, color=color, legend=legend, muted_alpha=mute, **kws)

################################################################################

def legend(fig, location='top_left', policy='mute'):
    fig.legend.location = location
    fig.legend.click_policy = policy

################################################################################

def circles(fig, XY, radius=10, color='red', **kwargs):
    '''Add circles at given XY positions'''
    ops = dict(fill_color=None, alpha=1, line_width=1)
    ops.update(kwargs)
    return fig.circle(XY[:, -1], XY[:, -2], radius=radius, line_color=color, **ops)

################################################################################

def alignments(fig, images, centers, weights, offsets, scale=2):
    if images is not None:
        images = [(4 * (255 * x) / x.max()).clip(0, 255).astype(np.uint8) for x in images]
        imshow(fig, rgba32(color_stack(*images, fill=(0, 0, 0, 255))))
    offsets = np.array(offsets)
    offsets -= offsets.mean(0)
    for c, w, o, k in zip(centers, weights, offsets, hexes(*'rgb')):
        circles(fig, c, radius=scale * np.sqrt(w / w.max()), color=k)
    circles(fig, centers[1] + offsets[1] - offsets[0], radius=scale * np.sqrt(weights[1] / weights[1].max()), color=hexes('b')[0])
    fig.add_tools(HoverTool(tooltips=[("index", "$index"), ("(x,y)", "($x, $y)")]))
    return fig

################################################################################

def intersections(fig, pairs, A, B, colors):
    i, j = (np.setdiff1d(np.arange(len(x[0])), p) for x, p in zip([A, B], pairs.T))
    X = 0.5 * (A[0][pairs[:,0]] + B[0][pairs[:,1]])
    W = 0.5 * (A[1][pairs[:,0]] + B[1][pairs[:,1]])
    a = circles(fig, A[0][i], A[1][i], color=colors[0])
    b = circles(fig, B[0][j], B[1][j], color=colors[1])
    return a, b, circles(fig, X, W, color=colors[2])

################################################################################

def histogram(fig, hist, edges, hover=True):
    hover = HoverTool(tooltips=[('Count', '@H'), ('Range', '[@L, @R]')])
    fig.add_tools(hover)
    src = Source(dict(L=edges[:-1], R=edges[1:], H=hist))
    return fig.quad(top='H', bottom=0, left='L', right='R', line_color="black", source=src)

################################################################################

def imshow(fig, I, bounds=None, clip=None, flip=False, set_axes=True):
    """Show a 2D image, clipping optional, RGBA if data type is int32"""
    if bounds is None:
        bounds = inclusive_bounds(I)
    if set_axes:
        fig.x_range, fig.y_range = (Range1d(*bounds[i]) for i in (1, 0))
    if I.dtype == np.uint32:
        fun = fig.image_rgba
        data = getattr(I, 'values', I)
    else:
        fun = fig.image
        data = I.data if clip is None else I.clip(*clip).data
    if flip:
        data = np.flipud(data)
    return fun(image=[data], x=bounds[1][0], y=bounds[0][0], dw=bounds[1][1] - bounds[1][0], dh=bounds[0][1] - bounds[0][0])

################################################################################

def export_pdf(fig, filename, **kws):
    '''Write svg and then convert svg to pdf'''
    import cairosvg
    with tempfile.TemporaryDirectory() as p:
        export_svgs(fig, filename=os.path.join(p, 'tmp.svg'), **kws)
        cairosvg.svg2pdf(url=os.path.join(p, 'tmp.svg'), write_to=filename)

SAVE_FUNCTIONS = dict(svg=('svg', export_svgs), png=('png', export_png), pdf=('svg', export_pdf))

def save_figure(fig, path, format=None, height=None, width=None, webdriver=None):
    '''Save bokeh figure into a path'''
    path = pathlib.Path(path)
    if hasattr(fig, 'savefig'):
        return fig.savefig(str(path), format=format)
    else:
        old = fig.output_backend
        if format is None:
            format = path.suffix[1:]
        fig.output_backend, fun = SAVE_FUNCTIONS[format]
        path.parent.mkdir(parents=True, exist_ok=True)
        out = fun(fig, filename=str(path), height=height, width=width, webdriver=webdriver)
        fig.output_backend = old
        return out

################################################################################

def label_matshow(axes, title, x, y):
    axes.set_title(str(title)+'\n')
    axes.set_xticks(np.arange(len(x)), labels=list(map(str, x)))
    axes.set_yticks(np.arange(len(y)), labels=list(map(str, y)))

################################################################################

