import numpy as np, seaborn as sb, logging
from bokeh.io import output_notebook

from .core import *
from .core.leica import *
from . import gui
from . import dots
from . import cpp

from .dots import detect_dots, box, partitions

palettes = {
    12: ["#b0b341", "#b05cc6", "#5db956", "#d74065", "#4fbaac", "#cc532f", "#6f7dcb", "#d89244", "#c16196", "#548646", "#bf6a5f", "#8f7a38"]
}

def configure_notebook(palette=12):
    '''Set up Bokeh for outputting to a notebook and set some color palette defaults'''
    np.set_printoptions(linewidth=120)
    sb.set_style('darkgrid')
    if palette is not None:
        sb.set_palette(palettes[palette])
    output_notebook()

def bound_solve(A, B, X=None, min=0, max=float('inf'), regularize=0, iters=10000, tolerance=1e-8):
    '''General purpose least-squares solve with scalar bounds
    Arguments:
        - A: Square matrix, left hand side in linear system
        - B: Rectangular matrix, right hand side in linear system
        - X: If give, initial guess for output matrix
        - min: minimum value of each element in X
        - max: maximum value of each element in X
        - regularize: regularization factor, effective added to the diagonal of A
        - iters: maximum number of iterations to solve a single column of B
        - tolerance: tolerance for the solution
    Returns: 
        - X
        - the achieved objective value totaled across each column of B
    '''
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)
    assert A.ndim == 2, 'A should be a matrix'
    assert A.shape[0] == A.shape[1], 'Matrix A should be square'
    assert B.ndim == 2, 'B should be a matrix'
    assert B.shape[0] == A.shape[0], 'A and B dimensions do not agree'
    assert A.dtype in (np.float64, np.float32), 'dtype not supported'
    assert A.dtype == B.dtype, 'dtype of A does not agree with B'
    if X is None:
        X = np.empty_like(B)
        warm = False
    else:
        X = np.asfortranarray(X)
        assert X.shape == B.shape, 'X should be same shape as B if supplied'
        assert X.dtype == B.dtype, 'X should have same dtype as A and B'
        warm = True
    return X, cpp.bound_solve(X, A, B, min=float(min), max=float(max), warm_start=warm, 
        regularize=float(regularize), iters=int(iters), tolerance=float(tolerance))


def nnls(A, img, min=0, max=float('inf'), regularize=0, iters=10000, tolerance=1e-8):
    '''
    Unmix an input image given an unmixing matrix and non-negativity constraints.
    Arguments:
        - A: Unmixing matrix (of dimension # detector indices x # channels)
        - img: Image to unmix
        - min: minimum value of each element in output
        - max: maximum value of each element in output
        - regularize: regularization factor, effectively added to the diagonal of A
        - iters: maximum number of iterations to solve a single pixel
        - tolerance: tolerance for the solution
    Returns: 
        - X: the unmixed image

    The implementation of `nnls` relies centrally on the more abstract `bound_solve` function. To use the `bound_solve` function, we will reshape our inputs into matrices, unmix, then reshape our inputs back into the original image dimensions.
    '''
    dims = [d for d in img.dims if d != 'l']
    B = img.transpose('l', *dims)
    A = A / np.tensordot(A.T, B, 1).max(tuple(range(1,len(dims)+1))) # We normalize our spectra by their appearance in the image; this does nothing but scale the different fluorophore results independently
    AB = np.tensordot(A.T, B, 1)
    AB = np.asfortranarray(AB).reshape(AB.shape[0], -1)
    X2, err = bound_solve(A.T @ A, AB, min=min, max=max, regularize=regularize, iters=iters, tolerance=tolerance)
    X = array(X2.reshape(-1, *B.shape[1:]), B.dims)
    X.coords.update({k: img.coords[k] for k in dims})
    return X


def cholesky_nnls(A, B):
    '''Non-negative least-squares solver using Cholesky decomposition
    Arguments:
        - A: Square matrix, left hand side in linear system
        - B: Rectangular matrix, right hand side in linear system
    Returns: 
        - X: matrix of same shape as B
    '''
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)
    assert A.ndim == 2, 'A should be a matrix'
    assert A.shape[0] == A.shape[1], 'Matrix A should be square'
    assert B.ndim == 2, 'B should be a matrix'
    assert B.shape[0] == A.shape[0], 'A and B dimensions do not agree'
    assert A.dtype in (np.float64, np.float32), 'dtype not supported'
    assert A.dtype == B.dtype, 'dtype of A does not agree with B'
    X = np.empty_like(B)
    cpp.cholesky_nnls(X, A, B)
    return X