"""

Stub implementation for stretching methods

"""

from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    NDA = numpy.ndarray

def sg_ugrd(*, axes: 'NDA', coords: 'NDA', sizes: 'NDA', ndim: int, smin: 'NDA', smax: 'NDA') -> None:
    for axis, (size, start, end) in enumerate(zip(sizes, smin, smax)):
        if axis < ndim and axis in axes:
            coords[axis] = numpy.linspace(start, end, size + 1)  

def sg_tanh(*, axes: 'NDA', coords: 'NDA', sizes: 'NDA', ndim: int, params: 'NDA', smin: 'NDA', smax: 'NDA') -> None:
    for axis, (size, start, end, p) in enumerate(zip(sizes, smin, smax, params)):
        if axis < ndim and axis in axes:
            coords[axis] = (end - start) * (numpy.tanh((-1.0 + 2.0 * numpy.linspace(0.0, 1.0, size + 1)) * numpy.arctanh(p)) / p + 1.0) / 2.0 + start

