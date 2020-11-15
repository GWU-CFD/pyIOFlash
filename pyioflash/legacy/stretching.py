"""

Stub implementation for stretching methods

"""

from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    NDA = numpy.ndarray

def sg_ugrd(*, meshMe: int, axes: 'NDA', coords: 'NDA', coordsGlb: 'NDA', cornerID: 'NDA',
            guards: 'NDA', gSizes: 'NDA', hiGc: 'NDA', loGc: 'NDA', ndim: int, smin: 'NDA', smax: 'NDA') -> None:
    # calculate block cell coordinates
    # start with a uniform distribution of points [0,1]
    sdelta = (smax - smin) / gSizes
    shalfd = sdelta / 2.0
    j = cornerID - guards - 1

    # uniform grid
    l, c, r = range(3)
    for axis, (coord, mn, delta, halfd, low, high, jj) in enumerate(zip(coords, smin, sdelta, shalfd, loGc, hiGc, j)):
        if axis < ndim and axis in axes:
            for i in range(low - 1, high):
                coord[l, i, 0] = mn + jj * delta
                coord[c, i, 0] = mn + jj * delta + halfd
                jj = jj + 1
                coord[r, i, 0] = mn + jj * delta

    # calculate global cell coordinates
    if meshMe == 0:
        j = j * 0
        for axis, (coord, mn, delta, halfd, low, high, jj) in enumerate(zip(coordsGlb, smin, sdelta, shalfd, [1, 1, 1], gSizes, j)):
            if axis < ndim and axis in axes:
                for i in range(low - 1, high):
                    coord[l, i, 0] = mn + jj * delta
                    coord[c, i, 0] = mn + jj * delta + halfd
                    jj = jj + 1
                    coord[r, i, 0] = mn + jj * delta

def sg_tanh(*, meshMe: int, axes: 'NDA', coords: 'NDA', coordsGlb: 'NDA', cornerID: 'NDA', guards: 'NDA',
            gSizes: 'NDA', hiGc: 'NDA', loGc: 'NDA', ndim: int, params: 'NDA', smin: 'NDA', smax: 'NDA') -> None:
    # calculate block cell coordinates
    # start with a uniform distribution of points [0,1]
    sdelta = 1.0 / gSizes
    shalfd = sdelta / 2.0
    j = cornerID - guards - 1

    # transform uniform to stretched using y(s) = Tanh((-1+2s) atan(a)) + 1 / 2a
    l, c, r = range(3)
    for axis, (coord, mn, mx, delta, halfd, p, low, high, jj) in enumerate(zip(coords, smin, smax, sdelta, shalfd, params, loGc, hiGc, j)):
        if axis < ndim and axis in axes:
            for i in range(low - 1, high):
                coord[l, i, 0] = (mx - mn) * (numpy.tanh((-1.0 + 2.0 * jj * delta        ) * numpy.arctanh(p)) / p + 1.0) / 2.0 + mn
                coord[c, i, 0] = (mx - mn) * (numpy.tanh((-1.0 + 2.0 * jj * delta + halfd) * numpy.arctanh(p)) / p + 1.0) / 2.0 + mn
                jj = jj + 1
                coord[r, i, 0] = (mx - mn) * (numpy.tanh((-1.0 + 2.0 * jj * delta        ) * numpy.arctanh(p)) / p + 1.0) / 2.0 + mn

    # calculate global cell coordinates
    if meshMe == 0:
        j = j * 0
        for axis, (coord, mn, mx, delta, halfd, p, low, high, jj) in enumerate(zip(coordsGlb, smin, smax, sdelta, shalfd, params, [1, 1, 1], gSizes, j)):
            if axis < ndim and axis in axes:
                for i in range(low - 1, high):
                    coord[l, i, 0] = (mx - mn) * (numpy.tanh((-1.0 + 2.0 * jj * delta        ) * numpy.arctanh(p)) / p + 1.0) / 2.0 + mn
                    coord[c, i, 0] = (mx - mn) * (numpy.tanh((-1.0 + 2.0 * jj * delta + halfd) * numpy.arctanh(p)) / p + 1.0) / 2.0 + mn
                    jj = jj + 1
                    coord[r, i, 0] = (mx - mn) * (numpy.tanh((-1.0 + 2.0 * jj * delta        ) * numpy.arctanh(p)) / p + 1.0) / 2.0 + mn

