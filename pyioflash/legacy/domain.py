"""
 Stub implementation for domain creation module

"""

from dataclasses import dataclass, field, InitVar
from typing import Any, Tuple, List, Dict, Union, Callable, TYPE_CHECKING
from functools import partial

import os

import h5py
import numpy

from pyioflash.preprocess.chaining.stretching import sg_ugrd, sg_tanh

MDIM = 3

if TYPE_CHECKING:
    NDA = numpy.ndarray

# define the module api
def __dir__() -> List[str]:
    return ['calc_coords', 'write_coords']

def calc_coords(*, guard: Dict[str, int], param: Dict[str, Dict[str, Union[int, float]]],
                procs: Dict[str, int], simmn: Dict[str, float], simmx: Dict[str, float], 
                sizes: Dict[str, int], stype: Dict[str, str], ndims: int = 2) -> Tuple['NDA', 'NDA', 'NDA', 
                                                                                       'NDA', 'NDA', 'NDA']:

    # create grid init parameters
    gr_axisNumProcs, gr_axisMesh = create_processor_grid(**procs)
    gr_guard = create_guards(**guard)
    gr_min, gr_max = create_bounds(mins=simmn, maxs=simmx)

    # create grid stretching parameters
    gr_strBool, gr_strType = create_stretching(methods=stype)
    gr_str = Stretching(gr_strType, Parameters(**param))

    # create grid index parameters as in GRID/UG/RegularGrid
    gr_ndim = ndims
    gr_lIndexSize, gr_gIndexSize = create_indexSize_fromLocal(**sizes, ijkProcs=gr_axisNumProcs)
    gr_loGc, gr_lo, gr_hi, gr_hiGc = get_blockPoints(lSizes=gr_lIndexSize, ndim=gr_ndim,
                                                     gSizes=gr_gIndexSize, guards=gr_guard)

    # Create grids
    gr_coords, gr_coordsGlb = get_blockCoords(axisMesh=gr_axisMesh, guards=gr_guard, gSizes=gr_gIndexSize, 
                                              hiGc=gr_hiGc, loGc=gr_loGc, lSizes=gr_lIndexSize, 
                                              methods=gr_str, ndim=gr_ndim, smin=gr_min, smax=gr_max)


    gr_iCoords, gr_jCoords, gr_kCoords = get_shapedCoords(coords=gr_coords)
    gr_iCoordsGlb, gr_jCoordsGlb, gr_kCoordsGlb = get_shapedCoords(coords=gr_coordsGlb, local=False)
    
    return gr_iCoords, gr_jCoords, gr_kCoords, gr_iCoordsGlb, gr_jCoordsGlb, gr_kCoordsGlb

def write_coords(*, coordinates: Tuple['NDA', 'NDA', 'NDA', 'NDA', 'NDA', 'NDA'],
                 path: str = '') -> None:

    # specify path
    cwd = os.getcwd()
    path = cwd + '/' + path
    filename = path + 'initGrid.h5'
    
    print("Creating Grid Initialization File")

    # create file
    with h5py.File(filename, 'w') as h5file:
        
        # write data to file
        for n, (axis, coords) in enumerate(zip(['x', 'y', 'z']*2, coordinates)): 
            for face, data in zip(('l', 'c', 'r'), coords):
                name = 'blk' if n < 3 else 'glb'
                h5file.create_dataset(axis + name + face, data=data)

def create_bounds(*, mins: Dict[str, float]={}, maxs: Dict[str, float]={}):
    def_mins = {key: 0.0 for key in ('i', 'j', 'k')}
    def_maxs = {key: 1.0 for key in ('i', 'j', 'k')}
    simmn = [mins.get(key, default) for key, default in def_mins.items()]
    simmx = [maxs.get(key, default) for key, default in def_maxs.items()]
    return tuple(numpy.array(item, float) for item in (simmn, simmx))

def create_guards(*, i: int = 1, j: int  = 1, k: int  = 1):
    return numpy.array([i, j, k], int)

def create_indexSize_fromGlobal(*, i: int  = 1, j: int  = 1, k: int  = 1, ijkProcs: 'NDA'):
    gSizes = [i, j, k]
    blocks = [size / procs for procs, size in zip(ijkProcs, gSizes)]
    return tuple(numpy.array(item, int) for item in (blocks, gSizes))

def create_indexSize_fromLocal(*, i = 1, j = 1, k = 1, ijkProcs: 'NDA'):
    blocks = [i, j, k]
    gSizes = [procs * nb for procs, nb in zip(ijkProcs, blocks)]
    return tuple(numpy.array(item, int) for item in (blocks, gSizes))

def create_processor_grid(*, i: int  = 1, j: int  = 1, k: int  = 1):
    iProcs, jProcs, kProcs = i, j, k
    proc = [iProcs, jProcs, kProcs]
    grid = [[i, j, k] for k in range(kProcs) for j in range(jProcs) for i in range(iProcs)]
    return tuple(numpy.array(item, int) for item in (proc, grid))

def create_stretching(*, methods: Dict[str, str] = {}):
    default = 'SG_UGRD'
    def_methods = {key: default for key in ('i', 'j', 'k')}
    strTypes = [methods.get(key, default) for key, default in def_methods.items()]
    strBools = [method == default for method in strTypes]
    return tuple(numpy.array(item) for item in (strBools, strTypes))

def get_blankCoords(*, gSizes: 'NDA', hiGc: 'NDA', loGc: 'NDA') -> Tuple['NDA', 'NDA']:
    coords = [numpy.zeros((3, high - low + 1, 1)) for high, low in zip(hiGc, loGc)]
    coordsGlb = [numpy.zeros((3, high, 1)) for high in gSizes]
    return coords, coordsGlb

def get_blockCoords(*, axisMesh: 'NDA', guards: 'NDA', gSizes: 'NDA', hiGc: 'NDA',
                    loGc: 'NDA', lSizes: 'NDA', methods: 'Stretching', ndim: int,
                    smin: 'NDA', smax: 'NDA') -> Tuple[List[List['NDA']], List[List['NDA']]]:
    # Create grids
    gr_coords = []
    for meshMe, axisMe in enumerate(axisMesh):

        #store lower left global index for each dim
        cornerID = get_blockCornerID(axisMe=axisMe, lSizes=lSizes)

        # Now create the grid and coordinates etc
        coords, coordsGlb = get_blankCoords(gSizes=gSizes, hiGc=hiGc, loGc=loGc, )

        for method, func in methods.stretch.items():
            if methods.any_axes(method):
                func(meshMe=meshMe, axes=methods.map_axes(method), coords=coords, coordsGlb=coordsGlb,
                     cornerID=cornerID, guards=guards, gSizes=gSizes, hiGc=hiGc, loGc=loGc,
                     ndim=ndim, smin=smin, smax=smax)

        gr_coords.append(coords)
        if meshMe == 0:
            gr_coordsGlb = coordsGlb

    return gr_coords, gr_coordsGlb

def get_blockCornerID(*, axisMe: 'NDA', lSizes: 'NDA') -> int:
    return axisMe * lSizes + 1

def get_blockPoints(*, ndim: int, guards: 'NDA', gSizes: 'NDA', lSizes: 'NDA'
                   ) -> Tuple['NDA', 'NDA', 'NDA', 'NDA']:
    loGc = numpy.ones(MDIM, int) * 1
    lo = loGc + guards
    hi = lo + lSizes - 1
    hiGc = hi + guards
    for axis in range(ndim, MDIM):
        loGc[axis] = 1
        lo[axis] = 1
        hi[axis] = 1
        hiGc[axis] = 1
        guards[axis] = 0
        gSizes[axis] = 1
    return loGc, lo, hi, hiGc

def get_shapedCoords(*, coords: List[List['NDA']], local: bool=True) -> Tuple['NDA', 'NDA', 'NDA']:
    if local:
        return [numpy.array([[block[n][f, :, 0] for block in coords] for f in range(3)]) for n in range(3)]
    else:
        return [numpy.array([coords[n][f, :, 0] for f in range(3)]) for n in range(3)]

@dataclass
class Parameters:
    alpha: Union[Dict, numpy.ndarray] = field(default_factory=dict)
    
    def __post_init__(self):
        def_alpha = {'i': 0.001, 'j': 0.001, 'k': 0.001}
        self.alpha = numpy.array([self.alpha.get(key, default) for key, default in def_alpha.items()])
        
@dataclass
class Stretching:
    methods: InitVar[numpy.ndarray] # length mdim specifying strType
    parameters: InitVar[Parameters]
            
    map_axes: Callable[[str], List[int]] = field(repr=False, init=False)
    any_axes: Callable[[str], bool] = field(repr=False, init=False)
    stretch: Dict[str, Callable[[Any], None]] = field(repr=False, init=False)
    default: str = 'SG_UGRD'
    
    def __post_init__(self, methods, parameters):
        self.map_axes = lambda stretch: [axis for axis, method in enumerate(methods) if method == stretch]
        self.any_axes = lambda stretch: len(self.map_axes(stretch)) > 0
        self.stretch = {'SG_UGRD': sg_ugrd,
                        'SG_TANH': partial(sg_tanh, params=parameters.alpha)}


