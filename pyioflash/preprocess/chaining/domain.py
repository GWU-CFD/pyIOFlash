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
    return ['calc_coords', 'get_blocks', 'write_coords']

def calc_coords(*, param: Dict[str, Dict[str, Union[int, float]]], procs: Dict[str, int] = {},
                simmn: Dict[str, float], simmx: Dict[str, float], sizes: Dict[str, int] = {}, 
                stype: Dict[str, str], ndims: int = 2) -> Tuple['NDA', 'NDA', 'NDA']:

    # create grid init parameters
    gr_axisNumProcs, gr_axisMesh = create_processor_grid(**procs)
    gr_min, gr_max = create_bounds(mins=simmn, maxs=simmx)
    gr_lIndexSize, gr_gIndexSize = create_indexSize_fromLocal(**sizes, ijkProcs=gr_axisNumProcs)
    gr_ndim = ndims

    # create grid stretching parameters
    gr_strBool, gr_strType = create_stretching(methods=stype)
    gr_str = Stretching(gr_strType, Parameters(**param))

    # Create grids
    return tuple(get_filledCoords(sizes=gr_gIndexSize, methods=gr_str, ndim=gr_ndim, smin=gr_min, smax=gr_max))

def get_blocks(*, procs: Dict[str, int] = {}, sizes: Dict[str, int] = {}, 
               coordinates: Tuple['NDA', 'NDA', 'NDA']) -> Tuple[Tuple['NDA', 'NDA', 'NDA'], 
                                                                 Tuple['NDA', 'NDA', 'NDA'],
                                                                 Tuple['NDA', 'NDA', 'NDA']]:

    # get the processor communicator layout and global arrays
    gr_axisNumProcs, gr_axisMesh = create_processor_grid(**procs)
    gr_lIndexSize, gr_gIndexSize = create_indexSize_fromLocal(**sizes, ijkProcs=gr_axisNumProcs)
    xfaces, yfaces, zfaces = coordinates

    # calculate the iaxis block coordinates
    xxxl = numpy.array([xfaces[0 + i * sizes['i']:0 + (i + 1) * sizes['i']] for i, _, _ in gr_axisMesh])
    xxxr = numpy.array([xfaces[1 + i * sizes['i']:1 + (i + 1) * sizes['i']] for i, _, _ in gr_axisMesh])
    xxxc = (xxxr + xxxl) / 2.0

    # calculate the jaxis block coordinates
    yyyl = numpy.array([yfaces[0 + j * sizes['j']:0 + (j + 1) * sizes['j']] for _, j, _ in gr_axisMesh])
    yyyr = numpy.array([yfaces[1 + j * sizes['j']:1 + (j + 1) * sizes['j']] for _, j, _ in gr_axisMesh])
    yyyc = (yyyr + yyyl) / 2.0

    # calculate the kaxis block coordinates
    zzzl = numpy.array([zfaces[0 + k * sizes['k']:0 + (k + 1) * sizes['k']] for _, _, k in gr_axisMesh])
    zzzr = numpy.array([zfaces[1 + k * sizes['k']:1 + (k + 1) * sizes['k']] for _, _, k in gr_axisMesh])
    zzzc = (zzzr + zzzl) / 2.0

    return (xxxl, xxxc, xxxr), (yyyl, yyyc, yyyr), (zzzl, zzzc, zzzr)

def get_shapes(*, procs: Dict[str, int] = {}, sizes: Dict[str, int] = {}) -> Dict[str, Tuple[int, int, int]]:

    # get the processor communicator layout and global arrays
    gr_axisNumProcs, gr_axisMesh = create_processor_grid(**procs)
    gr_lIndexSize, gr_gIndexSize = create_indexSize_fromLocal(**sizes, ijkProcs=gr_axisNumProcs)
   
    # create shape data as dictionary
    shapes = {'center': tuple([gr_axisNumProcs.prod()] + gr_lIndexSize[::-1].tolist())}
    shapes['facex'] = tuple([gr_axisNumProcs.prod()] + list(gr_lIndexSize[::-1] + (0, 0, 1))) 
    shapes['facey'] = tuple([gr_axisNumProcs.prod()] + list(gr_lIndexSize[::-1] + (0, 1, 0))) 
    shapes['facez'] = tuple([gr_axisNumProcs.prod()] + list(gr_lIndexSize[::-1] + (1, 0, 0))) 
    
    return shapes

def write_coords(*, coordinates: Tuple['NDA', 'NDA', 'NDA'], path: str = '') -> None:

    # specify path
    cwd = os.getcwd()
    path = cwd + '/' + path
    filename = path + 'initGrid.h5'
    
    print("Creating Grid Initialization File")

    # create file
    with h5py.File(filename, 'w') as h5file:
        
        # write data to file
        for axis, coords in zip(('x', 'y', 'z'), coordinates): 
            h5file.create_dataset(axis + 'Faces', data=coords)

def create_bounds(*, mins: Dict[str, float]={}, maxs: Dict[str, float]={}):
    def_mins = {key: 0.0 for key in ('i', 'j', 'k')}
    def_maxs = {key: 1.0 for key in ('i', 'j', 'k')}
    simmn = [mins.get(key, default) for key, default in def_mins.items()]
    simmx = [maxs.get(key, default) for key, default in def_maxs.items()]
    return tuple(numpy.array(item, float) for item in (simmn, simmx))

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

def get_blankCoords(sizes: 'NDA') -> List['NDA']:
    return [None] * len(sizes)

def get_filledCoords(*, sizes: 'NDA', methods: 'Stretching', ndim: int, smin: 'NDA', smax: 'NDA') -> List['NDA']:
    coords = get_blankCoords(sizes)

    for method, func in methods.stretch.items():
        if methods.any_axes(method):
            func(axes=methods.map_axes(method), coords=coords, sizes=sizes, ndim=ndim, smin=smin, smax=smax)

    return coords

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


