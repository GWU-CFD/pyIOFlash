"""
 Stub implementation for initial flow field methods 

"""

from typing import Tuple, List, Dict, TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    NDA = numpy.ndarray
    blocks = Tuple[Tuple['NDA', 'NDA', 'NDA'], Tuple['NDA', 'NDA', 'NDA'], Tuple['NDA', 'NDA', 'NDA']]

# define the module api
def __dir__() -> List[str]:
    return ['rb_plume3d']

def rb_plume3d(*, blocks: 'blocks', procs: Dict[str, int], 
               freq: float = 1.0, shift: float = 0.5, scale: float = 4.0) -> Dict[str, 'NDA']:
    
    (_, xxxc, _), (_, yyyc, _), (_, zzzc, _) = blocks

    x, y = [numpy.ravel(axis[:procs[index], :]) for axis, index in ((xxxc, 'i'), (yyyc, 'j'))]
    x, y = numpy.meshgrid(x, y)
    z = numpy.maximum(numpy.sin(freq * numpy.pi * (x - shift)) * 
                      numpy.sin(freq * numpy.pi * (y - shift)) / scale, 0.0)
    c = (0.5 - numpy.sum(z) / numpy.prod(z.shape))

    T = []
    for block, (x, y, z) in enumerate(zip(xxxc, yyyc, zzzc)):
        zz, yy, xx = numpy.meshgrid(z, y, x, indexing='ij')
        h = numpy.maximum(numpy.sin(freq* numpy.pi * (xx - shift)) * 
                          numpy.sin(freq * numpy.pi * (yy - shift)) / scale, 0.0) + c
        T.append(numpy.where(zz > h, 0.0, 1.0))
    
    return {'temp': numpy.array(T)} 
