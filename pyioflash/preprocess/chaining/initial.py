"""
 Stub implementation for domain creation module

"""

from typing import Any, Tuple, List, Dict, Union, Callable, TYPE_CHECKING

import numpy
import h5py
import os

from pyioflash.preprocess.chaining.flows import rb_plume3d

if TYPE_CHECKING:
    NDA = numpy.ndarray
    blocks = Tuple[Tuple['NDA', 'NDA', 'NDA'], Tuple['NDA', 'NDA', 'NDA'], Tuple['NDA', 'NDA', 'NDA']]

# define the module api
def __dir__() -> List[str]:
    return ['calc_flowField', 'write_flowField']

def calc_flowField(*, blocks: 'blocks', procs: Dict[str, int], 
                   method: Union[str, Callable], options: Dict[str, Any] = {}) -> Dict[str, 'NDA']:

    # define available default methods
    methods = {'rb_plume3d': rb_plume3d}

    if isinstance(method, str):
        fields = methods[method](blocks=blocks, procs=procs, **options)

    elif isinstance(method, Callable):
        fields = method(blocks=blocks, procs=procs, **options)

    else:
        pass

    return fields

def write_flowField(*, fields: Dict[str, 'NDA'], path: str = '', filename: str = 'initBlock.h5') -> None:
    
    # define supported input fields
    defaults = {'velx', 'vely', 'velz', 'temp'}

    # auto fill missing supported fields
    keys = fields.keys()
    first = next(iter(fields))
    for default in defaults:
        if default not in keys:
            fields[default] = numpy.zeros_like(fields[first])


    # specify path and filename
    filename = os.getcwd() + '/' + path + filename
    if os.path.exists(filename):
        os.remove(filename)

    print("Creating Block Initialization File")

    # create hdf5 file
    with h5py.File(filename, 'w-') as h5file:

        # write data to file
        for field, data in fields.items():
            h5file.create_dataset(field, data=data)

