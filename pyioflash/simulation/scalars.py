"""

This module defines the scalar custom type necessary for the pyio package.

The ScalarData derived type is implemented as data classes for simplicity and
lack of code duplication.

This module currently defines the following type and therefore data to be read from
the hdf5 output files catagorically:

    ScalarData

Todo:

"""
from dataclasses import dataclass, field
from typing import Tuple, List

import h5py

from pyioflash.simulation.types import _BaseData
from pyioflash.simulation.utility import _first_true

@dataclass
class ScalarData(_BaseData):
    """
    ScalarData is a derived class implementing the functionality to
    read the relavent scalar data (e.g., time, dt, iteration count)
    contained in the hdf5 output file.

    Attributes:
        _groups: specification of parameters used to import scalars from hdf5 file

    Note:
        The group specification attribute is required at the time of instanciation
        in order to read the desired data from the hdf5 output file.

    """
    # parameter specification, format of -- [(group, dataset, type), ...]
    _groups: List[Tuple[str, str, str]] = field(repr=True, init=True, compare=False)

    def __str__(self) -> str:
        return super()._str_keys()

    # pylint: disable=arguments-differ
    def _init_process(self, file: h5py.File, code: str, form: str) -> None:
        # pull relavent data from hdf5 file object
        real_scalars: List[Tuple[bytes, float]] = list(file['real scalars'])

        # initialize mappable keys
        self.key = float(_first_true(real_scalars, lambda l: 'time' in str(l[0]))[1])

         # initialize field data members
        for group, dataset, *name in self._groups: # pylint: disable=not-an-iterable
            if name == []:
                name = [dataset]
            setattr(self, *name, _first_true(list(file[group]), lambda l, data=dataset: data in str(l[0]))[1])

        # initialize list of class member names holding the data
        setattr(self, '_attributes', {group[-1] for group in self._groups}) # pylint: disable=not-an-iterable

