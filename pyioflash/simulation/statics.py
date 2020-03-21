
"""

This module defines the static custom type necessary for the pyio package.

The StaticData derived type is implemented as data classes for simplicity and
lack of code duplication.

This module currently defines the following type and therefore data to be read from
the hdf5 output files catagorically:

    StaticData

Todo:

"""
from dataclasses import dataclass, field
from typing import Tuple, List, Callable, Union

import h5py

from pyioflash.simulation.types import _BaseData
from pyioflash.simulation.utility import _first_true, _reduce_str

@dataclass
class StaticData(_BaseData):
    """
    StaticData is a derived class implementing the functionality to
    read desired data wich does not conform to geometry, field, or scalar
    data information (e.g., input parameters) contained in the hdf5 output file.

    Attributes:
        _groups: specification of parameters used to import scalars from hdf5 file

    Notes:
        The group specification attribute is required at the time of instanciation
        in order to read the desired data from the hdf5 output file.

        StaticData instances may be collected into a SortedDict, for example, in
        order to provide for simulation data which changes throughout the simulation
        but does not conform to geometry, field, or scalar information.

    """
    # parameter specification, format of -- [group ...]
    _groups: List[Tuple[str, Callable]] = field(repr=True, init=True, compare=False)

    def __str__(self) -> str:
        return super()._str_keys()

    @staticmethod
    def decode_label(label: Union[bytes, str]) -> str:
        """
        Class method used as a helper method to provide the functionality
        to appropriatly decode either the hdf5 dataset name/label or dataset values
        stored as byte array data into utf-8 strings.

        Args:
            label: data to be decoded, either a byte array or string

        Returns:
            decoded data represented as a utf-8 string
        """
        try:
            return label.decode('utf-8')
        except AttributeError:
            return str(label)

    # pylint: disable=arguments-differ
    def _init_process(self, file: h5py.File, code: str, form: str) -> None:
        # pull relavent data from hdf5 file object
        real_scalars: List[Tuple[bytes, float]] = list(file['real scalars'])

        # initialize mappable keys
        self.key = float(_first_true(real_scalars, lambda l: 'time' in str(l[0]))[1])

         # initialize class members by hdf5 file groups
        for group, wrap in self._groups: # pylint: disable=not-an-iterable
            setattr(self, _reduce_str(group), {_reduce_str(self.decode_label(dataset[0])):
                                               wrap(dataset[1]) for dataset in file[group]})

        # initialize list of class member names holding the data
        # pylint: disable=not-an-iterable
        setattr(self, '_attributes', {_reduce_str(group[0]) for group in self._groups})

    @staticmethod
    def pass_label(label: str) -> str:
        """
        Class method used as a helper method to provide a pass through when a callable
        is expected but the relavent data does not need to be decoded;
        see StaticData.decode_label().

        Args:
            label: data to be passed through

        Returns:
            Unmodified argument data
        """
        return label

