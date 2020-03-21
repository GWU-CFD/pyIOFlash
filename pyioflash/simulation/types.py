"""

This module defines the custom types necessary for the pyio package.

The custom type, BaseData, defines the base and common behavior for the derived
types which provide the functionality to import data from hdf5 output files.
The common features /members to all data are a key (time stamp), an HDF5 file object,
and a code and form string which define the code and file formating expected.
The BaseData and derived types are implemented as data classes for simplicity and
lack of code duplication.

This module currently defines the following types and therefore data to be read from
the hdf5 output files catagorically:

    GeometryData
    FieldData
    ScalarData
    StaticData
    DynamicData

    Note each of the preceeding types is defined in its own file

Todo:

"""

from abc import ABC as AbstractBase, abstractmethod
from dataclasses import dataclass, field, InitVar
from typing import Any, List, Dict, Callable

import h5py

from pyioflash.simulation.utility import open_hdf5

@dataclass(order=True)
class _BaseData(AbstractBase):
    """
    _BaseData is an abstract class implementing the common behavior
    for all derived dataclasses which provide the functionality to
    read the hdf5 output file data.

    Attributes:
        key: mappable for compositting into a sortable collection object; sorted by
        _attributes: list of named attributes loaded from file
        file: (InitVar) h5py file object
        form: (InitVar) the expected file format or data layout
        code: (InitVar) the expected code associated with the output file

    """

    key: float = field(repr=True, init=False, compare=True)
    _attributes: str = field(repr=False, init=False, compare=False)
    file: InitVar[h5py.File]
    form: InitVar[str]
    code: InitVar[str]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __post_init__(self, file, code, form, *args) -> None:
        supported: Dict[str, List[str]] = {'flash' : ['plt', 'chk']}

        # check for supported codes and formats
        if code not in supported:
            raise Exception(f'Code {code} is not supported; only code = {[*supported]}')

        if form not in supported[code]:
            raise Exception(f'File format {form} is not supported; only form = {supported[code]}')

        # process the file based on options
        self._init_process(file, code, form, *args)

        # verify properly initialized object
        if not hasattr(self, 'key'):
            raise NotImplementedError(f'{type(self)} does not initialize a key member')
        if not hasattr(self, '_attributes'):
            raise NotImplementedError(f'{type(self)} does not initialize a _attributes member')

    def __str__(self) -> str:
        return self._str_attrs(self._attributes)

    @abstractmethod
    def _init_process(self, file, code, form, *args) -> None:
        raise NotImplementedError(f'{type(self)} does not implement an _init_process method')

    def _str_keys(self) -> str:
        return self._str_base(self._attributes, lambda *args: '')

    def _str_attrs(self, attrs: list) -> str:
        return self._str_base(attrs, lambda inst, key: '=' + str(getattr(inst, key)))

    def _str_base(self, attrs: list, wrap: Callable) -> str:
        return f'{self.__class__.__name__}(key={self.key:.4f}, ' + ', '.join(
            key + wrap(self, key) for key in attrs) + ')'

    def keys(self) -> set:
        """
        Method to return the members (i.e., names of data fields) contained
        in the data object.

        Works like dict.keys(); except for the returned type

        Returns:
            A set containing the named data fields available.
        """
        return self._attributes

    def todict(self) -> dict:
        """
        Method to return the the data object as a key, value dictionary.

        Returns:
            A dict containing the all data fields available.
        """
        return {key : self[key] for key in self._attributes}
