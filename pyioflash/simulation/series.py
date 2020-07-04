"""

This module defines public methods and classes necessary for the 
SimulationData subpackage of the pyioflash library.

This module provides abstractions useful for interacting with the 
SimulationData package in a series or time-like manner.

Currently this module implements the following useful abstractions:

    NameData       -> provides an abstraction of the filenames output by FLASH
    DataPath       -> provides a why to locate data in the SimulationData object
    data_from_path -> provides an interface to extract data using DataPath 


Todo:
    Build typying information for SimulationData
"""

from dataclasses import dataclass, field
from collections import namedtuple
from typing import List, Iterable, Union, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray
    from pyioflash.simulation.data import SimulationData


DataPath = namedtuple('DataPath', ['data', 'module', 'type', 'name'], defaults=[None, None])


def data_from_path(path: DataPath, *,
                   index: Optional[Union[Iterable, slice, int]] = None,
                   times: Union[slice, float, int] = None
                   ) -> Union[str, int, float, 'ndarray']:
    """
    """
    
    # attach index if going to slice into data from path
    if index is not None:
        lookup = lambda entry, index : entry[index]
    else:
        lookup = lambda entry, index : entry

    # attache time-like slice if provided
    if times is not None:
        source = lambda entry, index : entry[index]
    else:
        source = lambda entry, index : entry

    # need to work with a filled out DataPath object
    if path.name is None:
        raise Exception(f'DataPath.name default of None provided in argument path; must provide path.name!')

    # apply correct lookup semantics based on provided path, times, and index 
    if path.module in {'fields', 'scalars'}:
        return lookup(source(getattr(path.data, path.module), times)[path.name], index)[0]

    elif path.module in {'dynamics'}:
        return lookup(source(getattr(path.data, path.module), times)[path.type][path.name], index)

    elif path.module in {'statics'}:
        return lookup(getattr(path.data, path.module)[path.type][path.name], index)

    elif path.module in {'geometry'}:
        return lookup(path.data.geometry[path.name], index)

    # cannot work with module provided
    else:
        raise Exception(f'DataPath.module provided does not match known objects is path.data object!') 


@dataclass
class NameData:
    """
    Storage class for containing file names as a list for hdf5 files to be processed;
    the list, NameData.files, is automatically generated based on keyword arguments

    NameData assumes file names (and paths) of the form:
        directory + header + numbers + footer + extension

    Example:

        *if the files to be processed were*:

            - ../out/INS_LidDr_Cavity_hdf5_plt_cnt_0000
            - ../out/INS_LidDr_Cavity_hdf5_plt_cnt_0001
            - ../out/INS_LidDr_Cavity_hdf5_plt_cnt_0002

        *the following would be the values for attributes*:

            - numbers = [0, 1, 2]
            - directory = '../out/'
            - basename = 'INS_LidDr_Cavity_'
            - header = 'hdf5_plt_cnt_'
            - footer = ''
            - extension = ''
            - numform = '04d'

    Args:
        numbers (Iterable): list of output file numbers (or names)
        directory (str): relative file path to output files
        basename (str): leading file name text
        header (str): pre-pended (before numbers) file name text
        footer (str): post-pended (after numbers) file name text
        extention (str): file name extention (must include '.')
        numform (str): number format for file names e.g., 04d -> 0000

    Attributes:
        length (int): number of filenames - calculated
        names (list): list of full filenames and paths - calculated
        geometry (str): full filename and path for geometry file - calculated

    """
    numbers: Iterable[Union[int, str]] = field(repr=False, init=True, compare=False)
    directory: str = field(repr=False, init=True, compare=False, default='')
    basename: str = field(repr=False, init=True, compare=False, default='')
    header: str = field(repr=False, init=True, compare=False, default='')
    footer: str = field(repr=False, init=True, compare=False, default='')
    extention: str = field(repr=False, init=True, compare=False, default='')
    numform: str = field(repr=False, init=True, compare=False, default='04d')
    geonumber: int = field(repr=False, init=True, compare=False, default=0)
    geometry: str = field(repr=False, init=False, compare=False)
    length: int = field(repr=False, init=False, compare=False)
    names: List[str] = field(repr=True, init=False, compare=True)

    def __post_init__(self):
        self.names = [self.directory + self.basename + self.header +  f'{n:{self.numform}}' +
                      self.footer + self.extention for n in self.numbers] # pylint: disable=not-an-iterable
        self.length = len(self.names)
        self.geometry = self.directory + self.basename + 'hdf5_grd_' + f'{self.geonumber:{self.numform}}'

    @classmethod
    def from_strings(cls, names: List[str], **kwargs):
        """
        Class method to create a NameData instance with a list of strings

        Note:
            Accepts same keyword arguments as NameData()

        Args:
            names (list): list of output file names

        """
        kwargs['numform'] = ''
        instance = cls(names, **kwargs)
        return instance

    @classmethod
    def from_name(cls, name: str, **kwargs):
        """
        Class method to create a NameData instance with a single string

        Note:
            Accepts same keyword arguments as NameData()

        Args:
            name (str): output file name

        """
        return cls.from_strings([name], **kwargs)
