"""

This module defines the utility methods and classes necessary for the pyio package.

Todo:
    None
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, List, Iterable, Callable, Union

import h5py

Namelist = Iterable[Union[int, str]]

def _first_true(iterable: Iterable, predictor: Callable[..., bool]) -> Any:
    """Returns the first true value in the iterable according to predictor."""
    return next(filter(predictor, iterable))

def _filter_transpose(source: List[Any], names: Iterable[str]) -> List[List[Any]]:
    """Returns a list of named members extracted from a
    collection object (source) based on a list (names)"""
    try:
        return [list(map(lambda obj, n=name: getattr(obj, n), source)) for name in names]
    except AttributeError:
        return [list(map(lambda obj, n=name: obj[n], source)) for name in names]

def _set_is_unique(source: Iterable, sequence: Iterable, mask: Iterable = None) -> bool:
    """Returns (True) if no member of sequence is found in source, with ability
    to mask members of source from comparision to members of sequence"""
    try:
        if mask is None:
            _first_true(sequence, lambda item: item in source)
        else:
            masked_source: set = {k for k in source if k not in mask}
            _first_true(sequence, lambda item: item in masked_source)
        return False
    except StopIteration:
        return True

def _reduce_str(value: str, sentinal: str = '_'):
    """ Provides reduced string with intervineing spaces replaced, and trailing removed"""
    return value.rstrip().replace(' ', sentinal)

@contextmanager
def open_hdf5(*args, **kwargs):
    """Context manager for working with a hdf5 file;
    using a h5py file handle"""
    file = h5py.File(*args, **kwargs)
    try:
        yield file
    finally:
        file.close()

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
            - header = 'INS_LidDr_Cavity_hdf5_plt_cnt\_'
            - footer = ''
            - extension = ''
            - numform = '04d'

    Args:
        numbers (Iterable): list of output file numbers (or names)
        directory (str): relative file path to output files
        header (str): leading file name text
        footer (str): following file name text
        extention (str): file name extention (must include '.')
        numform (str): number format for file names e.g., 04d -> 0000

    Attributes:
        length (int): number of filenames - calculated
        names (list): list of full filenames and paths - calculated

    """
    numbers: Namelist = field(repr=False, init=True, compare=False)
    directory: str = field(repr=False, init=True, compare=False, default='')
    header: str = field(repr=False, init=True, compare=False, default='')
    footer: str = field(repr=False, init=True, compare=False, default='')
    extention: str = field(repr=False, init=True, compare=False, default='')
    numform: str = field(repr=False, init=True, compare=False, default='04d')
    length: int = field(repr=False, init=False, compare=False)
    names: List[str] = field(repr=True, init=False, compare=True)

    def __post_init__(self):
        self.names = [self.directory + self.header +  f'{n:{self.numform}}' +
                      self.footer + self.extention for n in self.numbers] # pylint: disable=not-an-iterable
        self.length = len(self.names)

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

@dataclass
class Plane:
    """
    Data object for defining a 2d cut-plane at a give time for use in plotting

    Attributes:
        time: time or index (e.g., key) associated with the desired plane
    """
    time: Union[float, int] = None
    cut: float = None
    axis: str = 'z'
