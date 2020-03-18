"""

This module defines the utility methods and classes necessary for the pyio package.

Todo:
    None
"""

from dataclasses import dataclass, field
from typing import List, Iterable, Union

Namelist = Iterable[Union[int, str]]

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
    numbers: Namelist = field(repr=False, init=True, compare=False)
    directory: str = field(repr=False, init=True, compare=False, default='')
    basename: str = field(repr=False, init=True, compare=False, default='')
    header: str = field(repr=False, init=True, compare=False, default='')
    footer: str = field(repr=False, init=True, compare=False, default='')
    extention: str = field(repr=False, init=True, compare=False, default='')
    numform: str = field(repr=False, init=True, compare=False, default='04d')
    geometry: str = field(repr=False, init=False, compare=False)
    length: int = field(repr=False, init=False, compare=False)
    names: List[str] = field(repr=True, init=False, compare=True)

    def __post_init__(self):
        self.names = [self.directory + self.basename + self.header +  f'{n:{self.numform}}' +
                      self.footer + self.extention for n in self.numbers] # pylint: disable=not-an-iterable
        self.length = len(self.names)
        self.geometry = self.directory + self.basename + 'hdf5_grd_' + f'{self.numbers[0]:{self.numform}}'

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
