"""
Python module for implementing the SimulationData class

This class is the workhorse of the pyIOFlash package;
providing methods and members to read, process, and
store simulation output, allowing for post processing
and ploting of simulation data.
"""

from typing import Any, Tuple, List, Dict, Callable

from .pyio_utility import _reduce_str
from .pyio_utility import open_hdf5, NameData
from .pyio_collections import SortedDict, TransposableAsArray, TransposableAsSingle
from .pyio_types import GeometryData, FieldData, ScalarData, StaticData


class SimulationData:
    """
    ...
    """
    files: NameData
    code: str
    form: str
    geometry: SortedDict
    fields: SortedDict
    scalars: SortedDict
    dynamics: SortedDict
    statics: StaticData

    def __init__(self, files: NameData, *, form: str = None, code: str = None):
        """ ... """

        # initialize filenames
        self.files = files

        # initialize code and file type
        self.form = form
        if self.form is None:
            self.form = 'plt'
        self.code = code
        if self.code is None:
            self.code = 'flash'

        # initialize empty containers
        self.geometry = type('TransposableAsArray_SortedDict', (TransposableAsArray, SortedDict), {})([])
        self.fields = type('TransposableAsArray_SortedDict', (TransposableAsArray, SortedDict), {})([])
        self.scalars = type('TransposableAsArray_SortedDict', (TransposableAsArray, SortedDict), {})([])
        self.dynamics = type('TransposableAsSingle_SortedDict', (TransposableAsSingle, SortedDict), {})([])
        self.dynamics = type('TransposableAsSingle_SortedDict', (TransposableAsSingle, SortedDict), {})([])

        # read simulation files and store to member variables
        # -- future development of mpi4py version -> multiprocessing branch --

        # process flash simulation output (default option)
        if  self.code == 'flash':

            # process plot or checkpoint file (default option)
            if  self.form == 'plt' or  self.form == 'chk':
                self.__read_flash4__()
            else:
                raise Exception(f'Unknown hdf5 layout for FLASH4; filetype == plt or chk')

        # other codes not supported
        else:
            raise Exception(f'Codes other then FLASH4 not supported; code == flash')

    @classmethod
    def from_list(cls, numbers: List[int], *, numform: str = None,
                  path: str = None, header: str = None, footer: str = None, ext: str = None,
                  form: str = None, code: str = None):
        """
        ...
        """
        # create NameData instance and initialize member variables
        options: Dict[str, Any] = {'numbers' : numbers}
        if path is not None:
            options['directory'] = path
        if header is not None:
            options['header'] = header
        if footer is not None:
            options['footer'] = footer
        if ext is not None:
            options['extention'] = ext
        if numform is not None:
            options['numform'] = numform

        return cls(NameData(**options), code=code, form=form)

    def __read_flash4__(self):
        """
        Method for importing and processing a time series of FLASH4 HDF5 plot or checkpoint files
        """

        # definitions used to process a FLASH4 output file;
        # format = [(group, dataset, member_name), ...]
        def_scalars: List[Tuple[str, str, str]] = [
            ('real scalars', 'time', 't'),
            ('real scalars', 'dt '),
            ('integer scalars', 'nstep'),
            ('integer scalars', 'nbegin')]

        # format = [(group, function to process dataset values), ...]
        def_dynamics: List[Tuple[str, Callable]] = [
            ('integer scalars', StaticData.pass_label),
            ('logical scalars', StaticData.pass_label),
            ('real scalars', StaticData.pass_label),
            ('string scalars', lambda label: _reduce_str(StaticData.decode_label(label)))]

        # format = [(group, function to process dataset values), ...]
        def_statics: List[Tuple[str, Callable]] = [
            ('integer runtime parameters', StaticData.pass_label),
            ('logical runtime parameters', StaticData.pass_label),
            ('real runtime parameters', StaticData.pass_label),
            ('string runtime parameters', lambda label: _reduce_str(StaticData.decode_label(label))),
            ('sim info', lambda label: _reduce_str(StaticData.decode_label(label), ' '))]

        # process first FLASH4 hdf5 file
        with open_hdf5(self.files.names[0], 'r') as file:
            setattr(self, 'statics', StaticData(file, self.code, self.form, def_statics))

        # process FLASH4 hdf5 files
        for name in self.files.names:
            with open_hdf5(name, 'r') as file:
                geometry: GeometryData = GeometryData(file, self.code, self.form)
                self.geometry.append(geometry)
                self.fields.append(FieldData(file, self.code, self.form, geometry))
                self.scalars.append(ScalarData(file, self.code, self.form, def_scalars))
                self.dynamics.append(StaticData(file, self.code, self.form, def_dynamics))
