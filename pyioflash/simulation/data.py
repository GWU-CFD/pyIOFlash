"""A python module for implementing the SimulationData class.

This class is the workhorse of the pyIOFlash package; providing methods and members to read, process, and
store simulation output, allowing for convienent and intuitive post-processing and ploting of simulation data.

Todo:

"""

from typing import Any, Tuple, List, Dict, Callable

from pyioflash.simulation.series import NameData
from pyioflash.simulation.utility import _reduce_str, open_hdf5
from pyioflash.simulation.collections import SortedDict, TransposableAsArray, TransposableAsSingle
from pyioflash.simulation.geometry import GeometryData
from pyioflash.simulation.fields import FieldData
from pyioflash.simulation.scalars import ScalarData
from pyioflash.simulation.statics import StaticData


class SimulationData:
    """A class providing a data structure to store and an api to process hdf5 output files.

    When a SimulationData instance is created, the provided hdf5 files are opened, read into
    memory, and subsequently closed. The init method provides this functionality by instanciating
    the relavent empty collections (as composite objects) and subsequently calling the appropriate
    import method (e.g., __read_flash4__).::

        from pyio import SimulationData

        data = SimulationData.from_list(range(20), path='../out/',
                                        header='INS_LidDr_Cavity_hdf5_plt_cnt_')

        data.fields[20.0 : 60.0 : 2]['temp', 'pres'][:, :, :, :, :]

    The basic usage of the class is as follows: (1) import the class from the package, (2) create an instance
    of the class by providing the necessary specification for the hdf5 output files, and (3) use the class
    instance data members in a natural way (e.g., slicing and computations work as expected if you are
    familure with either numpy or matlab indexing).

    The general format for accessing the geometry, field, or scalar data is::

        instance.member[times or indicies]['name', ...][indicies, blocks, z(s), y(s), x(s)]

    The general format for accessing the dynamic data is::

        instance.dynamic[times or indicies]['name', ...][indicies]

    The general format for accessing the static data is::

        instance.static['name']

    Wherein the above times / indicies / blocks / x(s) are slices in the python or numpy format and the
    ['name', 'name', ...] may be either a single string or a list of srings associated with member names.

    Finally, the class instance is a needed input for the SimulationPlot class necessary to provide convienent,
    and intuitive plotting interface as opposed to manually writing matplotlib scripts.

    Attributes:
        files (NameData): list of filenames and paths to be processed
        code (str): flag for the code which produced the output (e.g., flash)
        form (str): flag for the format of the hdf5 output file (e.g., plt)
        geometry (GeometryData): geometry data/information from the processed hdf5 file (first)
        fields (SortedDict): vector and scalar field data from the processed hdf5 files
        scalars (SortedDict): scalar (e.g., time, dt) data from the processed hdf5 files
        dynamics (SortedDict): time varying information from the processed hdf5 files
        statics (StaticData):  non-time varying information from the processed hdf5 files

    Note:

        While each of fields, scalars, dynamics, and statics is a
        :class:`~pyioflash.pyio_collections.SortedDict` object, each is in actuality a composition of SortedDict
        and a derived class of :class:`~pyioflash.pyio_collections.BaseTransposable`; specifically, either
        :class:`~pyioflash.pyio_collections.TransposableAsArray` or
        :class:`~pyioflash.pyio_collections.TransposableAsSingle`. This is done to provide a convienent
        and intuitive indexing syntax for each object.

    """
    files: NameData
    code: str
    form: str
    geometry: GeometryData
    fields: SortedDict
    scalars: SortedDict
    dynamics: SortedDict
    statics: StaticData

    def __init__(self, files: NameData, *, form: str = None, code: str = None):
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
        self.fields = type('TransposableAsArray_SortedDict', (TransposableAsArray, SortedDict), {})([])
        self.scalars = type('TransposableAsArray_SortedDict', (TransposableAsArray, SortedDict), {})([])
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
    def from_list(cls, numbers: List[int], *, numform: str = None, path: str = None,
                  basename: str = None, header: str = None, footer: str = None, gnumber: int = None,
                  ext: str = None, form: str = None, code: str = None) -> 'SimulationData':
        """Creates a SimulationData instance from a list of file numbers.

        This class method provides the capability to supply a list of integers associated with
        hdf5 output file names along with relavent keyword arguments as the method of specifying
        the desired hdf5 files to be processed and read into memory.

        Args:
            numbers: list of output file numbers

        *Keyword*

        Args:
            code: code which produced the output (e.g., flash)
            form: format of the hdf5 output file (e.g., plt)
            numform: number format for file names (e.g., 04d -> 0000)
            path: relative file path to output files
            header: leading file name text
            footer: following file name text
            ext: file name extention (must include '.')

        Returns:
            A data object containing the processed simulation output

        """
        # create NameData instance and initialize member variables
        options: Dict[str, Any] = {'numbers' : numbers}
        if path is not None:
            options['directory'] = path
        if basename is not None:
            options['basename'] = basename
        if header is not None:
            options['header'] = header
        if footer is not None:
            options['footer'] = footer
        if ext is not None:
            options['extention'] = ext
        if numform is not None:
            options['numform'] = numform
        if gnumber is not None:
            options['geonumber'] = gnumber

        return cls(NameData(**options), code=code, form=form)

    def __read_flash4__(self):
        """Method for importing and processing a time series of FLASH4 HDF5 plot or checkpoint files.

        This private method provides the defualt options and processing for reading FLASH4 HDF5 files into
        memory.  This class may be used as a template for providing similar functionality for other output
        formats from FLASH4 or other similar simulation code.

        Todo:

            # provide an interface for the user to change the defualt behavior at the time of instanciation

        """

        # definitions used to process a FLASH4 output file;
        # format = [(group, dataset, member_name), ...]
        def_scalars: List[Tuple[str, str, str]] = [
            ('real scalars', 'time', 't'),
            ('real scalars', 'dt'),
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
            setattr(self, 'geometry', GeometryData(file, self.code, self.form, self.files.geometry))
            setattr(self, 'statics', StaticData(file, self.code, self.form, def_statics))

        # process FLASH4 hdf5 files
        for name in self.files.names:
            with open_hdf5(name, 'r') as file:
                self.fields.append(FieldData(file, self.code, self.form, self.geometry))
                self.scalars.append(ScalarData(file, self.code, self.form, def_scalars))
                self.dynamics.append(StaticData(file, self.code, self.form, def_dynamics))
