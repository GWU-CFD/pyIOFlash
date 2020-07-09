"""A python module for implementing the SimulationData class.

This class is the workhorse of the pyIOFlash package; providing methods and members to read, process, and
store simulation output, allowing for convienent and intuitive post-processing and ploting of simulation data.

Todo:

"""


from typing import Any, Tuple, List, Dict, Iterable, Union, Optional, Callable
from functools import partial
from sys import stdout


from pyioflash.simulation.collections import SortedDict, TransposableAsArray, TransposableAsSingle
from pyioflash.simulation.fields import FieldData
from pyioflash.simulation.geometry import GeometryData
from pyioflash.simulation.scalars import ScalarData
from pyioflash.simulation.series import NameData, DataPath, data_from_path
from pyioflash.simulation.statics import StaticData
from pyioflash.simulation.utility import _blocks_from_plane, _blocks_from_line
from pyioflash.simulation.utility import _get_indices, _get_times, _reduce_str, open_hdf5


# define the module api
def __dir__() -> List[str]:
    return ['SimulationData']

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
        utility (Utility): collection of helper methods for extending SimulationData functionality

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
    utility: 'Utility'

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

        # intialize utility functions
        self.utility = Utility(self)

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
            print("\n############    Building SImulationData Object   ############\n")
            print("Processing metadata from: " + self.files.names[0])
            setattr(self, 'geometry', GeometryData(file, self.code, self.form, self.files.geometry))
            setattr(self, 'statics', StaticData(file, self.code, self.form, def_statics))

        # process FLASH4 hdf5 files
        for name in self.files.names:
            with open_hdf5(name, 'r') as file:
                stdout.write("Processing file: " + name + "\r")
                stdout.flush()
                self.fields.append(FieldData(file, self.code, self.form, self.geometry))
                self.scalars.append(ScalarData(file, self.code, self.form, def_scalars))
                self.dynamics.append(StaticData(file, self.code, self.form, def_dynamics))

        print("\n\n#############################################################\n\n")

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


class Utility:
    """ A simple class providing helper methods to support extending SimulationData functionality.

    When a SimulationData instance is created, a collection of helper methods is also provided 
    under the umbrella member class 'utility'. These memeber fuctions provide functionality for 
    simplifying common tasks such as looking up simulation times as well as assisting accomplishment 
    of more complicated tasks such as looking up the simulation blocks intersected by a line or plane.

    Attributes:
        data_from_path: assists in the lookup of simulation data
        indices: assists in the lookup of simulation indices 
        times: assists in the lookup of simulation times
        blocks_from_plane: provides blocks from intersecting plane
        blocks_from_line: provides blocks from intersecting line

    """
    
    def __init__(self, data):
        self._data = data
        self._reference = data.fields
        self._geometry = data.geometry
        
    def blocks_from_line(self, axes: Tuple[str] = ('y', 'z'), 
                         values: Tuple[float] = (0.0, 0.0)) -> List[int]:
        """
        Provides a list of blocks which are intersected by the provided line, axes = values.

        Attributes:
            axes: named normal axes of the desired line (optional)
            value: points which define the normal plane (optional)

        Note:
            Intersections are treated as the open interval, lows <= values < highs.

        Todo:
            Add support for none axis alined normals and points

        """
        return _blocks_from_line(self._geometry, axes, values)

    def blocks_from_plane(self, axis: str = 'z', value: float = 0.0) -> List[int]:
        """
        Provides a list of blocks which are intersected by the provided plane, axis = value.

        Attributes:
            axis: named normal axis of the desired plane (optional)
            value: point which defines the plane (optional)

        Note:
            Intersections are treated as the open interval, low <= value < high.

        Todo:
            Add support for none axis alined normals and points

        """
        return _blocks_from_plane(self._geometry, axis, value)

    def data_from_path(self, module: str, name: str, *,
                       sub: Optional[str] = None,
                       index: Optional[Union[Iterable, slice, int]] = None,
                       times: Union[slice, float, int] = None) -> Union[str, int, float, 'ndarray']:
        """
        A helper method to provide desired data from the simulation output using a simple interface.

        Attributes:
            module: where is the data in the SimulationData object (e.g., dynamics)
            name: what is the desired data to retrieve (e.g., time)
            sub: if necessary, where in the specified module is the data (e.g., real_scalars) -- (optional)
            index: index the data once retrieved (optional)
            times: from which times to retrieve the data, if not all (optional)

        Notes:

        Todo: 
            allow providing an iterable as times

        """
        return data_from_path(DataPath(self._data, module, sub, name), index=index, times=times) 

    def indices(self, keys: Union[int, float, slice, Iterable] = slice(None)) -> List[int]:
        """
        Provides a list of indices associated with a set of simulation times (or indices)

        Attributes:
            keys: simulation times or indices from which to lookup indices (optional)

        Notes: 
            the provided times may be approximate and the indices matching each nearest 
            time will be returned; for example, keys = [..., 30.0, ...] would return the list
            [..., 27, ...]  from the simulation times   [..., 29.002593, 30.003594, ...] and 
                                    associated indices  [..., 27, ...].   

        Todo:

        """
        return _get_indices(self._reference, keys)

    def times(self, keys: Union[int, float, slice, Iterable] = slice(None)) -> List[Union[int, float]]:
        """
        Provides a list of times associated with a set of simulation indices (or times)

        Attributes:
            keys: simulation times or indices from which to lookup times (optional)

        Notes: 
            the provided times may be approximate and the indices matching each nearest 
            time will be returned; for example, keys = [..., 30.0, ...] would return the list
            [..., 30.003594, ...]   from the simulation times   [..., 29.002593, 30.003594, ...] and 
                                            associated indices  [..., 27, ...].   

        Todo:

        """
        return _get_times(self._reference, keys)
