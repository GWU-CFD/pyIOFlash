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


Todo:
    * Provide more correct and higher performance guard data filling
    * Provide differed or Just-In-Time block data loading for large simulations
"""

from abc import ABC as AbstractBase, abstractmethod
from dataclasses import dataclass, field, InitVar
from typing import Any, Tuple, List, Set, Dict, Callable, Union
from functools import partial

import numpy
import h5py

from .pyio_utility import _first_true, _reduce_str

@dataclass(order=True)
class _BaseData(AbstractBase):
    """
    _BaseData is an abstract class implementing the common behavior
    for all derived dataclasses which provide the functionality to
    read the hdf5 output file data.

    Attributes:
        _key: mappable for compositting into a sortable collection object; sorted by
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

@dataclass
class GeometryData(_BaseData):
    """
    GeometryData is a derived class implementing the functionality to
    read the relavent geometry data contained in the hdf5 output file.

    Attributes:
        blk_num: total number of blocks in the simulation (at current timestep)
        blk_num_x: number of blocks in x direction (at current timestep)
        blk_num_y: number of blocks in y direction (at current timestep)
        blk_num_z: number of blocks in z direction (at current timestep)
        blk_size_x: simulation points of each block in x direction
        blk_size_y: simulation points of each block in y direction
        blk_size_z: simulation points of each block in z direction
        blk_coords: coordinates of each block center (at current timestep)
        blk_bndbox: bounding box coordinates of each block (at current timestep)
        blk_tree_str: tree structure containing block neighbors, parents,
                      and children (at current timestep)
        blk_refine: refinement level of each block (at current timestep)
        blk_neighbors: list of neighbors and type (e.g., same refinement level)
                       for each block (at current timestep)
        grd_type: type of grid in the simulation (e.g., uniform or paramesh)
        grd_dim: dimentionality of the simulation (e.g., 2d or 3d)
        grd_bndbox: bouding box coordinates of the simulation
        grd_mesh_x: mesh data for block data, in x direction (at current timestep)
        grd_mesh_y: mesh data for block data, in y direction (at current timestep)
        grd_mesh_z: mesh data for block data, in z direction (at current timestep)

    Note:
        The grid mesh data attributes return mesh coordinate data for each block
        without filling in relavent guard cell neighbor data; if this data is
        desired, the attribute name should be prepended with an underscore.
    """
    blk_num: int = field(repr=False, init=False, compare=False)
    blk_num_x: int = field(repr=True, init=False, compare=False)
    blk_num_y: int = field(repr=True, init=False, compare=False)
    blk_num_z: int = field(repr=True, init=False, compare=False)
    blk_size_x: int = field(repr=True, init=False, compare=False)
    blk_size_y: int = field(repr=True, init=False, compare=False)
    blk_size_z: int = field(repr=True, init=False, compare=False)
    blk_coords: numpy.ndarray = field(repr=False, init=False, compare=False)
    blk_bndbox: numpy.ndarray = field(repr=False, init=False, compare=False)
    blk_tree_str: List[List[int]] = field(repr=False, init=False, compare=False)
    blk_filtered: List[List[int]] = field(repr=False, init=False, compare=False)
    blk_refine: List[List[int]] = field(repr=False, init=False, compare=False)
    blk_neighbors: List[Tuple[int, List[int]]] = field(repr=False, init=False, compare=False)

    grd_type: str = field(repr=True, init=False, compare=False)
    grd_dim: int = field(repr=True, init=False, compare=False)
    grd_bndbox: List[Tuple[float, float]] = field(repr=False, init=False, compare=False)
    _grd_mesh_x: numpy.ndarray = field(repr=False, init=False, compare=False)
    _grd_mesh_y: numpy.ndarray = field(repr=False, init=False, compare=False)
    _grd_mesh_z: numpy.ndarray = field(repr=False, init=False, compare=False)

    @staticmethod
    def _get_filtered_blocks(blk_tree_str, refine, level=None):
        level = max(refine) if level is None else (min(refine) + 1 if level < 0 else level)
        return [(id, blocks) for id, blocks in enumerate(blk_tree_str) if
                refine[id] == level or (sum(blocks[-4:]) == -4 and refine[id] < level)]

    @staticmethod
    def _get_neighbors(blocks, blk_tree_str, directions):
        nbr_inflg = -1
        neighbors = []
        for direct in directions:
            if blocks[direct] >= 0:
                neighbors.append((0, blocks[direct]))
            elif blocks[direct] == nbr_inflg:
                neighbors.append((1, blk_tree_str[blocks[4]][direct]))
            else:
                neighbors.append((2, blocks[direct]))
        return neighbors

    def _init_process(self, file, code, form) -> None: # pylint: disable=arguments-differ
        # pull relavent data from hdf5 file object
        sim_info: List[Tuple[int, bytes]] = list(file['sim info'])
        coordinates: numpy.ndarray = file['coordinates']
        boundingbox: numpy.ndarray = file['bounding box']
        tree_struct: List[List[int]] = file['gid'][()].tolist()
        refine_lvls: List[List[int]] = file['refine level'][()].tolist()
        int_runtime: List[Tuple[bytes, int]] = list(file['integer runtime parameters'])
        real_runtime: List[Tuple[bytes, int]] = list(file['real runtime parameters'])
        int_scalars: List[Tuple[bytes, int]] = list(file['integer scalars'])
        real_scalars: List[Tuple[bytes, float]] = list(file['real scalars'])

        # initialize mappable keys
        self.key = float(_first_true(real_scalars, lambda l: 'time' in str(l[0]))[1])

        # initialize grid type
        setup_call: str = _first_true(sim_info, lambda l: l[0] == 9)[1].decode('utf-8')
        if setup_call.find('+ug') != -1:
            self.grd_type = 'uniform'
        elif setup_call.find('+pm4dev') != -1:
            self.grd_type = 'paramesh'
        else:
            raise Exception(f'Unable to determine grid type from sim info field')

        # initialize grid dimensionality
        self.grd_dim = _first_true(int_scalars, lambda l: 'dimensionality' in str(l[0]))[1]

        # initialize grid bounding box
        self.grd_bndbox = [(_first_true(real_runtime, lambda l: 'xmin' in str(l[0]))[1],
                            _first_true(real_runtime, lambda l: 'xmax' in str(l[0]))[1]),
                           (_first_true(real_runtime, lambda l: 'ymin' in str(l[0]))[1],
                            _first_true(real_runtime, lambda l: 'ymax' in str(l[0]))[1]),
                           (_first_true(real_runtime, lambda l: 'zmin' in str(l[0]))[1],
                            _first_true(real_runtime, lambda l: 'zmax' in str(l[0]))[1])]

        # initialize block data
        if self.grd_type == 'uniform':
            self.blk_num = _first_true(int_scalars, lambda l: 'globalnumblocks' in str(l[0]))[1]
            self.blk_num_x = _first_true(int_runtime, lambda l: 'iprocs' in str(l[0]))[1]
            self.blk_num_y = _first_true(int_runtime, lambda l: 'jprocs' in str(l[0]))[1]
            self.blk_num_z = 1
            self.blk_size_x = _first_true(int_scalars, lambda l: 'nxb' in str(l[0]))[1]
            self.blk_size_y = _first_true(int_scalars, lambda l: 'nyb' in str(l[0]))[1]
            self.blk_size_z = _first_true(int_scalars, lambda l: 'nzb' in str(l[0]))[1]

        elif self.grd_type == 'paramesh':
            self.blk_num = _first_true(int_scalars, lambda l: 'globalnumblocks' in str(l[0]))[1]
            self.blk_num_x = _first_true(int_runtime, lambda l: 'nblockx' in str(l[0]))[1]
            self.blk_num_y = _first_true(int_runtime, lambda l: 'nblocky' in str(l[0]))[1]
            self.blk_num_z = _first_true(int_runtime, lambda l: 'nblockz' in str(l[0]))[1]
            self.blk_size_x = _first_true(int_scalars, lambda l: 'nxb' in str(l[0]))[1]
            self.blk_size_y = _first_true(int_scalars, lambda l: 'nyb' in str(l[0]))[1]
            self.blk_size_z = _first_true(int_scalars, lambda l: 'nzb' in str(l[0]))[1]

        else:
            pass # other grid handling operations

        # initialize coordinates of block centers
        self.blk_coords = numpy.ndarray(coordinates.shape, dtype=numpy.dtype(float))
        self.blk_coords[:, :] = coordinates

        # initialize coordinates of block bounding boxes
        self.blk_bndbox = numpy.ndarray(boundingbox.shape, dtype=numpy.dtype(float))
        self.blk_bndbox[:, :, :] = boundingbox

        # initialize tree structure and filter a max refinement level
        if self.grd_type == 'uniform':
            self.blk_tree_str = tree_struct.copy()
            self.blk_filtered = GeometryData._get_filtered_blocks(tree_struct, refine_lvls)

        elif self.grd_type == 'paramesh':
            self.blk_tree_str = [GeometryData._shift_block_ids(blocks, -1) for blocks in tree_struct]
            self.blk_filtered = GeometryData._get_filtered_blocks(tree_struct, refine_lvls, -1)

        else:
            pass # other grid handling operations

        # intialize neighbors type and ids for each block
        self.blk_neighbors = [GeometryData._get_neighbors(blocks, self.blk_tree_str,
                                                          range(int(2 * self.grd_dim)))
                              for blocks in self.blk_tree_str]

        # intialize mesh grids for cell centered fields
        self._grd_mesh_x = numpy.ndarray((self.blk_num, self.blk_size_z + 1,
                                          self.blk_size_y + 1, self.blk_size_x + 1), dtype=numpy.dtype(float))
        self._grd_mesh_y = numpy.ndarray((self.blk_num, self.blk_size_z + 1,
                                          self.blk_size_y + 1, self.blk_size_x + 1), dtype=numpy.dtype(float))
        self._grd_mesh_z = numpy.ndarray((self.blk_num, self.blk_size_z + 1,
                                          self.blk_size_y + 1, self.blk_size_x + 1), dtype=numpy.dtype(float))

        for block in range(self.blk_num):
            bndbox = self.blk_bndbox[block]
            sz_box = (self.blk_size_x, self.blk_size_y, self.blk_size_z)
            x = numpy.linspace(bndbox[0][0], bndbox[0][1], sz_box[0] + 1, True)
            y = numpy.linspace(bndbox[1][0], bndbox[1][1], sz_box[1] + 1, True)
            z = numpy.linspace(bndbox[2][0], bndbox[2][1], sz_box[2] + 1, True)

            Z, Y, X = numpy.meshgrid(z, y, x, indexing='ij')
            self._grd_mesh_x[block, :, :, :] = X
            self._grd_mesh_y[block, :, :, :] = Y
            self._grd_mesh_z[block, :, :, :] = Z

        # initialize list of class member names holding the data
        setattr(self, '_attributes', {
            'blk_num', 'blk_num_x', 'blk_num_y', 'blk_num_z', 'blk_size_x',
            'blk_size_y', 'blk_size_z', 'blk_coords', 'blk_bndbox',
            'grd_type', 'grd_dim', 'grd_mesh_x', 'grd_mesh_y', 'grd_mesh_z'})

    @staticmethod
    def _shift_block_ids(blocks, shift):
        try:
            return [(block + shift) if block >= 0 else block for block in blocks]
        except TypeError:
            return (blocks + shift) if blocks >= 0 else blocks

    def __str__(self) -> str:
        fields = ['grd_type', 'blk_num', 'blk_num_x', 'blk_num_y', 'blk_num_z',
                  'blk_size_x', 'blk_size_y', 'blk_size_z']
        return super()._str_attrs(fields)

    @property
    def grd_mesh_x(self):
        """
        Method to provide the property grd_mesh_x.

        Returns:
            Mesh data for block data, in x direction (at current timestep)
        """
        return self._grd_mesh_x[:, :-1, :-1, :-1]

    @grd_mesh_x.setter
    def grd_mesh_x(self, value):
        self._grd_mesh_x[:, :-1, :-1, :-1] = value

    @property
    def grd_mesh_y(self):
        """
        Method to provide the property grd_mesh_y.

        Returns:
            Mesh data for block data, in y direction (at current timestep)
        """
        return self._grd_mesh_y[:, :-1, :-1, :-1]

    @grd_mesh_y.setter
    def grd_mesh_y(self, value):
        self._grd_mesh_y[:, :-1, :-1, :-1] = value

    @property
    def grd_mesh_z(self):
        """
        Method to provide the property grd_mesh_z.

        Returns:
            Mesh data for block data, in z direction (at current timestep)
        """
        return self._grd_mesh_z[:, :-1, :-1, :-1]

    @grd_mesh_z.setter
    def grd_mesh_z(self, value):
        self._grd_mesh_z[:, :-1, :-1, :-1] = value

@dataclass
class FieldData(_BaseData):
    """
    FieldData is a derived class implementing the functionality to
    read the relavent scalar and vector field data contained in the
    hdf5 output file.

    Attributes:
        geometry: (InitVar) corrisponding GeometryData instance required for initialization
        _groups: set of named field data in the hdf4 output file

    Notes:
        The FieldData instance also contains attributes corrisponding to each named
        field in the output file (i.e., _groups) that is created dynamically as the
        hdf5 output file is read.

        The field data attributes return the named field data for each block
        without filling in relavent guard cell neighbor data; if this data is
        desired, the attribute name should be prepended with an underscore.
    """
    geometry: InitVar[GeometryData]
    _groups: Set[str] = field(repr=True, init=False, compare=False)

    @staticmethod
    def _fill_guard_data(data, geometry):
        map_dir = {'x': lambda block, index: numpy.index_exp[block, :, :, index],
                   'y': lambda block, index: numpy.index_exp[block, :, index, :],
                   'z': lambda block, index: numpy.index_exp[block, index, :, :]}

        map_type = {0: lambda data, _, guard, dir: data[map_dir[dir](guard, 0)],
                    1: lambda data, block, _, dir: data[map_dir[dir](block, -2)] * 0.0 - 10.0,
                    2: lambda data, block, _, dir: data[map_dir[dir](block, -2)]}
        y_fix = 2 if geometry.grd_type == 'uniform' else 3

        for block, neighbors in enumerate(geometry.blk_neighbors):
            data[block, :, :, -1] = map_type[neighbors[1][0]](data, block, neighbors[1][1], 'x')
            data[block, :, -1, :] = map_type[neighbors[y_fix][0]](data, block, neighbors[y_fix][1], 'y')
            if geometry.grd_dim == 3:
                data[block, -1, :, :] = map_type[neighbors[5][0]](data, block, neighbors[5][1], 'z')

    # pylint: disable=arguments-differ
    def _init_process(self, file: h5py.File, code: str, form: str, geometry: GeometryData) -> None:
        # pull relavent data from hdf5 file object
        real_scalars: List[Tuple[bytes, float]] = list(file['real scalars'])
        unknown_names: List[bytes] = list(file['unknown names'][:, 0])

        # initialize mappable keys
        self.key = float(_first_true(real_scalars, lambda l: 'time' in str(l[0]))[1])

        # initialize named fields
        self._groups = {k.decode('utf-8') for k in unknown_names}
        if form == 'chk':
            if geometry.grd_dim == 3:
                self._groups.update({'fcx2', 'fcy2', 'fcz2'})

            elif geometry.grd_dim == 2:
                self._groups.update({'fcx2', 'fcy2'})

            else:
                pass

        # initialize field data members (for cell centered data)
        for group in self._groups:

            # allow for guard data at axis upper extent
            shape = file[group].shape
            shape = tuple([shape[0]]) + tuple([length + 1 for length in shape[1:]])

            # read dataset from file
            data = numpy.ndarray(shape, dtype=numpy.dtype(float))
            data[:, :-1, :-1, :-1] = file[group][()]

            # fill guard data (twice to ensure corners are valid)
            FieldData._fill_guard_data(data, geometry)
            FieldData._fill_guard_data(data, geometry)

            # attach dataset to FieldData instance
            setattr(self, '_' + group, data)
            setattr(FieldData, group, property(partial(self._get_attr, attr='_' + group),
                                               partial(self._set_attr, attr='_' + group)))

        # initialize list of class member names holding the data
        setattr(self, '_attributes', {group for group in self._groups})

    def _set_attr(self, _, value, attr):
        getattr(self, attr)[:, :-1, :-1, :-1] = value

    def _get_attr(self, _, attr):
        return getattr(self, attr)[:, :-1, :-1, :-1]

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
