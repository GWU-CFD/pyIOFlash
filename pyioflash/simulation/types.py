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
    * Eliminate branches in favor of casting for _fill_guard_data method

"""

from abc import ABC as AbstractBase, abstractmethod
from dataclasses import dataclass, field, InitVar
from typing import Any, Tuple, List, Set, Dict, Callable, Union
from functools import partial

import numpy
import h5py

from pyioflash.simulation.utility import _first_true, _reduce_str, open_hdf5
from pyioflash.simulation.support import _guard_cells_from_data, _bound_cells_from_data

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
        blk_guards: guard cell points of each block in each direction
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
    gridfilename: InitVar[str]
    blk_num: int = field(repr=False, init=False, compare=False)
    blk_num_x: int = field(repr=True, init=False, compare=False)
    blk_num_y: int = field(repr=True, init=False, compare=False)
    blk_num_z: int = field(repr=True, init=False, compare=False)
    blk_size_x: int = field(repr=True, init=False, compare=False)
    blk_size_y: int = field(repr=True, init=False, compare=False)
    blk_size_z: int = field(repr=True, init=False, compare=False)
    blk_guards: int = field(repr=False, init=False, compare=False)
    blk_coords: numpy.ndarray = field(repr=False, init=False, compare=False)
    blk_bndbox: numpy.ndarray = field(repr=False, init=False, compare=False)
    blk_tree_str: List[List[int]] = field(repr=False, init=False, compare=False)
    blk_neighbors: List[Dict[str, int]] = field(repr=False, init=False, compare=False)
    grd_type: str = field(repr=True, init=False, compare=False)
    grd_dim: int = field(repr=True, init=False, compare=False)
    grd_bndbox: List[Tuple[float, float]] = field(repr=False, init=False, compare=False)
    grd_bndcnds: Dict[str, Dict[str, str]] = field(repr=False, init=False, compare=False)
    grd_bndvals: Dict[str, Dict[str, float]] = field(repr=False, init=False, compare=False)
    _grd_mesh_x: numpy.ndarray = field(repr=False, init=False, compare=False)
    _grd_mesh_y: numpy.ndarray = field(repr=False, init=False, compare=False)
    _grd_mesh_z: numpy.ndarray = field(repr=False, init=False, compare=False)

    @staticmethod
    def _get_neighbors(tree, dim):
        names = ["left", "right", "front", "back", "up", "down"]
        return [{names[face] : block for face, block in enumerate(struct[:2*dim]) 
                 if block >= 0} 
                for struct in tree]

    # pylint: disable=arguments-differ
    def _init_process(self, file: h5py.File, code: str, form: str, gridfilename: str) -> None:

        # pull relavent data from hdf5 file object  
        sim_info: List[Tuple[int, bytes]] = list(file['sim info'])
        coordinates: numpy.ndarray = file['coordinates']
        boundingbox: numpy.ndarray = file['bounding box']
        tree_struct: List[List[int]] = file['gid'][()].tolist()
        int_runtime: List[Tuple[bytes, int]] = list(file['integer runtime parameters'])
        real_runtime: List[Tuple[bytes, float]] = list(file['real runtime parameters'])
        str_runtime: List[Tuple[bytes, bytes]] = list(file['string runtime parameters'])
        int_scalars: List[Tuple[bytes, int]] = list(file['integer scalars'])
        real_scalars: List[Tuple[bytes, float]] = list(file['real scalars'])

        # initialize mappable keys
        self.key = float(_first_true(real_scalars, lambda l: 'time' in str(l[0]))[1])

        # initialize grid type
        setup_call: str = _first_true(sim_info, lambda l: l[0] == 9)[1].decode('utf-8')
        if setup_call.find('+ug') != -1:
            self.grd_type = 'uniform'
        elif setup_call.find('+rg') != -1:
            self.grd_type = 'regular'
        elif setup_call.find('+pm4dev') != -1:
            raise Exception(f'Paramesh grid type not currently supported')
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

        # initialize grid boundary conditions
        bndcnds = {"velc" : {"left"  : "xl_boundary_type", "right" : "xr_boundary_type",
                             "front" : "yr_boundary_type", "back"  : "yl_boundary_type",
                             "up"    : "zr_boundary_type", "down"  : "zl_boundary_type"},
                   "temp" : {"left"  : "txl_boundary_type", "right" : "txr_boundary_type",
                             "front" : "tyr_boundary_type", "back"  : "tyl_boundary_type",
                             "up"    : "tzr_boundary_type", "down"  : "tzl_boundary_type"}}
        bndvals = {"temp" : {"left"  : "txl_boundary_value", "right" : "txr_boundary_value",
                             "front" : "tyr_boundary_value", "back"  : "tyl_boundary_value",
                             "up"    : "tzr_boundary_value", "down"  : "tzl_boundary_value"}}
        self.grd_bndcnds = {field : {
            face : _first_true(str_runtime, lambda l: name == l[0].decode('utf-8').replace(' ','')
                               )[1].decode('utf-8').replace(' ','') 
            for face, name in faces.items()} for field, faces in bndcnds.items()}
        self.grd_bndvals = {field : {
            face : _first_true(real_runtime, lambda l: name == l[0].decode('utf-8').replace(' ',''))[1] 
            for face, name in faces.items()} for field, faces in bndvals.items()}

        # initialize block data
        if self.grd_type in {'uniform', 'regular'}:
            self.blk_num = _first_true(int_scalars, lambda l: 'globalnumblocks' in str(l[0]))[1]
            self.blk_num_x = _first_true(int_runtime, lambda l: 'iprocs' in str(l[0]))[1]
            self.blk_num_y = _first_true(int_runtime, lambda l: 'jprocs' in str(l[0]))[1]
            self.blk_num_z = _first_true(int_runtime, lambda l: 'kprocs' in str(l[0]))[1] 
            self.blk_size_x = _first_true(int_scalars, lambda l: 'nxb' in str(l[0]))[1]
            self.blk_size_y = _first_true(int_scalars, lambda l: 'nyb' in str(l[0]))[1]
            self.blk_size_z = _first_true(int_scalars, lambda l: 'nzb' in str(l[0]))[1]

        elif self.grd_type == 'paramesh':
            pass # paramesh grid handling operations

        else:
            pass # other grid handling operations

        # initialize coordinates of block centers
        self.blk_coords = numpy.ndarray(coordinates.shape, dtype=numpy.dtype(float))
        self.blk_coords[:, :] = coordinates

        # initialize coordinates of block bounding boxes
        self.blk_bndbox = numpy.ndarray(boundingbox.shape, dtype=numpy.dtype(float))
        self.blk_bndbox[:, :, :] = boundingbox

        # initialize number of guard cells for each block per direction
        self.blk_guards = _first_true(int_runtime, lambda l: 'iguard' in str(l[0]))[1]
        self.blk_guards = int(self.blk_guards / self.blk_num_x)

        # initialize tree structure and filter a max refinement level
        if self.grd_type in {'uniform', 'regular'}:
            self.blk_tree_str = tree_struct.copy()

        elif self.grd_type == 'paramesh':
            pass # paramesh grid handling operations

        else:
            pass # other grid handling operations

        # intialize neighbors type and ids for each block
        self.blk_neighbors = GeometryData._get_neighbors(self.blk_tree_str, self.grd_dim)

        # create mesh grids for cell centered and face fields (guard data in both directions per axis)
        # FUTURE -- only load on demand
        self._grd_mesh_x = numpy.zeros((3, self.blk_num,
                                        self.blk_size_z + self.blk_guards,
                                        self.blk_size_y + self.blk_guards, 
                                        self.blk_size_x + self.blk_guards), dtype=numpy.dtype(float))
        self._grd_mesh_y = numpy.zeros((3, self.blk_num, 
                                        self.blk_size_z + self.blk_guards,
                                        self.blk_size_y + self.blk_guards, 
                                        self.blk_size_x + self.blk_guards), dtype=numpy.dtype(float))
        self._grd_mesh_z = numpy.zeros((3, self.blk_num, 
                                        self.blk_size_z + self.blk_guards,
                                        self.blk_size_y + self.blk_guards, 
                                        self.blk_size_x + self.blk_guards), dtype=numpy.dtype(float))
        self._grd_mesh_ddx = numpy.zeros((3, self.blk_num, 
                                        self.blk_size_z + self.blk_guards,
                                        self.blk_size_y + self.blk_guards, 
                                        self.blk_size_x + self.blk_guards), dtype=numpy.dtype(float))
        self._grd_mesh_ddy = numpy.zeros((3, self.blk_num, 
                                        self.blk_size_z + self.blk_guards,
                                        self.blk_size_y + self.blk_guards, 
                                        self.blk_size_x + self.blk_guards), dtype=numpy.dtype(float))
        self._grd_mesh_ddz = numpy.zeros((3, self.blk_num, 
                                        self.blk_size_z + self.blk_guards,
                                        self.blk_size_y + self.blk_guards, 
                                        self.blk_size_x + self.blk_guards), dtype=numpy.dtype(float))

        # initialize mesh grids for cell centered and face fields
        # FUTURE -- only load on demand
        if self.grd_type == 'uniform':
            grds = self.blk_guards
            size = (self.blk_size_x, self.blk_size_y, self.blk_size_z)
            gsft = (self.blk_guards - 1) / 2

            for face, fsft in enumerate({-1/2, 0, 1/2}):
                for block in range(self.blk_num):
                    bbox = self.blk_bndbox[block]

                    dx = (bbox[0][1] - bbox[0][0]) / size[0]
                    dy = (bbox[1][1] - bbox[1][0]) / size[1]
                    dz = (bbox[2][1] - bbox[2][0]) / size[2]

                    x = numpy.linspace(bbox[0][0] + (fsft - gsft) * dx, 
                                       bbox[0][1] + (fsft + gsft) * dx, size[0] + grds, True)
                    y = numpy.linspace(bbox[1][0] + (fsft - gsft) * dy, 
                                       bbox[1][1] + (fsft + gsft) * dy, size[1] + grds, True)
                    z = numpy.linspace(bbox[2][0] + (fsft - gsft) * dz, 
                                       bbox[2][1] + (fsft + gsft) * dz, size[2] + grds, True)

                    Z, Y, X = numpy.meshgrid(z, y, x, indexing='ij')
                    self._grd_mesh_x[face, block, :, :, :] = X
                    self._grd_mesh_y[face, block, :, :, :] = Y
                    self._grd_mesh_z[face, block, :, :, :] = Z
                    self._grd_mesh_ddx[face, block, :, :, :] = 1 / dx
                    self._grd_mesh_ddy[face, block, :, :, :] = 1 / dy
                    if self.grd_dim == 3:
                        self._grd_mesh_ddz[face, block, :, :, :] = 1 / dz

        elif self.grd_type == 'regular':
            grids = {"x" : ["xxxl", "xxxc", "xxxr"],
                     "y" : ["yyyl", "yyyc", "yyyr"],
                     "z" : ["zzzl", "zzzc", "zzzr"],
                     "ddx" : ["ddxl", "ddxc", "ddxr"],
                     "ddy" : ["ddyl", "ddyc", "ddyr"],
                     "ddz" : ["ddzl", "ddzc", "ddzr"]}
            with open_hdf5(gridfilename, 'r') as gridfile:
                for face, name in enumerate(grids["x"]):
                    self._grd_mesh_x[face, :, 1:-1, 1:-1, 1:-1] = gridfile[name][()]
                    _guard_cells_from_data(self._grd_mesh_x[face, :, :, :, :], self)
                    _bound_cells_from_data(self._grd_mesh_x[face, :, :, :, :], self, name)
                for face, name in enumerate(grids["y"]):
                    self._grd_mesh_y[face, :, 1:-1, 1:-1, 1:-1] = gridfile[name][()]
                    _guard_cells_from_data(self._grd_mesh_y[face, :, :, :, :], self)
                    _bound_cells_from_data(self._grd_mesh_y[face, :, :, :, :], self, name)
                for face, name in enumerate(grids["z"]):
                    self._grd_mesh_z[face, :, 1:-1, 1:-1, 1:-1] = gridfile[name][()]
                    _guard_cells_from_data(self._grd_mesh_z[face, :, :, :, :], self)
                    _bound_cells_from_data(self._grd_mesh_z[face, :, :, :, :], self, name)
                for face, name in enumerate(grids["ddx"]):
                    self._grd_mesh_ddx[face, :, 1:-1, 1:-1, 1:-1] = gridfile[name][()]
                    _guard_cells_from_data(self._grd_mesh_ddx[face, :, :, :, :], self)
                    _bound_cells_from_data(self._grd_mesh_ddx[face, :, :, :, :], self, name)
                for face, name in enumerate(grids["ddy"]):
                    self._grd_mesh_ddy[face, :, 1:-1, 1:-1, 1:-1] = gridfile[name][()]
                    _guard_cells_from_data(self._grd_mesh_ddy[face, :, :, :, :], self)
                    _bound_cells_from_data(self._grd_mesh_ddy[face, :, :, :, :], self, name)
                for face, name in enumerate(grids["ddz"]):
                    self._grd_mesh_ddz[face, :, 1:-1, 1:-1, 1:-1] = gridfile[name][()]
                    _guard_cells_from_data(self._grd_mesh_ddz[face, :, :, :, :], self)
                    _bound_cells_from_data(self._grd_mesh_ddz[face, :, :, :, :], self, name)

        elif self.grd_type == 'paramesh':
            pass # paramesh grid handling operations
   
        else:
            pass # other mesh grid handling operations


        # initialize list of class member names holding the data
        setattr(self, '_attributes', {
            'blk_num', 'blk_num_x', 'blk_num_y', 'blk_num_z', 'blk_size_x',
            'blk_size_y', 'blk_size_z', 'blk_guards', 'blk_coords', 'blk_bndbox',
            'grd_type', 'grd_dim', 'grd_mesh_x', 'grd_mesh_y', 'grd_mesh_z',
            'grd_mesh_ddx', 'grd_mesh_ddy', 'grd_mesh_ddz'})

    def __str__(self) -> str:
        fields = ['grd_type', 'blk_num', 'blk_num_x', 'blk_num_y', 'blk_num_z',
                  'blk_size_x', 'blk_size_y', 'blk_size_z']
        return super()._str_attrs(fields)

    @property
    def grd_mesh_x(self):
        """
        Method to provide the property grd_mesh_x.

        Returns:
            Mesh data for block data, in x direction 
        """
        g = int(self.blk_guards / 2)
        return self._grd_mesh_x[:, :, g:-g, g:-g, g:-g]

    @grd_mesh_x.setter
    def grd_mesh_x(self, value):
        g = int(self.blk_guards / 2)
        self._grd_mesh_x[:, :, g:-g, g:-g, g:-g] = value

    @property
    def grd_mesh_y(self):
        """
        Method to provide the property grd_mesh_y.

        Returns:
            Mesh data for block data, in y direction 
        """
        g = int(self.blk_guards / 2)
        return self._grd_mesh_y[:, :, g:-g, g:-g, g:-g]

    @grd_mesh_y.setter
    def grd_mesh_y(self, value):
        g = int(self.blk_guards / 2)
        self._grd_mesh_y[:, :, g:-g, g:-g, g:-g] = value

    @property
    def grd_mesh_z(self):
        """
        Method to provide the property grd_mesh_z.

        Returns:
            Mesh data for block data, in z direction
        """
        g = int(self.blk_guards / 2)
        return self._grd_mesh_z[:, :, g:-g, g:-g, g:-g]

    @grd_mesh_z.setter
    def grd_mesh_z(self, value):
        g = int(self.blk_guards / 2)
        self._grd_mesh_z[:, :, g:-g, g:-g, g:-g] = value

    @property
    def grd_mesh_ddx(self):
        """
        Method to provide the property grd_mesh_ddx.

        Returns:
            Metric mesh data for block data, in x direction
        """
        g = int(self.blk_guards / 2)
        return self._grd_mesh_ddx[:, :, g:-g, g:-g, g:-g]

    @grd_mesh_ddx.setter
    def grd_mesh_ddx(self, value):
        g = int(self.blk_guards / 2)
        self._grd_mesh_ddx[:, :, g:-g, g:-g, g:-g] = value

    @property
    def grd_mesh_ddy(self):
        """
        Method to provide the property grd_mesh_ddy.

        Returns:
            Metric mesh data for block data, in y direction
        """
        g = int(self.blk_guards / 2)
        return self._grd_mesh_ddy[:, :, g:-g, g:-g, g:-g]

    @grd_mesh_ddy.setter
    def grd_mesh_ddy(self, value):
        g = int(self.blk_guards / 2)
        self._grd_mesh_ddy[:, :, g:-g, g:-g, g:-g] = value

    @property
    def grd_mesh_ddz(self):
        """
        Method to provide the property grd_mesh_ddz.

        Returns:
            Metric mesh data for block data, in z direction
        """
        g = int(self.blk_guards / 2)
        return self._grd_mesh_ddz[:, :, g:-g, g:-g, g:-g]

    @grd_mesh_ddz.setter
    def grd_mesh_ddz(self, value):
        g = int(self.blk_guards / 2)
        self._grd_mesh_ddz[:, :, g:-g, g:-g, g:-g] = value

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
    _guards: int = field(repr=False, init=False, compare=False)

    @staticmethod
    def _fill_guard(data, geometry):
        _guard_cells_from_data(data, geometry)
                
    @staticmethod
    def _fill_bound(data, geometry, field):
        _bound_cells_from_data(data, geometry, field)

    # pylint: disable=arguments-differ
    def _init_process(self, file: h5py.File, code: str, form: str, geometry: GeometryData) -> None:

        # pull relavent data from hdf5 file object
        real_scalars: List[Tuple[bytes, float]] = list(file['real scalars'])
        unknown_names: List[bytes] = list(file['unknown names'][:, 0])

        # initialize number of guard cells for each block per direction
        self._guards = geometry.blk_guards

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

        # initialize field names and shapes (for face centered data)
        g = int(self._guards / 2)
        vel_grp = {'fcx2', 'fcy2', 'fcz2'}
        vel_map = {'fcx2' : [2*g, 2*g, 1*g],
                   'fcy2' : [2*g, 1*g, 2*g],
                   'fcz2' : [1*g, 2*g, 2*g]}

        # initialize field data members (for cell centered data)
        for group in self._groups:

            # allow for guard data at axis upper extent
            shape = file[group].shape
            if group not in {'fcx2', 'fcy2', 'fcz2'}:
                shape = tuple([shape[0]]) + tuple([length + 2 for length in shape[1:]])
            else:
                shape = tuple([shape[0]]) + tuple([length +
                                                   vel_map[group][i] for i, length in enumerate(shape[1:])])

            # read dataset from file
            data = numpy.zeros(shape, dtype=numpy.dtype(float))
            if group not in vel_grp:
                data[:, g:-g, g:-g, g:-g] = file[group][()]
            elif group == 'fcx2':
                data[:, g:-g, g:-g, g-1:-g] = file[group][()]
            elif group == 'fcy2':
                data[:, g:-g, g-1:-g, g:-g] = file[group][()]
            elif group == 'fcz2':
                data[:, g-1:-g, g:-g, g:-g] = file[group][()]
            else:
                raise Exception(f'requested field not found!')

            # fill guard and bound cell data
            FieldData._fill_guard(data, geometry)
            #FieldData._fill_bound(data, geometry, group)

            # attach dataset to FieldData instance
            setattr(self, '_' + group, data)
            setattr(FieldData, group, property(partial(FieldData._get_attr, attr='_' + group),
                                               partial(FieldData._set_attr, attr='_' + group)))

        # initialize list of class member names holding the data
        setattr(self, '_attributes', {group for group in self._groups})

    def _set_attr(self, value, attr):
        g = int(self._guards / 2)
        getattr(self, attr)[:, g:-g, g:-g, g:-g] = value

    def _get_attr(self, attr):
        g = int(self._guards / 2)
        return getattr(self, attr)[:, g:-g, g:-g, g:-g]

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
