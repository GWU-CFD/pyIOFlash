"""

This module defines the geometry custom type necessary for the pyio package.

The GeometryData derived type is implemented as data classes for simplicity and
lack of code duplication.

This module currently defines the following type and therefore data to be read from
the hdf5 output files catagorically:

    GeometryData

Todo:
    * Provide differed or Just-In-Time block data loading for large simulations

"""
from dataclasses import dataclass, field, InitVar
from typing import Tuple, List, Dict

import numpy
import h5py

from pyioflash.simulation.types import _BaseData
from pyioflash.simulation.utility import _first_true, open_hdf5
from pyioflash.simulation.support import _guard_cells_from_data, _bound_cells_from_data

@dataclass
class GeometryData(_BaseData):
    """
    GeometryData is a derived class implementing the functionality to
    read the relavent geometry data contained in the hdf5 output file.

    Attributes:
        blk_num: total number of blocks in the simulation
        blk_num_x: number of blocks in x direction
        blk_num_y: number of blocks in y direction
        blk_num_z: number of blocks in z direction
        blk_size_x: simulation points of each block in x direction
        blk_size_y: simulation points of each block in y direction
        blk_size_z: simulation points of each block in z direction
        blk_guards: guard cell points of each block in each direction
        blk_coords: coordinates of each block center
        blk_bndbox: bounding box coordinates of each block
        blk_tree_str: tree structure containing block neighbors, parents, and children
        blk_neighbors: list of neighbors for each block
        grd_type: type of grid in the simulation (e.g., uniform or regular)
        grd_dim: dimentionality of the simulation (e.g., 2d or 3d)
        grd_bndbox: bouding box coordinates of the simulation
        grd_mesh_x: mesh data for block data, in x direction
        grd_mesh_x_max: max of mesh data, in x direction
        grd_mesh_x_min: min of mesh data, in x direction
        grd_mesh_y: mesh data for block data, in y direction
        grd_mesh_y_max: max of mesh data, in y direction
        grd_mesh_y_min: min of mesh data, in y direction
        grd_mesh_z: mesh data for block data, in z direction
        grd_mesh_z_max: max of mesh data, in z direction
        grd_mesh_z_min: min of mesh data, in z direction
        grd_mesh_ddx: mesh metric data for block data, in x direction
        grd_mesh_ddx_max: max of mesh metric data, in x direction
        grd_mesh_ddx_min: min of mesh metric data, in x direction
        grd_mesh_ddy: mesh metric data for block data, in y direction
        grd_mesh_ddy_max: max of mesh metric data, in y direction
        grd_mesh_ddy_min: min of mesh metric data, in y direction
        grd_mesh_ddz: mesh metric data for block data, in z direction
        grd_mesh_ddz_max: max of mesh metric data, in z direction
        grd_mesh_ddz_min: min of mesh metric data, in z direction

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
    _grd_mesh_ddx: numpy.ndarray = field(repr=False, init=False, compare=False)
    _grd_mesh_ddy: numpy.ndarray = field(repr=False, init=False, compare=False)
    _grd_mesh_ddz: numpy.ndarray = field(repr=False, init=False, compare=False)

    @staticmethod
    def _get_neighbors(tree, dim):
        names = ["left", "right", "front", "back", "up", "down"]
        return [{names[face] : block for face, block in enumerate(struct[:2*dim]) 
                 if block >= 0} 
                for struct in tree]

    @staticmethod
    def _fill_guard(data, geometry, field):
        _guard_cells_from_data(data, geometry)
        _bound_cells_from_data(data, geometry, field) 

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
        self.blk_guards = 2 # do not need to use more than 1 currently
        g = int(self.blk_guards / 2)

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

        # create mesh grid and metric extreme value storage arrays
        self._grd_mesh_x_max = numpy.zeros(3, dtype=numpy.dtype(float))
        self._grd_mesh_x_min = numpy.zeros(3, dtype=numpy.dtype(float))        
        self._grd_mesh_y_max = numpy.zeros(3, dtype=numpy.dtype(float))
        self._grd_mesh_y_min = numpy.zeros(3, dtype=numpy.dtype(float))  
        self._grd_mesh_z_max = numpy.zeros(3, dtype=numpy.dtype(float))
        self._grd_mesh_z_min = numpy.zeros(3, dtype=numpy.dtype(float))
        self._grd_mesh_ddx_max = numpy.zeros(3, dtype=numpy.dtype(float))
        self._grd_mesh_ddx_min = numpy.zeros(3, dtype=numpy.dtype(float))        
        self._grd_mesh_ddy_max = numpy.zeros(3, dtype=numpy.dtype(float))
        self._grd_mesh_ddy_min = numpy.zeros(3, dtype=numpy.dtype(float))  
        self._grd_mesh_ddz_max = numpy.zeros(3, dtype=numpy.dtype(float))
        self._grd_mesh_ddz_min = numpy.zeros(3, dtype=numpy.dtype(float)) 

        # initialize mesh grids for cell centered and face fields
        # FUTURE -- only load on demand
        if self.grd_type == 'uniform':
            print('Building Uniform Grid')

            grds = self.blk_guards
            size = (self.blk_size_x, self.blk_size_y, self.blk_size_z)
            gsft = (self.blk_guards - 1) / 2

            for face, fsft in enumerate({-1/2, 0, 1/2}):

                # build mesh grids and metrics by block
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

        # read from file the mesh grids for cell centered and face fields
        # FUTURE -- only load on demand
        elif self.grd_type == 'regular':
            print('Reading Grid from File')

            grids = {"x" : ["xxxl", "xxxc", "xxxr"],
                     "y" : ["yyyl", "yyyc", "yyyr"],
                     "z" : ["zzzl", "zzzc", "zzzr"],
                     "ddx" : ["ddxl", "ddxc", "ddxr"],
                     "ddy" : ["ddyl", "ddyc", "ddyr"],
                     "ddz" : ["ddzl", "ddzc", "ddzr"]}

            with open_hdf5(gridfilename, 'r') as gridfile:

                for face, name in enumerate(grids["x"]):
                    self._grd_mesh_x[face, :, g:-g, g:-g, g:-g] = gridfile[name][()]
                    GeometryData._fill_guard(self._grd_mesh_x[face, :, :, :, :], self, name)

                for face, name in enumerate(grids["y"]):
                    self._grd_mesh_y[face, :, g:-g, g:-g, g:-g] = gridfile[name][()]
                    GeometryData._fill_guard(self._grd_mesh_y[face, :, :, :, :], self, name)

                for face, name in enumerate(grids["z"]):
                    self._grd_mesh_z[face, :, g:-g, g:-g, g:-g] = gridfile[name][()]
                    GeometryData._fill_guard(self._grd_mesh_z[face, :, :, :, :], self, name)

                for face, name in enumerate(grids["ddx"]):
                    self._grd_mesh_ddx[face, :, g:-g, g:-g, g:-g] = gridfile[name][()]
                    GeometryData._fill_guard(self._grd_mesh_ddx[face, :, :, :, :], self, name)

                for face, name in enumerate(grids["ddy"]):
                    self._grd_mesh_ddy[face, :, g:-g, g:-g, g:-g] = gridfile[name][()]
                    GeometryData._fill_guard(self._grd_mesh_ddy[face, :, :, :, :], self, name)

                for face, name in enumerate(grids["ddz"]):
                    self._grd_mesh_ddz[face, :, g:-g, g:-g, g:-g] = gridfile[name][()]
                    GeometryData._fill_guard(self._grd_mesh_ddz[face, :, :, :, :], self, name)

        elif self.grd_type == 'paramesh':
            pass # paramesh grid handling operations
   
        else:
            pass # other mesh grid handling operations

        # initialize grid and metric extreme values
        for mesh in {'grd_mesh_x', 'grd_mesh_y', 'grd_mesh_z',
                     'grd_mesh_ddx', 'grd_mesh_ddy', 'grd_mesh_ddz'}:
            setattr(self, '_' + mesh + '_max', getattr(self, '_' + mesh)[:, :, :, :, :].max((1,2,3,4)) )
            setattr(self, mesh + '_max', getattr(self, '_' + mesh)[:, :, g:-g, g:-g, g:-g].max((1,2,3,4)))
            setattr(self, '_' + mesh + '_min', getattr(self, '_' + mesh)[:, :, :, :, :].min((1,2,3,4)))
            setattr(self, mesh + '_min', getattr(self, '_' + mesh)[:, :, g:-g, g:-g, g:-g].min((1,2,3,4)))

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

