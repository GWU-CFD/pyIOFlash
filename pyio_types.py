from abc import ABC as AbstractBase, abstractmethod
from dataclasses import dataclass, field, InitVar
from typing import Any, Tuple, List, Set, Dict, Iterable, Callable, Union

import numpy
import h5py

from pyio_utility import _filter_transpose, _first_true, _set_is_unique, _reduce_str

@dataclass(order=True)
class _BaseData(AbstractBase):
    # mappable member for when composited into a sortable collection object; sorted by
    key: float = field(repr=True, init=False, compare=True)
        
    # initialization arguments; removed after initialization
    file: InitVar[h5py.File]
    form: InitVar[str]
    code: InitVar[str]
        
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
    
    def __post_init__(self, file, code, form) -> None:
        supported: Dict[str, List[str]] = {'flash' : ['plt', 'chk']}
        
        # check for supported codes and formats
        if code not in supported:
            raise Exception(f'Code {code} is not supported; only code = {[*supported]}')
            
        if form not in supported[code]:
            raise Exception(f'File format {form} is not supported; only form = {supported[code]}')

        # process the file based on options
        self._init_process(file, code, form)         

        # verify properly initialized object
        if not hasattr(self, 'key'):
            raise NotImplementedError(f'{type(self)} does not initialize a key member')
        if not hasattr(self, '_attributes'):
            raise NotImplementedError(f'{type(self)} does not initialize a _attributes member')
            
    def __str__(self) -> str:
        return self._str_attrs(self._attributes)
    
    @abstractmethod   
    def _init_process(self, file, code, form) -> None:
        raise NotImplementedError(f'{type(self)} does not implement an _init_process method')
        
    def _str_keys(self) -> str:
        return self._str_base(self._attributes, lambda *args: '')
    
    def _str_attrs(self, attrs: list) -> str:
        return self._str_base(attrs, lambda inst, key: '=' + str(getattr(inst, key)))
    
    def _str_base(self, attrs: list, wrap: Callable) -> str:
        return f'{self.__class__.__name__}(key={self.key:.4f}, ' + ', '.join(
            key + wrap(self, key) for key in attrs) + ')'        
            
    def keys(self) -> set:
        return self._attributes 
    
    def todict(self) -> dict:
        return {key : self[key] for key in self._attributes}        
    
@dataclass
class GeometryData(_BaseData):
    blk_num: int = field(repr=False, init=False, compare=False)
    blk_num_x: int = field(repr=True, init=False, compare=False)
    blk_num_y: int = field(repr=True, init=False, compare=False)
    blk_num_z: int = field(repr=True, init=False, compare=False)
    blk_size_x: int = field(repr=True, init=False, compare=False)
    blk_size_y: int = field(repr=True, init=False, compare=False)
    blk_size_z: int = field(repr=True, init=False, compare=False)
    blk_coords: numpy.ndarray = field(repr=False, init=False, compare=False)
    blk_bndbox: numpy.ndarray = field(repr=False, init=False, compare=False)
    blk_dict_x: Dict[float, int] = field(repr=False, init=False, compare=False)
    blk_dict_y: Dict[float, int] = field(repr=False, init=False, compare=False)
    blk_dict_z: Dict[float, int] = field(repr=False, init=False, compare=False)

    grd_type: str = field(repr=True, init=False, compare=False)
    grd_dim: int = field(repr=True, init=False, compare=False)
    grd_size: int = field(repr=False, init=False, compare=False)
    grd_size_x: int = field(repr=False, init=False, compare=False)
    grd_size_y: int = field(repr=False, init=False, compare=False)
    grd_size_z: int = field(repr=False, init=False, compare=False)
    grd_mesh_x: numpy.ndarray = field(repr=False, init=False, compare=False)
    grd_mesh_y: numpy.ndarray = field(repr=False, init=False, compare=False)
    grd_mesh_z: numpy.ndarray = field(repr=False, init=False, compare=False)            
        
    def __str__(self) -> str:
        fields = ['grd_type', 'blk_num', 'blk_num_x', 'blk_num_y', 'blk_num_z', 
                  'blk_size_x', 'blk_size_y', 'blk_size_z']
        return super()._str_attrs(fields)
    
    def _init_process(self, file, code, form) -> None:        
        # pull relavent data from hdf5 file object
        sim_info: List[Tuple[int, bytes]] = list(file['sim info'])
        coordinates: numpy.ndarray = file['coordinates']
        boundingbox: numpy.ndarray = file['bounding box']
        int_runtime: List[Tuple[bytes, int]] = list(file['integer runtime parameters'])
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

        # initialize coordinates of block centers
        self.blk_coords = numpy.ndarray(coordinates.shape, dtype=numpy.dtype(float))
        self.blk_coords[:, :] = coordinates
        
        # initialize coordinates of block bounding boxes
        self.blk_bndbox = numpy.ndarray(boundingbox.shape, dtype=numpy.dtype(float))
        self.blk_bndbox[:, :, :] = boundingbox

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
            self.blk_num_z = 1
            self.blk_size_x = _first_true(int_scalars, lambda l: 'nxb' in str(l[0]))[1]
            self.blk_size_y = _first_true(int_scalars, lambda l: 'nyb' in str(l[0]))[1]
            self.blk_size_z = _first_true(int_scalars, lambda l: 'nzb' in str(l[0]))[1]

        else:
            pass # other grid handling operations

        # initialize block to grid mapping dictionaries
        self.blk_dict_x = {}
        keys: List[int] = sorted(set(self.blk_coords[:, 0]))
        for key, val in zip(keys, range(self.blk_num_x)):
            self.blk_dict_x[key] = val

        self.blk_dict_y = {}
        keys = sorted(set(self.blk_coords[:, 1]))
        for key, val in zip(keys, range(self.blk_num_y)):
            self.blk_dict_y[key] = val 
          
        self.blk_dict_z = None

        # initialize grid data
        self.grd_size_x = self.blk_num_x * self.blk_size_x
        self.grd_size_y = self.blk_num_y * self.blk_size_y
        self.grd_size_z = self.blk_num_z * self.blk_size_z
        self.grd_size = self.grd_size_x * self.grd_size_y * self.grd_size_z
        
        # intialize mesh grids
        self.grd_mesh_x = numpy.ndarray((self.blk_num, self.blk_size_z, 
                                         self.blk_size_y, self.blk_size_x), dtype=numpy.dtype(float))
        self.grd_mesh_y = numpy.ndarray((self.blk_num, self.blk_size_z, 
                                         self.blk_size_y, self.blk_size_x), dtype=numpy.dtype(float))
        self.grd_mesh_z = numpy.ndarray((self.blk_num, self.blk_size_z, 
                                         self.blk_size_y, self.blk_size_x), dtype=numpy.dtype(float))
        
        for block in range(self.blk_num):
            bndbox = self.blk_bndbox[block]
            sz_box = (self.blk_size_x, self.blk_size_y, self.blk_size_z)
            x = numpy.linspace(bndbox[0][0], bndbox[0][1], sz_box[0], False)
            y = numpy.linspace(bndbox[1][0], bndbox[1][1], sz_box[1], False)
            z = numpy.linspace(bndbox[2][0], bndbox[2][1], sz_box[2], False)
            
            Z, Y, X = numpy.meshgrid(z, y, x, indexing='ij')
            self.grd_mesh_x[block, :, :, :] = X
            self.grd_mesh_y[block, :, :, :] = Y
            self.grd_mesh_z[block, :, :, :] = Z
            
        # initialize list of class member names holding the data 
        setattr(self, '_attributes', {
                'blk_num', 'blk_num_x', 'blk_num_y', 'blk_num_z', 'blk_size_x',
                'blk_size_y', 'blk_size_z', 'blk_coords', 'blk_bndbox', 'blk_dict_x',
                'blk_dict_y', 'blk_dict_z', 'grd_type', 'grd_dim', 'grd_size', 'grd_size_x', 
                'grd_size_y', 'grd_size_z', 'grd_mesh_x', 'grd_mesh_y', 'grd_mesh_z'})
    
@dataclass
class FieldData(_BaseData):
    _groups: Set[str] = field(repr=True, init=False, compare=False)
    
    def _init_process(self, file: h5py.File, code: str, form: str) -> None:
        # pull relavent data from hdf5 file object
        real_scalars: List[Tuple[bytes, float]] = list(file['real scalars'])
        int_scalars: List[Tuple[bytes, int]] = list(file['integer scalars'])
        unknown_names: List[bytes] = list(file['unknown names'][:, 0])
            
        # initialize mappable keys
        self.key = float(_first_true(real_scalars, lambda l: 'time' in str(l[0]))[1])
        
        # initialize grid dimensionality
        grd_dim : int = _first_true(int_scalars, lambda l: 'dimensionality' in str(l[0]))[1]
        
        # initialize named fields
        self._groups = set([k.decode('utf-8') for k in unknown_names])
        if form == 'chk':
            if grd_dim == 3:
                self._groups.update({'fcx2', 'fcy2', 'fcz2'})
            elif grd_dim == 2:
                self._groups.update({'fcx2', 'fcy2'})
            else:
                pass   
            
        # initialize field data members
        for group in self._groups:
            setattr(self, group, file[group][()])
            
        # initialize list of class member names holding the data 
        setattr(self, '_attributes', {group for group in self._groups}) 
    
@dataclass
class ScalarData(_BaseData):
    # specification of parameters used to import scalars from hdf5 file;
    # of the format -- [(group, dataset, type), ...]
    _groups: List[Tuple[str, str, str]] = field(repr=True, init=True, compare=False)
    
    def __str__(self) -> str:
        return super()._str_keys() 
    
    def _init_process(self, file: h5py.File, code: str, form: str) -> None:
        # pull relavent data from hdf5 file object
        real_scalars: List[Tuple[bytes, float]] = list(file['real scalars'])
            
        # initialize mappable keys
        self.key = float(_first_true(real_scalars, lambda l: 'time' in str(l[0]))[1])        
        
         # initialize field data members
        for group, dataset, *name in self._groups:
            if name == []:
                name = [dataset]
            setattr(self, *name, _first_true(list(file[group]), lambda l: dataset in str(l[0]))[1])
            
        # initialize list of class member names holding the data 
        setattr(self, '_attributes', {group[-1] for group in self._groups})          
    
@dataclass
class StaticData(_BaseData):
    # specification of parameters used to import static data from a hdf5;
    # of the format -- [group ...]
    _groups: List[Tuple[str, Callable]] = field(repr=True, init=True, compare=False)
                    
    def __str__(self) -> str:
        return super()._str_keys()
    
    @staticmethod
    def _decode_label(label: Union[bytes, str]) -> str:
        try:
            return label.decode('utf-8')
        except AttributeError:
            return str(label)
   
    def _init_process(self, file: h5py.File, code: str, form: str) -> None:
        # pull relavent data from hdf5 file object
        real_scalars: List[Tuple[bytes, float]] = list(file['real scalars'])
            
        # initialize mappable keys
        self.key = float(_first_true(real_scalars, lambda l: 'time' in str(l[0]))[1])        
        
         # initialize class members by hdf5 file groups
        for group, wrap in self._groups:
            setattr(self, _reduce_str(group), {_reduce_str(self._decode_label(dataset[0])) : 
                                               wrap(dataset[1]) for dataset in file[group]})
        
        # initialize list of class member names holding the data 
        setattr(self, '_attributes', {_reduce_str(group[0]) for group in self._groups})
       
    @staticmethod
    def _pass_label(label: str) -> str:
        return label 
