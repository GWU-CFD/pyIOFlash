"""
Python Module for providing methods to import and process FLASH4 HDF5 plot and chk files

   -- Revision 0.15 (alpha)

   -- needed feature list --
       [-] add block dimensions to GeometryData
       [-] add refine level to GeometryData
       [ ] others ???

   -- major revision roadmap --
       [-] initial functioning FLASH4 parser
       [ ] initial functioning Plot Utility
       [ ] initial multiprocessing functionality
       [ ] add parser for EDDY6 code
       [ ] others ???

 
    -- USAGE --

  |  from pyio import SimulationData
  |
  |  data = SimulationData.from_list(range(20), path='test/', header='INS_Rayleigh_Benard_hdf5_plt_cnt_')
  |
  |  data.fields[20.0 : 60.0 : 2]['temp', 'pres'][:, :, :, :, :]
                 ## [slicing based     ||                    ||
                 ##  on time or        ||                    ||
                 ##  file numbers]     ||                    ||
                                       ## [field names or    ||
                                       ##  plotfile parms]   ||
                                                             ## [time : block : z, y, x]

    -- SimulationData returns an object which contains the folowing data objects:
            geometry    -- geometry information
            fields      -- simulation unknowns
            scalars     -- (time, dt, nstep, nbegin)
            dynamics    -- remaining hdf5 plt file (time varying)
            statics     -- remaining hdf5 plt file (steady w/ time)

"""


from contextlib import contextmanager
from abc import ABC as AbstractBase, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field, InitVar
from typing import Any, Tuple, List, Set, Dict, Iterable, Callable, Union

import numpy
import h5py

def _first_true(iterable: Iterable, predictor: Callable[..., bool]) -> Any:
    """Returns the first true value in the iterable according to predictor."""
    return next(filter(predictor, iterable))

def _filter_transpose(source: List[Any], names: Iterable[str]) -> List[List[Any]]:
    """Returns a list of named members extracted from a 
    collection object (source) based on a list (names)"""
    try:
        return [list(map(lambda obj: getattr(obj, name), source)) for name in names]
    except AttributeError:
        return [list(map(lambda obj: obj[name], source)) for name in names]

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
            x = numpy.linspace(bndbox[0][0], bndbox[0][1], sz_box[0], True)
            y = numpy.linspace(bndbox[1][0], bndbox[1][1], sz_box[1], True)
            z = numpy.linspace(bndbox[2][0], bndbox[2][1], sz_box[2], True)
            
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

class Filterable:
    """Provides an abstraction to implement slicing of list-of-lists elements using [] syntax;
    including multi-key slicing and lookup vice slicing if applicable"""
    def __getitem__(self, keys: slice) -> Any:
        if hasattr(keys, '__iter__') and (
            isinstance(keys[0], str) or isinstance(keys[0], tuple)):
            return [[item[key] for item in self] for key in keys]
        else:
            return [item[keys] for item in self]

class BaseTransposable(AbstractBase):
    """Abstract base class for implementing a wrapper class used to provide 
    filtered, transpose-like viewing of multiple named members of a derived class, 
    wrapped in some fashion (must implement a _return_map method); super().self must be iterable
    """
    @abstractmethod   
    def _return_map(self, source):
        raise NotImplementedError(f'{self.__class__.__name__} does not implement an _return_map method') 
    
    def __getitem__(self, key: Iterable) -> Union[List[Any], Any]:      
        try:
            return super().__getitem__(key)
        except TypeError:
            if isinstance(key, str):
                rtn = _filter_transpose(self, [key])        
            elif hasattr(key, '__iter__'):
                rtn = _filter_transpose(self, list(key))
            else:
                raise TypeError(f'Indices of {self.__class__.__name__} objects must be integers, slices, or iterable')
            return type('FilterableList', (Filterable, list), {})(self._return_map(rtn))    

class TransposableAsArray(BaseTransposable):
    """Wrapper class used to provide filtered,transpose-like viewing of multiple named members of 
    a derived collection-like class, wrapped in numpy arrays; super().self must be iterable
    
    -- Create an instance with the following syntax --
    ta = type('MyObject', (TransposeAsArray, ...), {})(...)
    
    
    -- Use the returned instance as follows --
    ta[str, ...]            Returns a list-like object containing numpy arrays along the key-axis 
                            for each (outer list); matching the args to members of collection 
                            elements, operation is approxamatly a filtered, transpose with wrapping
                            
    ta[str, ...][:, ...]    The returned object has the added property that slicing may be applied 
                            to each inner numpy array elements simultaniously with the [] notation 
    
    """    
    def _return_map(self, source):
        return map(numpy.array, source) 

class TransposableAsSingle(BaseTransposable):
    """Wrapper class used to provide filtered viewing of a single named member of a derived class, 
    wrapped as a list of items; super().self must be iterable
    
    -- Create an instance with the following syntax --
    ta = type('MyObject', (TransposeAsArray, ...), {})(...)
    
    
    -- Use the returned instance as follows --
    ta[str]                 Returns a list-like object containing a member's values along the key-axis; 
                            matching the arg to amember of collection elements, 
                            operation is approximatly a filtered, transpose with wrapping
                            
    ta[str][: or str]       The returned object has the added property that lookup or slicing, as 
                            appropriate, may be applied to the list elements with the [] notation 
    
    """    
    def _return_map(self, source):
        return source[0]

class SortedDict:
    """
    Mutable collection of unique item (that must implement an item.key), providing the functionality
      (1) Sequence-like fast indexable access; [i or slice]
      (2) Dictionary-like fast mappable access; [key of slice]
      (3) Collection remains ordered (invariant) based on each element's [item.key] value
      (4) Insert/remove (slow), append (fast if sorted), pop (fast)
      (5) Collection can only contain unique (w.r.t item.key) items


    --------------------------------------------------------------------------------------
    ----                        High Level Usage Documentation                        ----
    --------------------------------------------------------------------------------------
    series = SortedDict(iter)               creates collection using an iterable
    
    series = SortedDict.with_sorted(iter)   creates collection using a pre-sorted iterable
                                   - possible undefined behavior if iterable is not sorted

    ---------------------------------------------------------------
    --    Common Sequence/Mapping behavior                       --
    ---------------------------------------------------------------
    clear()         Removes all the items from the collection
    copy()          Return a shallow copy of the collection

    sc = sa + sb    Returns the concatenation of rhs collections
    sa += sb        Extends the collection with the contents of sb   
    sa == sb        Returns logical (equal) comparison of collection's items;
                        also implements analogous <, <=, !=, >=, and >

    ---------------------------------------------------------------
    --   Sequence-like behavior                                  --
    ---------------------------------------------------------------
    append(x)        Add an item to the end of the collection
    extend(t)        Extends collection with the contents of t
    insert(i, x)     Inserts x into collection at the index given by i 
    pop([i])         Returns the item at i and removes it from the collection; default i = -1
    sort()           Forces a sort of the collection; otherwise defered until the last possible moment


    ---------------------------------------------------------------
    --   Mapping-like behavior                                   --    
    ---------------------------------------------------------------
    get(key[, default])     Return the value for key if key is in the dictionary, else default; 
    items()                 Return an iterator of the collection's items ((key, value) pairs)
    keys()                  Returns an iterator of the collection's keys
    pop([key, default])     Returns the key and removes it from the collection, if present, else defualt; 
                                if no key, default is last element
    popitem()               Remove and return a (key, value) pair from collection; returned in LIFO order    
    setdefault(default)     If default.key is in collection, return value; otherwise insert/return default
    update([other])         Update the collection with the key/value pairs from other; overwriting keys
    values()                Returns an iterator of the collections values
    
    ** note type(key) is float  /// future update to generalize type ///

    ---------------------------------------------------------------
    --   Additional notes w.r.t. access using the [] operator    --
    ---------------------------------------------------------------
    sd[index] = value           Assignment will overwrite item @ index and sort if necessary;
                                    exeption if value.key nonunique and series[index].key != value.key
    sd[key] = value             Assignment will insert or overwrite existing item w/ equal key             

    sd[low : high : set] 
        (1) Works (nearly) identical to list if args are integers
        (2) Returns a consitant skice matching a closed interval if args are floats, set must be int 
        
    Finally, it is again noted that while the sorted property of SortedDict is invariant; 
    the sorting operation is defered until the last possible moment (e.g., access, index based assignment)
        
    """    
    _data: List[Any]
    _keys: Dict[float, int]
    _is_valid: bool
        
    def __add__(self, other: Iterable) -> 'SortedDict':
        self.__make_valid__()
        if not _set_is_unique(self._keys, {item.key for item in other}):
            raise ValueError(f'Nonunique key, location mismatch; identical keys found in collection') 
        self._is_valid = False
        return self.__class__(self._data + other)
    
    def __contains__(self, other: Iterable) -> bool:
        self.__make_valid__()
        return other in self._data
    
    def __eq__(self, other: Iterable) -> bool:
        self.__make_valid__()
        if isinstance(other, SortedDict):
            return self._data == other._data
        else:
            return self._data == SortedDict(other)._data   
        
    def __ge__(self, other: Iterable) -> bool:
        self.__make_valid__()
        if isinstance(other, SortedDict):
            return self._data >= other._data
        else:
            return self._data >= SortedDict(other)._data 
    
    def __getitem__(self, key: Union[int, float, slice]) -> Union[List[Any], Any]:
        start: int
        stop: int        
        self.__make_valid__()
        
        # dictionary-like behavior
        if isinstance(key, float):
            return self.__class__.from_sorted([self._data[self._keys[key]]])
        
        # list-like behavior
        elif isinstance(key, int):
            return self.__class__.from_sorted([self._data[key]])
        
        # slicing behavior; both list and dict** like
        elif isinstance(key, slice):
            
            # dictionary-like behavior; if a dict were 'slice-able'
            if isinstance(key.start, float) or isinstance(key.stop, float):
                if key.start is None:
                    start = 0
                else:
                    start = _first_true(self._keys.items(), lambda x: key.start <= x[0])[1]
                if key.stop is None:
                    stop = len(self._data)
                else:
                    stop = _first_true(reversed(self._keys.items()), lambda x: x[0] <= key.stop)[1] + 1
                return self.__class__.from_sorted(self._data[start : stop : key.step])
            
            # list-like behavior
            else:
                return self.__class__.from_sorted(self._data[key])
        
        else:
            raise TypeError(f'SeriesData indices must be integers, floats, or slices')
            
    def __gt__(self, other: Iterable) -> bool:
        self.__make_valid__()
        if isinstance(other, SortedDict):
            return self._data > other._data
        else:
            return self._data > SortedDict(other)._data             
            
    def __iadd__(self, other: Iterable) -> 'SortedDict':
        self.extend(other)
        return self            
        
    def __init__(self, iterable: Iterable) -> None:
        self._data = list(iterable)
        self._is_valid = False
        
    def __iter__(self):
        self.__make_valid__()
        yield from self._data

    def __le__(self, other: Iterable) -> bool:
        self.__make_valid__()
        if isinstance(other, SortedDict):
            return self._data <= other._data
        else:
            return self._data <= SortedDict(other)._data          
        
    def __len__(self):
        return len(self._data)
    
    def __lt__(self, other: Iterable) -> bool:
        self.__make_valid__()
        if isinstance(other, SortedDict):
            return self._data < other._data
        else:
            return self._data < SortedDict(other)._data    
        
    def __make_keys__(self) -> None:
        self._keys = OrderedDict([(item.key, i) for i, item in enumerate(self._data)])
        
    def __make_valid__(self, force: bool = False) -> None:
        if force or not self._is_valid:
            #print('-- making valid --')
            self._data.sort(key=lambda item: item.key)
            self.__make_keys__()
            self._is_valid = True 

    def __ne__(self, other: Iterable) -> bool:
        result = self.__eq__(other)
        if result is not NotImplemented:
            return not result
        return NotImplemented            
        
    def __repr__(self) -> str:
        self.__make_valid__()
        return 'SortedDict[' + ',\n '.join(item.__repr__() for item in self._data) + ']'  
    
    def __reversed__(self):
        self.__make_valid__()
        yield from self._data[::-1]
            
    def __setitem__(self, key: Union[int, float, slice, Iterable], value: Union[List[Any], Any]) -> None:
        start: int
        stop: int
        self.__make_valid__()
                
        # mappable-like behavior
        if isinstance(key, float):    # // future type generalization //
            if value.key in self._keys:
                self._data[self._keys[key]] = value
            else:
                self.append(value)                
        
        # sequence-like behavior
        elif isinstance(key, int):            
            if value.key in self._keys and key != self._keys[value.key]:
                raise ValueError(f'Nonunique key, location mismatch; identical key @ {self._keys[value.key]}')    
            else:
                self._data[key] = value
                self._is_valid = False
                
        elif isinstance(key, slice):
            
            # mappable-like behavior; ** if dict allowed slice-like replacement
            if isinstance(key.start, float):
                start = _first_true(self._keys.items(), lambda x: key.start <= x[0])[1]
                stop = _first_true(reversed(self._keys.items()), lambda x: x[0] <= key.stop)[1] + 1           
            
            # sequence-like behavior
            else:
                start = key.start
                stop = key.stop           
            
            masked: set = {i.key for i in self._data[:start]} | {i.key for i in self._data[stop:]}   
            if _set_is_unique(masked, {item.key for item in value}):
                self._data[start : stop] = value
                self._is_valid = False                
            else:
                raise ValueError(f'Nonunique key, location mismatch; identical keys found outside of slice')
        
        else:
            raise TypeError(f'SortedDict indices must be integers, floats, slices, or iterables')
        
    def __str__(self) -> str:
        self.__make_valid__()
        return '[' + ',\n '.join(item.__str__() for item in self._data) + ']'
    
    def append(self, value: Any) -> None:
        self.__make_valid__()
        if value.key in self._keys:
            raise ValueError(f'Nonunique key provided; {value.key} @ index {self._keys[value.key]}')             
        if len(self._data) > 0 and value.key < next(reversed(self._keys)):
            self._is_valid = False
        self._data.append(value)
        self._keys[value.key] = len(self._data) - 1
    
    def clear(self) -> None:
        self._data.clear()
        self._keys = OrderedDict({})
        self._is_valid = True
        
    def copy(self) -> None:
        self.__make_valid__()
        return self.__class__(self._data)
    
    def extend(self, iterable: Iterable):
        self.__make_valid__()
        if not _set_is_unique(self._keys, {item.key for item in iterable}):
            raise ValueError(f'Nonunique key, location mismatch; identical keys found in collection')            
        self._data.extend(iterable)
        self._is_valid = False
    
    @classmethod
    def from_sorted(cls, iterable: Iterable):
        instance = cls(iterable)
        instance.__make_keys__()
        instance._is_valid = True
        return instance
    
    def get(self, key: float, defualt: Any = None) -> Any:
        self.__make_valid__()
        try:
            return self._data[self._keys[key]]
        except KeyError:
            return defualt
            
    def insert(self, key: int, value: Any) -> None:
        self.__make_valid__()
        self[key] = value
            
    def items(self):
        self.__make_valid__()
        yield from {item[0] : self._data[item[1]] for item in self._keys.items()}.items()
            
    def keys(self):
        self.__make_valid__()
        yield from self._keys.keys()
        
    def pop(self, key: Union[int, float] = -1, default: float = None) -> Union[Any, float]:
        self.__make_valid__()
        if isinstance(key, int):
            self._is_valid = False
            return self._data.pop(key)
        if key in self._keys:
            self._is_valid = False
            self._data.pop(self._keys[key])
            return key 
        if default is not None:
            return default            
        raise KeyError(f'Key is not in collection and no default provided') 
        
    def popitem(self) -> Tuple[float, Any]:
        self.__make_valid__()
        if len(self._data) == 0:
            raise KeyError(f'Method popitem called on an empty collection') 
        return (self._keys.popitem()[0], self._data.pop())
    
    def setdefault(self, default: Any) -> Any:
        try:      
            self.append(default)
            return default
        except ValueError:
            return self._data[self._keys[default.key]]
    
    def sort(self) -> None:
        self.__make_valid__()
        
    def tolist(self) -> list:
        self.__make_valid__()
        return list(self)
        
    def update(self, iterable: Iterable) -> None:
        self.__make_valid__()
        for item in iterable:
            self[item.key] = item
        
    def values(self):
        self.__make_valid__()
        return self.__iter__()

@dataclass
class NameData:
    """
    Storage class for containing file names as a list for hdf5 files to be processed;
    the list, NameData.files, is automatically generated based on keyword arguments
    """
    numbers: Iterable = field(repr=False, init=True, compare=False)
    directory: str = field(repr=False, init=True, compare=False, default='')
    header: str = field(repr=False, init=True, compare=False, default='')
    footer: str = field(repr=False, init=True, compare=False, default='')
    extention: str = field(repr=False, init=True, compare=False, default='')
    format: str = field(repr=False, init=True, compare=False, default='04d')
    length: int = field(repr=False, init=False, compare=False)
    names: List[str] = field(repr=True, init=False, compare=True)

    def __post_init__(self):
        self.names = [self.directory + self.header +  f'{n:{self.format}}' +
                      self.footer + self.extention for n in self.numbers]
        self.length = len(self.names)
        
    @classmethod
    def from_strings(cls, names, **kwargs):
        kwargs['format'] = ''
        instance = cls(names, **kwargs)
        return instance
    
    @classmethod
    def from_name(cls, name, **kwargs):
        return cls.from_strings([name], **kwargs)

class SimulationData:
    files: NameData
    code: str
    form: str   
    geometry: SortedDict
    fields: SortedDict
    scalars: SortedDict
    dynamics: SortedDict
    statics: StaticData    
        
    def __init__(self, files: NameData, *, form : str = None, code : str = None):
        """ ... """
        
        # initialize filenames
        self.files = files
        
        # initialize code and file type
        if form is None:
            self.form = 'plt'        
        if code is None:
            self.code = 'flash'

        # initialize empty containers
        self.geometry = type('TransposableAsArray_SortedDict', (TransposableAsArray, SortedDict), {})([])
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
    def from_list(cls, numbers: List[int], *, format: str = None,
                 path: str = None, header: str = None, footer: str = None, ext: str = None,
                 form: str = None, code: str = None):

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
        if format is not None:
            options['format'] = format
            
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
            ('integer scalars', StaticData._pass_label), 
            ('logical scalars', StaticData._pass_label),
            ('real scalars', StaticData._pass_label), 
            ('string scalars', lambda label: _reduce_str(StaticData._decode_label(label)))]
            
        # format = [(group, function to process dataset values), ...]    
        def_statics: List[Tuple[str, Callable]] = [
            ('integer runtime parameters', StaticData._pass_label), 
            ('logical runtime parameters', StaticData._pass_label),
            ('real runtime parameters', StaticData._pass_label), 
            ('string runtime parameters', lambda label: _reduce_str(StaticData._decode_label(label))), 
            ('sim info', lambda label: _reduce_str(StaticData._decode_label(label), ' '))]

        # process first FLASH4 hdf5 file
        with open_hdf5(self.files.names[0], 'r') as file:
            setattr(self, 'statics', StaticData(file, self.code, self.form, def_statics)) 
            
        # process FLASH4 hdf5 files
        for num, name in enumerate(self.files.names):
            with open_hdf5(name, 'r') as file:
                self.geometry.append(GeometryData(file, self.code, self.form))
                self.fields.append(FieldData(file, self.code, self.form))
                self.scalars.append(ScalarData(file, self.code, self.form, def_scalars))
                self.dynamics.append(StaticData(file, self.code, self.form, def_dynamics))
 