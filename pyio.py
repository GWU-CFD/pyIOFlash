"""
Python Module for providing methods to import and process FLASH4 HDF5 plot and chk files

   -- Revision 0.16 (alpha)

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


from abc import ABC as AbstractBase, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field, InitVar
from typing import Any, Tuple, List, Set, Dict, Iterable, Callable, Union

import numpy
import h5py

from pyio_utility import _filter_transpose, _first_true, _set_is_unique, _reduce_str
from pyio_utility import open_hdf5, NameData
from pyio_collections import SortedDict, TransposableAsArray, TransposableAsSingle
from pyio_types import GeometryData, FieldData, ScalarData, StaticData


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
 