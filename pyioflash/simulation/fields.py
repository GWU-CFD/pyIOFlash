"""

This module defines the field custom type necessary for the pyio package.

The FieldData derived type is implemented as data classes for simplicity and
lack of code duplication.

This module currently defines the following type and therefore data to be read from
the hdf5 output files catagorically:

    FieldData

Todo:
    * Provide differed or Just-In-Time block data loading for large simulations

"""
from dataclasses import dataclass, field, InitVar
from typing import Tuple, List, Set
from functools import partial

import numpy
import h5py

from pyioflash.simulation.types import _BaseData
from pyioflash.simulation.geometry import GeometryData
from pyioflash.simulation.utility import _first_true, _reduce_str
from pyioflash.simulation.support import _guard_cells_from_data, _bound_cells_from_data

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
    def _fill_guard(data, geometry, field):
        _guard_cells_from_data(data, geometry)
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
            FieldData._fill_guard(data, geometry, group)

            # attach dataset to FieldData instance
            setattr(self, '_' + group, data)
            setattr(FieldData, group, property(partial(FieldData._get_attr, attr='_' + group),
                                               partial(FieldData._set_attr, attr='_' + group)))

            # attach extrema of dataset to FieldData instance
            setattr(self, '_' + group + '_max', data.max())
            setattr(self, group + '_max', data[:, g:-g, g:-g, g:-g].max())
            setattr(self, '_' + group + '_min', data.min())
            setattr(self, group + '_min', data[:, g:-g, g:-g, g:-g].min())

        # initialize list of class member names holding the data
        setattr(self, '_attributes', {group for group in self._groups})

    def _set_attr(self, value, attr):
        g = int(self._guards / 2)
        getattr(self, attr)[:, g:-g, g:-g, g:-g] = value

    def _get_attr(self, attr):
        g = int(self._guards / 2)
        return getattr(self, attr)[:, g:-g, g:-g, g:-g]
