"""

This module defines the energy calculation methods of the post-processing
subpackage of the pyioflash lbrary; part of the 'source' set of routines.

This module currently defines the following methods:

    thermal --  --  --  -> thermal energy
    kinetic --  --  --  -> total instantanious kinetic energy
    kinetic_mean    --  -> mean (time averaged) kinetic energy
    kinetic_turbulant   -> turbulant instantanious kinetic energy

Todo:

"""
from typing import Optional, TYPE_CHECKING 

from pyioflash.postprocess.utility import _interpolate_ftc
from pyioflash.postprocess.elements import integral
from pyioflash.postprocess.analysis import series

if TYPE_CHECKING:
    from pyioflash.simulation.data import SimulationData
    from pyioflash.postprocess.utility import Type_Step, Type_Field, Type_Index


def thermal(data: 'SimulationData', step: 'Type_Step' = -1, *,
            scale: Optional[float] = None, index = Optional['Type_Index'] = None,
            keepdims: bool = True) -> 'Type_Field':
    """
    Provides a method for calculation of the thermal energy by 
    consuming a SimulationData object; must have a 'temp' attribute in the
    SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        step: time-like specification for which to process data, the key (optional)
        scale: used to convert returned quantity to dimensional units (optional)
        index: used for custom slicing operation; should be (blks, k, j, i) (optional)
        keepdims: retain unused dimensions for broadcasting, else drop them (optional)

    Note:
        The thermal energy is computed according to the formula

                E(t)~ijk~ = T(t)~ijk~
                
                *where        t = step,         step is float*
                           *  t = times[step],  step is int*
                           *ijk = all cells*

        The returned quantity is on the cell centered grid

    Todo:
        Need to implement dimensionality

    """
    # need the dimensionality
    dimension = data.geometry.grd_dim

    # need to define slicing operators based on dims
    if index is None:
        i_all = slice(None)
        index = (i_all, ) * 4 if (keepdims or dimension == 3) else (i_all, 0, i_all, i_all)

    # thermal energy is temp in nondimensional units
    energy = data.fields['temp'][step][0]

    # apply a dimensional scale
    if scale is not None:
        energy = energy * scale

    return energy[index]


def kinetic(data: 'SimulationData', step: 'Type_Step' = -1, *,
            scale : Optional[float] = None, index = Optional['Type_Index'] = None,
            keepdims: bool = True) -> 'Type_Field':
    """
    Provides a method for calculation of the total kinetic energy by 
    consuming a SimulationData object; must have 'fcx2', 'fcy2' ('fcz2' if 3d) 
    attributes in the SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        step: time-like specification for which to process data, the key (optional) 
        scale: used to convert returned quantity to dimensional units (optional)
        index: used for custom slicing operation; should be (blks, k, j, i) (optional)
        keepdims: retain unused dimensions for broadcasting, else drop them (optional)

    Note:
        The total kinetic energy is computed according to the formula

                E(t)~ijk~ = u(t)~ijk~^2^ + v(t)~ijk~^2^ + w(t)~ijk~^2^
                
                *where        t = step,         step is float*
                           *  t = times[step],  step is int*
                           *ijk = all cells*

                where the all terms are interpolated to cell centers

    Todo:

    """
    # need the dimensionality
    dimension = data.geometry.grd_dim

    # need to define slicing operators based on dims
    if index is None:
        i_all = slice(None)
        index = (i_all, ) * 4 if (keepdims or dimension == 3) else (i_all, 0, i_all, i_all)

    # get guard size
    guards = data.geometry.blk_guards

    # calculate kinetic energy
    energy = _interpolate_ftc(data.fields['_fcx2'][step][0], 0, guards, dimension)**2
    energy = _interpolate_ftc(data.fields['_fcy2'][step][0], 1, guards, dimension)**2 + energy
    if dimension == 3:
        energy = _interpolate_ftc(data.fields['_fcz2'][step][0], 2, guards, dimension)**2 + energy
    
    # apply a dimensional scale
    if scale is not None:
        energy = energy * scale

    return energy[index]


def kinetic_mean(data: 'SimulationData', steps: Optional['Type_Index'] = None, *,
                 start: Optional['Type_Step'] = None, stop: Optional['Type_Step'] = None, 
                 skip: Optional[int] = None, scale : Optional[float] = None, 
                 index = Optional['Type_Index'] = None, keepdims: bool = True) -> 'Type_Field':
    """
    Provides a method for calculation of the mean or time-averaged kinetic energy by 
    consuming a SimulationData object and a time interval specification; 
    must have 'fcx2', 'fcy2' ('fcz2' if 3d) attributes in the SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        steps: iterable time-like specification for which to process data, the keys (optional) 
        start: used to determine the starting time-like specification, start key (optional)
        stop: used to determine the ending time-like specification, stop key (optional)
        skip: used to determine the sampling interval for the specification (optional)
        scale: used to convert returned quantity to dimensional units (optional)
        index: used for custom slicing operation; should be (blks, k, j, i) (optional)
        keepdims: retain unused dimensions for broadcasting, else drop them (optional)

    Note:
        The mean kinetic energy is computed according to the formula

                E(t)~ijk~ = $\sum_{$\tau$=t~0~}^{t} (u($\tau$)~ijk~^2^ + v($\tau)~ijk~^2^ + w($tau$)~ijk~^2^) / N
                
                *where the all terms are interpolated to cell centers*

    Todo:

    """

    # need the dimensionality
    dimension = data.geometry.grd_dim

    # need to define slicing operators based on dims
    if index is None:
        i_all = slice(None)
        index = (i_all, ) * 4 if (keepdims or dimension == 3) else (i_all, 0, i_all, i_all)

    # use provided information to source times
    if steps is None:
        times = data.utility.indices(slice(start, stop, step)) 

    # use provided slice to source times
    elif isinstance(steps, slice):
        times = data.utility.indices(steps)

    # try steps; it should be an iterable
    else:
        times = steps

    # use time series analysis to retreve mean kinetic energy
    source = make_sourceable(source=kinetic, args=data, method='step', context=False)
    stack = make_stackable(element=integral.time, args=data, method='series', context=False)
    energy = series.simple(source=source, sourceby=times, stack=stack) 

    # apply a dimensional scale
    if scale is not None:
        energy = energy * scale

    return energy[index]



