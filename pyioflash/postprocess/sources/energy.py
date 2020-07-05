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


from typing import Dict, Optional, Union, TYPE_CHECKING 


from pyioflash.postprocess.utility import _interpolate_ftc, make_sourceable, make_stackable, Output
from pyioflash.postprocess.elements import integral
from pyioflash.postprocess.analyses import series


if TYPE_CHECKING:
    from pyioflash.simulation.data import SimulationData
    from pyioflash.postprocess.utility import Type_Step, Type_Field, Type_Index, Type_Output


def thermal(data: 'SimulationData', step: 'Type_Step' = -1, *,
            wrapped: bool = False, mapping: Dict[str, str] = {},
            scale: Optional[float] = None, index: Optional['Type_Index'] = None,
            withguard: bool = False, keepdims: bool = True) -> 'Type_Output':
    """
    Provides a method for calculation of the thermal energy by 
    consuming a SimulationData object; must have a 'temp' attribute in the
    SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        step: time-like specification for which to process data, the key (optional)
        wrapped: whether to wrap context around result of sourcing (optional)
        mapping: if wrapped, how to map context to options of the next operation (optional)
        scale: used to convert returned quantity to dimensional units (optional)
        index: used for custom slicing operation; should be (blks, k, j, i) (optional)
        withguard: retain guard cell data for ploting and other actions (optional)
        keepdims: retain unused dimensions for broadcasting, else drop them (optional)

    Note:
        The thermal energy is computed according to the formula

                E(t)~ijk~ = T(t)~ijk~
                
                *where        t = step,         step is float*
                           *  t = times[step],  step is int*
                           *ijk = all cells*

        The returned quantity is on the cell centered grid

        This function does not generate any dynamic context; this even if wrapping is desired and specified, the
        mapping attribute is ignored.

    Todo:

    """

    # convert to integer from key if necessary
    if isinstance(step, float):
        try:
            step, = data.utility.indices(step)
        except ValueError as error:
            print(error)
            print('Could not find provided step in simulation keys!')

    # need the dimensionality
    dimension = data.geometry.grd_dim

    # need to define slicing operators based on dims
    if index is None:
        i_all = slice(None)
        i_zax = 0 if not withguard else 1
        index = (i_all, ) * 4 if (keepdims or dimension == 3) else (i_all, i_zax, i_all, i_all)

    # define lookup based on desired guarding option
    name = 'temp'
    if withguard:
        name = '_' + name

    # thermal energy is temp in nondimensional units
    energy = data.fields[name][step][0]

    # apply a dimensional scale
    if scale is not None:
        energy = energy * scale

    # index results if desired
    energy = energy[index]

    # wrap result of integration if desired (no context to provide)
    wrap = {True: lambda source: Output(source), False: lambda source: source} 
    return wrap[wrapped](energy)


def kinetic(data: 'SimulationData', step: 'Type_Step' = -1, *,
            wrapped: bool = False, mapping: Dict[str, str] = {},
            scale : Optional[float] = None, index: Optional['Type_Index'] = None,
            withguard: bool = False, keepdims: bool = True) -> 'Type_Output':
    """
    Provides a method for calculation of the total kinetic energy by 
    consuming a SimulationData object; must have 'fcx2', 'fcy2' ('fcz2' if 3d) 
    attributes in the SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        step: time-like specification for which to process data, the key (optional) 
        wrapped: whether to wrap context around result of sourcing (optional)
        mapping: if wrapped, how to map context to options of the next operation (optional)
        scale: used to convert returned quantity to dimensional units (optional)
        index: used for custom slicing operation; should be (blks, k, j, i) (optional)
        withguard: retain guard cell data for ploting and other actions (optional)
        keepdims: retain unused dimensions for broadcasting, else drop them (optional)

    Note:
        The total kinetic energy is computed according to the formula

                E(t)~ijk~ = u(t)~ijk~^2^ + v(t)~ijk~^2^ + w(t)~ijk~^2^
                
                *where        t = step,         step is float*
                           *  t = times[step],  step is int*
                           *ijk = all cells*

                where the all terms are interpolated to cell centers

        This function does not generate any dynamic context; this even if wrapping is desired and specified, the
        mapping attribute is ignored.

    Todo:

    """

    # convert to integer from key if necessary
    if isinstance(step, float):
        try:
            step, = data.utility.indices(step)
        except ValueError as error:
            print(error)
            print('Could not find provided step in simulation keys!')

    # need the dimensionality
    dimension = data.geometry.grd_dim

    # get guard size
    guards = data.geometry.blk_guards

    # need to define slicing operators based on dims
    if index is None:
        i_all = slice(None)
        i_zax = 0 if not withguard else int(guards / 2) 
        index = (i_all, ) * 4 if (keepdims or dimension == 3) else (i_all, i_zax, i_all, i_all)

    # calculate kinetic energy
    energy = _interpolate_ftc(data.fields['_fcx2'][step][0], 0, guards, dimension, withguard=withguard)**2
    energy = _interpolate_ftc(data.fields['_fcy2'][step][0], 1, guards, dimension, withguard=withguard)**2 + energy
    if dimension == 3:
        energy = _interpolate_ftc(data.fields['_fcz2'][step][0], 2, guards, dimension, 
                                  withguard=withguard)**2 + energy
    
    # apply a dimensional scale
    if scale is not None:
        energy = energy * scale

    # index results if desired
    energy = energy[index]

    # wrap result of integration if desired (no context to provide)
    wrap = {True: lambda source: Output(source), False: lambda source: source} 
    return wrap[wrapped](energy)


def kinetic_mean(data: 'SimulationData', steps: Optional['Type_Index'] = slice(None), *,
                 start: Optional['Type_Step'] = None, stop: Optional['Type_Step'] = None, skip: Optional[int] = None,
                 wrapped: bool = False, mapping: Dict[str, str] = {},
                 scale : Optional[float] = None, index: Optional['Type_Index'] = None, 
                 withguard: bool = False, keepdims: bool = True) -> Union['Type_Field', Output]:
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
        wrapped: whether to wrap context around result of sourcing (optional)
        mapping: if wrapped, how to map context to options of the next operation (optional)
        scale: used to convert returned quantity to dimensional units (optional)
        index: used for custom slicing operation; should be (blks, k, j, i) (optional)
        withguard: retain guard cell data for ploting and other actions (optional)
        keepdims: retain unused dimensions for broadcasting, else drop them (optional)

    Note:
        The mean kinetic energy is computed according to the formula

                E(t)~ijk~ = $\sum_{$\tau$=t~0~}^{t} (u($\tau$)~ijk~^2^ + v($\tau)~ijk~^2^ + w($tau$)~ijk~^2^) / N
                
                *where the all terms are interpolated to cell centers*

        This function does not generate any dynamic context; this even if wrapping is desired and specified, the
        mapping attribute is ignored.

    Todo:

    """

    # need the dimensionality
    dimension = data.geometry.grd_dim

    # get guard size
    guards = data.geometry.blk_guards

    # need to define slicing operators based on dims
    if index is None:
        i_all = slice(None)
        i_zax = 0 if not withguard else int(guards / 2)
        index = (i_all, ) * 4 if (keepdims or dimension == 3) else (i_all, i_zax, i_all, i_all)

    # use provided information to source times
    if start or stop:
        steps = slice(start, stop, skip)
    times = data.utility.times(steps)
    steps = data.utility.indices(steps)

    # use time series analysis to retreve mean kinetic energy
    source = make_sourceable(source=kinetic, args=data, method='step', options={'withguard': withguard})
    stack = make_stackable(element=integral.time, args=data, method='whole', options={'times': times})
    energy = series.simple(source=source, sourceby=steps, stack=stack) 

    # apply a dimensional scale
    if scale is not None:
        energy = energy * scale

    # index results if desired
    energy = energy[index]

    # wrap result of integration if desired (no context to provide)
    wrap = {True: lambda source: Output(source), False: lambda source: source} 
    return wrap[wrapped](energy)


def kinetic_turbulant(data: 'SimulationData', step: Optional['Type_Step'] = -1, *,
                      mean: Optional['Type_Field'] = None, start: Optional['Type_Step'] = None, 
                      stop: Optional['Type_Step'] = None, skip: Optional[int] = None,
                      wrapped: bool = False, mapping: Dict[str, str] = {},
                      scale : Optional[float] = None, index: Optional['Type_Index'] = None, 
                      withguard: bool = False, keepdims: bool = True) -> 'Type_Field':
    """
    Provides a method for calculation of the turbulant kinetic energy by 
    consuming a SimulationData object and a either a mean field or a time 
    interval specification to determine the mean field; must have 'fcx2', 
    'fcy2' ('fcz2' if 3d) attributes in the SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        step: time-like specification for which to process data, the key (optional)
        mean: provide mean turbulant kinetic energy to avoid calculating it (optional)
        start: used to determine the starting time-like specification, start key (optional)
        stop: used to determine the ending time-like specification, stop key (optional)
        skip: used to determine the sampling interval for the specification (optional)
        wrapped: whether to wrap context around result of sourcing (optional)
        mapping: if wrapped, how to map context to options of the next operation (optional)
        scale: used to convert returned quantity to dimensional units (optional)
        index: used for custom slicing operation; should be (blks, k, j, i) (optional)
        withguard: retain guard cell data for ploting and other actions (optional)
        keepdims: retain unused dimensions for broadcasting, else drop them (optional)

    Note:
        The mean kinetic energy is computed according to the formula

                E(t)~ijk~ = (u(t)~ijk~^2^ + v(t)~ijk~^2^ + w(t)~ijk~^2^) - mean

                *where the all terms are interpolated to cell centers*

        This function does not generate any dynamic context; this even if wrapping is desired and specified, the
        mapping attribute is ignored.

    Todo:

    """

    # convert to integer from key if necessary
    if isinstance(step, float):
        try:
            step, = data.utility.indices(step)
        except ValueError as error:
            print(error)
            print('Could not find provided step in simulation keys!')

    # need the dimensionality
    dimension = data.geometry.grd_dim

    # get guard size
    guards = data.geometry.blk_guards

    # need to define slicing operators based on dims
    if index is None:
        i_all = slice(None)
        i_zax = 0 if not withguard else int(guards / 2)
        index = (i_all, ) * 4 if (keepdims or dimension == 3) else (i_all, i_zax, i_all, i_all)

    # retieve mean kinetic energy if not provided
    if mean is None:
        mean = kinetic_mean(data, start=start, stop=stop, skip=skip, index=index, withguard=withguard, keepdims=keepdims)

    # turbulant energy
    energy = kinetic(data, step, scale=scale, index=index, withguard=withguard, keepdims=keepdims) - mean
                      
    # wrap result of integration if desired (no context to provide)
    wrap = {True: lambda source: Output(source), False: lambda source: source} 
    return wrap[wrapped](energy)
