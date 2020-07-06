"""

This module defines the field calculation methods of the post-processing
subpackage of the pyioflash lbrary; part of the 'source' set of routines.

This module currently defines the following methods:

    velocity_mean   --  -> mean velocity by component

Todo:

"""


from typing import List, Dict, Optional, Union, TYPE_CHECKING 


from pyioflash.simulation.series import DataPath
from pyioflash.postprocess.utility import _interpolate_ftc, make_stackable, Output
from pyioflash.postprocess.elements import integral
from pyioflash.postprocess.analyses import series


if TYPE_CHECKING:
    from pyioflash.simulation.data import SimulationData
    from pyioflash.postprocess.utility import Type_Step, Type_Field, Type_Index, Type_Output


# define the module api
def __dir__() -> List[str]:
    return ["velocity_mean"] 

def velocity_mean(data: 'SimulationData', steps: Optional['Type_Index'] = slice(None), *,
                  start: Optional['Type_Step'] = None, stop: Optional['Type_Step'] = None, skip: Optional[int] = None,
                  wrapped: bool = False, mapping: Dict[str, str] = {},
                  scale : Optional[float] = None, index: Optional['Type_Index'] = None, 
                  withguard: bool = False, keepdims: bool = True) -> 'Type_Output':
    """
    Provides a method for calculation of the mean velocity components by 
    consuming a SimulationData object and a time interval specification 
    in order to determine the mean fields by component; must have 'fcx2', 
    'fcy2' ('fcz2' if 3d) attributes in the SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        steps: iterable time-like specification for which to process data, the keys (optional) 
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
        Each mean velocity component is computed according to the formula

                u_bar(t)~ijk~ = 

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

    # use time series analysis to retreve mean velocity components
    stack = make_stackable(element=integral.time, args=data, method='whole', options={'times': times})
    u_bar = series.simple(source='_fcx2', sourceby=steps, stack=stack, path=DataPath(data, 'fields')) 
    v_bar = series.simple(source='_fcy2', sourceby=steps, stack=stack, path=DataPath(data, 'fields')) 
    if dimension == 3:
        w_bar = series.simple(source='_fcz2', sourceby=steps, stack=stack, path=DataPath(data, 'fields')) 

    # interpolate to cell-centers
    u_bar = _interpolate_ftc(u_bar, 0, guards, dimension, withguard=withguard)
    v_bar = _interpolate_ftc(v_bar, 1, guards, dimension, withguard=withguard)
    if dimension == 3:
        w_bar = _interpolate_ftc(w_bar, 2, guards, dimension, withguard=withguard)
    
    # apply a dimensional scale
    if scale is not None:
        u_bar = u_bar * scale
        v_bar = v_bar * scale
        if dimension == 3:
            w_bar = w_bar * scale

    # index results if desired 
    u_bar = u_bar[index]
    v_bar = v_bar[index]
    if dimension == 3:
        w_bar = w_ba[index]

    # collect velocity components
    if dimension !=3:
        components = (u_bar, v_bar)
    else:
        components = (u_bar, v_bar, w_bar)

    # wrap result of integration if desired (no context to provide)
    wrap = {True: lambda source: Output(source), False: lambda source: source} 
    return wrap[wrapped](components)
