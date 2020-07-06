"""

This module defines the integration methods of the post-processing
subpackage of the pyioflash library; part of the Stackable set of routines.

This module currently defines the following methods:

    space   -> perform simple spatial integration
    time    -> perform simple temporal integration

Todo:
    work on single and double spatial integration

"""


from typing import Tuple, Dict, Union, Iterable, Optional, TYPE_CHECKING


import numpy


from pyioflash.postprocess.utility import Output


if TYPE_CHECKING:
    from pyioflash.simulation.data import SimulationData
    from pyioflash.postprocess.utility import Type_Field, Type_Output, Type_Index 


# define the module api
def __dir__() -> List[str]:
    return ["space_full", "time"]


def space_full(data: 'SimulationData', field: 'Type_Field', *, 
               face: str = 'center',
               differential: bool = True, 
               wrapped: bool = False, mapping: Dict[str, str] = {},
               scale: Optional[float] = None, index: Optional['Type_Index'] = None, 
               withguard: bool = False, keepdims: bool = True) -> 'Type_Output':
    """
    Provides a method for calculation of the volumetric integral of a field by 
    consuming a SimulationData object; ...

    Attributes:
        data: object containing relavent flash simulation output
        field: numpy array for which to perform integration; [b, ...]
        face: which grid face to perform integration over [left, center, right] (optional)
        differential: whether to use the volume elements of integration (optional)
        wrapped: whether to wrap context around result of integration (optional)
        mapping: if wrapped, how to map context to options of the next operation (optional)
        scale: used to convert returned quantity to dimensional units (optional)
        index: used for custom slicing operation on differential; should be (blks, k, j, i) (optional)
        keepdims: input field has retained unused dimensions for broadcasting, else were dropped (optional)

    Note:
        The total kinetic energy is computed according to the formula

                sum( field{ijk} dV{ijk} ) over all ijk

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

    # calculate volume elements
    if differential:
        # need to define grid face to compute volume elements 
        i_grd = {"left" : 0, "center" : 1, "right" : 2}[face]

        # retrieve cell centered grid metrics
        if withguard:
            ddxc = data.geometry._grd_mesh_ddx[(i_grd, ) + index]
            ddyc = data.geometry._grd_mesh_ddy[(i_grd, ) + index]
        else: 
            ddxc = data.geometry.grd_mesh_ddx[(i_grd, ) + index]
            ddyc = data.geometry.grd_mesh_ddy[(i_grd, ) + index]
    
    
        # initialize volume element
        deltaV = 1/ddxc * 1/ddyc
        if dimension == 3:
            if withguard:
                ddzc = data.geometry._grd_mesh_ddz[(i_grd, ) + index]
            else:
                ddzc = data.geometry.grd_mesh_ddz[(i_grd, ) + index]
            deltaV = deltaV * 1/ddzc

    # use constant weights
    else:
        deltaV = 1.0

    # perform integration and index the result
    integral = numpy.sum(field * deltaV)

    # wrap result of integration if desired (no context to provide)
    wrap = {True: lambda integral: Output(integral), False: lambda integral: integral} 
    return wrap[wrapped](integral)


def space_single(data: 'SimulationData', field: 'Type_Field', *,
                 face: str = 'center', 
                 axis: Union[str, int] = 1, layout: Tuple[str] = ('b', 'z', 'y', 'x'),
                 differential: bool = True, 
                 wrapped: bool = False, mapping: Dict[str, str] = {},
                 scale: Optional[float] = None, index: Optional['Type_Index'] = None,
                 keepdims: bool = False) -> 'Type_Output':
    """
    Provides a method for calculation of the volumetric integral of a field by 
    consuming a SimulationData object; ...

    Attributes:
        data: object containing relavent flash simulation output
        field: numpy array for which to perform integration; [blocks, ...]
        face: which grid face to perform integration over [left, center, right] (optional)
        axis: which axis to integrate; can supply label and assiciated layout (optional)
        layout: indexable object containing the labeled dimentions of field (optional)
        differential: whether to use the volume elements of integration (optional)
        wrapped: whether to wrap context around result of integration (optional)
        mapping: if wrapped, how to map context to options of the next operation (optional)
        scale: used to convert returned quantity to dimensional units (optional)
        index: used for custom slicing operation; should be (blks, k, j, i) (optional)
        keepdims: input field has retained unused dimensions for broadcasting, else were dropped (optional)

    Note:

    Todo:
        need to perform a reduction operation and a renumbering of the blocks

    """

    # define axis and identify index to perform integration 
    mapping = {ax: layout.index(ax) for ax in layout}
    index = mapping.get(axis, axis)
    if index >= len(mapping) or not isinstance(index, int):
        Exception(f'Chosen axis is outside of provided or default layout!')
    if layout[index] not in {'x', 'y', 'z'} and differential:
        Exception(f'Chosen axis is not available in geometry data!')

    # calculate volume elements of appropriate dimension (to broadcast)
    if differential:
        i_face = {'left': 0, 'center': 1, 'right' : 2}[face],
        shrink = tuple(0 if ax != layout[index] else slice(None) for ax in layout) 
        ddnf = 1 / getattr(data.geometry, "grd_mesh_dd" + layout[index])[i_face + shrink]
        deltaV = ddnf[tuple(numpy.newaxis if ax != layout[index] else slice(None) for ax in layout)]

    # use constant weights
    else:
        deltaV = 1.0

    # perform integration and provide result
    return numpy.sum(field * deltaV, index)


def time(data: 'SimulationData', fields: 'Type_Output', *,
         method : str = 'center', 
         differential : bool = True,
         wrapped: bool = False, mapping: Dict[str, str] = {},
         scale : Optional[float] = None, steps: Optional['Type_Index'] = None, 
         starting: Union[int, float] = 0, times: Optional['Type_Index'] = None,
         force_steps: bool = False) -> 'Type_Output':
    """
    Provides a method for calculation of the temporal integral of fields or scalars by 
    consuming a SimulationData object and provided fields

    Attributes:
        data: object containing relavent flash simulation output
        fields: (list of) numpy arrays or floats/ints over which to perform integration
        method: choice of method of integration [left, center, right] (optional)
        differential: whether to use the temporal elements of integration (optional) 
        wrapped: whether to wrap context around result of integration (optional)
        mapping: if wrapped, how to map context to options of the next operation (optional)
        scale: used to convert returned quantity to dimensional units (optional)
        steps: time-like specification over which to process provided data (optional)
        starting: if data and fields are not aligned; provides a shift to align (optional)
        times: use different time-like specification of looking up temporal elements of integration (optional)
        force_steps: force the treatment of steps as an Iterable; needed for len(steps) < len(fields) (optional)

    Note:
        The total kinetic energy is computed according to the formula

                sum( field{t} dt{t} ) over all t


        If steps is an Iterable of slices the optional attibute times must be provided if the provided field
        is not aligned to the entries in the provided data object.

        If steps is provided (either as a slice or an Iterable), elements must be integers; specification using 
        floats is not yet supported and will result in runtime error.

        This function does not generate any dynamic context; this even if wrapping is desired and specified, the
        mapping attribute is ignored.

    Todo:
        Need to implement simpsons rule 
        Need to implement gaussian quadrature
        
    """

    # specify supported methods
    methods = {'center', 'left', 'right'}

    # strip context if was provided
    if isinstance(fields, Output):
        fields = fields.data

    # catch inappropriate specification for steps; if not could result in undetected issues
    if steps is not None and not (isinstance(steps, slice) or isinstance(steps, Iterable)):
        raise TypeError(f"Unsupported type '{type(steps).__name__}' provided for steps, must be 'slice' of 'Iterable'") 

    # determine if we can work with the provided fields
    if not hasattr(fields, '__len__') or type(fields[0]) not in {float, int, numpy.float64, numpy.ndarray}:
        raise TypeError(f"Unsupported type '{type(fields).__name__}' provided for fields!") 

    # lets work with a numpy array
    integrand = numpy.array(fields)

    # if desired slice or iterate into provided fields
    if steps is not None:
        if force_steps or (len(steps) > len(integrand.shape)):
            integrand = [integrand[step] for step in steps]
        else:
            integrand = integrand[steps]

    # calculate temporal elements
    if differential:
        
        # need to create list base on specifics of attributes provided
        if times is not None:
            taus = data.utility.times(times)
        elif steps is not None:
            if isinstance(steps, slice):
                start = starting if steps.start is None else steps.start + starting
                stop = steps.stop if steps.stop is None else steps.stop + starting
                taus = data.utility.times(slice(start, stop, steps.step))
                taus = taus[:len(integrand)] # needed aligned but different length fields and data
            elif starting == 0:
                taus = data.utility.times(steps)
            else:
                raise ValueError(f'If steps is Iterable, cannot specify starting; must specify times attribute!')
        else:
            if starting == 0:
                taus = data.utility.times(slice(None))
                taus = taus[:len(integrand)] # needed aligned but different length fields and data
            else:
                start = starting
                stop = starting + len(integrand)
                taus = data.utility.times(slice(start, stop))

        # calc time window widths
        dt = [right - left for right, left in zip(taus[1:], taus[:-1])]
        dt = numpy.array(dt)[(slice(None), ) + (numpy.newaxis, ) * (len(integrand.shape)-1)]

    # use constant weights
    else:
        dt = 1.0

    # catch issue with missaligned times and integrand
    if hasattr(dt, '__len__') and (len(dt) != (len(integrand) - 1)):
        raise ValueError(f'Unable to correctly determine differential elements for integral')

    # perform integration based on method
    if method in methods:

        if method == 'center':
            integral = numpy.sum(0.5 * (integrand[1:] + integrand[:-1]) * dt, 0)

        else:
            index = slice(1, None) if method == 'right' else slice(0, -1)
            integral = numpy.sum(integrands[index] * dt, 0)

    # method is not implemented
    else:
        raise ValueError(f"Unsupported method of integration '{method}'; must specify {methods}!") 

    # wrap result of integration if desired (no context to provide)
    wrap = {True: lambda integral: Output(integral), False: lambda integral: integral} 
    return wrap[wrapped](integral)
