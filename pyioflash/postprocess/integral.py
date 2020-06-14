"""

This module defines the integration methods of the post-processing
subpackage of the pyioflash lbrary.

This module currently defines the following methods:

    spacial  -> perform spacial integration
    temporal -> perform temporal integration

Todo:

"""
from typing import List, Iterable, Union

import numpy

from pyioflash.simulation.data import SimulationData


def integral_space(data : SimulationData, field : Union[str, numpy.ndarray], 
                   step : Union[int, float], differential : bool = True, 
                   scale : Union[None, float] = None, axis : str = "center") -> float:
    """
    Provides a method for calculation of the volumetric integral of a field by 
    consuming a SimulationData object; may provide either named field or array.

    Attributes:
        data: object containing relavent flash simulation output
        field: named field or numpy array for which to perform integration
        step: time-like specification for which to process data (key)
        differential: whether to use the volume elements of integration (optional)
        axis: which grid face to perform integration over [left, center, right] (optional)
        scale: used to convert returned quantity to dimensional units (optional)

    Note:
        The total kinetic energy is computed according to the formula

                sum( field{ijk} dV{ijk} ) over all ijk

    Todo:
        Need to thow if step is type(slice)

    """

    # need the dimensionality
    dims : int = data.geometry.grd_dim

    # need to define slicing operators based on dims
    i_zax = slice(None) if dims == 3 else 0
    i_all = slice(None)

    # calculate volume elements
    if volume:
        # need to define grid face to compute volume elements 
        i_cnt = {"left" : 0, "center" : 1, "right" : 2}[axis]

        # retrieve cell centered grid metrics
        ddxc : numpy.ndarray = data.geometry.grd_mesh_ddx[(i_cnt, i_all, i_zax, i_all, i_all)]
        ddyc : numpy.ndarray = data.geometry.grd_mesh_ddy[(i_cnt, i_all, i_zax, i_all, i_all)]
    
        # initialize volume element
        deltaV = 1/ddxc * 1/ddyc
        if dims == 3:
            ddzc : numpy.ndarray = data.geometry.grd_mesh_ddz[(i_cnt, i_all, i_zax, i_all, i_all)]
            deltaV = dv * 1/ddzc

    # use constant weights
    else:
        deltaV = 1.0

    # define the integrand based on arguments
    if isinstance(field, str):
        integrand = data.fields[field][step][0][:, i_zax, :, :]
    elif isinstance(field, numpy.ndarray):
        integrand = field
    else:
        pass

    # perform integration and provide result
    return numpy.sum(integrand * deltaV)


def integral_time(data : SimulationData, field : Union[str, List[numpy.ndarray], List[float], numpy.ndarray], 
                  steps : Union[Iterable, slice, None], differential : bool = True, differential : bool = True, 
                  scale : Union[None, float] = None, method : str = "center",
                  times : Union[None, Iterable] = None) -> Union[float, numpy.ndarray]:
    """
    Provides a method for calculation of the temporal integral of a field or scalar by 
    consuming a SimulationData object; may provide either named field or array.

    Attributes:
        data: object containing relavent flash simulation output
        field: named field or list of numpy arrays of floats for which to perform integration
        steps: time-like specification over which to process data (keys)
        differential: whether to use the temporal elements of integration (optional) 
        scale: used to convert returned quantity to dimensional units (optional)
        method: choice of method of integration [left, center, right] (optional)
        times: use different time-like specification of looking up temporal elements of integration (optional)

    Note:
        The total kinetic energy is computed according to the formula

                sum( field{t} dt{t} ) over all t

    Todo:
        Need to thow if field is not understood type
        Need to thow if method is not implemented
        Need to implement simpsons rule 
        Need to implement gaussian quadrature
        
    """
    
    # calculate temporal elements
    if differential:

        # get times using keys, then calc widths
        #   not in one step as keys may not be int
        time = data.scalars['t'][:][0]
        keys = times if times is not None else steps
        taus = [time[key] for for key in keys]
        dt = [right - left for right, left in zip(taus[1:], taus[:-1])]

    # use constant weights
    else:
        dt = 1.0

    # define integrand based on arguments
    if isinstance(field, str):

        # use data object and steps to create
        if isinstance(steps, slice):
            integrands = data.fields[field][steps][0]
        elif isinstance(steps, None):
            integrands = data.fields[field][:][0]
        else:
            integrands = [data.fields[field][step][0] for step in steps]

    # suitable for integration, field = [float, ...]
    elif isinstance(field[0], float):
        integrands = field

    # suitable for integration, field = [ndarray, ...] 
    elif len(field[0].shape) == 3 or len(field[0].shape) == 4:
        integrands = field

    # cannot work with provided field
    else:
        pass

    # perform integration based on method
    if method in {"left", "right", "center"}:

        if method == "center":
            integral = numpy.sum(0.5 * (integrands[1:] + integrands[:-1]) * dt, 0)

        else:
            index = slice(1, None) if method == "right" else slice(0, -1)
            integral = numpy.sum(integrands[index] * dt, 0)

    # method is not implemented
    else:
        pass

    return integral

