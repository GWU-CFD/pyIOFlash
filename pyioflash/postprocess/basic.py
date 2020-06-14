from typing import List, Iterable, Union
from sys import stdout
from functools import partial

import numpy

from pyioflash.simulation.data import SimulationData


# define pretty output formating
def _make_output(message : str, display : bool):

    def _write_step(step : int):
        stdout.write(message + " %d \r" % (step))
        stdout.flush()

    if not display:
        output = lambda i: None 
    else: 
        output = _write_step

    return output


def _interpolate_ftc(field : numpy.ndarray, axis : int, guards : int, dim : int) -> numpy.ndarray:
    
    # use one-sided guards
    guards = int(guards / 2)

    # define necessary slice operations
    iall = slice(None)
    icom = slice(guards, -guards)
    izcm = icom if dim == 3 else 1
    idif = slice(guards - 1, -(guards + 1))

    # define the upper axis; velocity on staggered grid where upper bound is on
    #   the domain boundary & the outer most interior cell on the high side of axis
    high = (iall, izcm, icom, icom)

    # define the lower axis; velocity on staggered grid where lower bound is on
    #   the domain boundary & the inner most guard cell on the low side of axis
    if axis == 0:
        low = (iall, izcm, icom, idif) 
    elif axis == 1:
        low = (iall, izcm, idif, icom)
    elif axis == 2:
        low = (iall, idif, icom, icom)
    else:
        pass

    return (field[high] + field[low]) / 2.0


def energy_thermal(data : SimulationData, step : Union[int, float, slice], 
                   scale : Union[None, float] = None) -> numpy.ndarray:
    """
    Provides a method for calculation of the thermal energy by 
    consuming a SimulationData object; must have a 'temp' attribute in the
    SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        step: time-like specification for which to process data (key)
        scale: used to convert returned quantity to dimensional units (optional)

    Note:
        The thermal energy is computed according to the formula

                E{ijk} = T{ijk}

        The returned quantity is on the cell centered grid

    Todo:
        Need to implement dimensionality

    """
    # need the dimensionality
    dims : int = data.geometry.grd_dim

    # need to define slicing operators based on dims
    i_all = slice(None)
    i_zax = slice(None) if dims == 3 else 0
    index = (i_all, izax, i_all, i_all)
    
    # thermal energy is temp in nondimensional units
    energy = data.fields["temp"][step][0][index]

    # apply a dimensional scale
    if scale is not None:
        energy = energy * scale

    return energy


def energy_kinetic(data : SimulationData, step : Union[int, float, slice], 
                   scale : Union[None, float] = None) -> numpy.ndarray:
    """
    Provides a method for calculation of the total kinetic energy by 
    consuming a SimulationData object; must have 'fcx2', 'fcy2' ('fcz2' if 3d) 
    attributes in the SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        step: time-like specification for which to process data (key)
        scale: used to convert returned quantity to dimensional units (optional)

    Note:
        The total kinetic energy is computed according to the formula

                {u*u + v*v + w*w}ijk

                where the all terms are interpolated to cell centers

    Todo:

    """
    # need the dimensionality
    dims : int = data.geometry.grd_dim

    # need to define slicing operators based on dims
    i_all = slice(None)
    index = (i_all, i_all, i_all, i_all) if dims == 3 else \
            (i_all, i_all, i_all)

    # get guard size
    g : int = data.geometry.blk_guards

    # calculate kinetic energy
    energy = (_interpolate_ftc(data.fields["_fcx2"][step][0], 0, g, dims)**2)[index]
    energy = energy + (_interpolate_ftc(data.fields["_fcy2"][step][0], 1, g, dims)**2)[index]
    if dims == 3:
        energy = energy + _interpolate_ftc(data.fields["_fcz2"][step][0], 2, g, dims)**2)[index]
    
    # apply a dimensional scale
    if scale is not None:
        energy = energy * scale

    return energy


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

