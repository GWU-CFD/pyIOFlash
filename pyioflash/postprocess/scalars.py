from typing import List, Iterable
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

def abs_thermal_energy(data : SimulationData, steps : Iterable, display : bool = True) -> List[float]:
    """
    Provides a method for calculation of the integral thermal energy of the domain over a
    range of times by consuming a SimulationData object; must have a 'temp' attribute in the
    SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        steps: time-like slices over which to process data (keys)
        display: (optional) whether to display in-prossess messaging

    Note:
        The integral thermal energy is computed according to the formula

                sum( T dx dy dz) over all ijk

    """
    # define pretty output and alias to name
    message = "Processing absolute thermal energy at step "
    output = _make_output(message, display)

    # need the dimensionality
    dims : int = data.geometry.grd_dim

    # need to define slicing operators based on dims
    #   z axis slice is index 1 for 2d and all for 3d,
    #   we need the cell centered metric using index 1,
    #   we need to define a slice for all indicies,
    #   and to define index for x, y[, z] field data
    i_zax = slice(0, None) if dims == 3 else 0
    i_cnt = 1
    i_all = slice(None)
    index = (i_all, i_all, i_all, i_all) if dims == 3 else \
            (i_all, 0, i_all, i_all)

    # retrieve cell centered grid metrics
    ddxc : numpy.ndarray = data.geometry.grd_mesh_ddx[(i_cnt, i_all, i_zax, i_all, i_all)]
    ddyc : numpy.ndarray = data.geometry.grd_mesh_ddy[(i_cnt, i_all, i_zax, i_all, i_all)]
    ddzc : numpy.ndarray = data.geometry.grd_mesh_ddz[(i_cnt, i_all, i_zax, i_all, i_all)]

    # initialize volume element
    volume = 1/ddxc * 1/ddyc
    if dims == 3:
        volume = volume * 1/ddzc

    energy = []
    for step in steps:
        output(step)
        energy.append( numpy.sum(data.fields["temp"][step][0][index] * volume) )

    print("") if display else None
    return energy


def rel_thermal_energy(data : SimulationData, steps : Iterable, absolute : List[float] = None, 
                       scale : float = 1.0, static : bool = True, tau : int = 0, 
                       display : bool = True) -> List[float]:
    """
    Provides a method for calculation of the integral thermal energy expressed as the
    (transient) relative percent difference over the domain over a range of times by consuming
    a SimulationData object; must have a 'temp' attribute in the SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        steps: time-like slices over which to process data (keys)
        absolute: (optional) absolute thermal energy over range of steps
        scale: (optional) scale multiplier on relative result; default is 1.0
        static: (optional) whether to perform a static or running reference; default is True
        tau: (optional) step for reference state if static, delta steps if not static; default is 1
        display: (optional) whether to display in-prossess messaging; default is True

    Note:
        The integral thermal energy is computed according to the formula

                E_rel = scale * abs( E(t) - E(t - tau) ) / ( abs(E(t) + abs(E(t - tau)) )

                where:  E = sum( T dx dy dz) over all ijk

    """
    # define pretty output and alias to name
    message = "Processing relative thermal energy at step "
    output = _make_output(message, display)

    # need to define refernce state
    if static:
        delta = lambda step: tau 
    else:
        delta = lambda step: step - tau

    # need to calc energy if not provided in call
    if not absolute:
        absolute = abs_thermal_energy(data, steps, display) 

    # need step through tau if not static
    if static:
        energy = [0.0 for step in range(tau)]
        steps = steps[tau:]
    else:
        energy = []

    # calculate relative energy
    for step in steps:
        output(step)
        inst = absolute[step]
        refn = absolute[delta(step)]
        energy.append(scale * abs(inst - refn) / (abs(inst) + abs(refn))) 

    print("") if display else None
    return energy


def abs_kinetic_energy(data : SimulationData, steps : Iterable, display : bool = True) -> List[float]:
    """
    Provides a method for calculation of the integral total kinetic energy of the domain over a
    range of times by consuming a SimulationData object; must have 'fcx2', 'fcy2' ('fcz2' if 3d) 
    attributes in the SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        steps: time-like slices over which to process data (keys)
        display: (optional) whether to display in-prossess messaging

    Note:
        The integral total kinetic energy is computed according to the formula

                sum( (u*u + v*v + w*w) dx dy dz) over all ijk

                where the all terms are interpolated to cell centers

    """
    # define pretty output and alias to name
    message = "Processing absolute total kinetic energy at step "
    output = _make_output(message, display)

    # need the dimensionality
    dims : int = data.geometry.grd_dim

    # need to define slicing operators based on dims
    #   z axis slice is index 1 for 2d and all for 3d,
    #   we need the cell centered metric using index 1,
    #   we need to define a slice for all indicies,
    #   and to define index for x, y[, z] field data
    i_zax = slice(None) if dims == 3 else 0
    i_cnt = 1
    i_all = slice(None)
    index = (i_all, i_all, i_all, i_all) if dims == 3 else \
            (i_all, i_all, i_all)

    # retrieve cell centered grid metrics
    ddxc : numpy.ndarray = data.geometry.grd_mesh_ddx[(i_cnt, i_all, i_zax, i_all, i_all)]
    ddyc : numpy.ndarray = data.geometry.grd_mesh_ddy[(i_cnt, i_all, i_zax, i_all, i_all)]
    ddzc : numpy.ndarray = data.geometry.grd_mesh_ddz[(i_cnt, i_all, i_zax, i_all, i_all)]

    # initialize volume element
    volume = 1/ddxc * 1/ddyc
    if dims == 3:
        volume = volume * 1/ddzc

    # get guard size
    g = data.geometry.blk_guards

    # define build for z axis if present
    if dims == 3:
        wfun = lambda step, index: (_interpolate_ftc(data.fields["_fcz2"][step][0], 2, g, dims)**2)[index]
    else:
        wfun = lambda step, index: 0.0

    energy = []
    for step in steps:
        output(step)
       
        uuxc = _interpolate_ftc(data.fields["_fcx2"][step][0], 0, g, dims)**2
        vvyc = _interpolate_ftc(data.fields["_fcy2"][step][0], 1, g, dims)**2
        wwzc = wfun(step, index)

        energy.append( numpy.sum((uuxc[index] + vvyc[index] + wwzc) * volume) )

    print("") if display else None
    return energy


def rel_kinetic_energy(data : SimulationData, steps : Iterable, absolute : List[float] = None,
                       scale : float = 1.0, static : bool = True, tau : int = 0,
                       display : bool = True) -> List[float]:
    """
    Provides a method for calculation of the integral total kinetic energy expressed as the
    (transient) relative percent difference over the domain over a range of times by consuming
    a SimulationData object; must have a 'temp' attribute in the SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        steps: time-like slices over which to process data (keys)
        absolute: (optional) absolute total kinetic energy over range of steps
        scale: (optional) scale multiplier on relative result; default is 1.0
        static: (optional) whether to perform a static or running reference; default is True
        tau: (optional) step for reference state if static, delta steps if not static; default is 1
        display: (optional) whether to display in-prossess messaging; default is True

    Note:
        The integral thermal energy is computed according to the formula

                E_rel = scale * abs( E(t) - E(t - tau) ) / ( abs(E(t) + abs(E(t - tau)) )

                where:  E = sum( (u*u + v*v + w*w) dx dy dz) over all ijk

    """
    # define pretty output and alias to name
    message = "Processing relative total kinetic energy at step "
    output = _make_output(message, display)

    # need to define refernce state
    if static:
        delta = lambda step: tau
    else:
        delta = lambda step: step - tau

    # need to calc energy if not provided in call
    if not absolute:
        absolute = abs_kinetic_energy(data, steps, display)

    # need step through tau if not static
    if static:
        energy = [0.0 for step in range(tau)]
        steps = steps[tau:]
    else:
        energy = []

    # calculate relative energy
    for step in steps:
        output(step)
        inst = absolute[step]
        refn = absolute[delta(step)]
        energy.append(scale * abs(inst - refn) / (abs(inst) + abs(refn)))

    print("") if display else None
    return energy


