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

    return energy

