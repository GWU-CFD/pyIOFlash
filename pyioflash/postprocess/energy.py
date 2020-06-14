"""

This module defines the energy calculation methods of the post-processing
subpackage of the pyioflash lbrary.

This module currently defines the following methods:

    thermal -> thermal energy
    kinetic -> total instantanious kinetric energy

Todo:
    Add method for turbulent kinetic energy
    Add method for mean kinetic energy

"""
from typing import Union

import numpy

from pyioflash.simulation.data import SimulationData
from pyioflash.postprocess.utility import _interpolate_ftc


def thermal(data : SimulationData, step : Union[int, float, slice], 
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


def kinetic(data : SimulationData, step : Union[int, float, slice], 
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

