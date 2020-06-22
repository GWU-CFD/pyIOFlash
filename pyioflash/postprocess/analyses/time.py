from typing import List, Iterable, Union, Callable, TYPE_CHECKING
from sys import stdout
from functools import partial

from pyioflash.postprocess.utility import _make_output
from pyioflash.postprocess.series import data_from_path

if TYPE_CHECKING:
    from numpy import ndarray
    from pyioflash.simulation.data import SimulationData


def absolute(data: 'SimulationData', 
             source: Union[str, 
                           List[Union[float, int, str]], List['ndarray'], 'ndarray', 
                           Callable[[Union[int, float, slice]], Union[float, 'ndarray']]],
             stack: Union[None, Iterable] = None,
             where: Union[None, DataPath] = None,
             steps: Union[Iterable, slice, None] = None, 
             scale: float = 1.0, 
             display: bool = True
             ) -> Union[List[Union[float, int, str]], List['ndarray']]:
    """
    Provides a method for calculation of the integral total kinetic energy expressed as the
    (transient) relative percent difference over the domain over a range of times by consuming
    a SimulationData object; must have a 'temp' attribute in the SimulationData.fields object.

    Attributes:
        data: object containing relavent flash simulation output
        steps: time-like slices over which to process data (keys)
        absolute: (optional) absolute total kinetic energy over range of steps
        scale: (optional) scale multiplier on relative result; default is 1.0
        display: (optional) whether to display in-prossess messaging; default is True

    Note:

    """
    # define pretty output and alias to name
    message = "Processing absolute series @ step "
    output = _make_output(message, display)

    # define source based on arguments
    if isinstance(source, str) and where is not None:
        where = where._replace(data=source)
        series = data_from_path(data, where)

    elif

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

#def relative():


