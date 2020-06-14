"""

This module defines the utility methods and classes necessary for the 
post-processing subpackage of pyioflash.

Todo:
    None

"""

from typing import Tuple, Union
from sys import stdout

import numpy

# define pretty output formating
def _make_output(message : str, display : bool) -> Callable[int]:
    """
    Provides a method for writing to screen for progress information

    Attributes:
        message: prepended message to write line
        display: whether or not to display message

    Todo:
        Provide more progress bar and or logging methods

    """

    # create the writing function; overwrites line
    def write_step(step : int) -> None:
        stdout.write(message + " %d \r" % (step))
        stdout.flush()

    # attach the output function or not
    output : Callable[int]
    if not display:
        output = lambda i: None 
    else: 
        output = write_step

    return output


def _interpolate_ftc(field : numpy.ndarray, axis : int, guards : int, dimension : int) -> numpy.ndarray:
    """
    Provides a method for interpolation from the face centered grids to the cell centered grid.

    Attributes:
        field: face centered field from which to interpolate
        axis: which face centered grid {0 : i, 1 : j, 2 : k}
        guards: how many guard cells in each spacial dimension in field array
        dimension: what is the spacial dimensionality of field

    Note:
        Performs simple linear two-point grid interpolation along relavent axis.

    Todo:
        Implement more advanced interpolation schemes

    """
    # use one-sided guards
    guard : int = int(guards / 2)

    # define necessary slice operations
    iall : slice = slice(None)
    icom : slice = slice(guard, -guard)
    izcm : int = icom if dimension == 3 else 1
    idif : slice = slice(guard - 1, -(guard + 1))

    # define the upper axis; velocity on staggered grid where upper bound is on
    #   the domain boundary & the outer most interior cell on the high side of axis
    high : Tuple[Union[slice, int]] = (iall, izcm, icom, icom)

    # define the lower axis; velocity on staggered grid where lower bound is on
    #   the domain boundary & the inner most guard cell on the low side of axis
    low : Tuple[Union[slice, int]]
    if axis == 0:
        low = (iall, izcm, icom, idif) 
    elif axis == 1:
        low = (iall, izcm, idif, icom)
    elif axis == 2:
        low = (iall, idif, icom, icom)
    else:
        pass

    return (field[high] + field[low]) / 2.0


