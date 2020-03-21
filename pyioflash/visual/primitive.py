"""

This module defines the utility methods and classes necessary for the pyio package.

Todo:
    None
"""

from dataclasses import dataclass
from typing import Union

@dataclass
class Plane:
    """
    Data object for defining a 2d cut-plane at a give time for use in plotting

    Attributes:
        time: time or index (e.g., key) associated with the desired plane
    """
    time: Union[float, int] = None
    cut: float = None
    axis: str = 'z'

