from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class FigureOptions:
    """ -- """
    title: str = ''
    size: Tuple[int, int] = (15, 11)  # not used
    size_single: Tuple[int, int] = (11, 7)
    font_size: int = 14
    font_face: str = 'Cambria'
    save: bool = False  # not used
    show: bool = True   # not used
    show_differed: bool = True

@dataclass
class PlotOptions:
    title: str = None  # computed default
    labels: Tuple[str, str, str] = ('x [-]', 'y [-]', 'z [-]')
    font_size: int = 10
    font_face: str = 'Calibri'
    type: str = 'contourf'
    colorbar: bool = True       # not used
    colorbar_lvls: int = 10     # not used
    colormap: str = 'viridis'   # not used
    vrange: Tuple[float, float] = (0.0, 1.0)    # not used
    vrange_ext: float = 0.0                     # not used
    vrange_auto: bool = True                    # not used
    vrange_lvls: int = 21                       # not used
    contours_skip: int = 2          # not used
    contours_alpha: float = 0.6     # not used
    persist: bool = False                       

@dataclass
class AnimationOptions:
    print_time: bool = True     # not used
    print_dt: bool = True       # not used
    interval: int = 70          # not used
    repeat_delay: int = 1000    # not used
    blit: bool = True           # not used
