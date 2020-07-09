"""A python module for providing optional behavior for the SimulationPlot class.

The classes provided in this module define the default and modifiable options specifying the
behavior of the SimulationPlot class plotting operations. The options are segregated between
figure wide and plot specific options.

To change the default behavior of a specific option, pass a key-value dictionary containing the
desired options as a keyword argument of the appropriate SimulationPlot instance method.
"""
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class FigureOptions:
    """A class for specifying the figure-wide default and modifiable plotting options.

    Attributes:
        title: Figure title; default is blank
        size_multiple: Figure size when multiple plots are requested;
        size_single: Figure size; default is 11x7
        font_size: Figure title font size; default is 14
        font_face: Figure title font face; default is Cambria
        save: Save the figure to file when the show method is used; default is False
        show: Show the figure to screen when the show method is used; default is True
        show_differed: Require a call to the show method (do not show after each plot method); default is True
    """
    size_multiple: Tuple[int, int] = (15, 11)
    size_single: Tuple[int, int] = (15, 11)
    header: str = ''
    header_size: int = 14
    header_face: str = 'DejaVu Sanse'
    save: bool = False  
    show: bool = True
    show_differed: bool = True # Not Used

@dataclass
class PlotOptions:
    """A class for specifying the plot specific default and modifiable plotting options.

    Attributes:
        title: Plot title; default is computed from requested plot method inputes
        labels: Spacial axis labels; default if x [-], y [-], and z [-]
        font_size: plot title font size; default is 10
        font_face: Plot title font face; default is Calibri
        type: Plot type to be displayed; default is contourf
        colorbar: Show a colorbar on the plot; NOT USED
        colormap: Colormap to use for the plot; NOT USED
        vrange: Specify a minimum and maximum plot value; NOT USED
        vrange_ext: Specify a percent exention for plotted values; NOT USED
        vrange_auto: Automatically determine a plot range; NOT USED
        vrange_lvls: Specify the number of contour levels to plot; NOT USED
        contours_skip: Specify how many contour levels to skip; NOT USED
        contours_alpha: Specify the alpha to use when drawing contours: NOT USED
        persist: Plotting options persist between plot methods; default is False
    """
    title: str = '' 
    labels: Tuple[str, str, str] = ('x [-]', 'y [-]', 'z [-]')
    font_size: int = 10
    font_face: str = 'DejaVu Sans'
    plot_type: str = 'density'
    colorbar: bool = True
    colorbar_skip: int = 10     
    colormap: str = 'viridis'   
    vrange: Optional[Tuple[float, float]] = None
    vrange_ext: str = 'neither'         
    vrange_auto: bool = True                    
    contours_lvls: int = 21                       
    contours_skip: int = 2          
    contours_alpha: float = 0.6     
    persist: bool = False

@dataclass
class AnimationOptions:
    """A class for specifying the plot specific default and modifiable plotting options.

    Attributes:
        print_time: Show time value on animation; NOT USED
        print_dt: Show time-step size on animation; NOT USED
        interval: Specify the refresh interval between frames (ms); NOT USED
        repeat_delay: Specify time delay between repeating the animation; NOT USED
        blit: Use blitting when showing an animation; default is contourf
    """
    print_time: bool = True     # not used
    print_dt: bool = True       # not used
    interval: int = 70          # not used
    repeat_delay: int = 1000    # not used
    blit: bool = True           # not used
