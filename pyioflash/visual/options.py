"""A python module for providing optional behavior for the visual subpackage of pyioflash.

The classes provided in this module define the default and modifiable options specifying the
behavior of the visual subpackage plotting operations. The options are segregated between
figure wide and plot specific options.

To change the default behavior of a specific option, pass a key-value dictionary containing the
desired options as a keyword argument of the appropriate plotting methods.
"""


from dataclasses import dataclass, InitVar
from typing import Tuple, List, Optional, TYPE_CHECKING
from functools import partial


import numpy
from matplotlib import cm, colors


from pyioflash.simulation.utility import _map_mesh_axes


if TYPE_CHECKING:
    from matplotlib import axes, contour
    from numpy import ndarray
    Type_Axis = axes._subplots.AxesSubplot
    Type_Contour = contour.QuadContourSet
    Type_Array = ndarray


# define the module api
def __dir__() -> List[str]:
    return ['FigureOptions', 'PlotOptions', 'AnimationOptions']


def _contour(ax: 'Type_Axis', x: 'Type_Array', y: 'Type_Array', z: 'Type_Array', 
              **kwargs) -> 'Type_Contour':
    """Defines the wrapper around matplotlib.pyplot.contour method"""
    return ax.contour(x[1:,1:], y[1:,1:], z[1:,1:], **kwargs)


def _contourp(ax: 'Type_Axis', x: 'Type_Array', y: 'Type_Array', z: 'Type_Array', 
              **kwargs) -> 'Type_Contour':
    """Defines the wrapper around matplotlib.pyplot.contour(f) methods"""
    ax.contour(x[1:,1:], y[1:,1:], z[1:,1:], alpha=1.0, **kwargs)
    return ax.contourf(x[1:,1:], y[1:,1:], z[1:,1:], alpha=0.5, **kwargs)


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
    """
    size_multiple: Tuple[int, int] = (15, 11)
    size_single: Tuple[int, int] = (15, 11)
    header: str = ''
    header_size: int = 14
    header_face: str = 'DejaVu Sanse'
    save: bool = False  
    show: bool = True


@dataclass
class PlotOptions:
    """A class for specifying the plot specific default and modifiable plotting options.

    Attributes:
        title: Plot title
        labels: Plot axis labels, depending on orientation; 
                defaults are x [-], y [-], and z [-]
        font_size: plot title font size; default is 10
        font_face: Plot title font face; default is DejaVu Sans
        method: Plot type to be displayed; default is density
        reverse: Whether or not to plot the blocks in reverse order; default is True
        colorbar: Show a colorbar on the plot;
        colormap: Colormap to use for the plot; default is viridis
        vrange: Specify a minimum and maximum plot value
        vrange_auto: Automatically determine a plot range, 
                     otherwise set to min/max; default is True 
        vrange_ext: Specify method of extending contours, options are
                     'neither', 'both', 'min', or 'max'; defualt is 'neither'
        contour_lvls: Specify the number of contour levels to plot; default is 81
        contours_skip: Specify how many contour levels to skip; default is 4
        contours_alpha: Specify the alpha to use when drawing contours: default is 1.0
    """
    plot_axis: InitVar[Optional['Type_Axis']] = None
    plot_field: InitVar[Optional['Type_Array']] = None
    plot_plane: InitVar[str] = 'z'
    title: str = '' 
    labels: Tuple[str, str, str] = ('x [-]', 'y [-]', 'z [-]')
    font_size: int = 12
    font_face: str = 'DejaVu Sans'
    method: str = 'density'
    reverse: bool = True
    colorbar: bool = True
    colorbar_skip: int = 8     
    colormap: str = 'viridis'   
    vrange: Optional[Tuple[float, float]] = None
    vrange_auto: Optional[bool] = True
    vrange_ext: str = 'neither'         
    contours_lvls: int = 81                       
    contours_skip: int = 4          
    contours_alpha: float = 1.0     

    def __post_init__(self, plot_axis, plot_field, plot_plane):
                
        # Determine the vrange if plot_field provided
        if self.vrange is None and plot_field is not None:
            if self.vrange_auto:
                ext = numpy.min(plot_field), numpy.max(plot_field)
                sgn = numpy.sign(ext)
                ext = numpy.abs(ext)
                pwr = numpy.log10(ext+1E-6)
                mag = numpy.round(10**(pwr - numpy.floor(pwr)))
                self.vrange = tuple(sgn * mag * 10**numpy.floor(pwr))
            else:
                self.vrange = (numpy.min(plot_field), numpy.max(plot_field)) 

        # Determine additional Plot Properties
        self._cmap = getattr(cm, self.colormap)
        self._lbls = tuple(self.labels[ax] for ax in _map_mesh_axes[plot_plane])    
        self._order = {True: reversed, False: lambda _: _}[self.reverse]
        if self.vrange is not None:
            self._lvls = numpy.linspace(self.vrange[0], self.vrange[1], self.contours_lvls)
            self._norm = colors.Normalize(vmin=self.vrange[0], vmax=self.vrange[1])

        # Determine appropriate plot function if able
        if plot_axis is not None and self.vrange is not None:
            self._plot = {
                'contour': partial(_contour, plot_axis, 
                                   levels=self._lvls[::self.contours_skip],
                                   vmin=self.vrange[0], vmax=self.vrange[1], extend=self.vrange_ext,
                                   norm=self._norm, cmap=self._cmap, alpha=self.contours_alpha),

                'contourf': partial(plot_axis.contourf, levels=self._lvls[::self.contours_skip],
                                    vmin=self.vrange[0], vmax=self.vrange[1], extend=self.vrange_ext,
                                    norm=self._norm, cmap=self._cmap, alpha=self.contours_alpha),

                'density': partial(plot_axis.pcolormesh, 
                                    vmin=self.vrange[0], vmax=self.vrange[1],
                                    norm=self._norm, cmap=self._cmap, antialiased=True),

                'contour+': partial(_contourp, plot_axis, 
                                    levels=self._lvls[::self.contours_skip],
                                    vmin=self.vrange[0], vmax=self.vrange[1], extend=self.vrange_ext,
                                    norm=self._norm, cmap=self._cmap)
                          }[self.method]


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
