"""

This module defines the utility metpods and classes necessary for the pyio package.

Todo:
    None
"""


from typing import Tuple, List, Dict, Optional, Callable, TYPE_CHECKING
from functools import partial


import numpy
from matplotlib import pyplot, cm, colors


from pyioflash.simulation.data import SimulationData
from pyioflash.simulation.utility import _first_true
from pyioflash.simulation.utility import Plane
from pyioflash.simulation.utility import _map_mesh_grid, _map_grid_inds
from pyioflash.visual.options import FigureOptions, PlotOptions


if TYPE_CHECKING:
    from matplotlib import figure, axes, collections
    from numpy import ndarray
    Type_Figure = figure.Figure
    Type_Axis = axes._subplots.AxesSubplot
    Type_QuadMesh = collections.QuadMesh
    Type_Array = ndarray


# define the module api
def __dir__() -> List[str]:
    return ['_simple_plot2D']


def _make_figure(figure: 'Type_Figure', axis: 'Type_Axis') -> Tuple['Type_Figure', 'Type_Axis']:
    if figure is None or axis is None:
        fig, ax = pyplot.subplots(figsize=(14, 9))
    else:
        fig, ax = figure, axis
    return fig, ax


def _simple_plot2D(data: SimulationData, z: 'Type_Array', plane: Plane, 
                   figure: Optional['Type_Figure'] = None, axis: Optional['Type_Axis'] = None, 
                   **options) -> Tuple['Type_Figure', 'Type_Axis']:

    # pre-plotting operations (e.g., make figure and options)
    blocks = data.utility.blocks_from_plane(plane.axis, plane.cut)
    index, _ = data.utility.cut_from_plane(plane.axis, plane.cut, plane.face, 
                                           blocks=blocks, withguard=True)
    fig, ax = _make_figure(figure, axis)
    plot = PlotOptions(ax, z, plane.axis, **options)

    # plot each block in desired order
    cax = {}
    for b, block in enumerate(plot._order(blocks)):
        cax[b] = _plot2d_from_block(data, plane, plot._plot, index, block, z[block])

    # format plot output
    _set_plot_labels(ax, plot)
    _set_colorbar(fig, ax, cax, plot)

    return fig, ax


def _plot2d_from_block(data: SimulationData, plane: Plane, 
                       plot: Callable[..., 'Type_QuadMesh'], 
                       index: int, block: int, field: 'Type_Array') -> 'Type_QuadMesh':
    grid = _map_mesh_grid[plane.axis]
    inds = _map_grid_inds[plane.axis](plane.face, block, index)
    mesh = tuple(data.geometry[g][inds] for g in grid)
    return plot(*mesh, field)


def _set_colorbar(figure: 'Type_Figure', axis: 'Type_Axis', 
                  artists: Dict[int, 'Type_QuadMesh'], plot: PlotOptions) -> None:
    if plot.colorbar:  
        figure.colorbar(artists[0], ax=axis, ticks=plot._lvls[::plot.colorbar_skip])

def _set_plot_labels(axis: 'Type_Axis', plot: PlotOptions) -> None:
    axis.set_title(plot.title, fontsize=plot.font_size, fontfamily=plot.font_face)
    axis.set_xlabel(plot._lbls[0], fontsize=plot.font_size, fontfamily=plot.font_face)
    axis.set_ylabel(plot._lbls[1], fontsize=plot.font_size, fontfamily=plot.font_face);
