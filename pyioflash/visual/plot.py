"""

This module defines the simple plot methods necessary for the pyio package.

Todo:
    None
"""


from typing import Tuple, List, Dict, Optional, Callable, TYPE_CHECKING
from functools import partial


import numpy
from matplotlib import pyplot, cm, colors


from pyioflash.simulation.utility import _first_true
from pyioflash.simulation.utility import Plane
from pyioflash.simulation.utility import _map_mesh_grid, _map_grid_inds, _map_data_inds
from pyioflash.visual.options import FigureOptions, PlotOptions


if TYPE_CHECKING:
    from pyioflash.simulation.data import SimulationData
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


def _simple_plot2D(data: 'SimulationData', field: 'Type_Array', *,
                   plane: Plane = Plane(), blocks: Optional[List[int]] = None,
                   figure: Optional['Type_Figure'] = None, axis: Optional['Type_Axis'] = None, 
                   x: Optional['Type_Array'] = None, y: Optional['Type_Array'] = None,
                   withguard: bool = True, **options) -> Tuple['Type_Figure', 'Type_Axis']:
    """ Plots a 2d plane from determined or provided blocks of a field""" 

    # pre-plotting operations (e.g., make figure and options)
    fig, ax = _make_figure(figure, axis)
    options = PlotOptions(ax, field, plane.axis, **options)

    # identify blocks for plotting the 2D plane
    if all(axis is not None for axis in (x, y)) and blocks is None:
        raise ValueError(f'must provide blocks if x and y are provided!')
    if blocks is None:
        blocks = data.utility.blocks_from_plane(plane.axis, plane.cut)
        index, _ = data.utility.cut_from_plane(plane.axis, plane.cut, plane.face, blocks=blocks, withguard=withguard)
        f_ind = index

    # plot each block in desired order with or without geometry data and try catch issues if can
    cax = {}
    if all(axis is not None for axis in (x, y)):
        if not all(hasattr(item, 'shape') for item in (x, y, field)):
            raise ValueError(f'Cannot use provided x, y, and field as all must be numpy arrays')
        if any(len(item.shape) != 3 for item in (x, y, field)):
            raise ValueError(f'Cannot use provided x, y, and field as shape must be (blocks, Nj, Ni)')

        for b, block in enumerate(options._order(blocks)):
            cax[b] = options._plot(x[block], y[block], field[block])

    else:
        if plane.axis != 'z' and data.geometry.grd_dim == 2:
            raise ValueError(f'Cannot use provided plane with 2 dimensional simulation data')
        if not hasattr(field, 'shape'):
            raise ValueError(f'Cannot use provided field as it must be a numpy array!')
        if len(field.shape) not in (3, 4):
            raise ValueError(f'Cannot use provided field as shape must be (blocks, Nj, Ni) or (blocks, Nk, Nj, Ni)')
        if len(field.shape) == 3:
            field = field[:, numpy.newaxis, :, :]
            f_ind = 0

        for b, block in enumerate(options._order(blocks)):
            cax[b] = _plot2d_from_block(data=data, plane=plane, plot=options._plot, index=index, block=block, 
                                        field=field[block][_map_data_inds[plane.axis](f_ind)], withguard=withguard)

    # format plot output
    _set_plot_labels(axis=ax, options=options)
    _set_colorbar(figure=fig, axis=ax, artists=cax, options=options)

    return fig, ax


def _plot2d_from_block(*, data: 'SimulationData', plane: Plane, plot: Callable[..., 'Type_QuadMesh'], 
                       index: int, block: int, field: 'Type_Array', withguard: bool = True) -> 'Type_QuadMesh':
    """plots a 2d block, assuming keepdims is False, using geometry data"""
    grid = _map_mesh_grid[withguard][plane.axis]
    inds = _map_grid_inds[plane.axis](plane.face, block, index)
    mesh = tuple(data.geometry[g][inds] for g in grid)
    return plot(*mesh, field)


def _set_colorbar(*, figure: 'Type_Figure', axis: 'Type_Axis', 
                  artists: Dict[int, 'Type_QuadMesh'], options: PlotOptions) -> None:
    """Method for addressing the colorbar""" 
    if options.colorbar:
        figure.colorbar(artists[0], ax=axis, ticks=options._lvls[::options.colorbar_skip])

def _set_plot_labels(*, axis: 'Type_Axis', options: PlotOptions) -> None:
    """Method for addressing labeling of plot"""
    axis.set_title(options.title, fontsize=options.font_size, fontfamily=options.font_face)
    axis.set_xlabel(options._lbls[0], fontsize=options.font_size, fontfamily=options.font_face)
    axis.set_ylabel(options._lbls[1], fontsize=options.font_size, fontfamily=options.font_face)


