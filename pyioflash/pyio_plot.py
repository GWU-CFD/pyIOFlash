"""A python module for implementing the SimulationPlot class.

This class is one of the most useful of the pyIOFlash package; providing methods to
convienently and intuitively create 2D and Line plots of simulation output data.
"""
from typing import Tuple, List, Dict, Union, Any
from dataclasses import replace

import numpy
import matplotlib
from matplotlib import pyplot, cm, lines
from matplotlib.font_manager import FontProperties

from .pyio import SimulationData
from .pyio_utility import _first_true
from .pyio_utility import Plane
from .pyio_options import FigureOptions, PlotOptions, AnimationOptions

_map_axes = {'x': 0, 'y': 1, 'z': 2}
_map_grid = {'x': 'grd_mesh_x', 'y': 'grd_mesh_y', 'z': 'grd_mesh_z'}
_map_points = {'x': lambda block : numpy.index_exp[0, block, 0, 0, :],
             'y': lambda block : numpy.index_exp[0, block, 0, :, 0],
             'z': lambda block : numpy.index_exp[0, block, :, 0, 0]}
_map_mesh = {'x': ('_grd_mesh_y', '_grd_mesh_z'),
             'y': ('_grd_mesh_x', '_grd_mesh_z'),
             'z': ('_grd_mesh_x', '_grd_mesh_y')}
_map_mesh_line = {'xy': '_grd_mesh_z',
                  'xz': '_grd_mesh_y',
                  'yx': '_grd_mesh_z',
                  'yz': '_grd_mesh_x',
                  'zx': '_grd_mesh_y',
                  'zy': '_grd_mesh_x'}
_map_mesh_axis = {'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}
_map_plane = {'x': lambda block, index : numpy.index_exp[0, block, :, :, index],
              'y': lambda block, index : numpy.index_exp[0, block, :, index, :],
              'z': lambda block, index : numpy.index_exp[0, block, index, :, :]}
_map_line = {'xy': lambda block, index1, index2 : numpy.index_exp[0, block, :, index2, index1],
             'xz': lambda block, index1, index2 : numpy.index_exp[0, block, index2, :, index1],
             'yx': lambda block, index1, index2 : numpy.index_exp[0, block, :, index1, index2],
             'yz': lambda block, index1, index2 : numpy.index_exp[0, block, index2, index1, :],
             'zx': lambda block, index1, index2 : numpy.index_exp[0, block, index1, :, index2],
             'zy': lambda block, index1, index2 : numpy.index_exp[0, block, index1, index2, :]}
_map_plot = {'contour' : lambda ax: ax.contour,
             'contourf' : lambda ax: ax.contourf}
_map_label = {'x': lambda options, ax: (options.labels[1], options.labels[2])[_map_axes[ax]],
              'y': lambda options, ax: (options.labels[0], options.labels[2])[_map_axes[ax]],
              'z': lambda options, ax: (options.labels[0], options.labels[1])[_map_axes[ax]],}
_map_line_label = {'xy': lambda options: options.labels[2],
                   'xz': lambda options: options.labels[1],
                   'yx': lambda options: options.labels[2],
                   'yz': lambda options: options.labels[0],
                   'zx': lambda options: options.labels[1],
                   'zy': lambda options: options.labels[0]}


class SimulationPlot:
    """A class providing methods and helper routines to plot hdf5 output file data.

    When a SimulationPlot instance is created with a SimulationData instance, the user is provided
    methods to plot simple 2D contour or density plots, Quiver plots, and Line plots.::

        from pyio import SimulationPlot

        ... create SimulationData instance, data

        visual = SimulationPlot(data, fig_options={key : value, ...},
                                      plot_options={key : value, ...},
                                      anim_options={key : value, ...})

        visual.plot(axis='x', cut=0.5, field='temp')

        visual.show()

    Note:

        None of fig_options, plot_options, or anim_options values are needed if the defualt
        behavior is acceptable for the desired plotting.

    The basic usage of the class is as follows: (1) import the class from the package, (2) create an instance
    of the class by providing the necessary SimulationData instance, and (3) use the class
    instance methods to create desired plot output.

    The general format for plotting 2D field data is::

        instance.plot[axis=__, cut=__, field='name', options={key : value, ...}]

    The general format for plotting Lines from field data is ::

        instance.plot[axis=__, cut=__, field='name', line=__, cutlines=[__, ...],
                                                     scale=__, options={key : value, ...}]

    The general format for modifying figure options after instanciation

        *methods not currenlty implemented*

    *Wherein the following definitions are used*:

    - The planer cut is defined by equation axis = cut (e.g. z = 0.5)
    - For line plots, plotted lines are the product of two planer intersections,
      where the secondary plane may be several values of a dimension to produce multiple
      co-plotted lines for convienence, as follows:

        - The primary plane, axis = cut (e.g., z = 0.5)
        - The secondary plane, line = cutline[0], cutline[1], ...

    - For both 2D contour and Line plots, field='name', provides the name of the data
    - Scale defines the scale multipler applied, if desired, to the field data before
      plotting the relavent lines.

    Finally, the class ploting methods return the relavent generated plot figures and  axes
    such that additional manual plot additions and modifications can be made befor the call
    to the instance show() method, as is possible with just the figure and axis handles.

    Attributes:
        data (SimulationData): data object containing the processed simulation output
        fig_options (FigureOptions): useful figure options for modification
        plot_options (PlotOptions): useful plot options for modification
        anim_options (AnimationOptions): useful animation options for modification

    """
    data: SimulationData
    fig_options: FigureOptions
    plot_options: PlotOptions
    anim_options: AnimationOptions

    def __init__(self, data: SimulationData, *,
                 fig_options: Union[FigureOptions, Dict[str, Any]] = None,
                 plot_options: Union[PlotOptions, Dict[str, Any]] = None,
                 anim_options: Union[AnimationOptions, Dict[str, Any]] = None):
        self.data = data

        if fig_options is None:
            self.fig_options = FigureOptions()
        elif isinstance(fig_options, FigureOptions):
            self.fig_options = fig_options
        else:
            self.fig_options = replace(FigureOptions(), **fig_options)

        if plot_options is None:
            self.plot_options = PlotOptions()
        elif isinstance(plot_options, PlotOptions):
            self.plot_options = plot_options
        else:
            self.plot_options = replace(PlotOptions(), **plot_options)


        if anim_options is None:
            self.anim_options = AnimationOptions()
        elif isinstance(anim_options, AnimationOptions):
            self.anim_options = anim_options
        else:
            self.anim_options = replace(AnimationOptions(), **anim_options)

    def plot(self, *, field: str, cut: float = None, axis: str = 'z', time: Union[float, int] = -1,
             line: str = None, cutlines: List[float] = None, scale: float = 1.0,
             options: Dict[str, Any] = None) -> Tuple[Any, Any]:
        if cut is None:
            low, high = self.data.geometry[time].tolist()[0].grd_bndbox[_map_axes[axis]]
            cut = 0.5 * (high - low) + low

        if not self.plot_options.persist:
            self.plot_options = PlotOptions()
        if options is not None:
            self.plot_options = replace(self.plot_options, **options)
        if self.plot_options.title is None:
            time = time if isinstance(time, float) else self.data.geometry[time].tolist()[0].key
            cut_text = f'' if self.data.geometry[time].tolist()[0].grd_dim == 2 else f'{axis} = {cut} and '
            self.plot_options = replace(self.plot_options, **{'title' : f"Field = '{field}'   @ " +
                                                              cut_text + f'time = {time:.2f}'})

        if line is None:
            fig, ax = self._simple_plot2D(field=field, plane=Plane(time=time, axis=axis, cut=cut))
        else:
            fig, ax = self._simple_plotLine(field=field, plane1=Plane(time=time, axis=axis, cut=cut),
                                            planes=[Plane(time=time, axis=line, cut=cuts) for cuts in cutlines],
                                            scale=scale)

        if not self.fig_options.show_differed:
            self.show()

        return fig, ax

    def show(self) -> None:
        pyplot.show()

    def _simple_plot2D(self, *, field: str, plane: Plane) -> Tuple[Any, Any]:
        fig, ax = self._make_figure()

        if self.data.geometry[plane.time].tolist()[0].grd_dim == 3:
            blocks = self._blocks_from_plane(plane)
            blocks = [block for block in blocks if
                      block in map(lambda x: x[0], self.data.geometry[plane.time].tolist()[0].blk_filtered)]
            index, _ = self._cut_from_plane(plane, blocks)
        else:
            plane.axis = 'z'
            blocks = [block[0] for block in self.data.geometry[plane.time].tolist()[0].blk_filtered]
            index, _ = (0, 0.5)

        # plot field @ plane by blocks
        for block in blocks[:]:
            self._plot2D_from_block(plane=plane, field=field, block=block, index=index, axes=ax)

        # set figure and plot font settings
        matplotlib.rcParams.update({'font.size': self.plot_options.font_size,
                                    'font.family': self.plot_options.font_face})
        font = FontProperties()
        font.set_size(self.plot_options.font_size)
        font.set_name(self.plot_options.font_face)

        font_fig = FontProperties()
        font_fig.set_size(self.fig_options.font_size)
        font_fig.set_name(self.fig_options.font_face)

        # set figure options
        fig.suptitle(self.fig_options.title, ha='center', fontproperties=font_fig)

        # set plot options
        ax.set_title(self.plot_options.title, loc='left', fontproperties=font)
        ax.set_xlim(self.data.geometry[plane.time].tolist()[0].grd_bndbox[_map_mesh_axis[plane.axis][0]])
        ax.set_ylim(self.data.geometry[plane.time].tolist()[0].grd_bndbox[_map_mesh_axis[plane.axis][1]])
        ax.set_xlabel(_map_label[plane.axis](self.plot_options, 'x'), fontproperties=font)
        ax.set_ylabel(_map_label[plane.axis](self.plot_options, 'y'), fontproperties=font)
        ax.tick_params(axis='both', which='major', labelsize=self.plot_options.font_size)
        ax.tick_params(axis='both', which='minor', labelsize=self.plot_options.font_size - 2)

        # arange figure and margin
        fig.tight_layout(pad=1.10, rect=(0, 0, 1, 0.95))

        # give figure and axes to caller
        return fig, ax

    def _simple_plotLine(self, *, field: str, plane1: Plane,
                         planes: List[Plane], scale: float) -> Tuple[Any, Any]:
        cmap = cm.Dark2 # pylint: disable=no-member
        fig, ax = self._make_figure()

        if self.data.geometry[plane1.time].tolist()[0].grd_dim == 3:
            blocks = self._blocks_from_plane(plane1)
            blocks = [block for block in blocks if
                      block in map(lambda x: x[0], self.data.geometry[plane1.time].tolist()[0].blk_filtered)]
            index1, _ = self._cut_from_plane(plane1, blocks)
        else:
            plane1.axis = 'z'
            blocks = [block[0] for block in self.data.geometry[plane1.time].tolist()[0].blk_filtered]
            index1, _ = (0, 0.5)

        label = []
        for line, plane in enumerate(planes):
            label.append(f'{plane.axis} = {plane.cut}')
            if self.data.geometry[plane1.time].tolist()[0].grd_dim == 3:
                blocks2 = [block for block in blocks if
                           block in self._blocks_from_plane(plane)]
                index2, _ = self._cut_from_plane(plane, blocks2)
            else:
                blocks2 = [block for block in blocks if
                           block in self._blocks_from_plane(plane)]
                index2, _ = self._cut_from_plane(plane, blocks2)

            # plot field @ plane by blocks
            for block in blocks2[:]:
                self._plotLine_from_block(plane1=plane1, plane2=plane, field=field, block=block, scale=scale,
                                          index1=index1, index2=index2, color=cmap(line), axes=ax)

        # set figure and plot font settings
        plane2 = planes[0]
        matplotlib.rcParams.update({'font.size': self.plot_options.font_size,
                                    'font.family': self.plot_options.font_face})
        font = FontProperties()
        font.set_size(self.plot_options.font_size)
        font.set_name(self.plot_options.font_face)

        font_fig = FontProperties()
        font_fig.set_size(self.fig_options.font_size)
        font_fig.set_name(self.fig_options.font_face)

        # set figure options
        fig.suptitle(self.fig_options.title, ha='center', fontproperties=font_fig)

        # set plot options
        ax.set_title(self.plot_options.title, loc='left', fontproperties=font)
        ax.set_xlim(self.data.geometry[plane2.time].tolist()[0].grd_bndbox[_map_mesh_axis[plane2.axis][0]])
        ax.set_xlabel(_map_line_label[plane1.axis + plane2.axis](self.plot_options), fontproperties=font)
        ax.tick_params(axis='both', which='major', labelsize=self.plot_options.font_size)
        ax.tick_params(axis='both', which='minor', labelsize=self.plot_options.font_size - 2)

        hdls = [lines.Line2D([], [], color=cmap(i), linestyle='-', label=label[i]) for i in range(len(planes))]
        ax.legend(handles=hdls)

        # arange figure and margin
        fig.tight_layout(pad=1.10, rect=(0, 0, 1, 0.95))

        # give figure and axes to caller
        return fig, ax

    def _make_figure(self) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        fig = pyplot.figure(figsize=self.fig_options.size_single)
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax

    def _blocks_from_plane(self, plane: Plane) -> List[int]:
        return list(map(lambda block: block[0], filter(lambda bound: bound[1][0] < plane.cut <= bound[1][1],
            enumerate(self.data.geometry[plane.time]['blk_bndbox'][0, :, _map_axes[plane.axis]][0].tolist()
                      ))))

    def _cut_from_plane(self, plane: Plane, blocks: List[int]) -> Tuple[int, float]:
        points = self.data.geometry[plane.time][
            _map_grid[plane.axis]][_map_points[plane.axis](blocks[0])][0].tolist()

        try:
            index = _first_true(enumerate(points), lambda point: point[1] > plane.cut)[0] - 1
        except StopIteration:
            index = len(points) - 1

        return index, points[index]

    def _plot2D_from_block(self, *, plane: Plane, field: str, block: int, index: int,
                         axes: matplotlib.axes.Axes) -> None:

        _map_plot[self.plot_options.type](axes)(
            *tuple(self.data.geometry[plane.time][_map_mesh[plane.axis]][_map_plane[plane.axis](block, index)]),
            self.data.fields[plane.time]['_' + field][_map_plane[plane.axis](block, index)][0],
            levels=numpy.linspace(0, 1, 15), extend='both')

    def _plotLine_from_block(self, *, plane1: Plane, plane2: Plane, field: str, block: int,
                             index1: int, index2: int, scale: float,
                             color: Tuple[float, float, float, float],
                             axes: matplotlib.axes.Axes) -> None:

        pyplot.plot(
            self.data.geometry[plane1.time][_map_mesh_line[plane1.axis + plane2.axis]][
                _map_line[plane1.axis + plane2.axis](block, index1, index2)][0],
            self.data.fields[plane1.time]['_' + field][
                _map_line[plane1.axis + plane2.axis](block, index1, index2)][0] * scale, color=color)
