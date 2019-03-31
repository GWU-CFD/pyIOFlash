from dataclasses import dataclass, replace
from typing import Tuple, List, Dict, Union, Any

import numpy
import matplotlib
import matplotlib
from matplotlib import pyplot
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
_map_mesh = {'x': ('grd_mesh_y', 'grd_mesh_z'),
             'y': ('grd_mesh_x', 'grd_mesh_z'),
             'z': ('grd_mesh_x', 'grd_mesh_y')}
_map_plane = {'x': lambda block, index : numpy.index_exp[0, block, :, :, index], 
              'y': lambda block, index : numpy.index_exp[0, block, :, index, :], 
              'z': lambda block, index : numpy.index_exp[0, block, index, :, :]}
_map_plot = {'contour' : lambda ax: ax.contour,
             'contourf' : lambda ax: ax.contourf}
_map_label = {'x': lambda options, ax: (options.labels[1], options.labels[2])[_map_axes[ax]],
              'y': lambda options, ax: (options.labels[0], options.labels[2])[_map_axes[ax]],
              'z': lambda options, ax: (options.labels[0], options.labels[1])[_map_axes[ax]],}


class SimulationPlot:
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

    def plot(self, *, axis: str, cut: float, field: str, time: Union[float, int] = -1, options: Dict[str, Any] = None) -> None:
        if options is not None:
            self.plot_options = replace(self.plot_options, **options)

        self._simple_plot(field=field, plane=Plane(time=time, axis=axis, cut=cut))

    def show(self) -> None:
        pyplot.show()

    def _simple_plot(self, *, field: str, plane: Plane) -> None:
        fig, ax = self._make_figure()

        blocks = self._blocks_from_plane(plane)
        index, point = self._cut_from_plane(plane, blocks)

        for block in blocks:
            self._plot_from_block(plane=plane, field=field, block=block, index=index, axes=ax)

        matplotlib.rcParams.update({'font.size': self.plot_options.font_size,
                                    'font.family': self.plot_options.font_face})
        font = FontProperties()
        font.set_size(self.plot_options.font_size)
        font.set_name(self.plot_options.font_face)

        font_fig = FontProperties()
        font_fig.set_size(self.fig_options.font_size)
        font_fig.set_name(self.fig_options.font_face)

        fig.suptitle(self.fig_options.title, ha='center', fontproperties=font_fig)
    
        ax.set_title(self.plot_options.title, loc='left', fontproperties=font)
        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.0])
        ax.set_xlabel(_map_label[plane.axis](self.plot_options, 'x'), fontproperties=font)
        ax.set_ylabel(_map_label[plane.axis](self.plot_options, 'y'), fontproperties=font)
        ax.tick_params(axis='both', which='major', labelsize=self.plot_options.font_size)
        ax.tick_params(axis='both', which='minor', labelsize=self.plot_options.font_size - 2)

        fig.tight_layout(pad=1.10, rect=(0, 0, 1, 0.95))

    def _make_figure(self) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        fig = pyplot.figure(figsize=self.fig_options.size_single)
        ax = fig.add_subplot(1, 1, 1)     
        return fig, ax
 
    def _blocks_from_plane(self, plane: Plane) -> List[int]:
        return list(map(lambda block: block[0], filter(lambda bound: bound[1][0] < plane.cut <= bound[1][1],
            enumerate(self.data.geometry[plane.time: plane.time + 1]['blk_bndbox'][0, :, _map_axes[plane.axis]][0].tolist()
                      ))))

    def _cut_from_plane(self, plane: Plane, blocks: List[int]) -> Tuple[int, float]:
        points = self.data.geometry[plane.time: plane.time + 1][_map_grid[plane.axis]][_map_points[plane.axis](blocks[0])][0].tolist()

        try: 
            index = _first_true(enumerate(points), lambda point: point[1] > plane.cut)[0] - 1
        except StopIteration:
            index = len(points) - 1

        return index, points[index]

    def _plot_from_block(self, *, plane: Plane, field: str, block: int, index: int, 
                         axes: matplotlib.axes.Axes) -> None:
        _map_plot[self.plot_options.type](axes)(
            *tuple(self.data.geometry[plane.time: plane.time + 1][_map_mesh[plane.axis]][_map_plane[plane.axis](block, index)]),
            self.data.fields[plane.time: plane.time + 1][field][_map_plane[plane.axis](block, index)][0],
            levels=numpy.linspace(0, 1, 15))