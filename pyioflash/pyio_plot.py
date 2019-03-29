from dataclasses import dataclass
from typing import List, Union

import numpy
from matplotlib import pyplot

from .pyio import SimulationData
from .pyio_utility import _first_true


@dataclass
class Plane:
    time: Union[float, int]
    axis: str
    cut: float

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

def _blocks_from_plane(data: SimulationData, plane: Plane) -> List[int]:
    return list(map(lambda block: block[0], filter(lambda bound: bound[1][0] < plane.cut <= bound[1][1],
        enumerate(data.geometry[plane.time: plane.time + 1]['blk_bndbox'][0, :, _map_axes[plane.axis]][0].tolist()
                  ))))

def _cut_from_plane(data: SimulationData, plane: Plane, blocks: List[int]):
    points = data.geometry[plane.time: plane.time + 1][_map_grid[plane.axis]][_map_points[plane.axis](blocks[0])][0].tolist()

    try: 
        index = _first_true(enumerate(points), lambda point: point[1] > plane.cut)[0] - 1
    except StopIteration:
        index = len(points) - 1

    return index, points[index]

def simple_contour(data: SimulationData, plane: Plane, field: str) -> None:
    blocks = _blocks_from_plane(data, plane)
    index, point = _cut_from_plane(data, plane, blocks)

    for block in blocks:
        pyplot.contourf(
            *tuple(data.geometry[plane.time: plane.time + 1][_map_mesh[plane.axis]][_map_plane[plane.axis](block, index)]),
            data.fields[plane.time: plane.time + 1][field][_map_plane[plane.axis](block, index)][0],
            levels=numpy.linspace(0, 1, 15))

    pyplot.xlim(0, 1.0)
    pyplot.ylim(0, 1.0)
    pyplot.show()
