import numpy
from matplotlib import pyplot

from pyio import SimulationData

data = SimulationData.from_list(range(20), path='../../../qual2/', header='INS_Rayleigh_Benard_hdf5_plt_cnt_')



map_axes = {'x': 0, 'y': 1, 'z': 2}
map_mesh = {'x': 'grd_mesh_x', 'y': 'grd_mesh_y', 'z': 'grd_mesh_z'}
map_slice = {'x': lambda block : numpy.index_exp[0, block, 0, 0, :], 
             'y': lambda block : numpy.index_exp[0, block, 0, :, 0], 
             'z': lambda block : numpy.index_exp[0, block, :, 0, 0]}
map_meshs = {'x': ('grd_mesh_y', 'grd_mesh_z'),
             'y': ('grd_mesh_x', 'grd_mesh_z'),
             'z': ('grd_mesh_x', 'grd_mesh_y')}
map_plane = {'x': lambda block, index : numpy.index_exp[0, block, :, :, index], 
             'y': lambda block, index : numpy.index_exp[0, block, :, index, :], 
             'z': lambda block, index : numpy.index_exp[0, block, index, :, :]}


time = 60.7
plane = ('z', 0.5)
field = 'temp'

blocks = list(map(lambda item: item[0], filter(lambda item: item[1][0] < plane[1] <= item[1][1], 
            enumerate(data.geometry[time: time + 1]['blk_bndbox'][0, :, map_axes[plane[0]]][0].tolist()))))

points = data.geometry[time: time + 1][map_mesh[plane[0]]][map_slice[plane[0]](blocks[0])][0].tolist()

try: 
    index = next(filter(lambda item: item[1] > plane[1], enumerate(points)))[0] - 1
except StopIteration:
    index = len(points) - 1
point = points[index]

print(blocks)
print(index)
print(point)

for block in blocks:
    pyplot.contourf(*tuple(data.geometry[time : time + 1][map_meshs[plane[0]]][map_plane[plane[0]](block, index)]),
                    data.fields[time : time + 1][field][map_plane[plane[0]](block, index)][0],
                   levels=numpy.linspace(0, 1, 15))
pyplot.xlim(0, 1.0)
pyplot.ylim(0, 1.0)
pyplot.show()