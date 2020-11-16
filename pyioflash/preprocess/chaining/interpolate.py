"""
 Stub implementation for flow field interpolation methods 

"""
from typing import List
from sys import stdout
import os

import h5py
import numpy
from scipy.interpolate import griddata

from pyioflash import SimulationData

# define the module api
def __dir__() -> List[str]:
    return ['simple']

epsilon = 0.0
termwid = 100

def screen_out(*, message: str, percent: float) -> None:
    total = 20
    progress = '  [' + '#' * int(total * percent) + '-' * int(total * (1 - percent)) + ']'
    message = message + progress
    stdout.write('\r' + message.ljust(termwid, ' '))
    stdout.flush()

def simple(*, lowpath: str, hghpath: str, basename: str, header: str, low: int, high: int = 0, method: str = 'linear') -> None:
    
    # get low resolution geometry and field data
    lowpath = os.getcwd() + '/' + lowpath
    data = SimulationData.from_list([low], path=lowpath, basename=basename, header=header)
    lowdata = data.geometry
    lowvals = data.fields
    del[data]

   # get high resolution geometry 
    hghpath = os.getcwd() + '/' + hghpath
    data = SimulationData.from_list([high], path=hghpath, basename=basename, header=header)
    hghdata = data.geometry
    del[data]

    # create field name associations and empty lists
    names = {'temp': '_temp', 'velx': '_fcx2', 'vely': '_fcy2', 'velz': '_fcz2'}
    grids = {name: [] for name in names.keys()}

    # traverse high resolution domain and interpolate
    for block, bbox in enumerate(hghdata.blk_bndbox):
        blocks = _blocks_from_bbox(lowdata, bbox)
    
        message = f'Interpolating blocks for block {block} are {blocks}'
        screen_out(message=message, percent=0.05)

        xxxc = lowdata._grd_mesh_x[1, blocks, :, :, :]
        yyyc = lowdata._grd_mesh_y[1, blocks, :, :, :]
        zzzc = lowdata._grd_mesh_z[1, blocks, :, :, :]
        screen_out(message=message, percent=0.10)

        xxxr = lowdata._grd_mesh_x[2, blocks, :, :, :]
        yyyr = lowdata._grd_mesh_y[2, blocks, :, :, :]
        zzzr = lowdata._grd_mesh_z[2, blocks, :, :, :]
        screen_out(message=message, percent=0.15)
    
        fields = {name: value for name, value in zip(names.keys(), lowvals[list(names.values())][0, blocks, :, :, :])}
        screen_out(message=message, percent=0.20)
        
        # Interpolate Temperature
        x = hghdata._grd_mesh_x[1, block, :, :, :]
        y = hghdata._grd_mesh_y[1, block, :, :, :]
        z = hghdata._grd_mesh_z[1, block, :, :, :]
        points = numpy.array([[i, j, k] for i, j, k in zip(xxxc.ravel(), yyyc.ravel(), zzzc.ravel())])     
        values = fields['temp'].ravel()
        grids['temp'].append(griddata(points, values, (x, y, z), method=method))
        screen_out(message=message, percent=0.40)
    
        # Interpolate faceX
        x = hghdata._grd_mesh_x[2, block, :, :, :]
        y = hghdata._grd_mesh_y[1, block, :, :, :]
        z = hghdata._grd_mesh_z[1, block, :, :, :]
        points = numpy.array([[i, j, k] for i, j, k in zip(xxxr.ravel(), yyyc.ravel(), zzzc.ravel())])     
        values = fields['velx'].ravel()
        grids['velx'].append(griddata(points, values, (x, y, z), method=method))
        screen_out(message=message, percent=0.60)
    
        # Interpolate faceY
        x = hghdata._grd_mesh_x[1, block, :, :, :]
        y = hghdata._grd_mesh_y[2, block, :, :, :]
        z = hghdata._grd_mesh_z[1, block, :, :, :]
        points = numpy.array([[i, j, k] for i, j, k in zip(xxxc.ravel(), yyyr.ravel(), zzzc.ravel())])     
        values = fields['vely'].ravel()
        grids['vely'].append(griddata(points, values, (x, y, z), method=method))
        screen_out(message=message, percent=0.80)

        # Interpolate faceZ
        x = hghdata._grd_mesh_x[1, block, :, :, :]
        y = hghdata._grd_mesh_y[1, block, :, :, :]
        z = hghdata._grd_mesh_z[2, block, :, :, :]
        points = numpy.array([[i, j, k] for i, j, k in zip(xxxc.ravel(), yyyc.ravel(), zzzr.ravel())])
        values = fields['velz'].ravel()
        grids['velz'].append(griddata(points, values, (x, y, z), method=method))
        screen_out(message=message, percent=1.00)

    # repackage list of blocks as numpy array
    grids = {key: numpy.array(value)[:, :, :, :] for key, value in grids.items()} 

    # specify path and filename
    filename = hghpath + 'initBlock.h5'
    if os.path.exists(filename):
        os.remove(filename)

    # create hdf5 file
    with h5py.File(filename) as h5file:

        # write data to file
        for field, data in zip(names.keys(), grids.values()):
            h5file.create_dataset(field, data=data)

def _blocks_from_bbox(data, bbox):

    # return when within a open interval; corner in [low, high)
    def within(corner, box, dimension):
        eps = epsilon
        return all([low - eps <= check and check < high + eps
                    for axis, (check, (low, high)) in enumerate(zip(corner, box))
                    if axis < dimension])

    # unpack bbox object and identify corners
    corners = [(i, j, k) for k in bbox[2] for j in bbox[1] for i in bbox[0]]

    # retrieve bounding boxes and dimensionality
    boxes = data.blk_bndbox
    dimension = data.grd_dim

    # return blocks which are intersected by bounding box
    return [block for block, box in enumerate(boxes)
            if any([within(corner, box, dimension) for corner in corners])]
