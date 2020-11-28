###########################################################
###########################################################

# define the parameters for the interpolation
high_path = 'trb_ra08/initial/neumann/'
low_path  = 'trb_ra06/initial/neumann/'
low_file  = '0000'
basename  = 'INS_Rayleigh_' 
gridname  = 'hdf5_grd_0000'
plotname  = 'hdf5_plt_cnt_'

# define desired datasets in the low resolution data to interpolate
fld_names = {'temp': 'temp', 'velx': 'fcx2', 'vely': 'fcy2', 'velz': 'fcz2'}
grd_names = {axis * 3 + face for axis in 'xyz' for face in 'cr'}
fld_adjsp = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0))

###########################################################
###########################################################
termwid = 120
epsilon = 0.0

from sys import stdout
import os

import h5py
import numpy
from scipy.interpolate import griddata

from mpi4py import MPI

def _first_true(iterable, predictor):
    return next(filter(predictor, iterable))

def _blocks_from_bbox(low_blk_bndbox, low_grd_dim, bbox):
    def within(corner, box, dimension):
        eps = epsilon
        return all([low - eps <= check and check < high + eps
                    for axis, (check, (low, high)) in enumerate(zip(corner, box))
                    if axis < dimension])
    corners = [(i, j, k) for k in bbox[2] for j in bbox[1] for i in bbox[0]]
    boxes = low_blk_bndbox
    dimension = low_grd_dim
    return [block for block, box in enumerate(boxes)
            if any([within(corner, box, dimension) for corner in corners])]

def screen_out(*, message: str, inner: float, outer: float) -> None:
    total = 16
    progress = '  [' + '#' * int(total * inner) + '-' * int(total * (1 - inner)) + ']' + f' with {100*outer:0.1f}% Complete'
    message = message + progress
    stdout.write('\r' + message.ljust(termwid, ' '))
    stdout.flush()

# get mpi parameters
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Write braces to the screen
if rank == 0:
    print()
    print('#' * termwid)
    print('Begin Interpolation of Simulation Results    ')
    print()

# read simulation information from high resolution
filename = os.getcwd() + '/' + high_path + basename + plotname + '0000'
with h5py.File(filename, 'r') as file:

    int_runtime = list(file['integer runtime parameters'])
    int_scalars = list(file['integer scalars'])
    boundingbox = file['bounding box']

    high_blk_num = _first_true(int_scalars, lambda l: 'globalnumblocks' in str(l[0]))[1]
    high_blk_num_x = _first_true(int_runtime, lambda l: 'iprocs' in str(l[0]))[1]
    high_blk_num_y = _first_true(int_runtime, lambda l: 'jprocs' in str(l[0]))[1]
    high_blk_num_z = _first_true(int_runtime, lambda l: 'kprocs' in str(l[0]))[1]
    high_blk_size_x = _first_true(int_scalars, lambda l: 'nxb' in str(l[0]))[1]
    high_blk_size_y = _first_true(int_scalars, lambda l: 'nyb' in str(l[0]))[1]
    high_blk_size_z = _first_true(int_scalars, lambda l: 'nzb' in str(l[0]))[1]

    high_blk_bndbox = numpy.ndarray(boundingbox.shape, dtype=numpy.dtype(float))
    high_blk_bndbox[:, :, :] = boundingbox    

# read simulation information from low resolution
filename = os.getcwd() + '/' + low_path + basename + plotname + low_file
with h5py.File(filename, 'r') as file:
    
    int_scalars = list(file['integer scalars'])
    boundingbox = file['bounding box']

    low_blk_num = _first_true(int_scalars, lambda l: 'globalnumblocks' in str(l[0]))[1]
    low_blk_size_x = _first_true(int_scalars, lambda l: 'nxb' in str(l[0]))[1]
    low_blk_size_y = _first_true(int_scalars, lambda l: 'nyb' in str(l[0]))[1]
    low_blk_size_z = _first_true(int_scalars, lambda l: 'nzb' in str(l[0]))[1]

    low_grd_dim = _first_true(int_scalars, lambda l: 'dimensionality' in str(l[0]))[1]
    low_blk_bndbox = numpy.ndarray(boundingbox.shape, dtype=numpy.dtype(float))
    low_blk_bndbox[:, :, :] = boundingbox    

# read and calculate grid data from low resolution
filename = os.getcwd() + '/' + low_path + basename + gridname  
with h5py.File(filename, 'r') as file:
    low_grd_cords = {name: file[name][:, :] for name in grd_names}
    low_grd_adjsp = {'velx': file['xxxl'][:, 0],
                     'vely': file['yyyl'][:, 0],
                     'velz': file['zzzl'][:, 0]}
    low_grd_shape = {name: (low_blk_num, low_blk_size_z + k, 
                            low_blk_size_y + j, low_blk_size_x + i) for 
                     name, (k, j, i) in zip(fld_names.keys(), fld_adjsp)}

# read and calculate grid data from high resolution
filename = os.getcwd() + '/' + high_path + basename + gridname  
with h5py.File(filename, 'r') as file:
    high_grd_cords = {name: file[name][:, :] for name in grd_names}
    high_grd_adjsp = {'velx': file['xxxl'][:, 0],
                      'vely': file['yyyl'][:, 0],
                      'velz': file['zzzl'][:, 0]}
    high_grd_shape = {name: (high_blk_num, high_blk_size_z + k, 
                             high_blk_size_y + j, high_blk_size_x + i) for 
                      name, (k, j, i) in zip(fld_names.keys(), fld_adjsp)}

# distribute across communicator
avg, res    = divmod(high_blk_num, size)
index_width = avg + 1 if rank < res else avg 
index_low   =  rank      * (avg + 1)     if rank < res else res * (avg + 1) + (rank - res    ) * avg
index_high  = (rank + 1) * (avg + 1) - 1 if rank < res else res * (avg + 1) + (rank - res + 1) * avg - 1

# only print from root process
write_out = {True: screen_out, False: (lambda **_: None)}[rank == 0]

inp_name = os.getcwd() + '/' + low_path + basename + plotname + low_file
out_name = os.getcwd() + '/' + 'initBlock.h5' 
with h5py.File(inp_name, 'r') as inp_file, h5py.File(out_name, 'w', driver='mpio', comm=comm) as out_file:

    # create datasets in output file
    dsets = {name: out_file.create_dataset(name, high_grd_shape[name], dtype=numpy.float) for name in fld_names.keys()}

    # interpolate over assigned blocks
    for step, (block, bbox) in enumerate(zip(range(index_low,index_high+1), high_blk_bndbox[index_low:index_high+1])):
        blocks = _blocks_from_bbox(low_blk_bndbox, low_grd_dim, bbox)

        # define standard output message and initial percentage
        message = f'Interpolating blocks for block {block} are {blocks}'
        write_out(message=message, inner=0.20, outer=(step + 0.20)/index_width)

        # interpolate cell center
        xxx = low_grd_cords['xxxc'][blocks, None, None, :].repeat(low_blk_size_y, 2).repeat(low_blk_size_z, 1).ravel()
        yyy = low_grd_cords['yyyc'][blocks, None, :, None].repeat(low_blk_size_x, 3).repeat(low_blk_size_z, 1).ravel()
        zzz = low_grd_cords['zzzc'][blocks, :, None, None].repeat(low_blk_size_x, 3).repeat(low_blk_size_y, 2).ravel()
        
        x = high_grd_cords['xxxc'][block, None, None, :].repeat(high_blk_size_y, 1).repeat(high_blk_size_z, 0)
        y = high_grd_cords['yyyc'][block, None, :, None].repeat(high_blk_size_x, 2).repeat(high_blk_size_z, 0)
        z = high_grd_cords['zzzc'][block, :, None, None].repeat(high_blk_size_x, 2).repeat(high_blk_size_y, 1)
        
        values = inp_file[fld_names['temp']][blocks, :, :, :].ravel()
        dsets['temp'][block] = griddata((xxx, yyy, zzz), values, (x, y, z), method='nearest')
        write_out(message=message, inner=0.40, outer=(step + 0.40)/index_width)


        # interpolate faceX
        xxx = numpy.empty( (len(blocks), ) + low_grd_shape['velx'][1:], dtype=numpy.float) 
        yyy, zzz = numpy.empty_like(xxx), numpy.empty_like(xxx) 
        xxx[:, :, :, 0] = low_grd_adjsp['velx'][blocks, None, None]
        xxx[:, :, :,1:] = low_grd_cords['xxxr'][blocks, None, None, :]
        yyy[:, :, :, :] = low_grd_cords['yyyc'][blocks, None, :, None] 
        zzz[:, :, :, :] = low_grd_cords['zzzc'][blocks, :, None, None] 
        xxx, yyy, zzz = xxx.ravel(), yyy.ravel(), zzz.ravel()

        x = numpy.empty(high_grd_shape['velx'][1:], dtype=numpy.float)
        y, z = numpy.empty_like(x), numpy.empty_like(x)
        x[ :, :, 0] = high_grd_adjsp['velx'][block]
        x[ :, :,1:] = high_grd_cords['xxxr'][block, None, None, :]
        y[ :, :, :] = high_grd_cords['yyyc'][block, None, :, None]
        z[ :, :, :] = high_grd_cords['zzzc'][block, :, None, None]

        values = inp_file[fld_names['velx']][blocks, :, :, :].ravel()
        dsets['velx'][block] = griddata((xxx, yyy, zzz), values, (x, y, z), method='nearest')
        write_out(message=message, inner=0.60, outer=(step + 0.60)/index_width)


        # interpolate faceY
        xxx = numpy.empty( (len(blocks), ) + low_grd_shape['vely'][1:], dtype=numpy.float) 
        yyy, zzz = numpy.empty_like(xxx), numpy.empty_like(xxx) 
        yyy[:, :, 0, :] = low_grd_adjsp['vely'][blocks, None, None]
        xxx[:, :, :, :] = low_grd_cords['xxxc'][blocks, None, None, :] 
        yyy[:, :,1:, :] = low_grd_cords['yyyr'][blocks, None, :, None]
        zzz[:, :, :, :] = low_grd_cords['zzzc'][blocks, :, None, None] 
        xxx, yyy, zzz = xxx.ravel(), yyy.ravel(), zzz.ravel()

        x = numpy.empty(high_grd_shape['vely'][1:], dtype=numpy.float)
        y, z = numpy.empty_like(x), numpy.empty_like(x)
        y[ :, 0, :] = high_grd_adjsp['vely'][block]
        x[ :, :, :] = high_grd_cords['xxxc'][block, None, None, :]
        y[ :,1:, :] = high_grd_cords['yyyr'][block, None, :, None]
        z[ :, :, :] = high_grd_cords['zzzc'][block, :, None, None]

        values = inp_file[fld_names['vely']][blocks, :, :, :].ravel()
        dsets['vely'][block] = griddata((xxx, yyy, zzz), values, (x, y, z), method='nearest')
        write_out(message=message, inner=0.80, outer=(step + 0.80)/index_width)


        # interpolate faceZ
        xxx = numpy.empty( (len(blocks), ) + low_grd_shape['velz'][1:], dtype=numpy.float) 
        yyy, zzz = numpy.empty_like(xxx), numpy.empty_like(xxx) 
        zzz[:, 0, :, :] = low_grd_adjsp['velz'][blocks, None, None]
        xxx[:, :, :, :] = low_grd_cords['xxxc'][blocks, None, None, :] 
        yyy[:, :, :, :] = low_grd_cords['yyyc'][blocks, None, :, None]
        zzz[:,1:, :, :] = low_grd_cords['zzzr'][blocks, :, None, None] 
        xxx, yyy, zzz = xxx.ravel(), yyy.ravel(), zzz.ravel()

        x = numpy.empty(high_grd_shape['velz'][1:], dtype=numpy.float)
        y, z = numpy.empty_like(x), numpy.empty_like(x)
        z[ 0, :, :] = high_grd_adjsp['velz'][block]
        x[ :, :, :] = high_grd_cords['xxxc'][block, None, None, :]
        y[ :, :, :] = high_grd_cords['yyyc'][block, None, :, None]
        z[1:, :, :] = high_grd_cords['zzzr'][block, :, None, None]

        values = inp_file[fld_names['velz']][blocks, :, :, :].ravel()
        dsets['velz'][block] = griddata((xxx, yyy, zzz), values, (x, y, z), method='nearest')
        write_out(message=message, inner=1.00, outer=(step + 1.00)/index_width)

# clear the screen
if rank == 0:
    print()
    print()
    print('#' * termwid)
    print()

