###########################################################
###########################################################

# define the parameters for the interpolation
high_path = 'trb_ra06/initial/neumann/'
high_file = '0000'
basename  = 'INS_Rayleigh_' 
gridname  = 'hdf5_grd_0000'
plotname  = 'hdf5_plt_cnt_'

# define desired datasets in the low resolution data to interpolate
fld_names = {'temp': 'temp', 'velx': 'fcx2', 'vely': 'fcy2', 'velz': 'fcz2'}
grd_names = {axis * 3 + face for axis in 'xyz' for face in 'cr'}
fld_adjsp = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0))

###########################################################
###########################################################

import os
import numpy
import h5py
import pyvista

def _first_true(iterable, predictor):
    return next(filter(predictor, iterable))

# read simulation information from high resolution
filename = os.getcwd() + '/' + high_path + basename + plotname + high_file
with h5py.File(filename, 'r') as file:

    int_runtime = list(file['integer runtime parameters'])
    int_scalars = list(file['integer scalars'])

    high_blk_num = _first_true(int_scalars, lambda l: 'globalnumblocks' in str(l[0]))[1]
    high_blk_num_x = _first_true(int_runtime, lambda l: 'iprocs' in str(l[0]))[1]
    high_blk_num_y = _first_true(int_runtime, lambda l: 'jprocs' in str(l[0]))[1]
    high_blk_num_z = _first_true(int_runtime, lambda l: 'kprocs' in str(l[0]))[1]
    high_blk_size_x = _first_true(int_scalars, lambda l: 'nxb' in str(l[0]))[1]
    high_blk_size_y = _first_true(int_scalars, lambda l: 'nyb' in str(l[0]))[1]
    high_blk_size_z = _first_true(int_scalars, lambda l: 'nzb' in str(l[0]))[1]

# read and calculate grid data from file
filename = os.getcwd() + '/' + high_path + basename + gridname  
with h5py.File(filename, 'r') as file:
    high_grd_cords = {name: file[name][:, :] for name in grd_names}
    high_grd_adjsp = {'velx': file['xxxl'][:, 0],
                      'vely': file['yyyl'][:, 0],
                      'velz': file['zzzl'][:, 0]}
    high_grd_shape = {name: (high_blk_num, high_blk_size_z + k, 
                             high_blk_size_y + j, high_blk_size_x + i) for 
                      name, (k, j, i) in zip(fld_names.keys(), fld_adjsp)}


