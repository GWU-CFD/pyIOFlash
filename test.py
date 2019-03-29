from pyioflash import SimulationData, simple_contour

data = SimulationData.from_list(range(20), path='../../../qual2/', header='INS_Rayleigh_Benard_hdf5_plt_cnt_')

simple_contour(data, 60.7, ('z', 0.5), 'temp')


