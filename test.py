from pyioflash import SimulationData, SimulationPlot

data = SimulationData.from_list(range(20), path='../../../qual2/', header='INS_Rayleigh_Benard_hdf5_plt_cnt_')
visual = SimulationPlot(data)

visual.plot(axis='y', cut=0.5, time=60.7, field='temp')
visual.show()

