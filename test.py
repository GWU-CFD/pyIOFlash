from pyioflash import SimulationData, SimulationPlot

data = SimulationData.from_list(range(5), path='../../../qual2/', header='INS_Rayleigh_Benard_hdf5_plt_cnt_')
#data = SimulationData.from_list([681], path='../../../amr/', header='INS_LidDr_Cavity_hdf5_plt_cnt_')
visual = SimulationPlot(data, fig_options={'title': f'Rayleigh Benard Convection'})

visual.plot(axis='x', field='temp')
visual.plot(field='temp')
visual.show()



