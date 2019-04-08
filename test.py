from pyioflash import SimulationData, SimulationPlot

#data = SimulationData.from_list(range(5), path='../../../qual2/', header='INS_Rayleigh_Benard_hdf5_plt_cnt_')
data = SimulationData.from_list([681], path='../../../amr/', header='INS_LidDr_Cavity_hdf5_plt_cnt_')
visual = SimulationPlot(data, fig_options={'title': f'Rayleigh Benard Convection'})

#for block, neighbors in enumerate(data.geometry.tolist()[0].blk_neighbors):
#    print(f'{block + 1} \t--\t {neighbors}')

visual.plot(axis='x', cut=0.5, time=60.0, field='temp', options={'title': f'Temperature @ x=0.5 and t=60.0'})
visual.plot(cut=0.5, time=60.0, field='temp')
visual.show()


#print(data.geometry[60.0].tolist()[0])
#print(data.geometry[60.0 : 70.0]['grd_type', 'grd_dim'])
#print(data.dynamics[60.0 : 70.0]['integer_scalars']['nxb'])



