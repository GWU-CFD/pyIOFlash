from pyioflash import SimulationData, SimulationPlot
from pyioflash.pyio_utility import Plane

data = SimulationData.from_list(range(5), path='../../../qual2/', header='INS_Rayleigh_Benard_hdf5_plt_cnt_')
visual = SimulationPlot(data, fig_options={'title': f'Rayleigh Benard Convection'})

visual.plot(axis='y', cut=0.5, time=60.0, field='temp', options={'title': f'Temperature @ y=0.5 and t=60.0'})
visual.show()
