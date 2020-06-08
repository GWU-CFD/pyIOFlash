"""
Python Module for providing methods to import
and process FLASH4 HDF5 plt and chk files
"""

# Used to import and handle simulation data
from pyioflash.simulation import SimulationData
from pyioflash.simulation import NameData

# Used to create professional 2D plots
#from pyioflash.visual import SimulationPlot
from pyioflash.visual import FigureOptions, PlotOptions, AnimationOptions
from pyioflash.visual import Plane

# Used to post-process simulation data
from pyioflash.postprocess import abs_thermal_energy
from pyioflash.postprocess import rel_thermal_energy
from pyioflash.postprocess import abs_kinetic_energy
from pyioflash.postprocess import rel_kinetic_energy

# Used to generate and pre-process simulation data


