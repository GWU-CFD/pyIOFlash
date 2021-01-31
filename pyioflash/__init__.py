"""
Python package for providing methods to import, post-process,
and visualize simulation data; specifically, FLASH4 HDF5 files
"""


# Used to import and handle simulation data
from pyioflash.simulation import SimulationData

# Used as helper methods for accessing files and data
from pyioflash.simulation import NameData
from pyioflash.simulation import DataPath
from pyioflash.simulation import Plane, Line

# Used to post-process simulation data
from pyioflash.postprocess import sources 
from pyioflash.postprocess import elements
from pyioflash.postprocess import analyses 

# Used to create professional 2D plots
#from pyioflash.visual import _simple_plot2D 
#from pyioflash.visual import FigureOptions, PlotOptions, AnimationOptions

# Used to generate and pre-process simulation data


