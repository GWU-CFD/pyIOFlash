"""
Python sub-package for providing methods to  
post-process simulation data; specifically, methods 
to import, mutate, and conduct time-series or permutation
analysis.
"""

# Used to povide most simple importing of meaningful data
from pyioflash.postprocess.sources import energy

# Used to provide most simple operations on data
from pyioflash.postprocess.elements import integral
#from pyioflash.postprocess.elements import derivative
#from pyioflash.postprocess.elements import relative 

# Used to provide most simple time-series like analysis
from pyioflash.postprocess.analysis import series
