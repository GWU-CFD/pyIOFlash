"""
Python sub-sub-package for providing methods to perform 
time-series like post-processing of simulation data; specifically, 
methods to conduct time-series and permutation analysis.
"""

# Used to provide most simple time-series like analysis 
from pyioflash.postprocess.analyses import series

# Used to provide mechanism for consuming sources and elements
from pyioflash.postprocess.utility import make_sourceable
from pyioflash.postprocess.utility import make_stackable

#### Define other sub-sub-package modules as analyses ###
