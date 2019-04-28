pyIOFlash Readme
================

A simple python Module for providing methods to import, process, and plot FLASH4 HDF5 plot and chk files

*Insert more detailed description of the package here*


Information
-----------

:Authors:	Aaron Lentner, Akash Dhruv
:Revision:	1.0.34 (pre-alpha)

*Needed Feature List*

- general improvements to 'guard' data filling
- quiver and and animated plots
- out-of-core support
- [additional on github]

*Major revision roadmap*

- initial functioning Plot Utility
- initial out-of-core processing
- initial multiprocessing functionality
- add parser for EDDY6 code
- [additional on github]


Quick Reference
---------------------

*Usage*::

  from pyio import SimulationData

  data = SimulationData.from_list(range(20), path='test/',
                                  header='INS_Rayleigh_Benard_hdf5_plt_cnt_')

  data.fields[20.0 : 60.0 : 2]['temp', 'pres'][:, :, :, :, :]


*Reference*

+----------------------------------------------------------+
|SimulationData                                            |
+===============+==========================================+
|geometry       |geometry information                      |
+---------------+------------------------------------------+
|fields         |simulation unknowns                       |
+---------------+------------------------------------------+
|scalars        |(time, dt, nstep, nbegin)                 |
+---------------+------------------------------------------+
|dynamics       |remaining hdf5 plt file (time varying)    |
+---------------+------------------------------------------+
|statics        |remaining hdf5 plt file (steady w/ time)  |
+---------------+------------------------------------------+

+----------------------------------------------------------+
|SimulationData                                            |
+===============+==========================================+
|plot           |method for plotting 2D and line           |
+---------------+------------------------------------------+
|animate        |method for animating 2D and line plots    |
+---------------+------------------------------------------+
|show           |method for showing / saving plots         |
+---------------+------------------------------------------+
