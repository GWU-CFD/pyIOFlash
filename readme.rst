pyIOFlash Package
=================

A simple python Module for providing methods to import, process, and plot FLASH4 HDF5 plot and chk files

*Insert more detailed description of the package here*

Information
-----------

Full package documentation can be found at the GitHub Pages for pyIOFlash_.

.. _pyIOFlash: https://pyioflash.readthedocs.io

Jupyter Notebook based examples_ of using the library are also provided.

.. _examples: https://nbviewer.jupyter.org/github/GWU-CFD/pyIOFlash/tree/release/examples/ 

:Authors:	Aaron Lentner, Akash Dhruv
:Revision:	1.0.55 (pre-alpha)

*Needed Feature List*

- general improvements to 'boundary' data filling
- out-of-core and multiprossessing support
- [additional on github]

*Major revision roadmap*

- initial alpha release
- initial out-of-core processing
- initial multiprocessing functionality
- add parser for EDDY6 code
- [additional on github]


Quick Reference
---------------------

*Usage*::

  from pyio import SimulationData

  data = SimulationData.from_list(range(20), path='../out/',
                                  header='INS_LidDr_Cavity_hdf5_plt_cnt_')

  T, P = data.fields[20.0 : 60.0 : 2]['temp', 'pres'][:, :, :, :, :]
  u,   = data.fields[20.0 : 60.0 : 2]['fcx2'][:, :, :, :, :]


*Reference*

+-------------------------------------------------------------+
|SimulationData                                               |
+===============+=============================================+
|geometry       |geometry information                         |
+---------------+---------------------------------------------+
|fields         |simulation unknowns                          |
+---------------+---------------------------------------------+
|scalars        |(time, dt, nstep, nbegin)                    |
+---------------+---------------------------------------------+
|dynamics       |remaining hdf5 file info (time varying)      |
+---------------+---------------------------------------------+
|statics        |remaining hdf5 file info (steady w/ time)    |
+---------------+---------------------------------------------+
|utility        |helper methods for interacting with data     |
+---------------+---------------------------------------------+

+-------------------------------------------------------------+
|visual                                                       |
+===============+=============================================+
|plot           |method for plotting 2D and line              |
+---------------+---------------------------------------------+
|animate        |method for animating 2D and line plots       |
+---------------+---------------------------------------------+

+-------------------------------------------------------------+
|sources                                                      |
+===============+=============================================+
|energy         |methods for calculating energies             |
+---------------+---------------------------------------------+
|fields         |methods for calculating fields from data     |
+---------------+---------------------------------------------+

+-------------------------------------------------------------+
|elements                                                     |
+===============+=============================================+
|integral       |methods for calculating integrals of fields  |
+---------------+---------------------------------------------+

+-------------------------------------------------------------+
|analyses                                                     |
+===============+=============================================+
|series         |methods for simple time-series like analysis |
+---------------+---------------------------------------------+
