# pyIO
### A Simple FLASH4 HDF5 Input / Output Tool 
==================================

A simple python Module for providing methods to import, process, and plot FLASH4 HDF5 plot and chk files


   -- Revision 1.0.34 (pre-alpha)


   -- needed feature list --
   
      general improvements to 'guard' data filling
	  quiver and and animated plots
	  out-of-core support
	  [additional on github]

   -- major revision roadmap --
   
       initial functioning Plot Utility
	   initial out-of-core processing
       initial multiprocessing functionality
       add parser for EDDY6 code
       [additional on github]

### Usage
==================================

    from pyio import SimulationData
  
    data = SimulationData.from_list(range(20), path='test/', header='INS_Rayleigh_Benard_hdf5_plt_cnt_')
  
    data.fields[20.0 : 60.0 : 2]['temp', 'pres'][:, :, :, :, :]
                 -- [slicing based     ||                    ||
                 --  on time or        ||                    ||
                 --  file numbers]     ||                    ||
                                       -- [field names or    ||
                                       --  plotfile parms]   ||
                                                             -- [time : block : z, y, x]

    -- SimulationData returns an object which contains the folowing data objects:
            geometry    -- geometry information
            fields      -- simulation unknowns
            scalars     -- (time, dt, nstep, nbegin)
            dynamics    -- remaining hdf5 plt file (time varying)
            statics     -- remaining hdf5 plt file (steady w/ time)
