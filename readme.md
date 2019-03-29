# pyIO
### A Simple FLASH4 HDF5 Input / Output Tool 
==================================

A simple python Module for providing methods to import, process, and plot FLASH4 HDF5 plot and chk files


   -- Revision 0.16 (alpha)


   -- needed feature list --
   
       add block neighbors to GeometryData
       add refine level to GeometryData
       add ability to fill in 'guard' data for blocks
       others ???

   -- major revision roadmap --
   
       initial functioning Plot Utility
       initial multiprocessing functionality
       add parser for EDDY6 code
       others ???

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
