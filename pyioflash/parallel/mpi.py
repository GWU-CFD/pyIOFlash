from sys import platform, argv
from os import popen, getppid
from typing import Set, Any, ClassVar
from dataclasses import dataclass, field
from pkg_resources import get_distribution


@dataclass
class Parallel:
    __initialized: ClassVar[bool] = False
    __any_incomps: ClassVar[bool] = False
    __any_serials: ClassVar[bool] = False
    serial: bool = False
    MPI: Any = None
    comm: Any = None
    size: int = 1
    rank: int = 0 

    def __post_init__(self):

        # force serial if needed
        if Parallel.__any_incomps:
            print('ERROR -- Using an incompatible module; Please do not instance Parallel!')
            raise

        # initialize MPI library
        if not Parallel.__initialized:
            self.serial = Parallel.__setup()
        else:
            # do not initialize mpi more than once and raise exception
            print('ERROR -- Cannot initialize MPI; Please use a single instance of Parallel()!')
            raise

        # force serial if needed
        if Parallel.__any_serials and not self.serial:
            print('ERROR -- Using an unsupported module; Please execute without mpirun!')
            raise

        # initialize MPI Library
        if not self.serial:
            from mpi4py import MPI
            self.MPI = MPI
        else:
           return 

        # get mpi parameters
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        # Set completed init flag
        Parallel.__initialized = True

    @classmethod
    def __setup(cls):

        # Ascertain if mpi4py available
        try:
            mpi_isinst = True
            get_distribution('mpi4py')
        except:
            mpi_isinst = False

        # os specific implementation
        if platform == 'linux':

            # Ascertain if run with mpirun
            launch = popen("ps -p %d -oargs=" % getppid()).read().strip()
            option = {'mpirun', 'mpiexec'}
            mpirun_cmd = any(cmd in launch for cmd in option)

        else:
            option = ('--parallel', '--serial')
            if len(argv) > 1 and any(cmd in argv for cmd in option):
                parallel = True if option[0] in argv else False
            else:
                message = 'Error -- Please identify if using MPI; specify --parallel or --serial'
                print(message)
                raise

        # Need to determine parallel
        if mpirun_cmd and mpi_isinst:
            parallel = True
        elif mpirun_cmd and not mpi_isinst:
            print('ERROR -- mpi4py library is not available; Please execute in serial!')
            raise
        else:
            parallel = False
            print('WARNING -- MPI was not invoked; Switching to Serial!')
        
        return not parallel

    @classmethod
    def register_incompatible(cls):
        cls.__any_incomps = True

    @classmethod
    def register_serial_only(cls):
        cls.__any_serials = True




