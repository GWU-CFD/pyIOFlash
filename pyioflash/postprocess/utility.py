"""

This module defines the utility methods and classes necessary for the 
post-processing subpackage of the pyioflash library.

Todo:
    None

"""

from typing import Any, Tuple, List, Dict, Union, Callable, TYPE_CHECKING
from sys import stdout

if TYPE_CHECKING:
    from numpy import ndarray
    from pyioflash.simulation.data import SimulationData


# --- Define Types for static analysis ---
# Define types for slicing / indexing operations
Type_Step  = Union[int, float]          # defines a single key
Type_Slice = Union[Type_Step, slice]    # defines a single slice
Type_Index = Union[Type_Slice,          # defines multi-dimension slicing
                   Tuple[slice], Tuple[Type_Slice]]


# define types for single, array, and series data
Type_Basic  = Union[str, int, float]            # defines a basic piece of data
Type_Field  = 'ndarray'                         # defines a piece of field data 
Type_Series = Union[List[Type_Basic],           # defines time-series-like data
                    List['ndarray'], 'ndarray']
Type_Data   = Union[Type_Field, Type_Series]    # defines either series or data
Type_Output = Union[Type_Data, Output]          # define source and element output


# define type for flexable typing of a source and sourceby object 
Type_Source = Union[str, 'Sourceable', Type_Data]  
Type_SourceBy = Union[Type_Slice, Iterable]         
Type_Stack = Union['Stackable', Tuple['Stackable']]

# --- Define helper objects for providing flexable context attachemt ---
# define named object which wraps context around sources and stack elements
Sourceable = namedtuple('Sourceable', ['source', 'method', 'wrapped'], defaults=['step', False)
Stackable = namedtuple('Stackable', ['element', 'method', 'wrapped'], defaults=['step', False])


# define avalable methods to source data with a sourceby argument;
SourceByMethods = {'step', 'slice', 'iterable'}


# define available methods for how elements may operate on stack data       
StackableMethods = {'step', 'series'}


# define named objects which wrap context around source and element ouputs
Ouput = namedtuple('Output', ['data', 'context', 'mapping'], defaults=[{}, {}])


# define helper method to construct a sources with context attached for later ingestion 
def make_sourcable(source: Callable[..., Type_Output], args: Tuple[Any], *,
                   method: str = 'step', context: bool = False,
                   options: Dict[str, Any] = {}) -> Sourceable:         
    """
    Provides a helper method to construct sources with attached 
    context for use in ingestion in the analysis modules.

    Attributes:
        source: function which produces data on which to operate
        args: positional arguments to attach to sourcing function
        method: how will the source consume the sourceby passed to it (optional)
        context: whether or not source returns context wrapped output (optional)
        options: keyword arguments to attach to sourcing function (optional)
    """

    # make a tuple out of args if single provided
    if type(args) is not tuple:
        args = (args, )

    # return context wrapped sourcing function
    return Sourcable(partial(source, *args, **options), method)


# define helper methods to construct elements with context attached for later stacking
def make_stackable(element: Callable[..., Type_Output], args: Tuple[Any], *,
                   method: str = 'step', context: bool = False:
                   options: Dict[str, Any] = {}) -> Stackable:
    """
    Provides a helper method to construct elements with attached 
    context for use in constructing a stack for the analysis modules.

    Attributes:
        element: function which operates on data and returns modified results
        args: positional arguments to attach to stackable element
        method: how will the element operate on the data in the stack (optional)
        context: whether or not element returns context wrapped output (optional)
        options: keyword arguments to attach to stackable element (optional)
    """

    # make a tuple out of args if single provided
    if type(args) is not tuple:
        args = (args, )
    
    # return context wrapped stackable element
    return Stackable(partial(element, *args, **options), method, context)

# define an unwrapping factory for processing output
def _make_unwrapper():
    context, mapping, waswrap = {}, {}, False

    def unwrapper(result):
        def unwrap(result):
            nonlocal context, mapping, waswrap
            context, mapping, waswrap = result.context, result.mapping, True
            return result.data
        through = lambda result: result
        wrapped = {True: unwrap, False: through}
        return wrapped[isinstance(result, Output)](result)

    def wrapper(result):
        return Output(result, context, mapping) if waswrap else result

    def refresh():
        context, mapping, waswrap = {}, {}, False

    return unwrapper, wrapper, refresh


def _make_output(message: str, display: bool) -> Callable[[int], None]:
    """
    Provides a method (internal) for writing to screen for progress information

    Attributes:
        message: prepended message to write line
        display: whether or not to display message

    Todo:
        Provide more progress bar and or logging methods

    """

    # create the writing function; overwrites line
    def write_step(step: int) -> None:
        stdout.write(message + " %d \r" % (step))
        stdout.flush()

    # attach the output function or not
    output: Callable[int]
    if not display:
        output = lambda i: None 
    else: 
        output = write_step

    return output


def _ingest_source(source: Type_Source, sourceby: Type_SourceBy, *, 
                   path: DataPath = None) -> Type_Series:
    """
    Method (internal) provided to handle ingesting a source and returning
    an output suitable for use in a series method.

    Attributes:
        source: a source object, name, or function for producing a source
        sourceby: an object which provides specification of items in the source
        path: object specifing where to source the data; necessary if named source


    Todo:
        Verify flexability of method
        Throw more usefull exceptions

    """

    # injest a source using a path and name
    if isinstance(source, str):
        if not isinstance(path, DataPath):
            Exception(f'Must provide a DataPath for named sources')
        if sourceby is None:
            sourceby = path.data.utility.indices()
        output = [data_from_path(path, times=time) for time in sourceby]

    # injest a Sourceable using sourceby
    elif type(source, Sourceable):

        # using an available sourcing method
        if source.method in SourceByMethods:

            # source by steps of an iterable
            if source.method == 'step'

                # try to fill in steps from path
                if sourceby is None:
                    if not isinstance(path, DataPath):
                        Exception(f'Must provide a DataPath for step method sources when sourceby is None!')
                    sourceby = path.data.utility.indices()
            
                output = [source.source(step) for step in sourceby]

            # source by other available methods
            else:

                # try to fill in sourceby if able
                if sourceby is None:
                    if source.method == 'slice':
                        sourceby = slice(None)
                    else:
                        Exception(f'Cannot automatically generate sourceby for choosen source.method!')

                output = source.source(sourceby)

        # Unable to injest source with provided method
        else:
            Exception(f'Provided source.method not supported!')

    # provided source is directly usable as output
    elif type(source, ndarray) or type(source, list):
        if type(source[0]) is not in (str, int, float, ndarray):
            Exception(f'Provided source appears to be a collection, but cannot be used!')
        output = source

    # not able to injest source with provided information
    else:
        Exception(f'Cannot work with provided source!')

    return output


def _interpolate_ftc(field: Type_Field, axis: int, guards: int, dimension: int, *, 
                     keepdims: bool = True) -> Type_Field:
    """
    Provides a method for interpolation from the face centered grids to the cell centered grid.

    Attributes:
        field: face centered field from which to interpolate
        axis: which face centered grid {0 : i, 1 : j, 2 : k}
        guards: how many guard cells in each spacial dimension in field array
        dimension: what is the spacial dimensionality of field
        keepdims: retain unused dimensions for broadcasting, else drop them (optional)

    Note:
        Performs simple linear two-point grid interpolation along relavent axis.
        Returned array does not have guard cells included

    Todo:
        Implement more advanced interpolation schemes
        Add support for providing guard cells in returned array
        Reimplement select case construct as dictionary

    """
    # use one-sided guards
    guard = int(guards / 2)

    # define necessary slice operations
    iall = slice(None)
    icom = slice(guard, -guard)
    izcm = icom if (keepdims or dimension == 3) else 1
    idif = slice(guard - 1, -(guard + 1))

    # define the upper axis; velocity on staggered grid where upper bound is on
    #   the domain boundary & the outer most interior cell on the high side of axis
    high : Tuple[Union[slice, int]] = (iall, izcm, icom, icom)

    # define the lower axis; velocity on staggered grid where lower bound is on
    #   the domain boundary & the inner most guard cell on the low side of axis
    low : Tuple[Union[slice, int]]
    if axis == 0:
        low = (iall, izcm, icom, idif) 
    elif axis == 1:
        low = (iall, izcm, idif, icom)
    elif axis == 2:
        low = (iall, idif, icom, icom)
    else:
        pass

    return (field[high] + field[low]) / 2.0


