"""

"""


from typing import List, Optional, TYPE_CHECKING


from pyioflash.postprocess.utility import StackableMethods, Output, _ingest_source, _make_unwrapper


if TYPE_CHECKING:
    from pyioflash.postprocess.utility import Type_Source, Type_Sourceby, Type_Stack, Type_Output
    from pyioflash.simulation.utility import DataPath


# define the module api
def __dir__() -> List[str]:
    return ['simple']


def simple(source: 'Type_Source', sourceby: Optional['Type_SourceBy'] = None, 
           stack: Optional['Type_Stack'] = None, *, 
           path: Optional['DataPath'] = None) -> 'Type_Output':
    """
    Provides a method to perform a simple time-series like post-processing of 
    simulation data; specificaly, this method consumes a source + sourceby, and a 
    set of stackable methods in order to perform the desired time-series and/or
    permutation analysis.

    Attributes:


    """

    # injest source from provided information 
    result = _ingest_source(source, sourceby, path=path)
    
    # Use stack to operate on source if provided
    if stack is not None:

        # make stack if single provided
        if type(stack) is not tuple:
            stack = (stack, )

        # Operate on source result by each element in the stack in turn
        unwrap, rewrap, refresh = _make_unwrapper()
        for item in stack:

            # unwrap each and apply stack method by piece part before rewrapping the whole
            if item.method == 'part':

                # refesh factory context
                refresh()

                # need to pass context to element if previous element provided wrapped output
                if isinstance(result, Output):
                    refresh()
                    result = rewrap([unwrap(item.element(part, 
                                                         **{result.mapping[option]: value 
                                                            for option, value in result.context.items()
                                                            if option in result.mapping}))
                                     for part in result.data]) 

                # no context provided for previous element output
                else:
                    result = rewrap([unwrap(item.element(part))
                                     for part in result])

            # operate on the previous result as a whole with current element
            elif item.method == 'whole':

                # need to pass context to element if previous element provided wrapped output
                if isinstance(result, Output):
                    result = item.element(result,
                                          **{result.mapping[option]: value
                                             for option, value in result.context.items()
                                             if option in result.mapping})

                # no context provided for previous element output
                else:
                    result = item.element(result)

            # Unkown method of stack operation provided
            else:
                raise ValueError(f"Unsupported method of operation '{method}'; must specify {StackableMethods}!")

    return result

