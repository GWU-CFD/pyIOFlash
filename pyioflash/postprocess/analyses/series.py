"""

"""


from typing import Optional, TYPE_CHECKING
from pyioflash.postprocess.utility import _ingest_source, _make_unwrapper

if TYPE_CHECKING:
    from pyioflash.postprocess.utility import Type_Source, Type_Sourceby, Type_Stack, Type_Output
    from pyioflash.simulation.utility import DataPath



def simple(source: 'Type_Source', sourceby: Optional['Type_SourceBy'] = None, 
           stack: Optional['Type_Stack'] = None, *, 
           path: Optional['DataPath'] = None) -> 'Type_Output':
    """
    """

    # injest source from provided information 
    result = _ingest_source(source, sourceby, path=path)
    
    # Use stack to operate on source if provided
    if stack is not None:

        # make stack if single provided
        if type(stack) is not tuple:
            stack = (stack, )

        # Operate on source by each element in the stack
        unwrap, rewrap, refresh = _make_unwrapper()
        for item in stack:

            # unwrap each and apply stack method before rewrapping
            if item.method == 'step':
                refresh()
                result = rewrap([unwrap(item.element(part, 
                                                     **{result.mapping[option]: value 
                                                        for option, value in result.context.items()
                                                        if option in result.mapping}))
                                 for part in result.data]) 

            elif item.method == 'series':
                result = item.element(result)

            else:
                Exception(f'Stack element method of operation not supported!')


    return result

