from pyioflash.postprocessing.sources import energy
from pyioflash.postprocessing.elements import integral, relation
from pyioflash.postprocessing.analyses import series

def kinetic_mean(data: 'SimulationData', steps: Optional[Union[Iterable, slice]] = None, *,
                 start: Optional["Type_Step"] = None, stop: Optional["Type_step"] = None, 
                 skip: Optional[int] = None, scale: Optional[float] = None) -> 'Type_Field':
    """
    """

    # use provided information to source times
    if steps is None:
        steps = data.utility.indices(slice(start, stop, step)) 

    # use provided slice to source times
    elif isinstance(steps, slice):
        steps = data.utility.indices(steps)

    # try steps; it should be an iterable
    else:
        pass

    # use time series analysis to retreve mean kinetic energy
    source = make_sourceable(kinetic, data)
    stack = (make_stackable(integral.time, data, method='series', options={...}), )
    energy = series.simple(source=source, sourceby=steps, stack=stack) 

    # apply a dimensional scale
    if scale is not None:
        energy = energy * scale

    return energy


def kinetic_turbulant(data: 'SimulationData', step: 'Type_Step', *, 
                      mean: Optional['Type_Field'] = None, 
                      start: Optional["Type_Step"] = None, stop: Optional["Type_step"] = None, 
                      skip: Optional[int] = None, scale: Optional[float] = None) -> 'Type_Field':
    """
    """

    # retieve mean kinetic energy if not provided
    if mean is None:
        mean = kinetic_mean(data, start=start, stop=stop, skip=skip)

    return kinetic(data, step, scale=scale) - mean


# this is what calculating the mean turbulant kinetic energy should look like
mean = energy.kinetic_mean(data, start=30.6, stop=60.7)
source = make_sourceable(energy.kinetic_turbulent, data, options={'mean': mean})
sourceby = data.utility.indices(60.7, 160.7)
stack = (make_stackable(integral.space, data, options={...}),
         make_stackable(integral.time, data, method='series', options={...}))
turbulent = series.simple(source, sourceby, stack) 


# this is what calculating the transient relative total thermal energy should look like
thermal = series.simple(make_sourceable(energy.thermal, data), data.utility.indices(),
                        (make_stackable(integral.space, data), ), Relative(a=100.0))

# this is also what ...
series.simple(source="temp", stack=(make_stackable(integral.space, data), ), 
              mutate=Relative(a=100.0), path=DataPath(data, 'fields'))

# lookup time indices to source times
times = get_indices_from_times(data, slice(start, stop, step))
method = 'step'

if source.method == "step":
    if sourceby is None:
        sourceby = get_indices_from_times(data, slice(start, stop, step))
