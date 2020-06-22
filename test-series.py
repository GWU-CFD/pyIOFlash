from pyioflash.postprocessing.sources import energy
from pyioflash.postprocessing.elements import integral, relation
from pyioflash.postprocessing.analyses import series



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
