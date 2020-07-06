"""

This module defines the utility methods and classes necessary for the pyio package.

Todo:
    None
"""


from contextlib import contextmanager
from typing import Any, Tuple, List, Iterable, Union, Callable, TYPE_CHECKING


import h5py


if TYPE_CHECKING:
    from pyioflash.simulation.collections import SortedDict
    from pyioflash.simulation.geometry import GeometryData


def _blocks_from_plane(data: 'GeometryData', axis: str, value: float) -> List[int]:
    """Returns a list of block indices (using geometry data) which intersect a provided plane"""

    # define intersection truth function
    within = lambda low, high, check: low <= check and check < high
    
    # define axis of provided plane and retrieve bounding boxes
    index = {axis: index for index, axis in enumerate(['x', 'y', 'z'])}[axis]
    boxes = data.blk_bndbox[:, index, :]
 
    # return blocks which are intersected by plane; open interval value in [low, high)
    if data.grd_dim == 2 and axis == 'z':
        return list(range(len(boxes))) # open interval results in empty set for z-axis in 2d
    else:
        return [block for block, box in enumerate(boxes) if within(*(tuple(box) + (value, )))]


def _blocks_from_line(data: 'GeometryData', axes: Tuple[str], values: Tuple[float]) -> List[int]:
    """Returns a list of block indices (using geometry data) which intersect a provided line"""

    # define intersection truth function
    within = lambda low, high, check: low <= check and check < high

    # define axes of provided line and retrive bounding boxes
    mapping = {axis: index for index, axis in enumerate(['x', 'y', 'z'])}
    boxes0 = data.blk_bndbox[:, mapping[axes[0]], :]
    boxes1 = data.blk_bndbox[:, mapping[axes[1]], :]

    # return blocks which are intersected by a line; open interval values in [lows, highs)
    if data.grd_dim == 2 and 'z' in axes:  # open interval results in empty set for z-axis in 2d
        axis, value = [(axis, value) for axis, value in zip(axes, values) if axis != 'z'][0]
        return _blocks_from_plane(data, axis, value)
    else:
        return [block for block, (box0, box1) in enumerate(zip(boxes0, boxes1)) 
                if within(*(tuple(box0) + (values[0], ))) and within(*(tuple(box1) + (values[1], )))]

def _first_true(iterable: Iterable, predictor: Callable[..., bool]) -> Any:
    """Returns the first true value in the iterable according to predictor."""
    return next(filter(predictor, iterable))


def _filter_transpose(source: List[Any], names: Iterable[str]) -> List[List[Any]]:
    """Returns a list of named members extracted from a
    collection object (source) based on a list (names)"""
    try:
        return [list(map(lambda obj, n=name: getattr(obj, n), source)) for name in names]
    except AttributeError:
        return [list(map(lambda obj, n=name: obj[n], source)) for name in names]


def _get_indices(data: 'SortedDict', key: Union[int, float, slice, Iterable]) -> List[int]: 
    """Returns a list of indices associated with keys (or indices) from a SortedDict"""
    keys = list(data.keys())

    # dictionary-like behavior
    if isinstance(key, float):
        try:
            start = keys.index(_first_true(keys, lambda x: key <= x))
        except StopIteration:
            return []
        return [start]

    # list-like behavior
    elif isinstance(key, int):
        return [key] if key < len(keys) else []

    # slicing behavior; both list and dict** like
    elif isinstance(key, slice):

        # dictionary-like behavior; if a dict were 'slice-able'
        if isinstance(key.start, float) or isinstance(key.stop, float):
            if key.start is None:
                start = 0
            else:
                try:
                    start = keys.index(_first_true(keys, lambda x: key.start <= x))
                except StopIteration:
                    return []
            if key.stop is None:
                stop = len(keys)
            else:
                try:
                    stop = keys.index(_first_true(reversed(keys), lambda x: x <= key.stop)) + 1
                except StopIteration:
                    stop = len(keys)
            if key.step is not None:
                step = key.step
            else:
                step = 1
            return list(range(start, stop, step))

        # list-like behavior
        else:
            if key.start is None:
                start = 0
            elif key.start < len(keys):
                start = key.start
            else:
                return []
            if key.stop is not None and key.stop <= len(keys):
                stop = key.stop
            else:
                stop = len(keys)
            if key.step is not None:
                step = key.step
            else:
                step = 1
            return list(range(start, stop, step))
    
    # try consuming the key as an iterator as a last ditch effort
    elif hasattr(key, '__len__') and len(key) >= 1 and type(key[0]) in {int, float, slice}:
        return [_get_indices(data, k)[0] for k in key] 

    # cannot work with provided key
    else:
        raise TypeError(f'Provided key must be integers, floats, slices, or interable of such')


def _get_times(data: 'SortedDict', key: Union[int, float, slice, Iterable]) -> List[Union[int, float]]:
    """Returns a list of keys associated with the indicies (or keys) from a SortedDict"""
    keys = list(data.keys())
    #print(_get_indices(data, key))
    return [keys[k] for k in _get_indices(data, key)]


def _reduce_str(value: str, sentinal: str = '_'):
    """ Provides reduced string with intervineing spaces replaced, and trailing removed"""
    return value.rstrip().replace(' ', sentinal)


def _set_is_unique(source: Iterable, sequence: Iterable, mask: Iterable = None) -> bool:
    """Returns (True) if no member of sequence is found in source, with ability
    to mask members of source from comparision to members of sequence"""
    try:
        if mask is None:
            _first_true(sequence, lambda item: item in source)
        else:
            masked_source: set = {k for k in source if k not in mask}
            _first_true(sequence, lambda item: item in masked_source)
        return False
    except StopIteration:
        return True


@contextmanager
def open_hdf5(*args, **kwargs):
    """Context manager for working with a hdf5 file;
    using a h5py file handle"""
    file = h5py.File(*args, **kwargs)
    try:
        yield file
    finally:
        file.close()
