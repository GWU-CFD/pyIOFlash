"""

This module defines the utility methods and classes necessary for the pyio package.

Todo:
    None
"""

from contextlib import contextmanager
from typing import Any, List, Iterable, Callable

import h5py

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

def _reduce_str(value: str, sentinal: str = '_'):
    """ Provides reduced string with intervineing spaces replaced, and trailing removed"""
    return value.rstrip().replace(' ', sentinal)

@contextmanager
def open_hdf5(*args, **kwargs):
    """Context manager for working with a hdf5 file;
    using a h5py file handle"""
    file = h5py.File(*args, **kwargs)
    try:
        yield file
    finally:
        file.close()
