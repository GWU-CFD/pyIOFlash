"""

This module defines the custom collections necessary for the pyio package.

The custom collection, SortedDict, composites behavior of a list and dict,
with the added behaior that the elements remain sorted based on a key value
that each member of the collection is required to possess.

This module also defines a Base, and derived, class(es) which provide filtered,
transpose-like viewing of multiple named members of a derived class when used in
combination with a SortedDict.

Example:
    An example use of a SortedDict is as follows, assuming each element loaded
    into the SortedDict instance provides an attribute, key, of type float::

        series = SortedDict(iter)

    An example use of attaching traspose-like viewing to a collection class
    using the TransposableAsArray class::

        series = type('MyObject', (TransposeAsArray, ...), {})(...)

Todo:
    * Extend SortedDict to allow type(key) other than float
    * Improve code by refactoring out if/elif/else constructs

"""
from abc import ABC as AbstractBase, abstractmethod
from collections import OrderedDict
from typing import Any, Tuple, List, Dict, Iterable, Union

import numpy

from pyioflash.simulation.utility import _filter_transpose, _first_true, _set_is_unique

class SortedDict:
    """
    SortedDict is a custom collection implmenting sequence and map like behaviors.

    SortedDict is a mutable collection of unique items (that must implement an item.key),
    providing the following functionality:
    (1) Sequence-like fast indexable access,
    (2) Dictionary-like fast mappable access,
    (3) Collection remains ordered (invariant) based on each element's key value,
    (4) Insert/remove (slow), append (fast if sorted), pop (fast) operations,
    (5) Collection can only contain elements with a unique key (i.e., element.key).


    +-----------------------------------------+------------------------------------+
    |High Level Usage                         |                                    |
    +=========================================+====================================+
    |series = SortedDict(iter)                |Creates a collection from iter      |
    +-----------------------------------------+------------------------------------+
    |series = SortedDict.from_sorted(iter)    |Creates a collection from iter;     |
    |                                         |assuming iter is sorted             |
    +-----------------------------------------+------------------------------------+


    +------------------------------------------------------------------------+
    |Common Sequence/Mapping behavior                                        |
    +==============+=========================================================+
    |clear()       |Removes all the items from the collection                |
    +--------------+---------------------------------------------------------+
    |copy()        |Return a shallow copy of the collection                  |
    +--------------+---------------------------------------------------------+
    |sc = sa + sb  |Returns the concatenation of rhs collections             |
    +--------------+---------------------------------------------------------+
    |sa += sb      |Extends the collection with the contents of sb           |
    +--------------+---------------------------------------------------------+
    |sa == sb      |Returns logical (equal) comparison of collection's items;|
    |              |also implements analogous <, <=, !=, >=, and >           |
    +--------------+---------------------------------------------------------+


    +------------------------------------------------------------------------+
    |Common Sequence/Mapping behavior                                        |
    +==============+=========================================================+
    |append(x)     |Add an item to the end of the collection                 |
    +--------------+---------------------------------------------------------+
    |extend(x)     |Extends collection with the contents of t                |
    +--------------+---------------------------------------------------------+
    |insert(i, x)  |Inserts x into collection at the index given by i        |
    +--------------+---------------------------------------------------------+
    |pop([i])      |Returns the item at i and removes it from the collection;|
    |              |default i = -1                                           |
    +--------------+---------------------------------------------------------+
    |sort()        |Forces a sort of the collection's items;                 |
    |              |otherwise defered until the last possible moment         |
    +--------------+---------------------------------------------------------+


    +----------------------------------------------------------------------------------------------+
    |Map-like behavior                                                                             |
    +======================+=======================================================================+
    |get(key[, default])   |Return the value for key if key is in the dictionary, else default;    |
    +----------------------+-----------------------------------------------------------------------+
    |items()               |Return an iterator of the collection's items ((key, value) pairs)      |
    +----------------------+-----------------------------------------------------------------------+
    |keys()                |Returns an iterator of the collection's keys                           |
    +----------------------+-----------------------------------------------------------------------+
    |pop([key, default])   |Returns the key and removes it from the collection, if present,        |
    |                      |else defualt; if no key, default is last element                       |
    +----------------------+-----------------------------------------------------------------------+
    |popitem()             |Remove and return a (key, value) pair from collection;                 |
    |                      |returned in LIFO order                                                 |
    +----------------------+-----------------------------------------------------------------------+
    |setdefault(default)   |If default.key is in collection, return value;                         |
    |                      |otherwise insert/return default                                        |
    +----------------------+-----------------------------------------------------------------------+
    |update([other])       |Update the collection with the key/value pairs from other;             |
    |                      |overwriting keys                                                       |
    +----------------------+-----------------------------------------------------------------------+
    |values()              |Returns an iterator of the collections values                          |
    +----------------------+-----------------------------------------------------------------------+

    Currently the only key type supported is float. This is not a fundemental limitation, however,
    and an implmentation will be provided in a future update to generalize the key type.

    .. Warning::
       type(element.key) must be a float

    +----------------------------------------------------------------------------------------------+
    |Additional notes w.r.t. access using the [] operator                                          |
    +======================+=======================================================================+
    |sd[index] = value     |Assignment will overwrite item @ index and sort if necessary;          |
    |                      |exeption if value.key nonunique and series[index].key != value.key     |
    +----------------------+-----------------------------------------------------------------------+
    |sd[key] = value       |Assignment will insert or overwrite existing item w/ equal key         |
    +----------------------+-----------------------------------------------------------------------+
    |sd[low : high : set]  || (1) Works (nearly) identical to list if args are integers, and       |
    |                      || (2) returns a consistant slice matching a closed interval            |
    |                      ||     if args are floats, set must be int                              |
    +----------------------+-----------------------------------------------------------------------+


    .. Note::
       Finally, it is again noted that while the sorted property of SortedDict is invariant;
       the sorting operation is defered until the last possible moment (e.g., access, index based assignment)

    Attributes:
        _data: Internal list storing the elementes of the SortedDict
        _keys: Internal dictionary storing the key value and list index pairs
        _is_valid: Internal state-flag used to implement just-in-time sorting
    """
    _data: List[Any]
    _keys: Dict[float, int]
    _is_valid: bool

    def __add__(self, other: Iterable) -> 'SortedDict':
        self.__make_valid__()
        if not _set_is_unique(self._keys, {item.key for item in other}):
            raise ValueError(f'Nonunique key, location mismatch; identical keys found in collection')
        self._is_valid = False
        return self.__class__(self._data + other)

    def __contains__(self, other: Iterable) -> bool:
        self.__make_valid__()
        return other in self._data

    def __eq__(self, other: Iterable) -> bool:
        self.__make_valid__()
        if isinstance(other, SortedDict):
            return self._data == other._data # pylint: disable=protected-access
        else:
            return self._data == SortedDict(other)._data # pylint: disable=protected-access

    def __ge__(self, other: Iterable) -> bool:
        self.__make_valid__()
        if isinstance(other, SortedDict):
            return self._data >= other._data # pylint: disable=protected-access
        else:
            return self._data >= SortedDict(other)._data # pylint: disable=protected-access

    def __getitem__(self, key: Union[int, float, slice]) -> Union[List[Any], Any]:
        start: int
        stop: int
        self.__make_valid__()

        # dictionary-like behavior
        if isinstance(key, float):
            start = _first_true(self._keys.items(), lambda x: key <= x[0])[1]
            return self.__class__.from_sorted([self._data[start]])

        # list-like behavior
        elif isinstance(key, int):
            return self.__class__.from_sorted([self._data[key]])

        # slicing behavior; both list and dict** like
        elif isinstance(key, slice):

            # dictionary-like behavior; if a dict were 'slice-able'
            if isinstance(key.start, float) or isinstance(key.stop, float):
                if key.start is None:
                    start = 0
                else:
                    start = _first_true(self._keys.items(), lambda x: key.start <= x[0])[1]
                if key.stop is None:
                    stop = len(self._data)
                else:
                    stop = _first_true(reversed(self._keys.items()), lambda x: x[0] <= key.stop)[1] + 1
                return self.__class__.from_sorted(self._data[start : stop : key.step])

            # list-like behavior
            else:
                return self.__class__.from_sorted(self._data[key])

        else:
            raise TypeError(f'SeriesData indices must be integers, floats, or slices')

    def __gt__(self, other: Iterable) -> bool:
        self.__make_valid__()
        if isinstance(other, SortedDict):
            return self._data > other._data # pylint: disable=protected-access
        else:
            return self._data > SortedDict(other)._data # pylint: disable=protected-access

    def __iadd__(self, other: Iterable) -> 'SortedDict':
        self.extend(other)
        return self

    def __init__(self, iterable: Iterable) -> None:
        self._data = list(iterable)
        self._is_valid = False

    def __iter__(self):
        self.__make_valid__()
        yield from self._data

    def __le__(self, other: Iterable) -> bool:
        self.__make_valid__()
        if isinstance(other, SortedDict):
            return self._data <= other._data # pylint: disable=protected-access
        else:
            return self._data <= SortedDict(other)._data # pylint: disable=protected-access

    def __len__(self):
        return len(self._data)

    def __lt__(self, other: Iterable) -> bool:
        self.__make_valid__()
        if isinstance(other, SortedDict):
            return self._data < other._data # pylint: disable=protected-access
        else:
            return self._data < SortedDict(other)._data # pylint: disable=protected-access

    def __make_keys__(self) -> None:
        self._keys = OrderedDict([(item.key, i) for i, item in enumerate(self._data)])

    def __make_valid__(self, force: bool = False) -> None:
        if force or not self._is_valid:
            self._data.sort(key=lambda item: item.key)
            self.__make_keys__()
            self._is_valid = True

    def __ne__(self, other: Iterable) -> bool:
        result = self.__eq__(other)
        if result is not NotImplemented:
            return not result
        return NotImplemented

    def __repr__(self) -> str:
        self.__make_valid__()
        return 'SortedDict[' + ',\n '.join(item.__repr__() for item in self._data) + ']'

    def __reversed__(self):
        self.__make_valid__()
        yield from self._data[::-1]

    def __setitem__(self, key: Union[int, float, slice, Iterable], value: Union[List[Any], Any]) -> None:
        start: int
        stop: int
        self.__make_valid__()

        # mappable-like behavior
        if isinstance(key, float):    # // future type generalization //
            if value.key in self._keys:
                self._data[self._keys[key]] = value
            else:
                self.append(value)

        # sequence-like behavior
        elif isinstance(key, int):
            if value.key in self._keys and key != self._keys[value.key]:
                raise ValueError(f'Nonunique key, location mismatch; identical key @ {self._keys[value.key]}')
            else:
                self._data[key] = value
                self._is_valid = False

        elif isinstance(key, slice):

            # mappable-like behavior; ** if dict allowed slice-like replacement
            if isinstance(key.start, float):
                start = _first_true(self._keys.items(), lambda x: key.start <= x[0])[1]
                stop = _first_true(reversed(self._keys.items()), lambda x: x[0] <= key.stop)[1] + 1

            # sequence-like behavior
            else:
                start = key.start
                stop = key.stop

            masked: set = {i.key for i in self._data[:start]} | {i.key for i in self._data[stop:]}
            if _set_is_unique(masked, {item.key for item in value}):
                self._data[start : stop] = value
                self._is_valid = False
            else:
                raise ValueError(f'Nonunique key, location mismatch; identical keys found outside of slice')

        else:
            raise TypeError(f'SortedDict indices must be integers, floats, slices, or iterables')

    def __str__(self) -> str:
        self.__make_valid__()
        return '[' + ',\n '.join(item.__str__() for item in self._data) + ']'

    def append(self, value: Any) -> None:
        """
        Appends an element to the end or into the collection, as appropriate.

        Append maintains the collection sorted, although the sort operation,
        if needed, is defered until the last possible moment.

        Same as list.append(obj).

        Args:
            value: the object to be added to the collection.

        Returns:
            None

        """
        self.__make_valid__()
        if value.key in self._keys:
            raise ValueError(f'Nonunique key provided; {value.key} @ index {self._keys[value.key]}')
        if self._data and value.key < next(reversed(self._keys)):
            self._is_valid = False
        self._data.append(value)
        self._keys[value.key] = len(self._data) - 1

    def clear(self) -> None:
        """
        Clears all elements from the collection.

        Same as list.clear().

        Args:
            None

        Returns:
            None

        """
        self._data.clear()
        self._keys = OrderedDict({})
        self._is_valid = True

    def copy(self) -> None:
        """
        Returns a shallow copy of the collection.

        Same as list.copy().

        Args:
            None

        Returns:
            A SortedDict instance.

        """
        self.__make_valid__()
        return self.__class__(self._data)

    def extend(self, iterable: Iterable):
        """
        Extends the collection by an element; either onto the end or into the collection, as appropriate.

        Extend maintains the collection sorted, although the sort operation,
        if needed, is defered until the last possible moment.

        Same as list.extend(obj).

        Args:
            iterable: the object used to extend the collection

        Returns:
            None

        """
        self.__make_valid__()
        if not _set_is_unique(self._keys, {item.key for item in iterable}):
            raise ValueError(f'Nonunique key, location mismatch; identical keys found in collection')
        self._data.extend(iterable)
        self._is_valid = False

    @classmethod
    def from_sorted(cls, iterable: Iterable):
        """
        Class method used to instantiate a SortedDict with a known to be
        sorted collection of elements.

        SortedDict.from_sorted(obj) does not verify obj is sorted.
        Undefined behavior results from providing an unsorted obj.

        Args:
            iterable: object used to create a SortedDict instance

        Returns:
            An instance of SortedDict from args.
        """
        instance = cls(iterable)
        instance.__make_keys__()
        instance._is_valid = True # pylint: disable=protected-access
        return instance

    def get(self, key: float, defualt: Any = None) -> Any:
        """
        Return the value for key if key is in the collection, else default.
        If default is not given, it defaults to None, so that this method never raises a KeyError.

        Same as dict.get(key[, default]).

        Args:
            key: Unique key of the element to be returned
            default (optional): Return value if key is not in the collection

        Returns:
            The value at key if found, else, the default value.

        """
        self.__make_valid__()
        try:
            return self._data[self._keys[key]]
        except KeyError:
            return defualt

    def insert(self, key: int, value: Any) -> None:
        """
        Instert an element into the collection at location of key.

        Same as dict[key] = value; append value if key not in collection,
        else overwite element at key with value.

        Args:
            key: Key of the element to be added to the collection
            value: Value of the element to be added to the collection

        Returns:
            None

        """
        self.__make_valid__()
        self[key] = value

    def items(self):
        """
        Returns the items in the collection.

        Same as dict.items().

        Args:
            None

        Returns:
            List of key value pairs (as iterator) in the collection.

        """
        self.__make_valid__()
        yield from {item[0] : self._data[item[1]] for item in self._keys.items()}.items()

    def keys(self):
        """
        Returns the keys in the collection.

        Same as dict.keys().

        Args:
            None

        Returns:
            List of keys (as iterator) in the collection.

        """
        self.__make_valid__()
        yield from self._keys.keys()

    def pop(self, key: Union[int, float] = -1, default: float = None) -> Union[Any, float]:
        """
        If key is in the collection, remove it and return its value, else return default.
        If default is not given and key is not in the collection, a KeyError is raised.

        Same as dict.pop(key[, default]).

        Args:
            key: Key of the collection element to be removed
            default (optional): Return value if key is not in the collection

        Returns:
            Value of the element removed from the collection, if found, else, default.

        """
        self.__make_valid__()
        if isinstance(key, int):
            self._is_valid = False
            return self._data.pop(key)
        if key in self._keys:
            self._is_valid = False
            self._data.pop(self._keys[key])
            return key
        if default is not None:
            return default
        raise KeyError(f'Key is not in collection and no default provided')

    def popitem(self) -> Tuple[float, Any]:
        """
        Remove and return an arbitrary (key, value) pair from the collection.

        Same as dict.popitem().

        Args:
            None

        Returns:
            Key value pair of the element removed from the collection.

        """
        self.__make_valid__()
        if not self._data:
            raise KeyError(f'Method popitem called on an empty collection')
        return (self._keys.popitem()[0], self._data.pop())

    def setdefault(self, key: float, default: Any = None) -> Any:
        """
        If key is in the collection, return its value. If not, insert key
        with a value of default and return default. Default defaults to None.

        Same as dict.setdefault(key[, default]).

        Args:
            key: Key of the collection element to be found
            default (optional): Return value if key is not in the collection

        Returns:
            Value of the element if key is in the collection, else default.

        """
        try:
            return self._data[self._keys[key]]
        except ValueError:
            return default

    def sort(self) -> None:
        """
        Forces a sort of the collection if the collection is 'not sorted'.

        The sort() method will only sort if there is currently an awaiting defered sort operation.

        Args:
            key: Key of the collection element to be found
            default (optional): Return value if key is not in the collection

        Returns:
            Value of the element if key is in the collection, else default.

        """
        self.__make_valid__()

    def tolist(self) -> list:
        """
        Returns the elements of the collection as a list.

        Args:
            None

        Returns:
            A list of the elements.

        """
        self.__make_valid__()
        return list(self)

    def update(self, iterable: Iterable) -> None:
        """
        Update the collection with the key/value pairs from iterable, overwriting existing keys.

        Same as dict.update(obj).

        Args:
            iterable: Object providing key/value pairs for append or replacement

        Returns:
            None

        """
        self.__make_valid__()
        for item in iterable:
            self[item.key] = item

    def values(self):
        """
        Returns the values in the collection.

        Same as dict.values().

        Args:
            None

        Returns:
            List of values (as iterator) in the collection.
        """
        self.__make_valid__()
        return self.__iter__()

class Filterable:
    """
    Provides an abstraction to implement slicing of list-of-lists elements using [] syntax;
    including multi-key slicing and lookup vice slicing if applicable
    """
    def __getitem__(self, keys: slice) -> Any:
        if hasattr(keys, '__iter__') and not isinstance(keys, str) and isinstance(keys[0], (str, tuple)):
            return [[item[key] for item in self] for key in keys]
        else:
            return [item[keys] for item in self]

class BaseTransposable(AbstractBase):
    """
    Abstract base class for implementing a wrapper class used to provide
    filtered, transpose-like viewing of multiple named members of a derived class,
    wrapped in some fashion (must implement a _return_map method); super().self must be iterable
    """
    @abstractmethod
    def _return_map(self, source):
        raise NotImplementedError(f'{self.__class__.__name__} does not implement an _return_map method')

    def __getitem__(self, key: Iterable) -> Union[List[Any], Any]:
        try:
            return super().__getitem__(key) # pylint: disable=no-member
        except TypeError:
            if isinstance(key, str):
                rtn = _filter_transpose(self, [key])
            elif hasattr(key, '__iter__'):
                rtn = _filter_transpose(self, list(key))
            else:
                raise TypeError(
                    f'Indices of {self.__class__.__name__} objects must be integers, slices, or iterable')
            return type('FilterableList', (Filterable, list), {})(self._return_map(rtn))

class TransposableAsArray(BaseTransposable):
    """
    Wrapper class used to provide filtered,transpose-like viewing of multiple named members of
    a derived collection-like class, wrapped in numpy arrays; super().self must be iterable.

    An instance is created with the following syntax::

        ta = type('MyObject', (TransposeAsArray, ...), {})(...)

    The composited object may be used as before with the following exceptions

    +-------------------------------------------------------------------------------------------------+
    |Extended behavior as follows                                                                     |
    +======================+==========================================================================+
    |ta[str, ...]          || Returns a list-like object containing numpy arrays along the key-axis   |
    |                      || for each (outer list); matching the args to members of collection       |
    |                      || elements, operation is approxamatly a filtered, transpose with wrapping |
    +----------------------+--------------------------------------------------------------------------+
    |ta[str, ...][:, ...]  || The returned object has the added property that slicing may be applied  |
    |                      || to each inner numpy array elements simultaniously with the [] notation  |
    +----------------------+--------------------------------------------------------------------------+

    """
    def _return_map(self, source):
        return map(numpy.array, source)

class TransposableAsSingle(BaseTransposable):
    """
    Wrapper class used to provide filtered viewing of a single named member of a derived class,
    wrapped as a list of items; super().self must be iterable

    An instance is created with the following syntax::

        ts = type('MyObject', (TransposeAsSingle, ...), {})(...)

    The composited object may be used as before with the following exceptions

    +-------------------------------------------------------------------------------------------------+
    |Extended behavior as follows                                                                     |
    +======================+==========================================================================+
    |ts[str, ...]          || Returns a list-like object containing numpy arrays along the key-axis;  |
    |                      || matching the arg to amember of collection elements,                     |
    |                      || operation is approximatly a filtered, transpose with wrapping           |
    +----------------------+--------------------------------------------------------------------------+
    |ts[str, ...][:, ...]  || The returned object has the added property that lookup or slicing, as   |
    |                      || appropriate, may be applied to the list elements with the [] notation   |
    +----------------------+--------------------------------------------------------------------------+

    """
    def _return_map(self, source):
        return source[0]
