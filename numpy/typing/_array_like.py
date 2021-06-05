from __future__ import annotations

import sys
from typing import Any, Sequence, TYPE_CHECKING, Union, TypeVar

from numpy import (
    ndarray,
    dtype,
    generic,
    bool_,
    unsignedinteger,
    integer,
    floating,
    complexfloating,
    number,
    timedelta64,
    datetime64,
    object_,
    void,
    str_,
    bytes_,
)

from . import _HAS_TYPING_EXTENSIONS
from ._dtype_like import DTypeLike

if sys.version_info >= (3, 8):
    from typing import Protocol
elif _HAS_TYPING_EXTENSIONS:
    from typing_extensions import Protocol

_T = TypeVar("_T")
_ScalarType = TypeVar("_ScalarType", bound=generic)
_DType = TypeVar("_DType", bound="dtype[Any]")
_DType_co = TypeVar("_DType_co", covariant=True, bound="dtype[Any]")

if TYPE_CHECKING or _HAS_TYPING_EXTENSIONS:
    # The `_SupportsArray` protocol only cares about the default dtype
    # (i.e. `dtype=None` or no `dtype` parameter at all) of the to-be returned
    # array.
    # Concrete implementations of the protocol are responsible for adding
    # any and all remaining overloads
    class _SupportsArray(Protocol[_DType_co]):
        def __array__(self) -> ndarray[Any, _DType_co]: ...
else:
    _SupportsArray = Any

# TODO: Wait for support for recursive types
_NestedSequence = Union[
    _T,
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]
_RecursiveSequence = Sequence[Sequence[Sequence[Sequence[Sequence[Any]]]]]

# A subset of `npt.ArrayLike` that can be parametrized w.r.t. `np.generic`
_ArrayLike = _NestedSequence[_SupportsArray[dtype[_ScalarType]]]

# TODO: support buffer protocols once
#
# https://bugs.python.org/issue27501
#
# is resolved. See also the mypy issue:
#
# https://github.com/python/typing/issues/593
ArrayLike = Union[
    _RecursiveSequence,
    _ArrayLike[Any],
    _NestedSequence[Union[complex, str, bytes]],
]

# NOTE: As a reminder: `complex` is a supertype of `float`,
# which is a supertype of `int`, which is a superclass of `bool`

# `ArrayLike<X>_co`: array-like objects that can be coerced into `X`
# given the casting rules `same_kind`
_ArrayLikeBool_co = Union[
    _ArrayLike[bool_],
    _NestedSequence[bool],
]
_ArrayLikeUInt_co = Union[
    _ArrayLike[Union[bool_, unsignedinteger]],
    _NestedSequence[bool],
]
_ArrayLikeInt_co = Union[
    _ArrayLike[Union[bool_, integer]],
    _NestedSequence[int],
]
_ArrayLikeFloat_co = Union[
    _ArrayLike[Union[bool_, integer, floating]],
    _NestedSequence[float],
]
_ArrayLikeComplex_co = Union[
    _ArrayLike[Union[bool_, integer, floating, complexfloating]],
    _NestedSequence[complex],
]
_ArrayLikeNumber_co = Union[
    _ArrayLike[Union[bool_, integer, floating, complexfloating]],
    _NestedSequence[complex],
]
_ArrayLikeTD64_co = Union[
    _ArrayLike[Union[bool_, integer, timedelta64()]],
    _NestedSequence[int],
]
_ArrayLikeDT64_co = _ArrayLike[datetime64]
_ArrayLikeObject_co = _ArrayLike[object_]
_ArrayLikeVoid_co = _ArrayLike[void]
_ArrayLikeStr_co = Union[
    _ArrayLike[str_],
    _NestedSequence[str],
]
_ArrayLikeBytes_co = _ArrayLike[
    _ArrayLike[byte_],
    _NestedSequence[bytes],
]
_ArrayLikeInt = _ArrayLike[
    _ArrayLike[integer],
    _NestedSequence[int],
]
