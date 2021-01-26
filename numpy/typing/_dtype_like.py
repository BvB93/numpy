import sys
from typing import Any, List, Sequence, Tuple, Union, Type, TypeVar, TYPE_CHECKING

import numpy as np

from . import _HAS_TYPING_EXTENSIONS
from ._shape import _ShapeLike
from ._generic_alias import _DType as DType

if sys.version_info >= (3, 8):
    from typing import Protocol, TypedDict
elif _HAS_TYPING_EXTENSIONS:
    from typing_extensions import Protocol, TypedDict
else:
    from ._generic_alias import _GenericAlias as GenericAlias

from ._char_codes import (
    _BoolCodes,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _Complex64Codes,
    _Complex128Codes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _IntPCodes,
    _IntCodes,
    _LongLongCodes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _UIntPCodes,
    _UIntCodes,
    _ULongLongCodes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
    _DT64Codes,
    _TD64Codes,
    _StrCodes,
    _BytesCodes,
    _VoidCodes,
    _ObjectCodes,
)

_DTypeLikeNested = Any  # TODO: wait for support for recursive types
_DType_co = TypeVar("_DType_co", covariant=True, bound=DType[Any])

if TYPE_CHECKING or _HAS_TYPING_EXTENSIONS or sys.version_info >= (3, 8):
    # Mandatory keys
    class _DTypeDictBase(TypedDict):
        names: Sequence[str]
        formats: Sequence[_DTypeLikeNested]

    # Mandatory + optional keys
    class _DTypeDict(_DTypeDictBase, total=False):
        offsets: Sequence[int]
        titles: Sequence[Any]  # Only `str` elements are usable as indexing aliases, but all objects are legal
        itemsize: int
        aligned: bool

    # A protocol for anything with the dtype attribute
    class _SupportsDType(Protocol[_DType_co]):
        @property
        def dtype(self) -> _DType_co: ...

else:
    _DTypeDict = Any

    class _SupportsDType: ...
    _SupportsDType = GenericAlias(_SupportsDType, _DType_co)


# Would create a dtype[np.void]
_VoidDTypeLike = Union[
    # (flexible_dtype, itemsize)
    Tuple[_DTypeLikeNested, int],
    # (fixed_dtype, shape)
    Tuple[_DTypeLikeNested, _ShapeLike],
    # [(field_name, field_dtype, field_shape), ...]
    #
    # The type here is quite broad because NumPy accepts quite a wide
    # range of inputs inside the list; see the tests for some
    # examples.
    List[Any],
    # {'names': ..., 'formats': ..., 'offsets': ..., 'titles': ...,
    #  'itemsize': ...}
    _DTypeDict,
    # (base_dtype, new_dtype)
    Tuple[_DTypeLikeNested, _DTypeLikeNested],
]

# Anything that can be coerced into numpy.dtype.
# Reference: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
DTypeLike = Union[
    DType[Any],
    # default data type (float64)
    None,
    # array-scalar types and generic types
    Type[Any],  # TODO: enumerate these when we add type hints for numpy scalars
    # anything with a dtype attribute
    _SupportsDType[DType[Any]],
    # character codes, type strings or comma-separated fields, e.g., 'float64'
    str,
    _VoidDTypeLike,
]

# NOTE: while it is possible to provide the dtype as a dict of
# dtype-like objects (e.g. `{'field1': ..., 'field2': ..., ...}`),
# this syntax is officially discourged and
# therefore not included in the Union defining `DTypeLike`.
#
# See https://github.com/numpy/numpy/issues/16891 for more details.

# Aliases for commonly used dtype-like objects.
# Note that the precision of `np.number` subclasses is ignored herein.
_DTypeLikeBool = Union[
    Type[bool],
    Type[np.bool_],
    DType[np.bool_],
    _SupportsDType[DType[np.bool_]],
    _BoolCodes,
]
_DTypeLikeUInt = Union[
    Type[np.unsignedinteger],
    DType[np.unsignedinteger],
    _SupportsDType[DType[np.unsignedinteger]],
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _UIntPCodes,
    _UIntCodes,
    _ULongLongCodes,
]
_DTypeLikeInt = Union[
    Type[int],
    Type[np.signedinteger],
    DType[np.signedinteger],
    _SupportsDType[DType[np.signedinteger]],
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _IntPCodes,
    _IntCodes,
    _LongLongCodes,
]
_DTypeLikeFloat = Union[
    Type[float],
    Type[np.floating],
    DType[np.floating],
    _SupportsDType[DType[np.floating]],
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
]
_DTypeLikeComplex = Union[
    Type[complex],
    Type[np.complexfloating],
    DType[np.complexfloating],
    _SupportsDType[DType[np.complexfloating]],
    _Complex64Codes,
    _Complex128Codes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
]
_DTypeLikeDT64 = Union[
    Type[np.timedelta64],
    DType[np.timedelta64],
    _SupportsDType[DType[np.timedelta64]],
    _TD64Codes,
]
_DTypeLikeTD64 = Union[
    Type[np.datetime64],
    DType[np.datetime64],
    _SupportsDType[DType[np.datetime64]],
    _DT64Codes,
]
_DTypeLikeStr = Union[
    Type[str],
    Type[np.str_],
    DType[np.str_],
    _SupportsDType[DType[np.str_]],
    _StrCodes,
]
_DTypeLikeBytes = Union[
    Type[bytes],
    Type[np.bytes_],
    DType[np.bytes_],
    _SupportsDType[DType[np.bytes_]],
    _BytesCodes,
]
_DTypeLikeVoid = Union[
    Type[np.void],
    DType[np.void],
    _SupportsDType[DType[np.void]],
    _VoidCodes,
    _VoidDTypeLike,
]
_DTypeLikeObject = Union[
    type,
    DType[np.object_],
    _SupportsDType[DType[np.object_]],
    _ObjectCodes,
]

_DTypeLikeFlexible = Union[
    Type[np.flexible],
    "np.dtype[np.flexible]",
    "_SupportsDType[np.dtype[np.flexible]]",
    _StrCodes,
    _BytesCodes,
    _VoidCodes,
    _VoidDTypeLike,
]

_DTypeLikeComplex_co = Union[
    _DTypeLikeBool,
    _DTypeLikeUInt,
    _DTypeLikeInt,
    _DTypeLikeFloat,
    _DTypeLikeComplex,
]
