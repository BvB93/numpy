import os
import sys
import ctypes
from typing import (
    Any,
    List,
    Union,
    TypeVar,
    Type,
    Generic,
    Optional,
    overload,
    Iterable,
    ClassVar,
    Tuple,
    Sequence,
)

from numpy import (
    ndarray,
    dtype,
    generic,
    bool_,
    byte,
    short,
    intc,
    int_,
    intp,
    longlong,
    ubyte,
    ushort,
    uintc,
    uint,
    ulonglong,
    single,
    double,
    float_,
    complex_,
    str_,
    bytes_,
    void,
    _flagsobj,
)
from numpy.core._internal import _ctypes
from numpy.typing import (
    # Arrays
    ArrayLike,
    NDArray,
    _SupportsArrayInterface,
    _NestedSequence,
    _SupportsArray,

    # Shapes
    _ShapeLike,

    # DTypes
    DTypeLike,
    _DTypeLike,
    _SupportsDType,
    _DTypeLikeBool,
    _VoidDTypeLike,
    _BoolCodes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _UIntCodes,
    _ULongLongCodes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _IntCodes,
    _LongLongCodes,
    _SingleCodes,
    _DoubleCodes,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

# TODO: Add a proper `_Shape` bound once we've got variadic typevars
_DType = TypeVar("_DType", bound=dtype[Any])
_DTypeOptional = TypeVar("_DTypeOptional", bound=Optional[dtype[Any]])
_ScalarType = TypeVar("_ScalarType", bound=generic)

_FlagsKind = Literal[
    'C_CONTIGUOUS', 'CONTIGUOUS', 'C',
    'F_CONTIGUOUS', 'FORTRAN', 'F',
    'ALIGNED', 'A',
    'WRITEABLE', 'W',
    'OWNDATA', 'O',
    'UPDATEIFCOPY', 'U',
    'WRITEBACKIFCOPY', 'X',
]

_FlagsType = Union[None, _FlagsKind, Iterable[_FlagsKind], int, _flagsobj]

__all__: List[str]

# TODO: Add a shape typevar once we have variadic typevars (PEP 646)
class _ndptr(ctypes.c_void_p, Generic[_DTypeOptional]):
    # In practice these 4 classvars are defined in the dynamic class
    # returned by `ndpointer`
    _dtype_: ClassVar[_DTypeOptional]
    _shape_: ClassVar[None]
    _ndim_: ClassVar[Optional[int]]
    _flags_: ClassVar[Optional[List[_FlagsKind]]]

    @overload  # type: ignore[override]
    @classmethod
    def from_param(cls: Type[_ndptr[None]], obj: ndarray[Any, Any]) -> _ctypes: ...
    @overload
    @classmethod
    def from_param(cls: Type[_ndptr[_DType]], obj: ndarray[Any, _DType]) -> _ctypes: ...

class _concrete_ndptr(_ndptr[_DType]):
    _dtype_: ClassVar[_DType]
    _shape_: ClassVar[Tuple[int, ...]]  # type: ignore[assignment]
    @property
    def contents(self) -> ndarray[Any, _DType]: ...

def load_library(
    libname: str,
    loader_path: Union[str, os.PathLike[str]],
) -> ctypes.CDLL: ...

@overload
def ndpointer(
    dtype: None = ...,
    ndim: int = ...,
    shape: Optional[_ShapeLike] = ...,
    flags: _FlagsType = ...,
) -> Type[_ndptr[None]]: ...
@overload
def ndpointer(
    dtype: _DTypeLike[_ScalarType],
    ndim: int = ...,
    shape: Optional[_ShapeLike] = ...,
    flags: _FlagsType = ...,
) -> Type[_concrete_ndptr[dtype[_ScalarType]]]: ...
@overload
def ndpointer(
    dtype: DTypeLike,
    ndim: int = ...,
    shape: Optional[_ShapeLike] = ...,
    flags: _FlagsType = ...,
) -> Type[_concrete_ndptr[dtype[Any]]]: ...

c_intp = intp

# NOTE: `void` dtypes can return either a `ctypes.Union` or
# `ctypes.Structure` subclass
@overload
def as_ctypes_type(dtype: _DTypeLikeBool) -> Type[ctypes.c_bool]: ...
@overload
def as_ctypes_type(dtype: Union[_ByteCodes, _DTypeLike[byte]]) -> Type[ctypes.c_byte]: ...
@overload
def as_ctypes_type(dtype: Union[_ShortCodes, _DTypeLike[short]]) -> Type[ctypes.c_short]: ...
@overload
def as_ctypes_type(dtype: Union[_IntCCodes, _DTypeLike[intc]]) -> Type[ctypes.c_int]: ...
@overload
def as_ctypes_type(dtype: Union[_IntCodes, _DTypeLike[int_], int]) -> Type[ctypes.c_long]: ...
@overload
def as_ctypes_type(dtype: Union[_LongLongCodes, _DTypeLike[longlong]]) -> Type[ctypes.c_longlong]: ...
@overload
def as_ctypes_type(dtype: Union[_UByteCodes, _DTypeLike[ubyte]]) -> Type[ctypes.c_ubyte]: ...
@overload
def as_ctypes_type(dtype: Union[_UShortCodes, _DTypeLike[ushort]]) -> Type[ctypes.c_ushort]: ...
@overload
def as_ctypes_type(dtype: Union[_UIntCCodes, _DTypeLike[uintc]]) -> Type[ctypes.c_uint]: ...
@overload
def as_ctypes_type(dtype: Union[_UIntCodes, _DTypeLike[uint]]) -> Type[ctypes.c_ulong]: ...
@overload
def as_ctypes_type(dtype: Union[_ULongLongCodes, _DTypeLike[ulonglong]]) -> Type[ctypes.c_ulonglong]: ...
@overload
def as_ctypes_type(dtype: Union[_SingleCodes, _DTypeLike[single]]) -> Type[ctypes.c_float]: ...
@overload
def as_ctypes_type(dtype: Union[_DoubleCodes, _DTypeLike[double], float]) -> Type[ctypes.c_double]: ...
@overload
def as_ctypes_type(dtype: _VoidDTypeLike) -> Type[Any]: ...
@overload
def as_ctypes_type(dtype: str) -> Type[Any]: ...

@overload
def as_array(obj: ctypes._PointerLike, shape: Sequence[int]) -> NDArray[Any]: ...
@overload
def as_array(obj: _NestedSequence[_SupportsArray[_DType]], shape: Optional[_ShapeLike] = ...) -> ndarray[Any, _DType]: ...
@overload
def as_array(obj: _NestedSequence[str], shape: Optional[_ShapeLike] = ...) -> NDArray[str_]: ...
@overload
def as_array(obj: _NestedSequence[bytes], shape: Optional[_ShapeLike] = ...) -> NDArray[bytes_]: ...
@overload
def as_array(obj: _NestedSequence[bool], shape: Optional[_ShapeLike] = ...) -> NDArray[bool_]: ...
@overload
def as_array(obj: _NestedSequence[int], shape: Optional[_ShapeLike] = ...) -> NDArray[int_]: ...
@overload
def as_array(obj: _NestedSequence[float], shape: Optional[_ShapeLike] = ...) -> NDArray[float_]: ...
@overload
def as_array(obj: _NestedSequence[complex], shape: Optional[_ShapeLike] = ...) -> NDArray[complex_]: ...
@overload
def as_array(obj: object, shape: Optional[_ShapeLike] = ...) -> NDArray[Any]: ...

@overload
def as_ctypes(obj: NDArray[bool_]) -> ctypes.c_bool: ...
@overload
def as_ctypes(obj: NDArray[byte]) -> ctypes.c_byte: ...
@overload
def as_ctypes(obj: NDArray[short]) -> ctypes.c_short: ...
@overload
def as_ctypes(obj: NDArray[intc]) -> ctypes.c_int: ...
@overload
def as_ctypes(obj: NDArray[int_]) -> ctypes.c_long: ...
@overload
def as_ctypes(obj: NDArray[longlong]) -> ctypes.c_longlong: ...
@overload
def as_ctypes(obj: NDArray[ubyte]) -> ctypes.c_ubyte: ...
@overload
def as_ctypes(obj: NDArray[ushort]) -> ctypes.c_ushort: ...
@overload
def as_ctypes(obj: NDArray[uintc]) -> ctypes.c_uint: ...
@overload
def as_ctypes(obj: NDArray[uint]) -> ctypes.c_ulong: ...
@overload
def as_ctypes(obj: NDArray[ulonglong]) -> ctypes.c_ulonglong: ...
@overload
def as_ctypes(obj: NDArray[single]) -> ctypes.c_float: ...
@overload
def as_ctypes(obj: NDArray[double]) -> ctypes.c_double: ...
@overload
def as_ctypes(obj: NDArray[void]) -> Any: ...  # `ctypes.Union` or `ctypes.Structure`
@overload
def as_ctypes(obj: _SupportsArrayInterface) -> Any: ...
