"""Various protocols used by `~numpy.DType` for extracting property values
from its underlying `~numpy.generic`.
"""

import sys
from typing import (
    TYPE_CHECKING,
    Optional,
    Tuple,
    TypeVar,
    Optional,
    Tuple,
    Union,
    Any,
    List,
)

from . import _HAS_TYPING_EXTENSIONS
from ._shape import _Shape
from ._generic_alias import _DType as DType

if sys.version_info >= (3, 8):
    from typing import Protocol
elif TYPE_CHECKING or _HAS_TYPING_EXTENSIONS:
    from typing_extensions import Protocol

# Construct a runtime-subscriptable version of `MappingProxyType`
if TYPE_CHECKING or sys.version_info >= (3, 9):
    from types import MappingProxyType
else:
    from types import MappingProxyType as _MappingProxyType
    from ._generic_alias import _GenericAlias

    _KT = TypeVar("_KT")
    _VT = TypeVar("_VT")
    MappingProxType = _GenericAlias(_MappingProxType, (_KT, _VT))

if TYPE_CHECKING or _HAS_TYPING_EXTENSIONS or sys.version_info >= (3, 8):
    # See `dtype.fields`
    _FieldValue = Union[
        Tuple[DType[Any], int],
        Tuple[DType[Any], int, Any],
    ]

    _BoolType_co = TypeVar("_BoolType_co", bound=bool, covariant=True)
    _IntType_co = TypeVar("_IntType_co", bound=int, covariant=True)
    _StrType_co = TypeVar("_StrType_co", bound=str, covariant=True)
    _ListType_co = TypeVar("_ListType_co", bound=List[Any], covariant=True)
    _ShapeType_co = TypeVar("_ShapeType_co", bound=_Shape, covariant=True)
    _NameTupType_co = TypeVar(
        "_NameTupType_co", bound=Optional[Tuple[str, ...]], covariant=True
    )
    _FieldType_co = TypeVar(
        "_FieldType_co",
        bound=Optional[MappingProxyType[str, _FieldValue]],
        covariant=True,
    )
    _DTypeTupType = TypeVar(
        "_DTypeTupType",
        bound=Optional[Tuple[DType[Any], _Shape]],
        covariant=True,
    )

    class _SupportsHasObject(Protocol[_BoolType_co]):
        def _has_object(self) -> _BoolType_co: ...

    class _SupportsIsBuiltin(Protocol[_IntType_co]):
        def _isbuiltin(self) -> _IntType_co: ...

    class _SupportsIsAlignedStruct(Protocol[_BoolType_co]):
        def _isalignedstruct(self) -> _BoolType_co: ...

    class _SupportsItemSize(Protocol[_IntType_co]):
        def _itemsize(self) -> _IntType_co: ...

    class _SupportsKind(Protocol[_StrType_co]):
        def _kind(self) -> _StrType_co: ...

    class _SupportsName(Protocol[_StrType_co]):
        def _name(self) -> _StrType_co: ...

    class _SupportsNames(Protocol[_NameTupType_co]):
        def _names(self) -> _NameTupType_co: ...

    class _SupportsFields(Protocol[_FieldType_co]):
        def _fields(self) -> _FieldType_co: ...

    class _SupportsDescr(Protocol[_ListType_co]):
        def _descr(self) -> _ListType_co: ...

    class _SupportsNDim(Protocol[_IntType_co]):
        def _ndim(self) -> _IntType_co: ...

    class _SupportsShape(Protocol[_ShapeType_co]):
        def _shape(self) -> _ShapeType_co: ...

    class _SupportsSubDType(Protocol[_DTypeTupType]):
        def _subDType(self) -> _DTypeTupType: ...

else:
    _SupportsHasObject = Any
    _SupportsIsBuiltin = Any
    _SupportsIsAlignedStruct = Any
    _SupportsItemSize = Any
    _SupportsKind = Any
    _SupportsName = Any
    _SupportsNames = Any
    _SupportsFields = Any
    _SupportsDescr = Any
    _SupportsNDim = Any
    _SupportsShape = Any
    _SupportsSubDType = Any
