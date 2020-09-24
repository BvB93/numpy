import sys
from typing import Any, overload, Sequence, TYPE_CHECKING, Union, List, Tuple, Optional

from numpy import ndarray
from ._shape import _Shape
from ._dtype_like import DtypeLike

if sys.version_info >= (3, 8):
    from typing import Protocol, TypedDict
    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol, TypedDict
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True

if TYPE_CHECKING or HAVE_PROTOCOL:
    class _InterfaceDictBase(TypedDict):
        shape: _Shape
        typestr: str
        version: int

    class InterfaceDict(_InterfaceDictBase, total=False):
        descr: List[Union[
            Tuple[
                Union[str, Tuple[str, str]],
                Union[str, List[Any]],  # TODO: wait for recursive types
            ],
            Tuple[
                Union[str, Tuple[str, str]],
                Union[str, List[Any]],  # TODO: wait for recursive types
                Optional[_Shape],
            ]
        ]]

        # TODO: Allow `data` to be an object exposing the buffer interface
        # once we have an appropiate protocol
        data: Optional[Tuple[int, bool]]
        strides: Optional[_Shape]
        mask: Optional["_SupportsArrayInterface"]
        offset: int

    class _SupportsArrayInterface(Protocol):
        @property
        def __array_interface__(self) -> InterfaceDict: ...

    class _SupportsArrayStruct(Protocol):
        @property
        def __array_struct__(self) -> Any: ...  # PyCapsule

    class _SupportsArray(Protocol):
        @overload
        def __array__(self, __dtype: DtypeLike = ...) -> ndarray: ...
        @overload
        def __array__(self, dtype: DtypeLike = ...) -> ndarray: ...
else:
    InterfaceDict = Any
    _SupportsArrayInterface = Any
    _SupportsArrayStruct = Any
    _SupportsArray = Any

# TODO: support buffer protocols once
#
# https://bugs.python.org/issue27501
#
# is resolved. See also the mypy issue:
#
# https://github.com/python/typing/issues/593
ArrayLike = Union[bool, int, float, complex, Sequence,
    _SupportsArray, _SupportsArrayInterface, _SupportsArrayStruct,
]
