import sys
from typing import Callable, TypeVar, Collection, Tuple, Any, Dict, TYPE_CHECKING

if sys.version_info >= (3, 8):
    from typing import Protocol
    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True

if TYPE_CHECKING or HAVE_PROTOCOL:
    _T_co = TypeVar("_T_co", covariant=True)
    _ST_co = TypeVar("_ST_co", covariant=True, bound=str)
    _ST_contra = TypeVar("_ST_contra", contravariant=True, bound=str)

    class _DispatchFunc(Protocol[_ST_co]):
        @property
        def __name__(self) -> _ST_co:
            ...
        __call__: Callable

    class _SupportsArrayFunction(Protocol[_T_co, _ST_contra]):
        def __array_function__(
            self,
            func: _DispatchFunc[_ST_contra],
            types: Collection[type],
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> _T_co:
            ...

else:
    _SupportsArrayFunction = Any
    _DispatchFunc = Any
