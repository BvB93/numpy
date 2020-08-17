import sys
from typing import Collection, Tuple, Dict, Any, Optional, overload, TypeVar

import numpy as np
from numpy.typing import ArrayLike, DtypeLike, _DispatchFunc, _SupportsArrayFunction

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

T = TypeVar("T")

class TestCls:
    def __array_function__(
        self,
        func: _DispatchFunc[Literal["asarray"]],
        types: Collection[type],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> bool: ...
    def __array__(self, dtype: DtypeLike = ...) -> np.ndarray: ...

class _AsArray(_DispatchFunc[Literal["asarray"]]):
    @overload
    def __call__(
        self,
        a: _SupportsArrayFunction[T, Literal["asarray"]],
        dtype: DtypeLike = ...,
        order: Optional[str] = ...,
    ) -> T: ...
    @overload
    def __call__(
        self, a: ArrayLike, dtype: DtypeLike = ..., order: Optional[str] = ...
    ) -> np.ndarray: ...
asarray: _AsArray

class _TestFunc(_DispatchFunc[Literal["test_func"]]):
    @overload
    def __call__(self, a: _SupportsArrayFunction[T, Literal["test_func"]]) -> T: ...
    @overload
    def __call__(self, a: ArrayLike) -> np.ndarray: ...
test_func: _TestFunc

a = np.array([0])
b = [0]
c = TestCls()

reveal_type(asarray(a))  # E: numpy.ndarray
reveal_type(asarray(b))  # E: numpy.ndarray
reveal_type(asarray(c))  # E: bool

reveal_type(test_func(a))  # E: numpy.ndarray
reveal_type(test_func(b))  # E: numpy.ndarray
reveal_type(test_func(c))  # E: numpy.ndarray
