from typing import Dict, Any

import numpy as np
from numpy.typing import ArrayLike


class A:
    pass


class B:
    @property
    def __array_interface__(self) -> Dict[int, int]: ...


class C:
    def __array_struct__(self) -> Any: ...


x1: ArrayLike = (i for i in range(10))  # E: Incompatible types in assignment
x2: ArrayLike = A()  # E: Incompatible types in assignment
x3: ArrayLike = {1: "foo", 2: "bar"}  # E: Incompatible types in assignment
x4: ArrayLike = B()  # E: Incompatible types in assignment

# This should be failing, but it doesn't (mypy bug).
# Any object with the `__array_struct__` attribute/method is currently
# accepted, regardless of its signature

# x5: ArrayLike = C()  # E: Incompatible types in assignment

scalar = np.int64(1)
scalar.__array__(dtype=np.float64)  # E: Unexpected keyword argument
array = np.array([1])
array.__array__(dtype=np.float64)  # E: Unexpected keyword argument
