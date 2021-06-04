from typing import List, Tuple, overload, Any

from numpy import floating, complexfloating, object_
from numpy.typing import (
    NDArray,
    ArrayLike,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeObject_co,
)

from typing_extensions import Literal as L, SupportsIndex

_BinKind = L[
    "stone",
    "auto",
    "doane",
    "fd",
    "rice",
    "scott",
    "sqrt",
    "sturges",
]

__all__: List[str]

@overload
def histogram_bin_edges(  # type: ignore[misc]
    a: _ArrayLikeFloat_co,
    bins: _BinKind | SupportsIndex | _ArrayLikeFloat_co = ...,
    range: None | Tuple[float, float] = ...,
    weights: None | ArrayLike = ...,
) -> NDArray[floating[Any]]: ...
@overload
def histogram_bin_edges(
    a: _ArrayLikeComplex_co,
    bins: _BinKind | SupportsIndex | _ArrayLikeComplex_co = ...,
    range: None | Tuple[float, float] = ...,
    weights: None | ArrayLike = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def histogram_bin_edges(
    a: _ArrayLikeObject_co,
    bins: _BinKind | SupportsIndex | Any = ...,
    range: None | Tuple[float, float] = ...,
    weights: None | ArrayLike = ...,
) -> NDArray[object_]: ...

def histogram(a, bins=..., range=..., normed=..., weights=..., density=...): ...
def histogramdd(sample, bins=..., range=..., normed=..., weights=..., density=...): ...
