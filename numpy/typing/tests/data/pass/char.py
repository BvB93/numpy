from itertools import combinations_with_replacement, chain
from typing import Iterator, Iterable, Tuple, Union, cast, TYPE_CHECKING, Any

import numpy as np
from numpy import char

if TYPE_CHECKING:
    from numpy.char import _ArrayLikeChar, _ArrayLike

a = "a"
b = b"a"
A = np.array(["a", "b", "c"], dtype=str)
B = A.astype(bytes)
A.setflags(write=False)
B.setflags(write=False)

i = 0
I = np.arange(3)
I.setflags(write=False)

combinations_bin1 = cast(
    Iterator[Tuple["_ArrayLikeChar", "_ArrayLikeChar"]],
    combinations_with_replacement([a, b, A, B], 2),
)
for x1, x2 in combinations_bin1:
    char.equal(x1, x2)
    char.not_equal(x1, x2)
    char.greater_equal(x1, x2)
    char.less_equal(x1, x2)
    char.greater(x1, x2)
    char.less(x1, x2)
    char.compare_chararrays(x1, x2, "==", rstrip=False)

combinations_bin2 = cast(
    Iterator[Tuple["_ArrayLike[str]", "_ArrayLike[str]"]],
    chain(
        combinations_with_replacement([a, A], 2),
        combinations_with_replacement([b, B], 2),
    ),
)
for y1, y2 in combinations_bin2:
    char.add(y1, y2)
    char.center(y1, i, y2)
    char.center(y1, I, y2)
    char.join(y1, y2)
    char.ljust(y1, i, y2)
    char.ljust(y1, I, y2)
    char.lstrip(y1, y2)
    char.partition(y1, y2)
    char.replace(y1, y2, y2, count=i)
    char.replace(y1, y2, y2, count=I)
    char.rjust(y1, i, y2)
    char.rjust(y1, I, y2)
    char.rpartition(y1, y2)
    char.rsplit(y1, y2, maxsplit=i)
    char.rsplit(y1, y2, maxsplit=I)
    char.rstrip(y1, y2)
    char.split(y1, y2, maxsplit=i)
    char.split(y1, y2, maxsplit=I)
    char.strip(y1, y2)

combinations_mono = cast(Iterable["_ArrayLikeChar"], [a, A, b, B])
for x1 in combinations_mono:
    char.multiply(x1, i)
    char.multiply(x1, I)
    char.capitalize(x1)
    char.expandtabs(x1, i)
    char.expandtabs(x1, I)
    char.lower(x1)
    char.splitlines(x1, True)
    char.swapcase(x1)
    char.title(x1)
    char.upper(x1)
    char.zfill(x1, i)
    char.zfill(x1, I)

char.mod(["%s", "%s", "%s"], A)
char.mod([b"%s", b"%s", b"%s"], A)
char.encode(a)
char.encode(A, "utf8")
char.decode(b)
char.decode(B, "utf8")

# mod, encode, decode
