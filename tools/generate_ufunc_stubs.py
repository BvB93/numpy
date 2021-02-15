#!/usr/bin/env python
"""A script for generating `numpy.ufunc` annotations."""

from __future__ import annotations

import sys
import types
import hashlib
import argparse
import importlib
from itertools import chain
from typing import (
    Dict,
    IO,
    List,
    Tuple,
    NamedTuple,
    Generator,
    Mapping,
    Iterable,
    TYPE_CHECKING,
    MutableSequence,
    Sequence,
)

import numpy as np

if TYPE_CHECKING:
    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal

__all__ = [
    "generate_ufunc_stubs",
    "parse_types",
    "INSERT",
    "REPLACE",
    "DELETE",
    "ArgTuple",
    "CaseTuple",
]

TEMPLATE = '''"""{hash}"""

import sys
from typing import (
    TypeVar,
    Any,
    Tuple,
    Type,
    Union,
    overload,
    Optional,
    List,
    Callable,
    NoReturn,
    Callable,
    Sequence,
    type_check_only,
)

from numpy import (
    ndarray,
    generic,
    dtype,
    ufunc,
    bool_,
    unsignedinteger,
    signedinteger,
    floating,
    complexfloating,
    timedelta64,
    datetime64,
    object_,
    _Casting,
    _OrderKACF,
)
from numpy.typing import (
    ArrayLike,
    DTypeLike,
    _NestedSequence,
    _SupportsArray,
    _RecursiveSequence,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeTD64_co,
    _ArrayLikeDT64_co,
    _ArrayLikeObject_co,
    _ArrayLikeFlexible_co,
    _DTypeLikeBool,
    _DTypeLikeUInt,
    _DTypeLikeInt,
    _DTypeLikeFloat,
    _DTypeLikeComplex,
    _DTypeLikeTD64,
    _DTypeLikeDT64,
    _DTypeLikeObject,
    _DTypeLikeFlexible,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

_T = TypeVar("_T")
_ArrayType = TypeVar("_ArrayType", bound=ndarray[Any, Any])
_ST1 = TypeVar("_ST1", bound=generic)
_ST2 = TypeVar("_ST2", bound=generic)

_ScalarOrArray1 = Union[_ST1, ndarray[Any, dtype[_ST1]]]
_ScalarOrArray2 = Tuple[
    Union[_ST1, _ST2],
    Union[ndarray[Any, dtype[_ST1]], ndarray[Any, dtype[_ST2]]]
]

# In reality this should be a length of list 3 containing an
# int, an int, and a callable/`None`, but there's no way to express
# that.
_ExtObj = List[Union[int, Optional[Callable[[str, int], Any]]]]

_ArrayLikeTD64 = _NestedSequence[_SupportsArray[dtype[timedelta64]]]

_CastingSafe = Literal["no", "equiv", "safe", "same_kind"]
'''

KWARGS1: str = (
    "casting: _CastingSafe = ..., "
    "order: _OrderKACF = ..., "
    "subok: bool = ..., "
    "extobj: _ExtObj = ..., "
    "where: Optional[_ArrayLikeBool_co] = ..."
)
KWARGS2: str = (
    "order: _OrderKACF = ..., "
    "subok: bool = ..., "
    "extobj: _ExtObj = ..., "
    "where: Optional[_ArrayLikeBool_co] = ..."
)

UFUNC_TEMPLATE = """
@type_check_only
class {name}(ufunc):
{overloads}
    @property
    def __name__(self) -> {cls_name}: ...
    @property
    def nin(self) -> {nin}: ...
    @property
    def nout(self) -> {nout}: ...
    @property
    def nargs(self) -> {nargs}: ...
    @property
    def ntypes(self) -> {ntypes}: ...
    @property
    def identity(self) -> {identity}: ...
    @property
    def signature(self) -> {signature}: ...
"""

OVERLOAD_TEMPLATE = """
    @overload
    def __call__(self, {input}, out: None = ..., *, dtype: {dtype} = ..., {kwargs}) -> {output}: ...
""".strip("\n")

OVERLOAD_TEMPLATE_FLEXIBLE = """
    @overload
    def __call__(self, {input}, out: None = ..., *, dtype: Any = ..., signature: Any = ..., casting: _Casting = ..., {kwargs}) -> NoReturn: ...
""".strip("\n")

OVERLOAD_TEMPLATE_RECURSIVE = """
    @overload
    def __call__(self, {input}, out: None = ..., *, dtype: Any = ..., signature: Any = ..., casting: _Casting = ..., {kwargs}) -> {output}: ...
""".strip("\n")

OVERLOAD_TEMPLATE_SIGNATURE = """
    @overload
    def __call__(self, {input}, out: None = ..., *, signature: Any, casting: _Casting = ..., {kwargs}) -> {output}: ...
""".strip("\n")

OVERLOAD_TEMPLATE_OUT = """
    @overload
    def __call__(self, {input}, out: {out}, *, dtype: Any = ..., signature: Any = ..., casting: _Casting = ..., {kwargs}) -> {output}: ...
""".strip("\n")

OVERLOAD_TEMPLATE_UNSAFE = """
    @overload
    def __call__(self, {input}, out: None = ..., *, casting: Literal["unsafe"], dtype: Any = ..., signature: Any = ..., {kwargs}) -> {output}: ...
""".strip("\n")


#: A dictionary that maps `numpy.dtype.kind` to one of the
#: input parameters in `TEMPLATE`
KIND_MAPPING_INP: Dict[str, str] = {
    "b": "_ArrayLikeBool_co",
    "u": "_ArrayLikeUInt_co",
    "i": "_ArrayLikeInt_co",
    "f": "_ArrayLikeFloat_co",
    "c": "_ArrayLikeComplex_co",
    "m": "_ArrayLikeTD64_co",
    "M": "_ArrayLikeDT64_co",
    "O": "_ArrayLikeObject_co",
}

#: A dictionary that maps `numpy.dtype.kind` to one of the
#: `numpy.generic` output types in `TEMPLATE`
KIND_MAPPING_OUT: Dict[str, str] = {
    "b": "bool_",
    "u": "unsignedinteger[Any]",
    "i": "signedinteger[Any]",
    "f": "floating[Any]",
    "c": "complexfloating[Any, Any]",
    "m": "timedelta64",
    "M": "datetime64",
    "O": "Any",
}

#: A dictionary mapping a type-aliases/types to a priority;
#: used for sorting the list created by `parse_types`
PRIORITY_MAPPING: Dict[str, int] = {
    "_ArrayLikeBool_co": 0,
    "_ArrayLikeUInt_co": 1,
    "_ArrayLikeInt_co": 2,
    "_ArrayLikeFloat_co": 3,
    "_ArrayLikeComplex_co": 4,
    "_ArrayLikeTD64_co": 10,
    "_ArrayLikeDT64_co": 100,
    "_ArrayLikeObject_co": 1000,

    "bool_": 0,
    "unsignedinteger[Any]": 1,
    "signedinteger[Any]": 2,
    "floating[Any]": 3,
    "complexfloating[Any, Any]": 4,
    "timedelta64": 10,
    "datetime64": 100,
    "Any": 1000,
}

#: A dictionary mapping a priorities to scalar types
DTYPE_MAPPING = {
    0: "Optional[_DTypeLikeBool]",
    1: "Optional[_DTypeLikeUInt]",
    2: "Optional[_DTypeLikeInt]",
    3: "Optional[_DTypeLikeFloat]",
    4: "Optional[_DTypeLikeComplex]",
    10: "Optional[_DTypeLikeTD64]",
    100: "Optional[_DTypeLikeDT64]",
    1000: "Optional[_DTypeLikeObject]",
}

INSERT: Literal[0] = 0
REPLACE: Literal[1] = 1
DELETE: Literal[2] = 2


class ArgTuple(NamedTuple):
    #: The input types of a ufunc
    inp: Tuple[str, ...]

    #: The output types of a ufunc
    out: Tuple[str, ...]

    #: Allowed values of the `dtype` parameter. This should generally
    #: be a dtype-like object representing the same type as `ArgTuple.out`
    dtype: str = "Any"


class CaseTuple(NamedTuple):
    #: The operation to-be performed on the `ArgTuple` list:
    #: * `0`: Insert an element
    #: * `1`: Replace an element
    #: * `2`: Delete an element
    op: Literal[0, 1, 2]

    #: The index in the `ArgTuple` list where `op` is to-be performed
    idx: int

    #: The element to-be applied to `idx`.
    #: Only relevant when performing insertions or replacement operations.
    args: ArgTuple = ArgTuple((), (), "Any")


def _get_special_cases() -> Dict[np.ufunc, List[CaseTuple]]:
    """Construct the `SPECIAL_CASES` dictionary."""
    b_in = KIND_MAPPING_INP["b"]
    f_in = KIND_MAPPING_INP["f"]
    m_in = KIND_MAPPING_INP["m"]

    i_out = KIND_MAPPING_OUT["i"]
    f_out = KIND_MAPPING_OUT["f"]
    m_out = KIND_MAPPING_INP["m"]

    bool11 = ArgTuple(inp=(b_in,), out=(i_out,), dtype="None")
    bool21 = ArgTuple(inp=(b_in, b_in), out=(i_out,), dtype="None")
    bool22 = ArgTuple(inp=(b_in, b_in), out=(i_out, i_out), dtype="None")
    bool11_noreturn = ArgTuple(inp=(b_in,), out=("NoReturn",), dtype="Optional[_DTypeLikeBool]")
    bool21_noreturn = ArgTuple(inp=(b_in, b_in), out=("NoReturn",), dtype="Optional[_DTypeLikeBool]")

    return {
        # Ufuncs that contain `uu->u`-type signatures but lack `??->i`.
        # Explicitly insert a bool-based overload so they return `int8`
        # instead of `uint8`
        np.conjugate: [CaseTuple(INSERT, 0, bool11)],
        np.divmod: [CaseTuple(INSERT, 0, bool22)],
        np.fmod: [CaseTuple(INSERT, 0, bool21)],
        np.left_shift: [CaseTuple(INSERT, 0, bool21)],
        np.remainder: [CaseTuple(INSERT, 0, bool21)],
        np.power: [CaseTuple(INSERT, 0, bool21)],
        np.reciprocal: [CaseTuple(INSERT, 0, bool11)],
        np.right_shift: [CaseTuple(INSERT, 0, bool21)],
        np.square: [CaseTuple(INSERT, 0, bool11)],

        # Ufuncs that cannot operate on booleans
        np.gcd: [CaseTuple(INSERT, 0, bool21_noreturn)],
        np.lcm: [CaseTuple(INSERT, 0, bool21_noreturn)],
        np.negative: [CaseTuple(INSERT, 0, bool11_noreturn)],
        np.positive: [CaseTuple(INSERT, 0, bool11_noreturn)],
        np.sign: [CaseTuple(INSERT, 0, bool11_noreturn)],
        np.subtract: [CaseTuple(INSERT, 0, bool21_noreturn)],

        # Ufuncs that truly require `timedelta64` and
        # not just timedelta64-like objects
        np.divmod: [
            CaseTuple(REPLACE, 3, ArgTuple(inp=("_ArrayLikeTD64", "_ArrayLikeTD64"), out=(i_out, m_out))),
        ],
        np.true_divide: [
            CaseTuple(REPLACE, 3, ArgTuple(inp=("_ArrayLikeTD64", f_in), out=(m_out,))),
            CaseTuple(REPLACE, 4, ArgTuple(inp=("_ArrayLikeTD64", "_ArrayLikeTD64"), out=(f_out,))),
            CaseTuple(DELETE, 2),
        ],
        np.floor_divide: [
            CaseTuple(REPLACE, 5, ArgTuple(inp=("_ArrayLikeTD64", f_in), out=(m_out,))),
            CaseTuple(REPLACE, 6, ArgTuple(inp=("_ArrayLikeTD64", "_ArrayLikeTD64"), out=(i_out,))),
            CaseTuple(DELETE, 4),
            CaseTuple(INSERT, 0, bool21),
        ],
        np.remainder: [
            CaseTuple(REPLACE, 3, ArgTuple(inp=("_ArrayLikeTD64", "_ArrayLikeTD64"), out=(m_out,))),
        ],

        # Ufuncs containing both `OO->?` and `OO->O` signatures;
        # the former is removed
        np.equal: [CaseTuple(DELETE, 7)],
        np.not_equal: [CaseTuple(DELETE, 7)],
        np.greater: [CaseTuple(DELETE, 7)],
        np.greater_equal: [CaseTuple(DELETE, 7)],
        np.less: [CaseTuple(DELETE, 7)],
        np.less_equal: [CaseTuple(DELETE, 7)],
        np.logical_and: [CaseTuple(DELETE, 5)],
        np.logical_not: [CaseTuple(DELETE, 5)],
        np.logical_or: [CaseTuple(DELETE, 5)],
    }


#: A dictionary containing additional ufunc-specific instructions.
#: Used by `parse_types` case some special casing is required for specific ufuncs.
SPECIAL_CASES: Dict[np.ufunc, List[CaseTuple]] = _get_special_cases()


def _sort_func(key: ArgTuple) -> int:
    """A custom key function for sorting the output of `parse_types`."""
    ret = sum(PRIORITY_MAPPING[k] for k in key.inp)
    ret += sum(PRIORITY_MAPPING[k] for k in key.out)
    return ret


def parse_types(ufunc: np.ufunc) -> List[ArgTuple]:
    """Parse the `~numpy.ufunc.types` attribute of the passed ufunc.

    Examples
    --------
    .. code:: python

        >>> import numpy as  np

        >>> parse_types(np.add)
        [ArgTuple(inp=('_BoolLike', '_BoolLike'), out=('bool_',), dtype='Optional[_DTypeLike[bool_]]'),
         ArgTuple(inp=('_UIntLike', '_UIntLike'), out=('unsignedinteger[Any]',), dtype='Optional[_DTypeLike[unsignedinteger[Any]]]'),
         ArgTuple(inp=('_IntLike', '_IntLike'), out=('signedinteger[Any]',), dtype='Optional[_DTypeLike[signedinteger[Any]]]'),
         ArgTuple(inp=('_FloatLike', '_FloatLike'), out=('floating[Any]',), dtype='Optional[_DTypeLike[floating[Any]]]'),
         ArgTuple(inp=('_ComplexLike', '_ComplexLike'), out=('complexfloating[Any, Any]',), dtype='Optional[_DTypeLike[complexfloating[Any, Any]]]'),
         ArgTuple(inp=('_TD64Like', '_TD64Like'), out=('timedelta64',), dtype='Optional[_DTypeLike[timedelta64]]'),
         ArgTuple(inp=('datetime64', '_TD64Like'), out=('datetime64',), dtype='Optional[_DTypeLike[datetime64]]'),
         ArgTuple(inp=('_TD64Like', 'datetime64'), out=('datetime64',), dtype='Optional[_DTypeLike[datetime64]]'),
         ArgTuple(inp=('object_', 'object_'), out=('Any',), dtype='Optional[_DTypeLike[object_]]')]

    """
    ret_set = set()
    for expr in ufunc.types:
        _inp, _out = expr.split("->")
        inp = tuple(KIND_MAPPING_INP[np.dtype(k).kind] for k in _inp)
        out = tuple(KIND_MAPPING_OUT[np.dtype(k).kind] for k in _out)
        dtype = DTYPE_MAPPING[max(PRIORITY_MAPPING[k] for k in inp)]
        ret_set.add(ArgTuple(inp, out, dtype))

    ret_list = sorted(ret_set, key=_sort_func)
    _special_case(ufunc, ret_list)
    return ret_list


def _special_case(ufunc: np.ufunc, types_list: MutableSequence[ArgTuple]) -> None:
    """Apply special casing to a specific ufunc (if applicable)."""
    try:
        lst = SPECIAL_CASES[ufunc]
    except KeyError:
        return

    for op, i, tup in lst:
        if op == INSERT:
            types_list.insert(i, tup)
        elif op == DELETE:
            del types_list[i]
        elif op == REPLACE:
            types_list[i] = tup
        else:
            raise ValueError(f"Unknown op: {op!r}")


def _gather_ufuncs(module: types.ModuleType) -> Dict[str, np.ufunc]:
    """Search and return all ufuncs in the passed module."""
    iterator = ((k, getattr(module, k)) for k in dir(module))
    return {k: v for k, v in iterator if isinstance(v, np.ufunc)}


def _yield_overloads(
    type_list: Sequence[ArgTuple], out_type: str
) -> Generator[str, None, None]:
    """Yield a set of overloads for specific to every passed `ArgTuple`."""
    # An overload for removing `np.flexible`-based array-likes
    _inp = type_list[0][0]
    yield OVERLOAD_TEMPLATE_FLEXIBLE.format(
        input=", ".join(f"__x{i}: _ArrayLikeFlexible_co" for i, _ in enumerate(_inp, 1)),
        kwargs=KWARGS2,
    )

    # Ufunc-specific overloads
    for _inp, _out, dtype in type_list:
        inp = ", ".join(f"__x{i}: {j}" for i, j in enumerate(_inp, 1))
        out = ", ".join(item for item in _out)

        if out == "NoReturn":
            output = out
        else:
            output = f"{out_type}[{out}]"

        yield OVERLOAD_TEMPLATE.format(
            input=inp, output=output, dtype=dtype, kwargs=KWARGS1
        )

    # An overload for all >4D array-likes
    yield OVERLOAD_TEMPLATE_RECURSIVE.format(
        input=", ".join(f"__x{i}: _RecursiveSequence" for i, _ in enumerate(_inp, 1)),
        output=f"{out_type}[Any]",
        kwargs=KWARGS2,
    )

    # An overload for `casting="unsafe"`
    yield OVERLOAD_TEMPLATE_UNSAFE.format(
        input=", ".join(f"__x{i}: Any" for i, _ in enumerate(_inp, 1)),
        output=f"{out_type}[Any]",
        kwargs=KWARGS2,
    )

    # An overload for the `signature` parameter
    yield OVERLOAD_TEMPLATE_SIGNATURE.format(
        input=", ".join(f"__x{i}: Any" for i, _ in enumerate(_inp, 1)),
        output=f"{out_type}[Any]",
        kwargs=KWARGS2,
    )

    # An overload for the `out` parameter
    if len(_out) == 1:
        out = "Union[_ArrayType, Tuple[_ArrayType]]"
        output = "_ArrayType"
    else:
        out = output = "Tuple[_ArrayType, _ArrayType]"
    yield OVERLOAD_TEMPLATE_OUT.format(
        input=", ".join(f"__x{i}: Any" for i, _ in enumerate(_inp, 1)),
        out=out,
        output=output,
        kwargs=KWARGS2,
    )


def _construct_cls(ufunc: np.ufunc, types_list: Sequence[ArgTuple], name: str) -> str:
    """Create the main `~numpy.ufunc` subclass and its ``__call__`` method."""
    nout = range(ufunc.nout)
    if ufunc.nout == 1:
        out_type = "_ScalarOrArray1"
    elif ufunc.nout == 2:
        out_type = f"_ScalarOrArray2"
    else:
        raise NotImplementedError(f"{ufunc.__name__}.nout > 2")

    signature = f"Literal[{ufunc.signature!r}]" if ufunc.signature is not None else "None"

    if ufunc.identity is None:
        identity = "None"
    elif type(ufunc.identity) in (int, str, bool):
        identity = f"Literal[{ufunc.identity!r}]"
    else:
        identity = type(ufunc.identity).__name__

    overloads = _yield_overloads(types_list, out_type)
    return UFUNC_TEMPLATE.format(
        overloads="\n".join(i for i in overloads),
        name=name,
        cls_name=f"Literal[{ufunc.__name__!r}]",
        nin=f"Literal[{ufunc.nin}]",
        nout=f"Literal[{ufunc.nout}]",
        nargs=f"Literal[{ufunc.nargs}]",
        ntypes=f"Literal[{ufunc.ntypes}]",
        identity=identity,
        signature=signature,
    )

# TODO: Generate stubs for all other `ufunc` methods

def _construct_at(ufunc: np.ufunc, types_list: Iterable[ArgTuple]) -> str:
    return "    at: Callable[..., Any]  # TODO\n"


def _construct_reduce(ufunc: np.ufunc, types_list: Iterable[ArgTuple]) -> str:
    return "    reduce: Callable[..., Any]  # TODO\n"


def _construct_accumulate(ufunc: np.ufunc, types_list: Iterable[ArgTuple]) -> str:
    return "    accumulate: Callable[..., Any]  # TODO\n"


def _construct_reduceat(ufunc: np.ufunc, types_list: Iterable[ArgTuple]) -> str:
    return "    reduceat: Callable[..., Any]  # TODO\n"


def _construct_outer(ufunc: np.ufunc, types_list: Iterable[ArgTuple]) -> str:
    return "    outer: Callable[..., Any]  # TODO\n"


def _gather_n(ufuncs: Iterable[np.ufunc]) -> np.ndarray:
    """
    Identify the largest `~numpy.ufunc.nin`, `~numpy.ufunc.out` and `~numpy.ufunc.nargs`
    values in a series of ufuncs.
    """
    iterator = chain.from_iterable((u.nin, u.nout, u.nargs) for u in ufuncs)
    array = np.fromiter(iterator, dtype=np.int64).reshape(-1, 3)
    return array.max(axis=0)


def generate_ufunc_stubs(file: IO[str], ufuncs: Mapping[str, np.ufunc]) -> None:
    """Generate a stub file with `~numpy.ufunc`-specific subclasses.

    Parameters
    ----------
    file : file_like
        A :term:`python:file-like object` used for writing the output.
        The passed file should be opened in text mode.
    ufuncs : dict of ufuncs
        A mapping with ufunc names as keys and matching `~numpy.ufunc`
        instances as values (*e.g.* ``{"add": np.add}``).

    """
    with open(__file__, "rb") as f:
        sha256 = hashlib.sha256()
        sha256.update(f.read())

    # Figure out how many type-aliases we for a series
    # `ufunc.nout` and `ufunc.nargs` values
    _, nout, nargs = _gather_n(ufuncs.values())
    file.write(TEMPLATE.format(hash=sha256.hexdigest()))

    for name, ufunc in ufuncs.items():
        try:
            types_list = parse_types(ufunc)
        except KeyError as ex:
            raise NotImplementedError(f"{name}: unsupported dtype kind {ex}") from None

        file.write(_construct_cls(ufunc, types_list, name))
        if ufunc.nout == 1:
            file.write(_construct_at(ufunc, types_list))
            if ufunc.nin == 2 and ufunc.signature is None:
                file.write(_construct_reduce(ufunc, types_list))
                file.write(_construct_reduceat(ufunc, types_list))
                file.write(_construct_accumulate(ufunc, types_list))
                file.write(_construct_outer(ufunc, types_list))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--file",
        nargs="?",
        default=None,
        help="The file used for writing the output",
    )

    args = parser.parse_args()
    file = args.file

    if file is None:
        generate_ufunc_stubs(sys.stdout, ufuncs=_gather_ufuncs(np))
    else:
        with open(file, "w", encoding="utf8") as f:
            generate_ufunc_stubs(f, ufuncs=_gather_ufuncs(np))


if __name__ == "__main__":
    main()
