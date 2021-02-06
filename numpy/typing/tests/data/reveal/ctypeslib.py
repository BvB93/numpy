from typing import Any
import numpy as np
import numpy.ctypeslib as ct

class ArrayInterface:
    @property
    def __array_interface__(self) -> Any: ...

AR_bool: np.ndarray[Any, np.dtype[np.bool_]]
AR_ubyte: np.ndarray[Any, np.dtype[np.ubyte]]
AR_ushort: np.ndarray[Any, np.dtype[np.ushort]]
AR_uintc: np.ndarray[Any, np.dtype[np.uintc]]
AR_uint: np.ndarray[Any, np.dtype[np.uint]]
AR_ulonglong: np.ndarray[Any, np.dtype[np.ulonglong]]
AR_byte: np.ndarray[Any, np.dtype[np.byte]]
AR_short: np.ndarray[Any, np.dtype[np.short]]
AR_intc: np.ndarray[Any, np.dtype[np.intc]]
AR_int: np.ndarray[Any, np.dtype[np.int_]]
AR_longlong: np.ndarray[Any, np.dtype[np.longlong]]
AR_single: np.ndarray[Any, np.dtype[np.single]]
AR_double: np.ndarray[Any, np.dtype[np.double]]
AR_void: np.ndarray[Any, np.dtype[np.void]]

reveal_type(  # E: Type[numpy.ctypeslib._ndptr[None]]
    ct.ndpointer()
)
reveal_type(  # E: Type[numpy.ctypeslib._ndptr[None]]
    ct.ndpointer(np.int64)
)
reveal_type(  # E: Type[numpy.ctypeslib._ndptr[None]]
    ct.ndpointer(np.int64, shape=None)
)
reveal_type(  # E: Type[numpy.ctypeslib._ndptr[None]]
    ct.ndpointer(None, shape=(1,))
)
reveal_type(  # E: Type[numpy.ctypeslib._concrete_ndptr[numpy.dtype[{int64}]]]
    ct.ndpointer(np.int64, shape=(5,))
)
reveal_type(  # E: Type[numpy.ctypeslib._concrete_ndptr[numpy.dtype[Any]]]
    ct.ndpointer("i8", shape=(5,))
)

reveal_type(ct.c_intp())  # E: {intp}

reveal_type(ct.as_ctypes_type(np.bool_))  # E: Type[ctypes.c_bool]
reveal_type(ct.as_ctypes_type(np.ubyte))  # E: Type[ctypes.c_ubyte]
reveal_type(ct.as_ctypes_type(np.ushort))  # E: Type[ctypes.c_ushort]
reveal_type(ct.as_ctypes_type(np.uintc))  # E: Type[ctypes.c_uint]
reveal_type(ct.as_ctypes_type(np.uint))  # E: Type[ctypes.c_ulong]
reveal_type(ct.as_ctypes_type(np.ulonglong))  # E: Type[ctypes.c_ulong]
reveal_type(ct.as_ctypes_type(np.byte))  # E: Type[ctypes.c_byte]
reveal_type(ct.as_ctypes_type(np.short))  # E: Type[ctypes.c_short]
reveal_type(ct.as_ctypes_type(np.intc))  # E: Type[ctypes.c_int]
reveal_type(ct.as_ctypes_type(np.int_))  # E: Type[ctypes.c_long]
reveal_type(ct.as_ctypes_type(np.longlong))  # E: Type[ctypes.c_long]
reveal_type(ct.as_ctypes_type(np.single))  # E: Type[ctypes.c_float]
reveal_type(ct.as_ctypes_type(np.double))  # E: Type[ctypes.c_double]
reveal_type(ct.as_ctypes_type([("i8", np.int64), ("f8", np.float64)]))  # E: Type[Any]
reveal_type(ct.as_ctypes_type("i8"))  # E: Type[Any]
reveal_type(ct.as_ctypes_type("f8"))  # E: Type[Any]

reveal_type(ct.as_ctypes(AR_bool))  # E: ctypes.c_bool
reveal_type(ct.as_ctypes(AR_ubyte))  # E: ctypes.c_ubyte
reveal_type(ct.as_ctypes(AR_ushort))  # E: ctypes.c_ushort
reveal_type(ct.as_ctypes(AR_uintc))  # E: ctypes.c_uint
reveal_type(ct.as_ctypes(AR_uint))  # E: ctypes.c_ulong
reveal_type(ct.as_ctypes(AR_ulonglong))  # E: ctypes.c_ulong
reveal_type(ct.as_ctypes(AR_byte))  # E: ctypes.c_byte
reveal_type(ct.as_ctypes(AR_short))  # E: ctypes.c_short
reveal_type(ct.as_ctypes(AR_intc))  # E: ctypes.c_int
reveal_type(ct.as_ctypes(AR_int))  # E: ctypes.c_long
reveal_type(ct.as_ctypes(AR_longlong))  # E: ctypes.c_long
reveal_type(ct.as_ctypes(AR_single))  # E: ctypes.c_float
reveal_type(ct.as_ctypes(AR_double))  # E: ctypes.c_double
reveal_type(ct.as_ctypes(AR_void))  # E: Any
reveal_type(ct.as_ctypes(ArrayInterface()))  # E: Any

reveal_type(ct.as_array(AR_ubyte))  # E: numpy.ndarray[Any, numpy.dtype[{ubyte}]]
reveal_type(ct.as_array(1))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]
reveal_type(ct.as_array([1.0]))  # E: numpy.ndarray[Any, numpy.dtype[{float_}]]
reveal_type(ct.as_array([[1j]]))  # E: numpy.ndarray[Any, numpy.dtype[{complex_}]]
reveal_type(ct.as_array([True]))  # E: numpy.ndarray[Any, numpy.dtype[{bool_}]]
reveal_type(ct.as_array(['a']))  # E: numpy.ndarray[Any, numpy.dtype[{str_}]]
reveal_type(ct.as_array([b'test']))  # E: numpy.ndarray[Any, numpy.dtype[{bytes_}]]
reveal_type(ct.as_array(ArrayInterface()))  # E: numpy.ndarray[Any, numpy.dtype[Any]]
