from tempfile import TemporaryFile
import numpy as np

A = np.recarray((1, 1), dtype=[('a', np.float64)])
A.setflags(write=False)

reveal_type(A)  # E: numpy.recarray
reveal_type(A.real)  # E: numpy.recarray
reveal_type(A.imag)  # E: numpy.recarray
reveal_type(A.astype(np.int64))  # E: numpy.recarray
reveal_type(A.byteswap())  # E: numpy.recarray
reveal_type(A.copy())  # E: numpy.recarray
reveal_type(A.view(np.recarray))  # E: numpy.recarray
reveal_type(A.view(np.ndarray))  # E: numpy.ndarray
reveal_type(A.view(np.bool_, np.ndarray))  # E: numpy.ndarray
reveal_type(A.getfield(np.bool_))  # E: numpy.recarray
reveal_type(A.reshape(-1))  # E: numpy.recarray
reveal_type(A.transpose(0, 1))  # E: numpy.recarray
reveal_type(A.swapaxes(0, 1))  # E: numpy.recarray
reveal_type(A.flatten())  # E: numpy.recarray
reveal_type(A.ravel())  # E: numpy.recarray
reveal_type(A.squeeze(0))  # E: numpy.recarray

with TemporaryFile() as f:
    B = np.memmap(f, dtype=np.float64, shape=(1, 1))
    B.setflags(write=False)

    reveal_type(B)  # E: numpy.memmap
    reveal_type(B.real)  # E: numpy.memmap
    reveal_type(B.imag)  # E: numpy.memmap
    reveal_type(B.astype(np.int64))  # E: numpy.memmap
    reveal_type(B.byteswap())  # E: numpy.memmap
    reveal_type(B.copy())  # E: numpy.memmap
    reveal_type(B.view(np.memmap))  # E: numpy.memmap
    reveal_type(B.view(np.ndarray))  # E: numpy.ndarray
    reveal_type(B.view(np.bool_, np.ndarray))  # E: numpy.ndarray
    reveal_type(B.getfield(np.bool_))  # E: numpy.memmap
    reveal_type(B.reshape(-1))  # E: numpy.memmap
    reveal_type(B.transpose(0, 1))  # E: numpy.memmap
    reveal_type(B.swapaxes(0, 1))  # E: numpy.memmap
    reveal_type(B.flatten())  # E: numpy.memmap
    reveal_type(B.ravel())  # E: numpy.memmap
    reveal_type(B.squeeze(0))  # E: numpy.memmap
