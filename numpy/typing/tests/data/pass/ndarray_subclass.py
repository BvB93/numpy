from tempfile import TemporaryFile
import numpy as np

A = np.recarray((1, 1), dtype=[('a', int)])
A.setflags(write=False)

A.real
A.imag
A.astype(int)
A.byteswap()
A.copy()
A.view(np.recarray)
A.view(np.ndarray)
A.view(bool, np.ndarray)
A.getfield(bool)
A.reshape(-1)
A.transpose(0, 1)
A.swapaxes(0, 1)
A.flatten()
A.ravel()
A.squeeze(0)

with TemporaryFile() as f:
    B = np.memmap(f, dtype=int, shape=(1, 1))
    B.setflags(write=False)

    B.real
    B.imag
    B.astype(int)
    B.byteswap()
    B.copy()
    B.view(np.memmap)
    B.view(np.ndarray)
    B.view(bool, np.ndarray)
    B.getfield(bool)
    B.reshape(-1)
    B.transpose(0, 1)
    B.swapaxes(0, 1)
    B.flatten()
    B.ravel()
    B.squeeze(0)
