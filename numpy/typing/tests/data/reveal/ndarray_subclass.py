import numpy as np

A = np.recarray((1, 1), dtype=[('a', int)])
A.setflags(write=False)

reveal_type(A)  # E: numpy.recarray
reveal_type(A.real)  # E: numpy.recarray
reveal_type(A.imag)  # E: numpy.recarray
reveal_type(A.astype(int))  # E: numpy.recarray
reveal_type(A.byteswap())  # E: numpy.recarray
reveal_type(A.copy())  # E: numpy.recarray
reveal_type(A.view(np.recarray))  # E: numpy.recarray
reveal_type(A.view(np.ndarray))  # E: numpy.ndarray
reveal_type(A.view(bool, np.ndarray))  # E: numpy.ndarray
reveal_type(A.getfield(bool))  # E: numpy.recarray
reveal_type(A.reshape(-1))  # E: numpy.recarray
reveal_type(A.transpose(0, 1))  # E: numpy.recarray
reveal_type(A.swapaxes(0, 1))  # E: numpy.recarray
reveal_type(A.flatten())  # E: numpy.recarray
reveal_type(A.ravel())  # E: numpy.recarray
reveal_type(A.squeeze(0))  # E: numpy.recarray
