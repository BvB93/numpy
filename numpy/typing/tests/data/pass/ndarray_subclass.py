import numpy as np

A = np.recarray((1, 1), dtype=[('a', int)])
A.setflags(write=False)

A.real
A.imag
A.astype(int)
A.byteswap()
A.copy()
A.view(np.recarray)
A.view(bool, np.recarray)
A.getfield(int)
A.reshape(-1)
A.transpose(0, 1)
A.swapaxes(0, 1)
A.flatten()
A.ravel()
A.squeeze(0)
