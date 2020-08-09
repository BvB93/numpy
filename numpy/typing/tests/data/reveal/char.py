import numpy as np

a = 'a'
b = b'b'
A = ['a', 'b']
B = [b'a', b'b']

reveal_type(np.char.add(A, a))  # E: numpy.ndarray
reveal_type(np.char.add(B, b))  # E: numpy.ndarray
