from typing import Any, List

from numpy._pytesttester import PytestTester

__all__: List[str]
test: PytestTester

class LinAlgError(Exception): ...

def tensorsolve(a, b, axes=...): ...
def solve(a, b): ...
def tensorinv(a, ind=...): ...
def inv(a): ...
def matrix_power(a, n): ...
def cholesky(a): ...
def qr(a, mode=...): ...
def eigvals(a): ...
def eigvalsh(a, UPLO=...): ...
def eig(a): ...
def eigh(a, UPLO=...): ...
def svd(a, full_matrices=..., compute_uv=..., hermitian=...): ...
def cond(x, p=...): ...
def matrix_rank(A, tol=..., hermitian=...): ...
def pinv(a, rcond=..., hermitian=...): ...
def slogdet(a): ...
def det(a): ...
def lstsq(a, b, rcond=...): ...
def norm(x, ord=..., axis=..., keepdims=...): ...
def multi_dot(arrays, *, out=...): ...
