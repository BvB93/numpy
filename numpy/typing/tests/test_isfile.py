import os
from pathlib import Path

import pytest
import numpy as np
from numpy.testing import assert_

ROOT = Path(np.__file__).parents[0]


@pytest.mark.parametrize(
    "file",
    [
        ROOT / "py.typed",
        ROOT / "__init__.pyi",
        ROOT / "char.pyi",
        ROOT / "ctypeslib.pyi",
        ROOT / "rec.pyi",
        ROOT / "core" / "__init__.pyi",
        ROOT / "distutils" / "__init__.pyi",
        ROOT / "f2py" / "__init__.pyi",
        ROOT / "fft" / "__init__.pyi",
        ROOT / "lib" / "__init__.pyi",
        ROOT / "linalg" / "__init__.pyi",
        ROOT / "ma" / "__init__.pyi",
        ROOT / "matrixlib" / "__init__.pyi",
        ROOT / "polynomial" / "__init__.pyi",
        ROOT / "random" / "__init__.pyi",
        ROOT / "testing" / "__init__.pyi",
        ROOT / "typing" / "_ufuncs.pyi",
    ],
)
def test_isfile(file: Path) -> None:
    """Test if all ``.pyi`` files are properly installed."""
    assert_(os.path.isfile(file))
