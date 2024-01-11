import ctypes
import ctypes.util
import numpy as np
import dqclibs

# contains functions and constants that are used specifically for
# dft_utils.hamilton.intor files (no dependance on other files in dft_utils.hamilton.intor
# is required)

__all__ = [
    "NDIM", "CINT", "CGTO", "CPBC", "CSYMM", "c_null_ptr", "np2ctypes",
    "int2ctypes"
]

# CONSTANTS
NDIM = 3

CINT = dqclibs.CINT
CGTO = dqclibs.CGTO
CPBC = dqclibs.CPBC
# CVHF = dftlib.CVHF
CSYMM = dqclibs.CSYMM

c_null_ptr = ctypes.POINTER(ctypes.c_void_p)


def np2ctypes(a: np.ndarray) -> ctypes.c_void_p:
    """get the ctypes of the numpy ndarray

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> c_type = np2ctypes(a)

    Parameters
    ----------
    a : np.ndarray
        the numpy ndarray to get the ctypes.

    Returns
    -------
    ctypes.c_void_p
        the ctypes of the numpy ndarray.

    """
    return a.ctypes.data_as(ctypes.c_void_p)


def int2ctypes(a: int) -> ctypes.c_int:
    """convert the python's integer to ctypes' integer

    Examples
    --------
    >>> a = 1
    >>> c_type = int2ctypes(a)
    >>> c_type
    c_int(1)

    Parameters
    ----------
    a : int
        the python's integer to convert.

    Returns
    -------
    ctypes.c_int
        the ctypes' integer.

    """
    return ctypes.c_int(a)
