cimport numpy as np
import numpy as np

cdef extern from "kernel.cu":
    void calculateLegendre(int l_max, double *x, double *result)

def python_wrapper(int l_max, np.ndarray[np.double_t, ndim=1] x):
    cdef np.ndarray[np.double_t, ndim=2] result = np.empty((l_max + 1, x.shape[0]), dtype=np.double)
    calculateLegendre(l_max, &x[0], &result[0, 0])
    return result
