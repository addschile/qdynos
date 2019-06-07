from numpy cimport ndarray
cimport numpy as np
cimport cython

@cython.boundscheck(False)
def eom_pq(int nmodes, double coup, ndarray[np.float64_t, ndim=1] omegas,
           ndarray[np.float64_t, ndim=1] cs, ndarray[np.float64_t, ndim=1] Qs,
           ndarray[np.float64_t, ndim=1] Ps, ndarray[np.float64_t, ndim=1] dQs,
           ndarray[np.float64_t, ndim=1] dPs):
    # TODO parallelize this?
    for i in range(nmodes):
        dPs[i] = -omegas[i]**2.*Qs[i] - cs[i]*coup
        dQs[i] = Ps[i]
    return dPs , dQs
