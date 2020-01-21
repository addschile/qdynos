import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

import qdynos.constants as const

from .utils import dag,inner,norm,matmult

################################################################################
# Generalized krylov subspace method functions                                 #
################################################################################
def lanczos(A, nvecs=6, v0=None, nstates=None, ret_type=None, return_evecs=True):
    """
    """

    # is A a linear operator function or a matrix
    if not hasattr(A, '__call__'):
        # size of matrix
        m = A.shape[0]
        ret_type = A.dtype

    # is A a sparse matrix
    if isinstance(A, sp.csr_matrix):
        sparse = True

    # initialize v0
    if v0 is None:
        # initialize random v0
        if sparse:
            v0 = sp.random(m, 1, format='csr', dtype=ret_type)
        else:
            v0 = np.random.rand(m,1).astype('complex')

    nvprev = norm(v0)
    nvprev2 = nvprev

    # list of V's
    # use list here, because appending vectors is much faster
    # than storing a matrix with changing sparse structure
    V = []
    V.append( v0/np.sqrt(nvprev) )
    T = np.zeros((nvecs,nvecs))

    # form krylov vectors and tridiagonal matrix
    for i in range(nvecs-1):
        if not hasattr(A, '__call__'):
            V.append( matmult(A,V[-1]) )
        else:
            V.append( A(V[-1]) )
        # compute alpha 
        T[i,i] = inner(V[-2],V[-1]).real
        if i>0:
            V[-1] += -T[i,i]*V[-2] - T[i-1,i]*V[-3]
        else:
            V[-1] += -T[i,i]*V[-2]
        # normalize previous vector
        nvprev = norm(V[-1])
        V[-1] /= np.sqrt(nvprev)
        # compute beta
        T[i,i+1] = np.sqrt(nvprev)
        T[i+1,i] = T[i,i+1]
        w,v = np.linalg.eigh(T)
    T[-1,-1] = inner(V[-1], matmult(A,V[-1])).real

    if return_evecs:
        return T , V
    else:
        return T

def arnoldi(A, nvecs=6, v0=None, nstates=None, ret_type=None):

    # is A a linear operator function or a matrix
    if not hasattr(A, '__call__'):
        # size of matrix
        m = A.shape[0]
        ret_type = A.dtype

    # is A a sparse matrix
    if isinstance(A, sp.csr_matrix):
        sparse = True

    # initialize v0
    if v0 is None:
        # initialize random v0
        if sparse:
            v0 = sp.random(m, 1, format='csr', dtype=ret_type)
        else:
            v0 = np.random.rand(m,1).astype('complex')

    # matrix of v's
    V = []
    V.append( v0.copy() )

    # form krylov subspace and upper hessenberg matrix
    T = np.zeros((nvecs,nvecs), dtype=v0.dtype)
    for j in range(nvecs-1):
        if not hasattr(A, '__call__'):
            w = matmult(A, V[j])
        else:
            w = A(V[j])
        for i in range(j+1):
            T[i,j] = inner(w, V[i])
            w -= T[i,j]*V[i]
        if j < nvecs-1:
            T[j+1,j] = np.sqrt(norm(w))
            V.append( w/T[j+1,j] )

    return T , V

################################################################################
# Lanczos function that does propagation without storing all the vectors       #
################################################################################
def lanczos_lowmem(A, nvecs, v0, dt):
    """
    """

    # size of matrix
    m = v0.shape[0]

    # normalize v0 for stability
    nv0 = norm(v0)
    vprev = v0/np.sqrt(nv0)

    # initialize Krylov Hamiltonian
    T = np.zeros((nvecs,nvecs))

    # form Krylov vectors and tridiagonal matrix
    nvprev = norm(vprev)
    nvprev2 = nvprev
    for i in range(nvecs-1):
        vnext = matmult(A,vprev)
        # compute alpha 
        T[i,i] = inner(vprev,vnext).real
        if i>0:
            vnext += -T[i,i]*vprev - T[i-1,i]*vprev2
        else:
            vnext += -T[i,i]*vprev
        # set previous by two vector
        vprev2 = vprev
        # set previous vector
        nvprev = norm(vnext)
        vnext /= np.sqrt(nvprev)
        vprev = vnext
        # compute beta
        T[i,i+1] = np.sqrt(nvprev)
        T[i+1,i] = T[i,i+1]
    # one final matrix multiplication and alpha
    vnext = matmult(A,vprev)
    T[-1,-1] = inner(vprev,vnext).real

    # form propagator
    psiprop = expm(-1.j*dt*T/const.hbar)[:,0]

    # project back out of Krylov subspace
    # initialize new lanczos iteration
    vprev = v0/np.sqrt(norm(v0))
    nvprev = norm(vprev)
    nvprev2 = nvprev
    # initialize output psi
    psiout = psiprop[0]*vprev
    for i in range(nvecs-1):
        vnext = matmult(A,vprev)
        # compute alpha 
        if i>0:
            vnext += -T[i,i]*vprev - T[i-1,i]*vprev2
        else:
            vnext += -T[i,i]*vprev
        # set previous by two vector
        vprev2 = vprev
        # set previous vector
        nvprev = norm(vnext)
        vnext /= np.sqrt(nvprev)
        vprev = vnext
        psiout += psiprop[i+1]*vprev
    vnext = matmult(A,vprev)

    return psiout

################################################################################
# Functions that use krylov subspace methods for oring all the vectors         #
################################################################################
def propagate(V, T, dt):
    """
    """
    nvecs = len(V)
    psiprop = expm(dt*T)[:,0]
    for i in range(nvecs):
        if i==0:
            psiout = psiprop[i]*V[i]
        else:
            psiout += psiprop[i]*V[i]
    return psiout

def propagate_unitary(V, T, dt):
    """
    """
    nvecs = len(V)
    psiprop = expm(-1.j*dt*T/const.hbar)[:,0]
    for i in range(nvecs):
        if i==0:
            psiout = psiprop[i]*V[i]
        else:
            psiout += psiprop[i]*V[i]
    return psiout

def krylov_prop(A, nvecs, psi, dt, method, unitary=True, nstates=None, ret_type=None, lowmem=False, return_all=False):
    """
    """
    if lowmem:
        if method == 'arnoldi':
            raise ValueError("Arnoldi must store all vectors. No low memory option.")
        else:
            return lanczos_lowmem(A, nvecs, psi, dt)
    else:
        if method == 'arnoldi':
            T , V = arnoldi(A, nvecs=nvecs, v0=psi, nstates=nstates, ret_type=ret_type)
        else:
            T , V = lanczos(A, nvecs=nvecs, v0=psi, nstates=nstates, ret_type=ret_type)
    if unitary:
        psiout = propagate_unitary(V, T, dt)
    else:
        psiout = propagate(V, T, dt)
    if return_all:
        return psiout , T , V
    else:
        return psiout
