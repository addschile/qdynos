import numpy as np
# TODO
"""
I should change these to be general to take in either numpy matrices or 
scipy sparse matrices

I'll need to go back through the whole code to make sure that every bit
of linear algebra is called from here
"""

def dag(op):
    return op.conj().T

def commutator(op1,op2):
    return np.dot(op1,op2) - np.dot(op2,op1)

def anticommutator(op1,op2):
    return np.dot(op1,op2) + np.dot(op2,op1)

def norm(psi):
    if is_vector(psi):
        return np.dot(dag(psi),psi)[0,0].real
    if is_matrix(psi):
        return np.trace(psi).real

def is_hermitian(op):
    if is_matrix(op):
        if np.allclose(op.conj().T, op):
            return True
        else:
            raise ValueError('Physical observables must be Hermitian')
    else:
        raise ValueError('Hermiticity check requires matrix')

def is_vector(vec):
    return vec.shape[0]!=vec.shape[1]

def is_matrix(mat):
    return mat.shape[0]==mat.shape[1]

def is_tensor(tensor):
    return (len(tensor.shape)>2)

def to_liouville(rho):
    if len(rho.shape) == 2:
        # A matrix to a vector
        return rho.flatten().astype(np.complex)
    else:
        # A tensor to a matrix 
        ns = rho.shape[0]
        rho_mat = np.zeros((ns*ns,ns*ns), dtype=np.complex)
        I = 0
        for i in range(ns):
            for j in range(ns):
                J = 0
                for k in range(ns):
                    for l in range(ns):
                        rho_mat[I,J] = rho[i,j,k,l]
                        J += 1
                I += 1
        return rho_mat

def from_liouville(rho_vec, ns=None):
    if ns is None:
        ns = int(np.sqrt(len(rho_vec)))
    return rho_vec.reshape(ns,ns).astype(np.complex_)
