import numpy as np

def dag(op):
    return op.conj().T

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
