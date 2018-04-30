import numpy as np
import global_variables as gv
from utils import dag,is_hermitian,is_vector,is_matrix

class Hamiltonian(object):
    """
    Base Hamiltonian class.
    """

    def __init__(self, H, hbar=1.):
        """
        """
        gv.hbar = hbar
        self.nstates = H.shape[0]
        self.check_hermiticity(H)
        self.ham = H
        self.eigensystem()
        
    def check_hermiticity(self, H):
        """
        Check hermiticity of the Hamiltonian
        """
        if is_hermitian(H):
            self.is_hermitian = True
        else:
            raise ValueError('Hamiltonian is not Hermitian')

    def eigensystem(self):
        self.ev,self.ek = np.linalg.eigh(self.ham)
        self.compute_frequencies()

    def compute_frequencies(self):
        self.omega_diff = np.zeros((self.nstates,self.nstates))
        for i in range(self.nstates):
            for j in range(self.nstates):
                self.omega_diff[i,j] = (self.ev[i]-self.ev[j])/gv.hbar
        self.frequencies = list()
        for i in range(self.nstates):
            for j in range(self.nstates):
                omega_ij = self.omega_diff[i,j]
                if len(self.frequencies)!=0:
                    flag=1
                    for k in range(len(self.frequencies)):
                        if omega_ij==self.frequencies[k]: flag=0
                    if flag: self.frequencies.append(omega_ij)
                else: self.frequencies.append(omega_ij)

    def to_eigenbasis(self, op):
        if is_vector(op):
            return np.dot(dag(self.ek), op)
        elif is_matrix(op):
            return np.dot(dag(self.ek), np.dot(op, self.ek))
        else:
            raise AttributeError('Not a valid operator')

    def from_eigenbasis(self, op):
        if is_vector(op):
            return np.dot(self.ek, op)
        elif is_matrix(op):
            return np.dot(self.ek, np.dot(op, dag(self.ek)))
        else:
            raise AttributeError('Not a valid operator')

    def commutator(self, op):
        return np.dot(self.ham,op) - np.dot(op,self.ham)
