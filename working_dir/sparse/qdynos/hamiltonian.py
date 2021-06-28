import numpy as np
import qdynos.constants as const

from .utils import dag,is_hermitian,is_vector,is_matrix

class Hamiltonian(object):
    """
    Base Hamiltonian class.
    """

    def __init__(self, H, nstates=None, baths=None, hbar=1.):
        """
        Parameters
        ----------
        H: np.ndarray
            Hamiltonian matrix
        nstates: int
            Number that specifies the size of Hilbert space
        bath: list of Bath classes
            Baths that independently couple to the system
        hbar: float
        """
        const.hbar = hbar
        if nstates==None:
            self.nstates = H.shape[0]
        else:
            self.nstates = nstates
        self.check_hermiticity(H)
        self.ham = H
        #self.eigensystem()
        self.baths = baths
        if self.baths != None: self.nbaths = len(self.baths)
        
    def __repr__(self):
        return "Hamiltonian class"

    @property
    def Heig(self):
        return np.diag(self.ev)

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
        self.ev = self.ev[:self.nstates]
        self.compute_frequencies()

    def compute_frequencies(self):
        self.omegas = np.array([[(self.ev[i]-self.ev[j]) for j in range(self.nstates)]
                                    for i in range(self.nstates)])/const.hbar

    def compute_unique_freqs(self):
        self.frequencies = np.unique(self.omegas)

    def to_eigenbasis(self, op):
        if is_vector(op):
            return np.dot(dag(self.ek), op)[:self.nstates,:]
        elif is_matrix(op):
            return np.dot(dag(self.ek), np.dot(op, self.ek))[:self.nstates,:self.nstates]
        else:
            raise AttributeError("Not a valid operator")

    def from_eigenbasis(self, op, trunc=True):
        if is_vector(op):
            return np.dot(self.ek, op)[:self.nstates,:]
        elif is_matrix(op):
            return np.dot(self.ek, np.dot(op, dag(self.ek)))[:self.nstates,:self.nstates]
        else:
            raise AttributeError("Not a valid operator")

    def commutator(self, op, eig=True):
        if eig:
            return self.Heig@op - op@self.Heig
        else:
            return self.ham@op - op@self.ham

    def thermal_dm(self):
        if self.baths != None:
            rho_eq = np.zeros((self.nstates,self.nstates), dtype=complex)
            rho_eq += np.diag(np.exp(-self.ev/self.baths[0].kT))
            return rho_eq/np.trace(rho_eq)
        else:
            raise NotImplementedError("Bath must be initialized before calling")
