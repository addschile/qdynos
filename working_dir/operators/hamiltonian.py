import numpy as np
import scipy as sp
import qdynos.constants as const

from .operator import Operator

class Hamiltonian(Operator):
    """
    """

    def __init__(self, H, nstates=None, baths=None, units='au', convert=None):
        """
        """
        op = Operator(H, nstates=nstates)
        self.op = op.op
        self.type = op.type
        self.nstates = op.nstates
        assert(self.type in [np.ndarray, sp.spmatrix, FunctionType])
        self.ev = None

        # make sure Hamiltonian is Hermitian
        self.check_hermiticity()

        # set working units
        const.hbar = const.get_hbar(units)

        # set up environment
        self.baths = baths
        if self.baths != None: self.nbaths = len(self.baths)

    @property
    def Heig(self):
        """Returns the eigenvalues as a matrix

        Note: Won't ever get called in other places of the code if not using
        other types of operators (i.e. sparse or Function)
        """
        return Operator(np.diag(self.ev[:self.nstates]))

    def check_hermiticity(self, H):
        """Check hermiticity of the Hamiltonian."""
        if self.is_hermitian != True:
            raise ValueError('Hamiltonian is not Hermitian')

    def compute_frequencies(self):
        """Computes frequencies of Hamiltonian."""
        if self.ev is None:
            self.eigensystem()
        self.omegas = np.array([[(self.ev[i]-self.ev[j]) for j in range(self.nstates)]
                                    for i in range(self.nstates)])/const.hbar

    def compute_unique_freqs(self):
        """Computes unique frequencies of Hamiltonian."""
        self.frequencies = np.unique(self.omegas)

    def commutator(self, op, eig=True):
        """Computes unique frequencies of Hamiltonian."""
        if eig:
            return self.Heig.dot(op) - op.dot(self.Heig)
        else:
            return self.dot(op) - op.dot(self)

    def thermal_dm(self):
        if self.baths != None:
            if isinstance(self.op, np.ndarray):
                self.ev,ek = numpy.linalg.eigh(self.op)
                self.ek = Operator(ek)
            else:
                raise AttributeError('Operator must be np.ndarray for full eigensystem')
            rho_eq = np.zeros((self.nstates,self.nstates), dtype=complex)
            rho_eq += np.diag(np.exp(-self.ev/self.baths[0].kT))
            return rho_eq/np.trace(rho_eq)
        else:
            raise NotImplementedError("Bath must be initialized before calling")
