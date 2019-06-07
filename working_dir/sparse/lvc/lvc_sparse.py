import numpy as np

class LVC:

    def __init__(self, nel, nmodes, mode_size, es, omegas, intra, inter, mode_intra, mode_inter):
        self.nel        = nel
        self.nmodes     = nmodes
        self.mode_size  = mode_size
        # make iterators for modes
        self.mode_iters = []
        for mode in self.mode_size:
            self.mode_iters.append( range(mode) )
        self.es         = es
        self.omegas     = omegas
        self.intra      = intra
        self.inter      = inter
        self.mode_intra = mode_intra
        self.mode_inter = mode_inter
        # make diagonal parts of hamiltonian
        self.hs = []
        for i in range(nmodes):
            h = np.array([omegas[i]*(float(j)+0.5) for j in range(self.mode_size[i])])
            self.hs.append( h )
        # make position operators
        self.qs = []
        for i in range(nmodes):
            ns = self.mode_size[i]
            q = np.zeros((ns,)*2)
            for j in range(ns-1):
                q[j,j+1] = np.sqrt(0.5*float(j+1))
                q[j+1,1] = np.sqrt(0.5*float(j+1))
            self.qs.append( q )

    def init_sys(self, modes_init=None, wfn=True):
        """
        """
        if wfn:
            # initialize wavefunction
            self.state = np.zeros(self.mode_size, dtype=complex)
            if modes_init==None:
                self.state[(0,)*(self.nmodes+1)] = 1.
            else:
                self.state[modes_init] = 1.

    def act_H_from_left(self):
        """
        """
        psi = np.zeros_like(self.state)
        # act energy shift part
        for i,e in enumerate(self.es):
            psi[i,:] += e*self.state[i,:]
        # act diagonal part of single-mode hamiltonians
        for i in range(self.nel):
            ind = (i,)
            for j in range(self.nmodes):
                iter_new = self.mode_iters[:j] + [...] + self.mode_iters[(j+1):]
                for k in range(self.mode_size[j]):
                    psi[ind] += np.dot(self.hs[j], self.state[ind])
        # act holstein coupling part
        for i in range(nmodes):
        # act peierls coupling part
        for i in range(nmodes):
        # act diagonal inter-mode coupling part
        # act off-diagonal inter-mode coupling part
        return psi

    def run():
        """
        """

