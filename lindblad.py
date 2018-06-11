from __future__ import print_function,absolute_import

import numpy as np
import qdynos.constants as const

from .integrator import Integrator
from .dynamics import Dynamics
from .utils import commutator,dag,to_liouville
from .options import Options
from .results import Results
from numba import jit

@jit
def lb_deriv(rho,A,L0,LdL):
    drho = np.dot(A,rho) + np.dot(rho,dag(A))
    for i in range(len(L0)):
        drho += 2.0*np.dot(L0[i] , np.dot(rho , dag(L0[i])))
        drho -= np.dot(dag(L0[i]), np.dot(L0[i],rho))
        drho -= np.dot(rho, np.dot(dag(L0[i]), L0[i]))
    for i in range(len(LdL)):
        drho[ LdL[i][1][0] , LdL[i][1][0] ] +=  2.0*rho[ LdL[i][1][1] , LdL[i][1][1] ]*LdL[i][0]
        drho[ LdL[i][1][1] , :] -=  rho[ LdL[i][1][1] , :]*LdL[i][0]
        drho[ : , LdL[i][1][1] ] -= rho[ : , LdL[i][1][1]]*LdL[i][0]
    return drho

class Lindblad(Dynamics):
    """
    """

    def __init__(self, ham, time_dependent=False):
        """
        """
        super(Lindblad, self).__init__(ham)
        self.ham = ham
        self.time_dep = time_dependent

    def setup(self, options, results):
        # generic setup
        if options==None:
            self.options = Options()
        else:
            self.options = options
            if self.options.method == "exact":
                raise NotImplementedError
        if results==None:
            self.results = Results()
        else:
            self.results = results
            if self.results.map_ops:
                assert(repr(self.ham)=="Multidimensional Hamiltonian class")
                self.results.map_function = self.ham.compute_coordinate_surfaces

        self.ode = Integrator(self.dt, self.eom, self.options)

        self.make_lindblad_operators()

    def make_lindblad_operators(self):
        """
        """
        nstates = self.ham.nstates

        # compute unique frequencies
        self.ham.compute_unique_freqs()

        hcorr = np.zeros((nstates,nstates),dtype=complex)
        self.Ls = []
        self.LdL = []
        self.L0 = []
        for k,bath in enumerate(self.ham.baths):
            Ga = self.ham.to_eigenbasis( bath.c_op )
            for i in range(len(self.ham.frequencies)):
                omega = self.ham.frequencies[i]
                cf_real = bath.ft_bath_corr(-omega).real
                cf_imag = bath.ft_bath_corr(-omega).imag
                proj = np.zeros((nstates,nstates))
                for j in range(nstates):
                    for k in range(nstates):
                        omega = self.ham.omegas[j,k]
                        if omega==self.ham.frequencies[i]:
                            proj[j,k] = 1.
                L = proj*Ga
                hcorr += (cf_imag*np.dot(dag(L),L))
                L *= np.sqrt(cf_real)
                self.Ls.append(L.copy())
                if self.ham.frequencies[i] == 0.:
                    self.L0.append(L.copy())
                else:
                    for j in range(nstates):
                        for k in range(nstates):
                            if L[j,k] != 0.:
                                ldl_list = [] # first list is for |L_mn|^2 the second is for [m,n]
                                ldl_list.append( np.conj(L[j,k])*L[j,k] ) # store |L_jk|^2
                                ldl_list.append([j,k]) # store jk
                                self.LdL.append(ldl_list)

        self.A  = (-1.j/const.hbar)*(self.ham.site2eig(self.ham.sys) + hcorr)

    def eom(self, state, order):
        return lb_deriv(state,self.A,self.L0,self.LdL)

    def solve(self, rho0, times, options=None, results=None):
        """
        """
        self.dt = times[1]-times[0]
        self.setup(options, results)
        rho = self.ham.to_eigenbasis(rho0.copy())

        self.ode._set_y_value(rho, times[0])
        for i,time in enumerate(times):
            if i%self.results.every==0:
                if self.options.progress: print(i)
                self.results.analyze_state(i, time, self.ham.from_eigenbasis(self.ode.y))
            self.ode.integrate()

        return self.results
