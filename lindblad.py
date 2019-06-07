from __future__ import print_function,absolute_import

import numpy as np
from time import time

import qdynos.constants as const

from .integrator import Integrator
from .dynamics import Dynamics
from .utils import commutator,anticommutator,dag
from .options import Options
from .results import Results
from .log import *

class Lindblad(Dynamics):
    """
    """

    def __init__(self, ham, time_dependent=False):
        """
        """
        super(Lindblad, self).__init__(ham)
        self.ham = ham
        self.time_dep = time_dependent

    def setup(self, gam, L, options, results):
        """
        """
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

        self.ode = Integrator(self.dt, self.eom, self.options)
        if isinstance(L, list):
            self.gam_re = gam.real
            self.gam_im = gam.imag
            self.L = L.copy()
        else:
            self.gam_re = np.array([gam.real])
            self.gam_im = np.array([gam.imag])
            self.L = [L.copy()]
        self.compute_lamb_shift()

    def compute_lamb_shift(self):
        """
        """
        nstates = self.ham.nstates
        # TODO this complicates things with the Hamiltonian class
        # TODO potential fix: define Hamiltonian class inside of this class and only init with a matrix hamiltonian
        #lamb = np.zeros((nstates,nstates),dtype=complex)
        for i in range(len(self.L)):
            self.L[i] = self.ham.to_eigenbasis(self.L[i])
        #    lamb += self.gam_im[i]*np.dot(dag(self.L[i]),self.L[i])

    #def eom(self, state, order):
    #    drho = (-1.j/const.hbar)*self.ham.commutator(state)
    #    for i in range(len(self.L)):
    #               # TODO make gam a function to account for time-dependence
    #        rho += self.gam[i]*(np.dot(L, np.dot(rho, dag(L))) - 0.5*anticommutator(np.dot(dag(L),L),rho))
    #    return drho

    def eom(self, state, order):
        drho = (-1.j/const.hbar)*self.ham.commutator(state)
        for i in range(len(self.L)):
            drho += self.gam_re[i]*(np.dot(self.L[i], np.dot(state, dag(self.L[i]))) - 0.5*anticommutator(np.dot(dag(self.L[i]),self.L[i]), state))
        return drho

    def solve(self, rho0, times, gam, L, options=None, results=None):
        """
        """
        self.dt = times[1]-times[0]
        self.tobs = len(times)
        self.setup(gam, L, options, results)
        rho = self.ham.to_eigenbasis(rho0.copy())
        if self.results.e_ops != None:
            for i in range(len(self.results.e_ops)):
                self.results.e_ops[i] = self.ham.to_eigenbasis(self.results.e_ops[i])

        self.ode._set_y_value(rho, times[0])
        btime = time()
        for i,tau in enumerate(times):
            if self.options.progress:
                if i%int(self.tobs/10)==0:
                    etime = time()
                    print_progress((100*i/self.tobs),(etime-btime))
            if i%self.results.every==0:
                self.results.analyze_state(i, tau, self.ode.y)
            self.ode.integrate()

        return self.results
