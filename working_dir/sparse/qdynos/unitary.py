from __future__ import print_function,absolute_import

import numpy as np
from scipy.linalg import expm
from time import time
from copy import deepcopy

import qdynos.constants as const

from .integrator import Integrator
from .dynamics import Dynamics
from .options import Options
from .results import Results
from .log import print_method,print_stage,print_progress,print_time

class UnitaryWF(Dynamics):
    """
    Class for unitary evolution of a wavefunction
    """

    def __init__(self, hamiltonian):
        super(UnitaryWF, self).__init__(hamiltonian)
        self.ham = hamiltonian
        print_method("Unitary WF")

    def setup(self, options, results):
        """
        Sets up options class and results class for dynamics.
        """
        if options==None:
            self.options = Options(method='exact')
        else:
            self.options = options
        if results==None:
            self.results = Results()
        else:
            self.results = results
            if self.results.map_ops:
                assert(str(type(self.ham))=="<class 'qdynos.hamiltonian.MDHamiltonian'>")
                self.results.map_function = self.ham.compute_coordinate_surfaces

    def eom(self, state, order):
        return matmult(self.prop, state)

    def eom_lanczos(self, state, order):
        # TODO
        return matmult(self.prop, state)

    def solve(self, psi0, times, eig=False, options=None, results=None):
        """
        Solve time-dependent Schrodinger equation for density matrix.
        """
        self.setup(options,results)
        self.dt = times[1]-times[0]
        tobs = len(times)
        psi = psi0.copy()

        if self.options.method == 'exact':
            if eig:
                self.ham.eigensystem()
                for i in range(len(self.results.e_ops)):
                    self.results.e_ops[i] = self.ham.to_eigenbasis(self.results.e_ops[i])
                psi = self.ham.to_eigenbasis(psi)
                self.prop = np.diag(np.exp(-(1.j/const.hbar)*self.ham.ev*self.dt))
            else:
                self.prop = expm(-(1.j/const.hbar)*self.ham.ham*dt)
        elif self.options.method == 'lanczos':
            ode = Integrator(self.dt, self.eom_lanczos, self.options)
        elif self.options.method == 'rk4':
            self.prop = -(1.j/const.hbar)*self.ham.ham
            ode = Integrator(self.dt, self.eom, self.options)
            ode._set_y_value(psi, times[0])

        ode = Integrator(self.dt, self.eom, self.options)
        ode._set_y_value(psi, times[0])
        btime = time()
        print_stage("Propagating Equation of Motion")
        for i,tau in enumerate(times):
            if self.options.progress:
                if i%int(tobs/10)==0:
                    etime = time()
                    print_progress((100*i/tobs),(etime-btime))
                elif self.options.really_verbose: print(i)
            if i%self.results.every==0:
                self.results.analyze_state(i, tau, ode.y)
            ode.integrate()

        return self.results
