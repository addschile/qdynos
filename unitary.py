from __future__ import print_function,absolute_import

import numpy as np
import qdynos.constants as const

from .integrator import Integrator
from .dynamics import Dynamics
from .options import Options
from .results import Results

from copy import deepcopy

class UnitaryDM(Dynamics):
    """
    Class for unitary evolution of a density matrix.
    """

    def __init__(self, hamiltonian):
        """
        Parameters
        ----------
        hamiltonian : Hamiltonian
        """
        super(UnitaryDM, self).__init__(hamiltonian)
        self.ham = hamiltonian

    def __str__(self):
        s = ""
        s += "Hamiltonian\n"
        s += "-----------\n"
        s += str(self.ham.ham)
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

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

    def eom(self, state):
        return self.equation_of_motion(state)

    def solve(self, rho0, times, options=None, results=None):
        """
        Solve Liouville-von Neumann equation for density matrix.
        """
        self.setup(options, results)
        self.dt = times[1]-times[0]
        rho = rho0.copy()

        if self.options.method == 'exact':
            for i in range(len(self.results.e_ops)):
                self.results.e_ops[i] = self.ham.to_eigenbasis(self.results.e_ops[i])
            rho = self.ham.to_eigenbasis(rho)
            self.prop = np.exp(-1.j*self.ham.omegas*self.dt)
            self.equation_of_motion = lambda x: self.prop*x
            ode = Integrator(self.dt, self.eom, self.options)
            ode._set_y_value(rho, times[0])
            for i,time in enumerate(times):
                if i%self.results.every==0:
                    self.results.analyze_state(i, time, ode.y)
                ode.integrate()
        else:
            self.equation_of_motion = lambda x: (-1.j/const.hbar)*self.ham.commutator(x)
            ode = Integrator(self.dt, self.eom, self.options)
            ode._set_y_value(rho, times[0])
            for i,time in enumerate(times):
                if i%self.results.every==0:
                    self.results.analyze_state(i, time, ode.y)
                ode.integrate()

        return self.results

class UnitaryWF(Dynamics):
    """
    Class for unitary evolution of a wavefunction
    """

    def __init__(self, hamiltonian):
        super(UnitaryWF, self).__init__(hamiltonian)
        self.ham = hamiltonian

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

    def eom(self, state, ind):
        return np.dot(self.prop,state)

    def solve(self, psi0, times, options=None, results=None):
        """
        Solve time-dependent Schrodinger equation for density matrix.
        """
        self.setup(options,results)
        self.dt = times[1]-times[0]
        psi = psi0.copy()

        if self.options.method == 'exact':
            for i in range(len(self.results.e_ops)):
                self.results.e_ops[i] = self.ham.to_eigenbasis(self.results.e_ops[i])
            psi = self.ham.to_eigenbasis(psi)
            self.prop = np.diag(np.exp(-(1.j/const.hbar)*self.ham.ev*self.dt))
            ode = Integrator(self.dt, self.eom, self.options)
            ode._set_y_value(psi, times[0])
            for i,time in enumerate(times):
                if i%self.results.every==0:
                    print(i)
                    self.results.analyze_state(i, time, ode.y)
                ode.integrate()
        else:
            self.prop = -(1.j/const.hbar)*self.ham.ham
            ode = Integrator(self.dt, self.eom, self.options)
            ode._set_y_value(psi, times[0])

            for i,time in enumerate(times):
                if i%self.results.every==0:
                    print(i)
                    self.results.analyze_state(i, time, ode.y)
                ode.integrate()

        return self.results
