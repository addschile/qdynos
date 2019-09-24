from __future__ import print_function,absolute_import

import numpy as np
import scipy.sparse as sp
from time import time
from copy import deepcopy
from scipy.linalg import expm

import qdynos.constants as const

from .integrator import Integrator
from .dynamics import Dynamics
from .options import Options
from .results import Results
from .utils import dag,matmult,norm
from .linalg import krylov_prop
from .log import *

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
        print_method("Unitary DM")

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

    def eom(self, state, order):
        return self.equation_of_motion(state, order)

    def solve(self, rho0, times, options=None, results=None):
        """
        Solve Liouville-von Neumann equation for density matrix.
        """
        self.setup(options, results)
        self.dt = times[1]-times[0]
        tobs = len(times)
        rho = rho0.copy()

        if self.options.method == 'exact':
            for i in range(len(self.results.e_ops)):
                self.results.e_ops[i] = self.ham.to_eigenbasis(self.results.e_ops[i])
            rho = self.ham.to_eigenbasis(rho)
            self.prop = np.exp(-1.j*self.ham.omegas*self.dt)
            self.equation_of_motion = lambda x,y: self.prop*x
            ode = Integrator(self.dt, self.eom, self.options)
            ode._set_y_value(rho, times[0])
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
        else:
            self.equation_of_motion = (lambda x,y: (-1.j/const.hbar)*self.ham.commutator(x, eig=False))
            ode = Integrator(self.dt, self.eom, self.options)
            ode._set_y_value(rho, times[0])
            btime = time()
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

class UnitaryWF(Dynamics):
    """
    Class for unitary evolution of a wavefunction
    """

    def __init__(self, hamiltonian):
        super(UnitaryWF, self).__init__(hamiltonian)
        self.ham = hamiltonian
        print_method("Unitary WF")
        const.working_units()

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

    def eom_krylov(self, state, order):
        return krylov_prop(self.ham.ham, self.options.nlanczos, state, self.dt, self.options.method, lowmem=self.options.lanczos_lowmem)

    def eom(self, state, order):
        return matmult(self.prop, state)

    def solve(self, psi0, times, eig=True, options=None, results=None):
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
                psi = psi0
                self.prop = expm(-(1.j/const.hbar)*self.ham.ham*self.dt)
            ode = Integrator(self.dt, self.eom, self.options)
        elif self.options.method == 'lanczos' or self.options.method == 'arnoldi':
            # check if relevant matrices are csr matrices #
            # ham
            if not isinstance(self.ham.ham, sp.csr_matrix):
                self.ham.ham = sp.csr_matrix(self.ham.ham)
            # psi
            if not isinstance(psi0, sp.csr_matrix):
                psi = sp.csr_matrix(psi0)
            # expectation operators
            if self.results.e_ops != None:
                for i in range(len(self.results.e_ops)):
                    if not isinstance(self.results.e_ops[i], sp.csr_matrix):
                        self.results.e_ops[i] = sp.csr_matrix(self.results.e_ops[i])
            if self.options.method == 'lanczos' or self.options.method == 'arnoldi':
                ode = Integrator(self.dt, self.eom_krylov, self.options)
        elif self.options.method == 'rk4':
            print_basic("Running dynamics with RK4. This algorithm is far less stable than the others")
            print_basic("Try setting options.method to lanczos or arnoldi if full diagonalization is prohibited.")
            psi = psi0
            self.prop = -(1.j/const.hbar)*self.ham.ham
            ode = Integrator(self.dt, self.eom, self.options)
        elif self.options.method == 'euler':
            print_basic("Running dynamics with Forward Euler. This algorithm is far less stable than the others")
            print_basic("Try setting options.method to lanczos or arnoldi if full diagonalization is prohibited.")
            psi = psi0
            self.prop = -(1.j/const.hbar)*self.ham.ham
            ode = Integrator(self.dt, self.eom, self.options)
        else:
            # shouldn't ever get here because of options assert
            raise ValueError("Incorrect method specification")

        ode._set_y_value(psi, times[0])
        btime = time()
        print_stage("Propagating Equation of Motion")
        for i,tau in enumerate(times):
            if self.options.progress:
                if tobs>=10:
                    if i%int(tobs/10)==0:
                        etime = time()
                        print_progress((100*i/tobs),(etime-btime))
                else: 
                    print(i)
            if self.options.really_verbose:
                print(i)
            if i%self.results.every==0:
                self.results.analyze_state(i, tau, ode.y)
            ode.integrate()

        return self.results
