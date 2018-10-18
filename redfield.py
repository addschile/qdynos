from __future__ import print_function,absolute_import

import numpy as np
from time import time

import qdynos.constants as const

from .integrator import Integrator
from .dynamics import Dynamics
from .utils import commutator,dag,to_liouville
from .options import Options
from .results import Results
from .log import print_method,print_stage,print_progress,print_time

class Redfield(Dynamics):
    """Dynamics class for Redfield-like dynamics. Can perform both 
    time-dependent (TCL2) and time-independent dynamics with and without the 
    secular approximation.
    """

    def __init__(self, ham, time_dependent=False, is_secular=False, options=None):
        """Instantiates the Redfield class.

        Parameters
        ----------
        ham : Hamiltonian or MDHamiltonian class
        time_dependent : bool
        is_secular : bool
        """
        super(Redfield, self).__init__(ham)
        self.ham = ham
        self.time_dep = time_dependent
        self.is_secular = is_secular
        if self.time_dep:
            if self.is_secular:
                print_method("Secular TCL2")
            else:
                print_method("TCL2")
        else:
            if self.is_secular:
                print_method("Secular Redfield Theory")
            else:
                print_method("Redfield Theory")
        if options==None:
            self.options = Options()
        else:
            self.options = options
            if self.options.method == "exact":
                raise NotImplementedError

    def setup(self, times, results):
        self.dt = times[1]-times[0]
        self.tobs = len(times)
        if results==None:
            self.results = Results()
        else:
            self.results = results
            if self.results.map_ops:
                assert(repr(self.ham)=="Multidimensional Hamiltonian class")
                self.results.map_function = self.ham.compute_coordinate_surfaces

        self.ode = Integrator(self.dt, self.eom, self.options)
        if self.options.method != "exact":
            if self.time_dep:
                self.equation_of_motion = self.td_rf_eom
            else:
                self.equation_of_motion = self.rf_eom

    def make_redfield_operators(self):
        """Make and store the coupling operators and "dressed" copuling operators.
        """
        nstates = self.ham.nstates
        self.C = list()
        self.E = list()
        for k,bath in enumerate(self.ham.baths):
            if self.options.really_verbose: print("operator %d of %d"%(k+1,len(self.ham.baths)))
            Ga = self.ham.to_eigenbasis( bath.c_op )
            theta_zero = bath.ft_bath_corr(0.0)
            theta_plus = theta_zero*np.identity(nstates,dtype=complex)
            for i in range(nstates):
                if self.options.really_verbose: print("%d rows of %d"%(i,nstates))
                for j in range(nstates):
                    if i!=j:
                        theta_plus[i,j] = bath.ft_bath_corr(-self.ham.omegas[i,j])
            Ga_plus = Ga*theta_plus
            self.C.append(Ga.copy())
            self.E.append(Ga_plus.copy())

    def coupling_operators_setup(self):
        """Make coupling operators and initialize "dressing" for copuling 
        operators.
        """
        self.C = []
        self.gamma_n = [[]]*self.ham.nbaths
        self.gamma_n_1 = [[]]*self.ham.nbaths
        nstates = self.ham.nstates
        b = self.ode.b

        for op,bath in enumerate(self.ham.baths):
            self.C.append( self.ham.to_eigenbasis( bath.c_op ) )
            self.gamma_n[op] = list()
            self.gamma_n_1[op] = list()

        for op,bath in enumerate(self.ham.baths):
            for k in range(self.ode.order):
                t = b[k]*self.dt
                if k==0:
                    self.gamma_n[op].append( np.zeros((nstates,nstates),dtype=complex) )
                    theta_plus = np.exp(-1.j*self.ham.omegas*0.0)*bath.bath_corr_t(0.0)
                    self.gamma_n_1[op].append(theta_plus.copy())
                else:
                    theta_plus = np.exp(-1.j*self.ham.omegas*t)*bath.bath_corr_t(t)
                    self.gamma_n[op].append( self.gamma_n[op][k-1] + 0.5*(b[k]-b[k-1])*self.dt*(theta_plus + self.gamma_n_1[op][k-1]) )
                    self.gamma_n_1[op].append( theta_plus.copy() )

    def make_tcl2_operators(self, time):
        """Integrate "dressing" for copuling operators. Uses trapezoid rule 
        with grid of integration method (e.g., Runge-Kutta 4).
        """
        b = self.ode.b
        for op,bath in enumerate(self.ham.baths):
            for k in range(self.ode.order):
                t = time + b[k]*self.dt
                theta_plus = np.exp(-1.j*self.ham.omegas*t)*bath.bath_corr_t(t)
                if k==0:
                    self.gamma_n[op][k] = self.gamma_n[op][-1].copy()
                    self.gamma_n_1[op][k] = theta_plus.copy()
                else:
                    self.gamma_n[op][k] = self.gamma_n[op][k-1] + 0.5*self.dt*(b[k]-b[k-1])*(theta_plus + self.gamma_n_1[op][k-1])
                    self.gamma_n_1[op][k] = theta_plus.copy()

    def update_ops(self, time):
        """Update the dressed coupling operators by integrating Fourier-Laplace
        transform of bath correlation function in time.
        """
        self.E = [[]]*self.ham.nbaths
        for i in range(len(self.C)):
            self.E[i] = list()
            for j in range(self.ode.order):
                self.E[i].append(self.gamma_n[i][j]*self.C[i])
        if time < self.options.markov_time:
            self.make_tcl2_operators(time)

    def eom(self, state, order):
        return self.equation_of_motion(state, order)

    def rf_eom(self, state, order):
        dy = (-1.j/const.hbar)*self.ham.commutator(state)
        for j in range(len(self.ham.baths)):
            dy += (commutator(self.E[j]@state,self.C[j]) + commutator(self.C[j],state@dag(self.E[j])))/const.hbar**2.
        return dy

    def td_rf_eom(self, state, order):
        dy = (-1.j/const.hbar)*self.ham.commutator(state)
        for j in range(len(self.ham.baths)):
            dy += (commutator(self.E[j][order]@state,self.C[j]) + commutator(self.C[j],state@dag(self.E[j][order])))/const.hbar**2.
        return dy

    def propagate_eom(self, rho, times):

        if self.results.e_ops != None:
            for i in range(len(self.results.e_ops)):
                self.results.e_ops[i] = self.ham.to_eigenbasis(self.results.e_ops[i])

        if self.options.method == 'exact':
            raise NotImplementedError
        else:
            self.ode._set_y_value(rho, times[0])
            btime = time()
            for i,tau in enumerate(times):
                if self.options.progress:
                    if i%int(self.tobs/10)==0:
                        etime = time()
                        print_progress((100*i/self.tobs),(etime-btime))
                    elif self.options.really_verbose: print(i)
                if self.time_dep:
                    self.update_ops(tau)
                if i%self.results.every==0:
                    self.results.analyze_state(i, tau, self.ode.y)
                self.ode.integrate()

        return self.results

    def solve(self, rho0, times, results=None):
        """Solve the Redfield equations of motion.

        Parameters
        ----------
        rho_0 : np.array
        times : np.array
        options : Options class
        results : Results class

        Returns
        -------
        results : Results class
        """
        self.setup(times, results)
        rho = self.ham.to_eigenbasis(rho0.copy())

        if self.options.method != "exact":
            if self.options.verbose:
                print_stage("Initializing Coupling Operators")
                btime = time()
            if self.time_dep:
                self.coupling_operators_setup()
            else:
                self.make_redfield_operators()
            if self.options.verbose:
                etime = time()
                print_stage("Finished Constructing Operators")
                print_time(etime-btime)
                print_stage("Propagating Equation of Motion")

        return self.propagate_eom(rho, times)
