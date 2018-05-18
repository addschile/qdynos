import numpy as np
from .integrator import Integrator
from .dynamics import Dynamics
from .options import Options
from .results import Results
import .constants as const

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
        super().__init__(hamiltonian)
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

    def eom(self, state):
        return self.equation_of_motion(state)

    def dmsolve(self, rho_0, t_init, t_final, dt, options=None, results=None):
        """
        Solve Liouville-von Neumann equation for density matrix.
        """
        self.setup(options, results)
        times = np.arange(t_init, t_final, dt)
        rho = rho_0.copy()

        if self.options.method == 'exact':
            rho = self.ham.to_eigenbasis(rho)
            self.prop = np.exp(-1.j*self.ham.omegas*dt)
            self.equation_of_motion = lambda x: self.prop*x
            ode = Integrator(dt, self.eom, self.options)
            ode._set_y_value(rho, times[0])
            for time in times:
                if i%self.results.every==0:
                    self.results.analyze_state(i, time, self.ham.from_eigenbasis(ode.y))
                ode.integrate()
        else:
            self.equation_of_motion = lambda x: (-1.j/const.hbar)*self.ham.commutator(x)
            ode = Integrator(dt, self.eom, self.options)
            ode._set_y_value(rho, times[0])
            for time in times:
                if i%self.results.every==0:
                    self.results.analyze_state(i, time, ode.y)
                ode.integrate()

        return times , self.results

class UnitaryWF(Dynamics):
    """
    Class for unitary evolution of a wavefunction
    """

    def __init__(self, hamiltonian):
        """Initialize the Unitary evolution class. 

        Parameters
        ----------
        hamiltonian : HamiltonianSystem
            An instance of the pyrho HamiltonianSystem class.
        """
        super().__init__(hamiltonian)
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

    def eom(self, state):
        return np.dot(self.prop,state)

    def wfsolve(self, psi_0, t_init, t_final, dt, options=None, results=None):
        """
        Solve time-dependent Schrodinger equation for density matrix.
        """
        self.setup(options,results)
        times = np.arange(t_init, t_final, dt)
        psi = psi_0.copy()

        if self.options.verbose:
            print("")
        if self.options.method == 'exact':
            psi = self.ham.to_eigenbasis(psi)
            self.prop = np.diag(np.exp(-(1.j/const.hbar)*self.ham.ev*dt))
            ode = Integrator(dt, self.eom, self.options)
            ode._set_y_value(psi, times[0])
            for i,time in enumerate(times):
                if i%self.results.every==0:
                    self.results.analyze_state(i, time, self.ham.from_eigenbasis(ode.y))
                ode.integrate()
        else:
            self.prop = -(1.j/const.hbar)*self.ham.ham
            ode = Integrator(dt, self.eom, self.options)
            ode._set_y_value(psi, times[0])

            for i,time in enumerate(times):
                if i%self.results.every==0:
                    self.results.analyze_state(i, time, ode.y)
                ode.integrate()

        return times , self.results
