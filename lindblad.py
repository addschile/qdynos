from __future__ import print_function,absolute_import

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from time import time
from scipy.linalg import expm
from copy import deepcopy

import qdynos.constants as const

from .integrator import Integrator
from .dynamics import Dynamics
from .utils import dag,commutator,anticommutator,inner,matmult,norm,is_vector,is_matrix
from .options import Options
from .results import Results,add_results,avg_results
from .linalg import propagate,krylov_prop
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

    def setup(self, gam, L, options, results, eig=False, sparse=False):
        """
        """
        # options setup
        if options==None:
            self.options = Options()
        else:
            self.options = options
            if not self.options.unraveling:
                if self.options.method == "exact":
                    raise NotImplementedError

        # results setup
        if results==None:
            self.results = Results()
        else:
            self.results = results

        # simulation technique setup
        if self.options.unraveling:
            if self.options.which_unraveling == 'jump':
                print_method("Lindblad w/ Jumps")
        else:
            print_method("Lindblad QME")
            self.ode = Integrator(self.dt, self.eom, self.options)

        # Lindblad operators setup
        if isinstance(L, list):
            self.gam_re = gam.real
            self.gam_im = gam.imag
            self.L = L.copy()
        else:
            self.gam_re = np.array([gam.real])
            self.gam_im = np.array([gam.imag])
            self.L = [L.copy()]

        # make operators sparse if need be
        if sparse:
            for i in range(len(self.L)):
                if not isinstance(self.L[i], sp.csr_matrix):
                    self.L[i] = sp.csr_matrix(self.L[i])

        self.precompute_operators(eig, sparse)

    def precompute_operators(self, eig, sparse):
        """
        """
        nstates = self.ham.nstates
        if self.options.method == 'arnoldi':
            lamb = sp.csr_matrix((nstates,nstates))
            eig = False
            # check if relevant matrices are csr matrices #
            # ham
            if not isinstance(self.ham.ham, sp.csr_matrix):
                self.ham.ham = sp.csr_matrix(self.ham.ham)
            # expectation operators
            if self.results.e_ops != None:
                for i in range(len(self.results.e_ops)):
                    if not isinstance(self.results.e_ops[i], sp.csr_matrix):
                        self.results.e_ops[i] = sp.csr_matrix(self.results.e_ops[i])
        else:
            if sparse:
                lamb = sp.csr_matrix(np.zeros((nstates,nstates),dtype=complex))
            else:
                lamb = np.zeros((nstates,nstates),dtype=complex)
        if eig:
            self.ham.eigensystem()
        self.LdL = []
        for i in range(len(self.L)):
            # transform lindblad operators into eigenbasis
            if eig:
                self.L[i] = self.ham.to_eigenbasis(self.L[i])
            if self.options.method == 'arnoldi':
                if not isinstance(self.L[i], sp.csr_matrix):
                    self.L[i] = sp.csr_matrix(self.L[i])
            # make list of L^\dagger L for faster computations
            self.LdL.append( matmult(dag(self.L[i]),self.L[i]) )
            self.L[i] *= np.sqrt(self.gam_re[i])
            # compute lamb 
            lamb += self.gam_im[i]*self.LdL[i]
            self.LdL[i] *= self.gam_re[i]
        if eig:
            if sparse:
                self.A = sp.csr_matrix(self.ham.Heig) + lamb
            else:
                self.A = self.ham.Heig + lamb
        else:
            if sparse:
                self.A = sp.csr_matrix(self.ham.ham) + lamb
            else:
                self.A = self.ham.ham + lamb

        if self.options.unraveling:
            # add non-hermitian term
            for i in range(len(self.L)):
                self.A -= 0.5j*self.LdL[i]*const.hbar

        if not (self.options.method == 'arnoldi' or self.options.method == 'lanczos'):
            self.A *= -1.j/const.hbar

    def make_propagator(self, dt):
        if self.options.method == 'exact':
            self.expmA = expm(self.A*dt)
        elif self.options.method == 'arnoldi':
            if self.jumping:
                self.dt_jump = dt

    # TODO make time-dependent version
    #def eom_td(self, state, order):
    #    drho = (-1.j/const.hbar)*self.ham.commutator(state)
    #    for i in range(len(self.L)):
    #               # TODO make gam a function to account for time-dependence
    #        rho += self.gam[i]*(np.dot(L, np.dot(rho, dag(L))) - 0.5*anticommutator(np.dot(dag(L),L),rho))
    #    return drho

    def jump_probs(self, psi):
        p_n = np.zeros(len(self.L))
        for i in range(len(self.LdL)):
            p_n[i] = inner(psi, matmult(self.LdL[i], psi)).real
        p = np.sum(p_n)
        return p_n , p

    def jump(self, rand, psi):
    
        # compute jump probabilities
        p_n , p = self.jump_probs(psi)
    
        # see which one it jumped along
        if isinstance(psi, np.ndarray):
            psi_out = np.zeros_like(psi)
        else:
            psi_out = sp.csr_matrix(psi.shape, dtype=complex)
        p *= rand
        for count in range(len(self.L)):
            if p <= np.sum(p_n[:count+1]):
                psi_out = matmult(self.L[count], psi)
                return psi_out/np.sqrt(p_n[count]) , count
    
        assert(count<len(self.L))

    def eom_jump_arnoldi(self, state, order):
        """
        """
        if self.jumping:
            # searching for jump time, so only need to propagate with different dt
            return propagate(self.V, self.T, self.dt_jump)
        elif self.just_jumped:
            # just jumped so need to reform the krylov subspace
            # TODO do I need to return all the stuff
            state , self.T , self.V = krylov_prop(self.A, self.options.nlanczos, state, self.dt_jump, self.options.method, return_all=True)
            return state
        else:
            # routine propagation
            state , self.T , self.V = krylov_prop(self.A, self.options.nlanczos, state, self.dt, self.options.method, return_all=True)
            return state

    def eom_jump(self, state, order):
        """
        """
        return matmult(self.expmA, state)

    def eom(self, state, order):
        """
        """
        drho = commutator(self.A,state)
        for i in range(len(self.L)):
            drho += (np.dot(self.L[i], np.dot(state, dag(self.L[i]))) - 0.5*anticommutator(self.LdL[i], state))
        return drho

    def integrate_trajectories(self, psi0, times, ntraj):
        """
        """

        if self.options.seed==None:
            seeder = int(time())
            rng = np.random.RandomState(seed=seeder)
        else:
            seeder = self.options.seed
            rng = np.random.RandomState(seed=seeder)
            
        # make initial propagator
        self.just_jumped = 0
        self.jumping = 0
        self.make_propagator(self.dt)
        times = np.append(times,times[-1]+self.dt)
    
        btime = time()
        for i in range(ntraj):
            if self.options.progress:
                if ntraj >= 10:
                    if i%int(ntraj/10)==0:
                        etime = time()
                        print_progress((100*i/ntraj),(etime-btime))
                else:
                    print_basic(i)
            njumps = 0
            jumps = []
    
            # initialize time and a random number
            t = times[0]
            t_prev = times[0]
            rand = rng.uniform()
    
            # set initial value of the integrator
            if self.options.method == 'exact':
                self.ode = Integrator(self.dt, self.eom_jump, self.options)
            elif self.options.method == 'lanczos' or self.options.method == 'arnoldi' :
                self.ode = Integrator(self.dt, self.eom_jump_arnoldi, self.options)
                if isinstance(psi0, np.ndarray):
                    psi0 = sp.csr_matrix(psi0)
            self.ode._set_y_value(psi0.copy(), t)
            psi_track = psi0.copy()
    
            # set up results
            results_traj = deepcopy(self.results)
            # printing results file
            if results_traj.print_es:
                results_traj.es_file += "_traj_%d"%(i)
            # printing states file
            if results_traj.print_states:
                results_traj.states_file += "_traj_%d"%(i)
    
            for j in range(len(times)-1):

                # for each time do results stuff
                results_traj.analyze_state(j, times[j], psi_track)

                self.just_jumped = 0
                while self.ode.t != times[j+1]:
    
                    # data before integrating
                    t_prev = self.ode.t
                    psi_prev = self.ode.y.copy()
                    norm_prev = norm(self.ode.y)
    
                    # integrate without renormalization
                    self.ode.integrate(change_dt=0)
    
                    # data after integrating
                    norm_psi = norm(self.ode.y)
                    t_next = times[j+1]
    
                    if norm_psi <= rand:
                        self.just_jumped = 1
                        self.jumping = 1
    
                        # quantum jump has happened
                        njumps += 1
    
                        ii = 0
                        t_final = t_next
    
                        while ii < self.options.jump_time_steps:
    
                            ii += 1
    
                            # make a guess for when the jump occurs
                            if self.options.method == 'arnoldi' or self.options.method == 'lanczos':
                                t_guess = t_prev + 0.5*(t_final-t_prev)
                                # make propagator up to t_guess
                                self.make_propagator(t_guess-self.ode.t)
                            else:
                                t_guess = t_prev + np.log(norm_prev / rand) / \
                                    np.log(norm_prev / norm_psi)*(t_final-t_prev) 
                                # make propagator up to t_guess
                                self.make_propagator(t_guess-t_prev)
    
                            # integrate psi from t_prev to t_guess
                            norm_prev = norm(psi_prev)
                            self.ode.y = psi_prev.copy()
                            self.ode.integrate(change_dt=0)
                            norm_guess = norm(self.ode.y)
    
                            # determine what to do next
                            if (np.abs(norm_guess - rand) <= (self.options.jump_time_tol*rand)):
                                # t_guess was right!
                                self.ode.t = t_guess
    
                                # jump
                                self.ode.y /= np.sqrt(norm_guess)
                                rand = rng.uniform()
                                self.ode.y , ind = self.jump(rand, self.ode.y)
                                jumps.append( [t,ind] )

                                # make propagator up to t_next
                                self.make_propagator(t_next-t_guess)

                                # need to reform krylov subspace
                                self.jumping = 0

                                # choose a new random number for next jump
                                rand = rng.uniform()
                                break
                            elif (norm_guess < rand):
                                # t_guess > t_jump
                                t_final = t_guess
                                norm_psi = norm_guess
                            else:
                                # t_guess < t_jump
                                t_prev = t_guess
                                psi_prev = self.ode.y.copy()
                                norm_prev = norm_guess
                            if ii == self.options.jump_time_steps:
                                raise ValueError("Couldn't find jump time")
                    else:
                        # no jump update time
                        self.ode.t = times[j+1]
                        if self.just_jumped:
                            self.make_propagator(self.dt)
                            self.just_jumped = 0

                # store new normalized wavefunction for this timestep
                psi_track = self.ode.y.copy()
                psi_track /= np.sqrt(norm(psi_track))

            if self.results.jump_stats:
                results_traj.store_jumps(njumps, jumps)
            add_results(self.results, results_traj)

        return avg_results(ntraj, self.results)

    def solve(self, rho0, times, gam, L, ntraj=1000, eig=False, sparse=False, options=None, results=None):
        """
        """
        self.dt = times[1]-times[0]
        self.tobs = len(times)
        self.setup(gam, L, options, results, eig=eig, sparse=sparse)
        if eig:
            if self.results.e_ops != None:
                for i in range(len(self.results.e_ops)):
                    self.results.e_ops[i] = self.ham.to_eigenbasis(self.results.e_ops[i])
            rho0 = self.ham.to_eigenbasis(rho0)

        if self.options.unraveling:
            if is_vector(rho0):
                psi0 = rho0.copy()
            else:
                raise AttributeError("Initial condition must be a wavefunction")
            if sparse:
                if not isinstance(psi0, sp.csr_matrix):
                    psi0 = sp.csr_matrix(psi0)
            return self.integrate_trajectories(psi0, times, ntraj)
        else:
            if is_matrix(rho0):
                rho = rho0.copy()
                self.ode._set_y_value(rho, times[0])
            else:
                print_warning("Converting wavefunction to density matrix.")
                rho = matmult(rho0,dag(rho0))
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

# TODO
#def make_lindblad_ops(ham):
#    """
#    """
#    nstates = ham.nstates
#    for k,bath in enumerate(ham.baths):
#        Ga = ham.to_eigenbasis( bath.c_op )
#        theta_zero = bath.ft_bath_corr(0.0)
#        theta_plus = theta_zero*np.identity(nstates,dtype=complex)
#        for i in range(nstates):
#            for j in range(nstates):
#                if i!=j:
#                    theta_plus[i,j] = bath.ft_bath_corr(-ham.omegas[i,j])
#        if self.options.print_coup_ops:
#            np.save(self.options.coup_ops_file+"c_op_%d"%(k),Ga)
#            np.save(self.options.coup_ops_file+"theta_plus_%d"%(k),theta_plus)
#        # population transfer matrix
#        prop += 2.*np.einsum('ji,ij,ij->ij',Ga,Ga,theta_plus.real)/const.hbar**2.
#        # dephasing matrix
#        self.Rdep += np.einsum('jj,ii,ii->ij',Ga,Ga,theta_plus)
#        self.Rdep += np.einsum('jj,ii,jj->ij',Ga,Ga,theta_plus.conj().T)
#        same_ik = np.einsum('im,mi,mi->i',Ga,Ga,theta_plus)
#        same_lj = np.einsum('im,mi,im->i',Ga,Ga,theta_plus.conj().T)
#        for i in range(nstates):
#            self.Rdep[i,:] -= same_ik[i]
#            self.Rdep[:,i] -= same_lj[i]
