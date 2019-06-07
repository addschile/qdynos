from __future__ import print_function,absolute_import

import numpy as np
from time import time

import qdynos.constants as const

from .integrator import Integrator
from .dynamics import Dynamics
from .utils import commutator,anticommutator,dag,norm,is_vector,is_matrix
from .options import Options
from .results import Results,add_results,avg_results
from .log import *

from scipy.linalg import expm
from copy import deepcopy

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
                self.options.method = 'exact'
                #self.ode = Integrator(self.dt, self.eom_jump, self.options)
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
        self.precompute_operators()

    def precompute_operators(self):
        """
        """
        nstates = self.ham.nstates
        lamb = np.zeros((nstates,nstates),dtype=complex)
        self.LdL = []
        for i in range(len(self.L)):
            # transform lindblad operators into eigenbasis
            self.L[i] = self.ham.to_eigenbasis(self.L[i])
            # make list of L^\dagger L for faster computations
            self.LdL.append( np.dot(dag(self.L[i]),self.L[i]) )
            self.L[i] *= np.sqrt(self.gam_re[i])
            # compute lamb 
            lamb += self.gam_im[i]*self.LdL[i]
            self.LdL[i] *= self.gam_re[i]
        self.A = self.ham.Heig + lamb

        if self.options.unraveling:
            # add non-hermitian term
            for i in range(len(self.L)):
                self.A -= 0.5j*self.LdL[i]

        self.A *= -1.j/const.hbar

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
            p_n[i] = np.dot(dag(psi), np.dot(self.LdL[i], psi))[0,0].real
        p = np.sum(p_n)
        return p_n , p

    def jump(self, rand, psi):
    
        # compute jump probabilities
        p_n , p = self.jump_probs(psi)
    
        # see which one it jumped along
        psi_out = np.zeros_like(psi)
        p *= rand
        for count in range(len(self.L)):
            if p <= np.sum(p_n[:count+1]):
                psi_out = np.dot(self.L[count], psi)
                return psi_out/np.sqrt(p_n[count]) , count
    
        assert(count<len(self.L))

    def eom_jump(self, state, order):
        return np.dot(self.expmA, state)

    def eom(self, state, order):
        #drho = (-1.j/const.hbar)*self.ham.commutator(state)
        #drho = (-1.j/const.hbar)*commutator(self.A,state)
        drho = commutator(self.A,state)
        for i in range(len(self.L)):
            #drho += self.gam_re[i]*(np.dot(self.L[i], np.dot(state, dag(self.L[i]))) - 0.5*anticommutator(self.LdL[i], state))
            drho += (np.dot(self.L[i], np.dot(state, dag(self.L[i]))) - 0.5*anticommutator(self.LdL[i], state))
        return drho

    def integrate_trajectories(self, psi0, times, ntraj):

        if self.options.seed==None:
            seeder = int(time())
            np.random.seed( seeder )
        else:
            seeder = self.options.seed + int(time())
            np.random.seed( seeder )
            
        # initialize integrator class
        self.expmA = expm(self.A*self.dt)
    
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
            rand = np.random.uniform()
    
            # set initial value of the integrator
            self.ode = Integrator(self.dt, self.eom_jump, self.options)
            self.ode._set_y_value(psi0.copy(), t)
            psi_track = psi0.copy()
    
            # set up results
            results_traj = deepcopy(self.results)
    
            count = 0
            while self.ode.t <= times[-1]:
    
                # do stuff for results
                results_traj.analyze_state(count, self.ode.t, psi_track)
                count += 1
    
                # data before integrating
                t_prev = self.ode.t
                psi_prev = self.ode.y.copy()
                norm_prev = norm(self.ode.y)
    
                # integrate without renormalization
                self.ode.integrate()
    
                # data after integrating
                norm_psi = norm(self.ode.y)
                t_next = self.ode.t
    
                if norm_psi <= rand:
    
                    # quantum jump has happened
                    njumps += 1
    
                    ii = 0
                    t_final = t_next
    
                    while ii < self.options.norm_steps:
    
                        ii += 1
    
                        # make a guess for when the jump occurs
                        t_guess = t_prev + np.log(norm_prev / rand) / \
                            np.log(norm_prev / norm_psi)*(t_final-t_prev) 
    
                        self.expmA = expm(self.A*(t_guess-t_prev))
    
                        # integrate psi from t_prev to t_guess
                        norm_prev = norm(psi_prev)
                        self.ode.y = psi_prev.copy()
                        self.ode.integrate(change_dt=0)
                        norm_guess = norm(self.ode.y)
    
                        # determine what to do next
                        if (np.abs(norm_guess - rand) <= (self.options.norm_tol*rand)):
                            # t_guess was right!
                            t = t_guess
    
                            # jump
                            self.ode.y /= np.sqrt(norm_guess)
                            rand = np.random.uniform()
                            self.ode.y , ind = self.jump(rand, self.ode.y)
                            jumps.append( [t,ind] )

                            # integrate up to t_next
                            self.expmA = expm(self.A*(t_next-t))
                            self.ode.integrate(renorm=1, change_dt=0)

                            # choose a new random number for next jump
                            rand = np.random.uniform()

                            # unitary trasnformation 
                            psi_track = self.ode.y.copy()

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
                        if ii == self.options.norm_steps:
                            raise ValueError("Couldn't find jump time")
                else:
                    # no jump occurred
                    psi_track = self.ode.y.copy()
                    # renormalize the tracking wavefunction
                    psi_track /= np.sqrt(norm(psi_track))

            if self.results.jump_stats:
                results_traj.store_jumps(njumps, jumps)
            add_results(self.results, results_traj)

        return avg_results(ntraj, self.results)

    def solve(self, rho0, times, gam, L, ntraj=1000, options=None, results=None):
        """
        """
        self.dt = times[1]-times[0]
        self.tobs = len(times)
        self.setup(gam, L, options, results)
        if self.results.e_ops != None:
            for i in range(len(self.results.e_ops)):
                self.results.e_ops[i] = self.ham.to_eigenbasis(self.results.e_ops[i])

        if self.options.unraveling:
            if is_vector(rho0):
                psi0 = self.ham.to_eigenbasis(rho0.copy())
            else:
                raise AttributeError("Initial condition must be a wavefunction")
            return self.integrate_trajectories(psi0, times, ntraj)
        else:
            if is_matrix(rho0):
                rho = self.ham.to_eigenbasis(rho0.copy())
                self.ode._set_y_value(rho, times[0])
            else:
                raise AttributeError("Initial condition must be a density matrix")
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
