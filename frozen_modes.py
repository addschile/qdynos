import numpy as np
import qdynos.constants as const

from copy import deepcopy
from time import time
from .hamiltonian import Hamiltonian,MDHamiltonian
from .results import add_results,avg_results
from .dynamics import Dynamics
from .unitary import UnitaryDM
from .log import print_basic,print_progress

class Frozen:

    def __init__(self, H, baths, nstates=None, hbar=1., omega_stars=None,
                 PDs=None, hamtype="default", dynamics=None, options=None):
        self.nbath = len(baths)
        if nstates == None:
            self.nstates = H.shape[0]
        else:
            self.nstates = nstates
        self.baths = baths
        self.options = options
        self.hamtype = hamtype
        self.hbar = hbar

        if dynamics==None:
            if hamtype=="default":
                    self.ham = Hamiltonian(H, nstates=nstates, hbar=hbar)
            elif hamtype=="md":
                    self.ham = MDHamiltonian(H)
            self.dynamics = UnitaryDM(ham)
        else:
            self.dynamics = dynamics
            if not (isinstance(omega_stars, list) or isinstance(omega_stars, np.ndarray)):
                if self.nstates != 2:
                    raise ValueError("Automated omega star only for two-level system.")
                else:
                    omega_stars = [0.5*np.sqrt( 0.25*(H[0,0]-H[1,1])**2. + H[0,1]**2. )/const.hbar]*self.nbath
            if PDs==None:
                PDs = [False]*self.nbath
            # compute decomposed spectral densities for each bath #
            for i in range(self.nbath):
                self.baths[i].frozen_mode_decomp(omega_stars[i], PD=PDs[i])

    def solve(self, rho0, times, nmodes=300, ntraj=1000, sample="Boltzmann", results=None):
        results_out = deepcopy(results)

        every = None
        if ntraj >= 10: every = int(ntraj/10)
        else: every = 1
        btime = time()
        for traj in range(ntraj):
            if traj%every==0:
                print_progress(10*traj/every,time()-btime)
            dynamics_copy = deepcopy(self.dynamics)
            baths_copy = [deepcopy(self.baths[i]) for i in range(self.nbath)]
            
            Hb = self.dynamics.ham.ham.copy()
            if self.options.print_decomp:
                self.options.decomp_file = open(self.options._decomp_file+"_traj_%d"%(traj), "w")
            for i in range(self.nbath):
                omegas, c_ns, Ps, Qs = self.baths[i].sample_modes(nmodes, sample)
                if self.options.print_decomp:
                    for j in range(len(c_ns)):
                        self.options.decomp_file.write("%.8f %.8f %.8f %.8f\n"%(omegas[j],c_ns[j],Ps[j],Qs[j]))
                        self.options.decomp_file.flush()
                Hb += np.sum((c_ns*Qs)[:])*self.baths[i].c_op
            if self.options.print_decomp:
                self.options.decomp_file.close()

            if self.hamtype=="default":
                dynamics_copy.ham = Hamiltonian(Hb, nstates=self.nstates, baths=baths_copy, hbar=self.hbar)
            elif self.hamtype=="md":
                dynamics_copy.ham = MDHamiltonian(Hb, baths=baths_copy, hbar=self.hbar)

            rho = rho0.copy()
            traj_results = deepcopy(results)
            if self.options.traj_results:
                traj_results.print_es = True
                traj_results.fes = open(self.options.traj_results_file+"_traj_%d"%(traj), "w")
            if self.options.traj_state:
                traj_results.print_states = True
                traj_results.states_file = self.opitions.traj_states_file
            results_i = dynamics_copy.solve(rho, times, results=traj_results)
            results_out = add_results(results_out,results_i)

        return avg_results(ntraj, results_out)

def frozen_solve(rho0, times, H, baths, hbar=1., omega_stars=None, PDs=None, 
                 nmodes=300, ntraj=1000, sample="Boltzmann", hamtype="default", 
                 hamargs=None, dynamics=None, options=None, results=None):
    """
    """
    nbath = len(baths)
    nstates = H.shape[0]
    const.hbar = hbar

    if dynamics==None:
        if hamtype=="default":
                ham = Hamiltonian(H)
        elif hamtype=="md":
                ham = MDHamiltonian(H)
        dynamics = UnitaryDM(ham)
    else:
        if omega_stars==None:
            if nstates != 2:
                raise ValueError("Automated omega star only for two-level system.")
            else:
                omega_stars = [2.*np.sqrt( 0.25*(H[0,0]-H[1,1])**2. + H[0,1]**2. )/const.hbar]*nbath

        if PDs==None:
            PDs = [True]*nbath

        # compute decomposed spectral densities for each bath #
        for i in range(nbath):
            baths[i].frozen_mode_decomp(omega_stars[i], PD=PDs[i])

    results_out = deepcopy(results)

    avg_ham = np.zeros_like(H)
    for traj in range(ntraj):
        dynamics_copy = deepcopy(dynamics)
        baths_copy = [deepcopy(baths[i]) for i in range(nbath)]
        
        Hb = H.copy()
        for i in range(nbath):
            omegas, c_ns, Ps, Qs = baths[i].sample_modes(nmodes, sample)
            Hb += np.sum((c_ns*Qs)[:])*baths[i].c_op
        avg_ham += Hb

        if self.hamtype=="default":
            dynamics_copy.ham = Hamiltonian(Hb, baths=baths_copy, hbar=hbar)
        elif self.hamtype=="md":
            dynamics_copy.ham = MDHamiltonian(Hb, baths=baths_copy, hbar=hbar)

        rho = rho0.copy()
        results_i = dynamics_copy.solve(rho, times, options=options, results=deepcopy(results))
        results_out = add_results(results_out,results_i)

    return avg_results(ntraj, results_out) , (avg_ham/float(ntraj))
