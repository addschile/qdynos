import numpy as np
import qdynos.constants as const

from copy import deepcopy
from .hamiltonian import Hamiltonian,MDHamiltonian
from .results import add_results,avg_results
from .dynamics import Dynamics
from .unitary import UnitaryDM

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
            if omega_stars==None:
                if self.nstates != 2:
                    raise ValueError("Automated omega star only for two-level system.")
                else:
                    omega_stars = [2.*np.sqrt( 0.25*(H[0,0]-H[1,1])**2. + H[0,1]**2. )/const.hbar]*self.nbath
            if PDs==None:
                PDs = [True]*self.nbath
            # compute decomposed spectral densities for each bath #
            for i in range(self.nbath):
                self.baths[i].frozen_mode_decomp(omega_stars[i], PD=PDs[i])

    def solve(self, rho0, times, nmodes=300, ntraj=1000, sample="Boltzmann", results=None):
        results_out = deepcopy(results)

        #avg_ham = np.zeros_like(self.dynamics.ham.ham)
        for traj in range(ntraj):
            print(traj)
            dynamics_copy = deepcopy(self.dynamics)#, options=self.options)
            baths_copy = [deepcopy(self.baths[i]) for i in range(self.nbath)]
            
            Hb = self.dynamics.ham.ham.copy()
            for i in range(self.nbath):
                omegas, c_ns, Ps, Qs = self.baths[i].sample_modes(nmodes, sample)
                Hb += np.sum((c_ns*Qs)[:])*self.baths[i].c_op
            #avg_ham += Hb

            if self.hamtype=="default":
                dynamics_copy.ham = Hamiltonian(Hb, baths=baths_copy, hbar=self.hbar)
            elif self.hamtype=="md":
                dynamics_copy.ham = MDHamiltonian(Hb, baths=baths_copy, hbar=self.hbar)

            rho = rho0.copy()
            results_i = dynamics_copy.solve(rho, times, results=deepcopy(results))
            results_out = add_results(results_out,results_i)

        return avg_results(ntraj, results_out)# , (avg_ham/float(ntraj))

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
