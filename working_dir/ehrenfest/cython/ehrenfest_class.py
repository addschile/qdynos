from __future__ import print_function,absolute_import

import numpy as np
from cyeom import eom_pq

import qdynos.constants as const

from .integrator import Integrator
from .dynamics import Dynamics
from .utils import commutator,dag,to_liouville,from_liouville
from .options import Options
from .results import Results,add_results,avg_results
from .log import *

class Ehrenfest(Dynamics):

    def __init__(self, ham, options=None):
        """
        """
        super(Redfield, self).__init__(ham)
        self.ham = ham
        if options==None:
            self.options = Options()
        else:
            self.options = options

    def setup(self, times, results):
        """
        """
        self.dt = times[1]-times[0]
        self.tobs = len(times)
        if results==None:
            results = Results()
        self.ode = Integrator(self.dt, self.eom, self.options)
        return deepcopy(results)

    # TODO need to integrate classical dofs into this
    def eom(self, state, order):#H,sigz,cs,rho,Qs):
        """
        """
        Htmp = H + sigz*np.sum((cs*Qs)[:])
        drho = -1.j*comm(Htmp, rho)
        return drho

    def propagate_eom(self, rho, times):

        self.ode._set_y_value(rho, times[0])
        btime = time()
        for i,tau in enumerate(times):
            if self.options.progress:
                if i%int(self.tobs/10)==0:
                    etime = time()
                    print_progress((100*i/self.tobs),(etime-btime))
                elif self.options.really_verbose: print_basic(i)
            if i%self.results.every==0:
                self.results.analyze_state(i, tau, self.ode.y)
            self.ode.integrate()

    def solve(self, rho0, times, results=None):
        """Solve the Redfield equations of motion.
        Parameters
        ----------
        rho_0 : np.array
        times : np.array
        results : Results class
        Returns
        -------
        results : Results class
        """
        results_out = self.setup(times, results)

        every = None
        if ntraj >= 10: every = int(ntraj/10)
        else: every = 1
        btime = time()
        for traj in range(ntraj):
            # initialize classical bath modes #
            self.omegas, self.c_ns, self.Ps, self.Qs = self.baths[i].sample_modes(nmodes, sample)
            # propagate trjectory #
            self.results = deepcopy(results)
            self.propagate_eom(rho0.copy(), times)
            # add results to running average #
            results_out = add_results(results_out,self.results)
        return avg_results(ntraj, results_out)

def rk4(rho,Qs,Ps,dt,H,sigz,omegas,cs,nmodes):
    # containers for dq and dp
    dQs = np.zeros_like(Qs)
    dPs = np.zeros_like(Ps)
    # 1
    coup = np.trace(np.dot(sigz,rho)).real
    krho1 = eom_rho(H,sigz,cs,rho,Qs)
    kq1, kp1 = eom_pq(nmodes,coup,omegas,cs,Qs,Ps,dQs,dPs)
    krho1 *= dt
    kq1   *= dt
    kp1   *= dt
    # 2
    coup = np.trace(np.dot(sigz,rho+0.5*krho1)).real
    krho2 = eom_rho(H,sigz,cs,rho+0.5*krho1,Qs+0.5*kq1)
    kq2, kp2 = eom_pq(nmodes,coup,omegas,cs,Qs+0.5*kq1,Ps+0.5*kp1,dQs,dPs)
    krho2 *= dt
    kq2   *= dt
    kp2   *= dt
    # 3
    coup = np.trace(np.dot(sigz,rho+0.5*krho2)).real
    krho3 = eom_rho(H,sigz,cs,rho+0.5*krho2,Qs+0.5*kq2)
    kq3, kp3 = eom_pq(nmodes,coup,omegas,cs,Qs+0.5*kq2,Ps+0.5*kp2,dQs,dPs)
    krho3 *= dt
    kq3   *= dt
    kp3   *= dt
    # 4
    coup = np.trace(np.dot(sigz,rho+krho3)).real
    krho4 = eom_rho(H,sigz,cs,rho+krho3,Qs+kq3)
    kq4, kp4 = eom_pq(nmodes,coup,omegas,cs,Qs+kq3,Ps+kp3,dQs,dPs)
    krho4 *= dt
    kq4   *= dt
    kp4   *= dt
    return (rho + (krho1/6.) + (krho2/3.) + (krho3/3.) + (krho4/6.)) , (Qs + (kq1/6.) + (kq2/3.) + (kq3/3.) + (kq4/6.)) , (Ps + (kp1/6.) + (kp2/3.) + (kp3/3.) + (kp4/6.))
