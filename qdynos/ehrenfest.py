from __future__ import print_function,absolute_import

import numpy as np

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
    super(Ehrenfest, self).__init__(ham)
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

  def eom(self, state, order):
    """
    """
    Htmp = H
    for i in range(len(self.ham.baths)):
      # update for Ps and Qs in derivative
      Qs = self.Qstmp[i] + self.ode.b[order]*self.kQs[i]
      Ps = self.Pstmp[i] + self.ode.b[order]*self.kPs[i]
      coup = np.trace(np.dot(self.ham.baths[i].op,rho))
      # propagate Ps and Qs
      self.kPs[i] = self.dt*(-self.omegas[i]**2.*Qs - self.cs[i]*coup)
      self.kQs[i] = self.dt*Ps
      # rk update for Ps and Qs
      self.Ps[i] += self.ode.a[order]*self.kPs[i]
      self.Ps[i] += self.ode.a[order]*self.kPs[i]
      # compute new hamiltonian
      Htmp += self.ham.baths[i].op*np.sum((self.cs[i]*Qs)[:])
    # propagate rho
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
      # prep lists used in integration
      for j in range(len(self.ham.baths)):
        self.kQs[j] *= 0.0
        self.kPs[j] *= 0.0
        self.Pstmp[j] = self.Ps[j].copy()
        self.Qstmp[j] = self.Qs[j].copy()
      self.ode.integrate()

  def solve(self, rho0, times, results=None):
    """Solve the Ehrenfest equations of motion.
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
    # TODO 
    # add method in eigenbasis
    # add non-density matrix capabilities

    every = None
    if ntraj >= 10: every = int(ntraj/10)
    else: every = 1
    btime = time()
    for traj in range(ntraj):
      # initialize classical bath modes #
      self.omegas = []
      self.cs   = []
      self.Ps   = []
      self.Qs   = []
      self.Pstmp  = []
      self.Qstmp  = []
      for i in range(len(self.ham.baths)):
        omegas, cs, Ps, Qs = self.ham.baths[i].sample_modes(nmodes, sample)
        self.omegas.append( omegas )
        self.cs.append( cs )
        self.Ps.append( Ps )
        self.Qs.append( Qs )
        self.kQs.append( np.zeros_like(self.Qs[i]) )
        self.kPs.append( np.zeros_like(self.Ps[i]) )
      # propagate trjectory #
      self.results = deepcopy(results)
      self.propagate_eom(rho0.copy(), times)
      # add results to running average #
      results_out = add_results(results_out,self.results)
    return avg_results(ntraj, results_out)
