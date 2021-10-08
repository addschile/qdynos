from __future__ import print_function,absolute_import

import numpy as np
from time import time
from scipy.linalg import expm

import qdynos.constants as const

from .integrator import Integrator
from .dynamics import Dynamics
from .utils import commutator,dag,matmult,to_liouville,from_liouville
from .options import Options
from .results import Results
from .log import *

class Redfield(Dynamics):
  """Dynamics class for Redfield-like dynamics. Can perform both 
  time-dependent (TCL2) and time-independent dynamics with and without the 
  secular approximation.
  """

  def __init__(self, ham, time_dependent=False, is_secular=False, options=None):
    """Instantiates the Redfield class.
    Parameters
    ----------
    ham : Hamiltonian
    time_dependent : bool
    is_secular : bool
    options : Options class
    """
    super(Redfield, self).__init__(ham)
    #self.ham = ham
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
      if self.is_secular and not self.time_dep:
        self.options.method = "exact"
    else:
      self.options = options
      if self.options.method == "exact":
        if self.time_dep:
          if not self.is_secular and not self.options.space == "hilbert":
            raise NotImplementedError

  def setup(self, times, results):
    """
    """
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
    # TODO this is just stupid
    if self.options.method == "exact":
      if self.time_dep:
        if self.is_secular and self.options.space == "hilbert":
          self.equation_of_motion = self.super_rf_eom
        else:
          raise NotImplementedError
      elif self.options.space == "hilbert":
        if self.is_secular:
          self.equation_of_motion = self.super_rf_eom
        else:
          raise NotImplementedError
      else:
        self.equation_of_motion = self.super_rf_eom
    else:
      if self.time_dep:
        if self.options.space == 'hilbert':
          if self.is_secular:
            self.equation_of_motion = self.super_td_rf_eom
          else:
            self.equation_of_motion = self.td_rf_eom
        elif self.options.space == 'liouville':
          self.equation_of_motion = self.super_td_rf_eom
      else:
        if self.is_secular:
          self.equation_of_motion = self.super_rf_eom
        else:
          if self.options.space == 'hilbert':
            self.equation_of_motion = self.rf_eom
          elif self.options.space == 'liouville':
            self.equation_of_motion = self.super_rf_eom

  def make_lindblad_operators(self):
    """Make and store the coupling operators and "dressed" copuling operators 
    as Lindblad jump operators.
    """
    nstates = self.ham.nstates

    # compute unique frequencies
    if self.options.really_verbose: print_basic("Computing unique frequencies of the Hamiltonian")
    self.ham.compute_unique_freqs()
    self.ham.compute_unique_freqs()

    #self.hcorr = np.zeros((nstates,nstates),dtype=complex)
    self.gams = np.zeros(int(len(self.ham.baths)*len(self.ham.frequencies)),dtype=complex)
    self.Ls = []
    #self.LdL = []
    #self.L0 = []
    count = -1
    for k,bath in enumerate(self.ham.baths):
      if self.options.really_verbose: print_basic("operator %d of %d"%(k+1,len(self.ham.baths)))
      Ga = self.ham.to_eigenbasis( bath.c_op )
      for i in range(len(self.ham.frequencies)):
        count += 1
        omega = self.ham.frequencies[i]
        cf = 2.*bath.ft_bath_corr(-omega)
        self.gams[count] = (cf.real / const.hbar**2.) + ( 0.5j*cf.imag / const.hbar )
        if self.ham.frequencies[i] != 0:
          # TODO don't be dumb
          proj = np.zeros((nstates,nstates))
          for j in range(nstates):
            for k in range(nstates):
              omega = self.ham.omegas[j,k]
              if omega==self.ham.frequencies[i]:
                proj[j,k] = 1.
          L = proj*Ga
          #self.hcorr += (cf_imag*np.dot(dag(L),L))/const.hbar**2.
          #L *= np.sqrt(2.*cf_real)/const.hbar
          #self.Ls[count] = L.copy()
          self.Ls.append( L.copy() )
          #for j in range(nstates):
          #  for k in range(nstates):
          #    if L[j,k] != 0.:
          #      ldl_list = [] # first list is for |L_mn|^2 the second is for [m,n]
          #      ldl_list.append( np.conj(L[j,k])*L[j,k] ) # store |L_jk|^2
          #      ldl_list.append([j,k]) # store jk
          #      self.LdL.append(ldl_list)
        else:
          L = np.diag(np.diag(Ga))
          #self.hcorr += (cf_imag*np.dot(dag(L),L))/const.hbar**2.
          #L *= np.sqrt(2.*cf_real)/const.hbar
          #self.Ls[count] = L.copy()
          self.Ls.append(L.copy())

    #self.A  = (-1.j/const.hbar)*(self.ham.Heig + self.hcorr)

  def make_redfield_operators(self):
    """Make and store the coupling operators and "dressed" copuling operators.
    """
    nstates = self.ham.nstates
    if self.options.space == "hilbert":
      if self.is_secular:
        self.prop = np.zeros((nstates,nstates))
        self.Rdep = np.zeros((nstates,nstates),dtype=complex)
      else:
        self.C = list()
        self.E = list()
    elif self.options.space == "liouville":
      gamma_plus  = np.zeros((nstates,nstates,nstates,nstates),dtype=complex)
      gamma_minus = np.zeros((nstates,nstates,nstates,nstates),dtype=complex)
    for k,bath in enumerate(self.ham.baths):
      if self.options.really_verbose: print_basic("operator %d of %d"%(k+1,len(self.ham.baths)))
      Ga = self.ham.to_eigenbasis( bath.c_op )
      theta_zero = bath.ft_bath_corr(0.0)
      theta_plus = theta_zero*np.identity(nstates,dtype=complex)
      for i in range(nstates):
        if self.options.really_verbose: print_basic("%d rows of %d"%(i,nstates))
        for j in range(nstates):
          if i!=j:
            theta_plus[i,j] = bath.ft_bath_corr(-self.ham.omegas[i,j])
      if self.options.space == "hilbert":
        if self.is_secular:
          if self.options.print_coup_ops:
            np.save(self.options.coup_ops_file+"c_op_%d"%(k),Ga)
            np.save(self.options.coup_ops_file+"theta_plus_%d"%(k),theta_plus)
          # population transfer matrix
          self.prop += 2.*np.einsum('ji,ij,ij->ij',Ga,Ga,theta_plus.real)/const.hbar**2.
          # dephasing matrix
          self.Rdep += np.einsum('jj,ii,ii->ij',Ga,Ga,theta_plus)
          self.Rdep += np.einsum('jj,ii,jj->ij',Ga,Ga,theta_plus.conj().T)
          same_ik = np.einsum('im,mi,mi->i',Ga,Ga,theta_plus)
          same_lj = np.einsum('im,mi,im->i',Ga,Ga,theta_plus.conj().T)
          for i in range(nstates):
            self.Rdep[i,:] -= same_ik[i]
            self.Rdep[:,i] -= same_lj[i]
        else:
          Ga_plus = Ga*theta_plus
          self.C.append(Ga.copy())
          self.E.append(Ga_plus.copy())
          if self.options.print_coup_ops:
            np.save(self.options.coup_ops_file+"c_op_%d"%(k),self.C[k])
            np.save(self.options.coup_ops_file+"e_op_%d"%(k),self.E[k])
            np.save(self.options.coup_ops_file+"theta_plus_%d"%(k),theta_plus)
      elif self.options.space == "liouville":
        if self.options.print_coup_ops:
          np.save(self.options.coup_ops_file+"Ga_%d"%(k),Ga)
          np.save(self.options.coup_ops_file+"theta_plus_%d"%(k),theta_plus)
        gamma_plus  += np.einsum('lj,ik,ik->ljik', Ga, Ga, theta_plus)
        gamma_minus += np.einsum('lj,ik,lj->ljik', Ga, Ga, theta_plus.conj().T)
    if self.options.space == "hilbert":
      if self.is_secular:
        for i in range(nstates):
          self.prop[i,i] = 0.0
          self.prop[i,i] -= np.sum(self.prop[:,i])
        self.Rdep -= self.Rdep*np.eye(nstates)
        if self.options.print_coup_ops:
          np.save(self.options.coup_ops_file+"prop.npy",self.prop)
          np.save(self.options.coup_ops_file+"Rdep.npy",self.Rdep)
    elif self.options.space == "liouville":
      self.R = (gamma_plus.transpose(2,1,3,0) + gamma_minus.transpose(2,1,3,0) -\
           np.einsum('lj,irrk->ijkl', np.identity(nstates), gamma_plus) -\
           np.einsum('ik,lrrj->ijkl', np.identity(nstates), gamma_minus))
      if self.is_secular:
        for i in range(nstates):
          for j in range(nstates):
            for k in range(nstates):
              for l in range(nstates):
                if abs(self.ham.omegas[i,j]
                     -self.ham.omegas[k,l]) > 1e-6:
                  self.R[i,j,k,l] = 0.0
      self.Omega = -1.j*np.einsum('ij,ik,jl->ijkl', self.ham.omegas,
                   np.identity(nstates), np.identity(nstates))
      self.prop = to_liouville(self.Omega) + to_liouville(self.R)/const.hbar**2.

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
      if self.options.print_coup_ops:
        np.save(self.options.coup_ops_file+"c_op_%d"%(op),self.C[op])
      self.gamma_n[op] = list()
      self.gamma_n_1[op] = list()

    if self.options.method == "exact":
      for op,bath in enumerate(self.ham.baths):
        self.gamma_n[op].append( np.zeros((nstates,nstates),dtype=complex) )
        theta_plus = np.exp(-1.j*self.ham.omegas*0.0)*bath.bath_corr_t(0.0)
        self.gamma_n_1[op].append(theta_plus.copy())
        theta_plus = np.exp(-1.j*self.ham.omegas*self.dt)*bath.bath_corr_t(self.dt)
        self.gamma_n[op].append( self.gamma_n[op][0] + self.dt*(theta_plus + self.gamma_n_1[op][0]) )
        self.gamma_n_1[op].append( theta_plus.copy() )
    else:
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
    if self.options.space == "liouville":
      self.Omega = -1.j*np.einsum('ij,ik,jl->ijkl', self.ham.omegas,
                   np.identity(nstates), np.identity(nstates))
      self.Omega = to_liouville(self.Omega)
    if self.is_secular:
      if self.options.space == "hilbert":
        self.prop_n_1 = np.zeros((nstates,nstates))
        self.Rdep_n_1 = np.zeros((nstates,nstates),dtype=complex)

  def make_tcl2_operators(self, time):
    """Integrate "dressing" for copuling operators. Uses trapezoid rule 
    with grid of integration method (e.g., Runge-Kutta 4).
    """
    if self.options.method == 'exact':
      for op,bath in enumerate(self.ham.baths):
        # t
        t = time
        theta_plus = np.exp(-1.j*self.ham.omegas*t)*bath.bath_corr_t(t)
        self.gamma_n[op][0] = self.gamma_n[op][-1].copy()
        self.gamma_n_1[op][0] = theta_plus.copy()
        # t + dt
        t = time + self.dt
        theta_plus = np.exp(-1.j*self.ham.omegas*t)*bath.bath_corr_t(t)
        self.gamma_n_1[op][-1] = theta_plus.copy()
        self.gamma_n[op][-1] = self.gamma_n[op][0] + self.dt*(theta_plus + self.gamma_n_1[op][0])
    else:
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
    nstates = self.ham.nstates
    if self.options.space == "hilbert":
      if self.is_secular:
        if self.options.method == "exact":
          ### time integrate the operators ###
          self.prop_n = np.zeros((nstates,nstates))
          self.Rdep_n = np.zeros((nstates,nstates),dtype=complex)
          for j in range(len(self.C)):
            # population transfer matrix
            self.prop_n += 2.*np.einsum('ji,ij,ij->ij',self.C[j],self.C[j],self.gamma_n[j][-1].real)/const.hbar**2.
            # dephasing matrix
            self.Rdep_n += np.einsum('jj,ii,ii->ij',self.C[j],self.C[j],self.gamma_n[j][-1])
            self.Rdep_n += np.einsum('jj,ii,jj->ij',self.C[j],self.C[j],self.gamma_n[j][-1].conj().T)
            same_ik = np.einsum('im,mi,mi->i',self.C[j],self.C[j],self.gamma_n[j][-1])
            same_lj = np.einsum('im,mi,im->i',self.C[j],self.C[j],self.gamma_n[j][-1].conj().T)
            for i in range(nstates):
              self.Rdep_n[i,:] -= same_ik[i]
              self.Rdep_n[:,i] -= same_lj[i]
          for i in range(nstates):
            self.prop_n[i,i] = 0.0
            self.prop_n[i,i] -= np.sum(self.prop_n[:,i])
          self.Rdep_n -= self.Rdep_n*np.eye(nstates)
          self.prop = expm(self.prop_n-self.prop_n_1)
          self.prop_n_1 = self.prop_n.copy()
          self.Rdep = np.exp(self.dt*(-1.j*self.ham.omegas + (self.Rdep_n-self.Rdep_n_1)/const.hbar**2./self.dt))
          self.Rdep_n_1 = self.Rdep_n.copy()
        else:
          self.prop = list()
          self.Rdep = list()
          for i in range(self.ode.order):
            prop_n = np.zeros((nstates,nstates))
            Rdep_n = np.zeros((nstates,nstates),dtype=complex)
            for j in range(len(self.C)):
              # population transfer matrix
              prop_n += 2.*np.einsum('ji,ij,ij->ij',self.C[j],self.C[j],self.gamma_n[j][i].real)/const.hbar**2.
              # dephasing matrix
              Rdep_n += np.einsum('jj,ii,ii->ij',self.C[j],self.C[j],self.gamma_n[j][i])
              Rdep_n += np.einsum('jj,ii,jj->ij',self.C[j],self.C[j],self.gamma_n[j][i].conj().T)
              same_ik = np.einsum('im,mi,mi->i',self.C[j],self.C[j],self.gamma_n[j][i])
              same_lj = np.einsum('im,mi,im->i',self.C[j],self.C[j],self.gamma_n[j][i].conj().T)
              for k in range(nstates):
                Rdep_n[k,:] -= same_ik[k]
                Rdep_n[:,k] -= same_lj[k]
            for j in range(nstates):
              prop_n[j,j] = 0.0
              prop_n[j,j] -= np.sum(prop_n[:,j])
            Rdep_n -= Rdep_n*np.eye(nstates)
            self.prop.append( prop_n.copy() )
            self.Rdep.append( Rdep_n.copy() )
      else:
        self.E = [[]]*self.ham.nbaths
        for i in range(len(self.C)):
          self.E[i] = list()
          for j in range(self.ode.order):
            self.E[i].append(self.gamma_n[i][j]*self.C[i])
    elif self.options.space == "liouville":
      nstates = self.ham.nstates
      self.prop = list()
      for i in range(self.ode.order):
        gamma_plus  = np.zeros((nstates,nstates,nstates,nstates),dtype=complex)
        gamma_minus = np.zeros((nstates,nstates,nstates,nstates),dtype=complex)
        for j in range(len(self.C)):
          gamma_plus  += np.einsum('lj,ik,ik->ljik', self.C[j], self.C[j], self.gamma_n[j][i])
          gamma_minus += np.einsum('lj,ik,lj->ljik', self.C[j], self.C[j], self.gamma_n[j][i].conj().T)
        R = (gamma_plus.transpose(2,1,3,0) + gamma_minus.transpose(2,1,3,0) -\
          np.einsum('lj,irrk->ijkl', np.identity(nstates), gamma_plus) -\
          np.einsum('ik,lrrj->ijkl', np.identity(nstates), gamma_minus))
        if self.is_secular:
          for j in range(nstates):
            for k in range(nstates):
              for l in range(nstates):
                for m in range(nstates):
                  if abs(self.ham.omegas[j,k]
                       -self.ham.omegas[l,m]) > 1e-6:
                    R[j,k,l,m] = 0.0
        self.prop.append( self.Omega + to_liouville(R.copy())/const.hbar**2. )
    if time < self.options.markov_time:
      self.make_tcl2_operators(time)

  def eom(self, state, order):
    return self.equation_of_motion(state, order)

  def super_rf_eom(self, state, order):
    return matmult(self.prop , state)

  def rf_eom(self, state, order):
    dy = (-1.j/const.hbar)*self.ham.commutator(state)
    for j in range(len(self.E)):
      dy += (commutator(matmult(self.E[j],state),self.C[j]) + commutator(self.C[j],matmult(state,dag(self.E[j]))))/const.hbar**2.
    return dy

  def super_td_rf_eom(self, state, order):
    return matmult(self.prop[order], state)

  def td_rf_eom(self, state, order):
    dy = (-1.j/const.hbar)*self.ham.commutator(state)
    for j in range(len(self.E)):
      dy += (commutator(matmult(self.E[j][order],state),self.C[j]) + commutator(self.C[j],matmult(state,dag(self.E[j][order]))))/const.hbar**2.
    return dy

  def propagate_eom(self, rho, times):

    rho = self.ham.to_eigenbasis(rho.copy())

    if self.results.e_ops != None:
      for i in range(len(self.results.e_ops)):
        self.results.e_ops[i] = self.ham.to_eigenbasis(self.results.e_ops[i])

    if self.options.space == "liouville":
      rho = to_liouville(rho)
    elif self.options.space == "hilbert" and self.is_secular:
      rho_od = rho*(np.ones((self.ham.nstates,self.ham.nstates)) - np.eye(self.ham.nstates))
      rho = np.diag(rho.copy())
    self.ode._set_y_value(rho, times[0])
    btime = time()
    td_switch = 1
    for i,tau in enumerate(times):
      if self.options.progress:
        if i%int(self.tobs/10)==0:
          etime = time()
          print_progress((100*i/self.tobs),(etime-btime))
        elif self.options.really_verbose: print_basic(i)
      if self.time_dep:
        if tau < self.options.markov_time:
          self.update_ops(tau)
        else:
          if td_switch:
            for j in range(len(self.C)):
              self.E[j] = list()
              for k in range(self.ode.order):
                self.E[j].append(self.gamma_n[j][-1]*self.C[j])
          td_switch = 0
        if self.options.print_coup_ops:
          for j in range(len(self.E)):
            np.save(self.options.coup_ops_file+"e_op_%d_%d"%(j,i),self.E[j][0])
      if i%self.results.every==0:
        if self.options.space == "hilbert":
          if self.is_secular:
            self.results.analyze_state(i, tau, np.diag(self.ode.y)+rho_od)
          else:
            self.results.analyze_state(i, tau, self.ode.y)
        elif self.options.space == "liouville":
          self.results.analyze_state(i, tau, from_liouville(self.ode.y))
      if self.is_secular and self.options.space == "hilbert":
        if self.options.method == "exact":
          rho_od *= self.Rdep
        else:
          b = self.ode.b
          for j in range(self.ode.order-1):
            #rho_od *= np.exp((b[j+1]-b[j])*self.dt*self.Rdep[j+1]/const.hbar**2.)
            rho_od *= np.exp((b[j+1]-b[j])*self.dt*self.Rdep/const.hbar**2.)
          rho_od *= np.exp(-1.j*self.dt*self.ham.omegas)
      self.ode.integrate()

    return self.results

  def solve(self, rho0, times, eig=True, results=None):
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
    # diagonalize hamiltonian
    if eig:
      self.ham.eigensystem()

    if self.options.verbose:
      print_stage("Initializing Coupling Operators")
      btime = time()
    if self.time_dep:
      self.coupling_operators_setup()
    else:
      self.make_redfield_operators()
      if self.options.method == "exact":
        self.prop = expm(self.dt*self.prop)
      if self.options.space == "hilbert" and self.is_secular:
        self.Rdep = np.exp(self.dt*(-1.j*self.ham.omegas + self.Rdep/const.hbar**2.))
    if self.options.verbose:
      etime = time()
      print_stage("Finished Constructing Operators")
      print_time(etime-btime)
      print_stage("Propagating Equation of Motion")

    return self.propagate_eom(rho0, times)
