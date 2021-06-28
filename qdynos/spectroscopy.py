import numpy as np
import qdynos.constants as const
from .results import Results

def linear_absorption(times, omegas, rho0, mu_op_eig, dynamics, print_dipole=True, dipole_file='dipole_corr.dat'):

  raise NotImplementedError
  # convert to eigenbasis
  rho0 = dynamics.ham.to_eigenbasis(rho0)
  rho0 = np.dot(mu_op_eig, rho0)
  rho0 = dynamics.ham.from_eigenbasis(rho0)

  # run dynamics
  if print_dipole:
    results = Results(tobs=len(times), e_ops=[dynamics.ham.from_eigenbasis(mu_op_eig)], print_es=True, es_file=dipole_file)
  else:
    results = Results(tobs=len(times), e_ops=[dynamics.ham.from_eigenbasis(mu_op_eig)])
  output = dynamics.solve(rho0, times, results=results)

  # fourier transform of the correlation function
  dt = times[1]-times[0]
  chi = np.zeros(len(omegas),dtype=complex)
  for i,w in enumerate(omegas):
    corr = np.exp(1.j*w*times)
    chi[i] = (1.j/const.hbar)*np.trapz((corr*output.expect[0,:]),dx=dt)
  
  return chi

def two_d_ev(times1, times2, times3, omegas_e, omegas_v, mu_e, mu_v, dynamics):
  """
  """
  raise NotImplementedError
  dt = times1[1]-times1[0]
  corr = np.zeros(4, dtype=np.ndarray)
  
  # first pathway
  mu_e_eig = dynamics.ham.to_eigenbasis(mu_e)
  rho_0 = np.dot(mu_op_eig, dynamics.ham.thermal_dm)
  rho_0 = dynamics.ham.from_eigenbasis(rho_0)
  for i in range(len(times1)):
    _times1 = np.arange(times1[0],times[i],dt)
    for j in range(len(times2)):
      for k in range(len(times3)):

  # run dynamics
  results = Results(tobs=len(times), e_ops=mu_op)
  output = dynamics.solve(times, rho_0, results=Results)

  # fourier transform of the correlation function
  corr = np.exp(1.j*np.outer(omegas*times))
  chi = np.zeros(len(omegas),dtype=complex)
  for i,w in enumerate(omegas):
    chi[i] = (1.j/const.hbar)*np.trapz(corr[i,:]*output.expect[0,:],dx=dt)
  
  return chi.imag
