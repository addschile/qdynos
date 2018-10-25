import numpy as np
import qdynos.constants as const
from .results import Results

def linear_absorption(times, omegas, mu_op_eig, dynamics):

    dt = times[1]-times[0]

    # convert to eigenbasis
    #mu_op_eig = dynamics.ham.to_eigenbasis(mu_op)
    #rho_0 = np.dot(mu_op_eig, dynamics.ham.thermal_dm) - np.dot(dynamics.ham.thermal_dm, mu_op_eig)
    rho_0 = np.zeros((9,9),dtype=complex)
    #rho_0[0,0] = 1.
    rho_0[3,3] = -1.
    #rho_0[6,6] = -1.
    rho_0 = dynamics.ham.to_eigenbasis(rho_0)
    rho_0 = np.dot(mu_op_eig, rho_0)
    rho_0 = dynamics.ham.from_eigenbasis(rho_0)

    # run dynamics
    results = Results(tobs=len(times), e_ops=[dynamics.ham.from_eigenbasis(mu_op_eig)])#, print_es=True, es_file='dipole_corr.dat')
    output = dynamics.solve(rho_0, times, results=results)
    f = open('dipole_corr.dat','w')
    for i in range(len(times)):
        f.write('%.8f %.8f\n'%(times[i],output.expect[0,i]))
    f.close()

    # fourier transform of the correlation function
    chi = np.zeros(len(omegas),dtype=complex)
    for i,w in enumerate(omegas):
        corr = np.exp(1.j*w*times)
        chi[i] = (1.j/const.hbar)*np.trapz((corr*output.expect[0,:]),dx=dt)
    
    return chi

#def two_d_ev(times1, times2, times3, omegas_elec, omegas_vib, mu_elec_op, mu_vib_op, dynamics):
#
#    dt = times[1]-times[0]
#
#    # convert to eigenbasis
#    mu_op_eig = dynamics.ham.to_eigenbasis(mu_op)
#    rho_0 = np.dot(mu_op_eig, dynamics.ham.thermal_dm) - np.dot(dynamics.ham.thermal_dm, mu_op_eig)
#    rho_0 = dynamics.ham.to_eigenbasis(rho_0)
#
#    # run dynamics
#    results = Results(tobs=len(times), e_ops=mu_op)
#    output = dynamics.solve(times, rho_0, results=Results)
#
#    # fourier transform of the correlation function
#    corr = np.exp(1.j*np.outer(omegas*times))
#    chi = np.zeros(len(omegas),dtype=complex)
#    for i,w in enumerate(omegas):
#        chi[i] = (1.j/const.hbar)*np.trapz(corr[i,:]*output.expect[0,:],dx=dt)
#    
#    return chi.imag
