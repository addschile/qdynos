import numpy as np
import qdynos as qd

def main():

  nsite = 7
  nbath = 7

  # System Hamiltonian in cm-1
  HS = np.array([[12410, -87.7,   5.5,  -5.9,   6.7, -13.7,  -9.9],
                 [-87.7, 12530,  30.8,   8.2,   0.7,  11.8,   4.3],
                 [  5.5,  30.8, 12210, -53.5,  -2.2,  -9.6,   6.0],
                 [ -5.9,   8.2, -53.5, 12320, -70.7, -17.0, -63.3],
                 [  6.7,   0.7,  -2.2, -70.7, 12480,  81.1,  -1.3],
                 [-13.7,  11.8,  -9.6, -17.0,  81.1, 12630,  39.7],
                 [ -9.9,   4.3,   6.0, -63.3,  -1.3,  39.7, 12440]])

  lamda = 35.0
  for [tau,T] in [[50.,77.], [166.,300.]]:
    omega_c = 1.0/tau
    eta = 2.*lamda*omega_c
    kT = qd.qdconst.kbs['cm']*T
    e_ops = []
    for n in range(nbath):
      ham_sysbath_n = np.zeros((nsite,nsite))
      ham_sysbath_n[n,n] = 1.0
      e_ops.append( ham_sysbath_n.copy() )
    baths = [qd.DebyeBath(eta, omega_c, kT, op=e_ops[n]) for n in range(nbath)]

    ham = qd.Hamiltonian(HS, baths=baths, units='cm')

    # redfield dynamics
    dynamics = qd.Redfield(ham)
    rho_0 = np.zeros((nsite, nsite),dtype=complex)
    rho_0[0,0] = 1.0
    times = np.arange(0.0,1000.0,1.0)
    results = qd.Results(tobs=len(times), e_ops=e_ops, print_es=True, 
                      es_file='site_pops_rf_tau_%.0f_T_%.0f.txt'%(tau,T))
    output = dynamics.solve(rho_0, times, results=results)

    # TCL2 dynamics
    dynamics = qd.Redfield(ham, time_dependent=True)
    rho_0 = np.zeros((nsite, nsite),dtype=complex)
    rho_0[0,0] = 1.0
    times = np.arange(0.0,1000.0,1.0)
    results = qd.Results(tobs=len(times), e_ops=e_ops, print_es=True, 
                      es_file='site_pops_tcl2_tau_%.0f_T_%.0f.txt'%(tau,T))
    output = dynamics.solve(rho_0, times, results=results)

if __name__ == '__main__':
  main()
