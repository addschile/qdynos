import sys
import numpy as np
sys.path.append('/Users/addison/Software/')
from qdynos.hamiltonian import Hamiltonian
from qdynos.results import Results
from qdynos.options import Options
from qdynos.lindblad import Lindblad

cm2au = 4.556e-6 # H/cm^-1
ev2au = 0.0367493 # H/eV

def construct_ops():

    nc      = 20
    nt      = 20
    nsite   = 2*nc*nt
    # units in eV
    omega_c = 0.118
    omega_t = 0.074
    kappa1  = -0.105
    kappa2  = 0.149
    lamda   = 0.262
    E1      = 3.94
    E2      = 4.84

    # hc
    qc = np.zeros((nc,nc))
    for i in range(nc-1):
        qc[i,i+1] = np.sqrt(0.5*float(i+1))
        qc[i+1,i] = np.sqrt(0.5*float(i+1))
    hc = np.zeros((nc,nc))
    for i in range(nc):
        hc[i,i] = 0.5 + float(i)
    hc *= omega_c
    wc,vc = np.linalg.eigh(hc)
    Qc = np.kron(np.identity(2),np.kron(qc,np.identity(nt)))

    # ht
    qt = np.zeros((nt,nt))
    for i in range(nt-1):
        qt[i,i+1] = np.sqrt(0.5*float(i+1))
        qt[i+1,i] = np.sqrt(0.5*float(i+1))
    ht1 = np.zeros((nt,nt))
    ht2 = np.zeros((nt,nt))
    for i in range(nt):
        ht1[i,i] = 0.5 + float(i)
        ht2[i,i] = 0.5 + float(i)
    ht1 *= omega_t
    ht2 *= omega_t
    ht1 += kappa1*qt
    ht2 += kappa2*qt
    wt1,vt1 = np.linalg.eigh(ht1)
    wt2,vt2 = np.linalg.eigh(ht2)
    Qt = np.kron(np.array([[1.,0.],[0.,0.]]),np.kron(np.identity(nc),vt1.transpose().dot(qt.dot(vt1))))
    Qt += np.kron(np.array([[0.,0.],[0.,1.]]),np.kron(np.identity(nc),vt2.transpose().dot(qt.dot(vt2))))
    ovt = np.zeros((nt,nt))
    for i in range(nt):
        for j in range(nt):
            ovt[i,j] = np.sum((np.conj(vt1[:,i])*vt2[:,j])[:])

    # H
    # diabatic diagonal energies
    H = np.kron(np.identity(2),np.kron(np.diag(wc),np.identity(nt)))
    H += np.kron(np.array([[1.,0.],[0.,0.]]),np.kron(np.identity(nc),np.diag(wt1)))
    H += np.kron(np.array([[0.,0.],[0.,1.]]),np.kron(np.identity(nc),np.diag(wt2)))
    # off-diagonal coupling
    H += lamda*np.kron(np.array([[0.,1.],[0.,0.]]),np.kron(qc,ovt))
    H += lamda*np.kron(np.array([[0.,0.],[1.,0.]]),np.kron(qc,ovt.conj().T))
    # energy shifts
    H += E1*np.kron(np.array([[1.,0.],[0.,0.]]),np.kron(np.identity(nc),np.identity(nt)))
    H += E2*np.kron(np.array([[0.,0.],[0.,1.]]),np.kron(np.identity(nc),np.identity(nt)))

    ### initial condition ###
    # coupling ground state
    psic = np.zeros((nc,1),dtype=complex)
    psic[0,0] = 1.
    # make ground state tuning mode
    ht_0 = np.zeros((nt,nt))
    for i in range(nt):
        ht_0[i,i] = float(i)+0.5
    ht_0 *= omega_t
    wt0,vt0 = np.linalg.eigh(ht_0)
    # overlap it with 2 electronic state
    psi2t = np.zeros((nt,1),dtype=complex)
    for i in range(nt):
        psi2t[i,0] = vt2[:,i].transpose().dot(vt0[:,0])

    psi_0 = np.kron(np.array([[0.],[1.]]),np.kron(psic,psi2t))

    return nsite,H,Qt,Qc,psi_0

def main(argv):

    hbar = 0.658229     # eV * fs

    t_init  = 0.0
    t_final = 1000.
    dt      = 1.00
    times = np.arange(t_init,t_final,dt) # fs
    tobs = len(times)

    nsite,H,qt,qc,psi_0=construct_ops()

    p1 = np.zeros((nsite,nsite))
    p2 = np.zeros((nsite,nsite))
    for i in range(int(nsite/2)):
        p1[i,i] = 1.
        p2[int(nsite/2)+i,int(nsite/2)+i] = 1.

    ham = Hamiltonian(H, nstates=500, units='ev')
    dynamics = Lindblad(ham)
    results = Results(tobs=tobs,e_ops=[p1,p2])
    options = Options(unraveling=True, seed=1)
    output = dynamics.solve(psi_0, times, np.array([0.1,0.1]), [qt,qc], ntraj=1, eig=True, options=options, results=results)
    output.print_expectation(es_file='db_pops_lindblad_jumps_eig.dat')

if __name__ == '__main__':
    main(sys.argv)
