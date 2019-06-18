import numpy as np

import sys
sys.path.append('/Users/addisonschile/Software/')
from qdynos.hamiltonian import Hamiltonian
from qdynos.unitary import UnitaryWF
from qdynos.results import Results
from qdynos.options import Options
import qdynos.constants as const

def kron(*mats):
    out = mats[0].copy()
    for mat in mats[1:]:
        out = np.kron(out,mat)
    return out

def phi(n1,n2=None,nel=None):
    if nel==None:
        phiout = np.zeros((2,2))
    else:
        phiout = np.zeros((nel,nel))
    if n2==None:
        phiout[n1,n1] = 1.
    else:
        phiout[n1,n2] = 1.
    return phiout

def eye(n):
    return np.eye(n)

def make_ho_q(n):
    qout = np.zeros((n,n))
    for i in range(n-1):
        qout[i,i+1] = np.sqrt(float(i+1)*0.5)
        qout[i+1,i] = np.sqrt(float(i+1)*0.5)
    return qout

def make_ho_h(n,omega,kappa=0.0,q=None):
    hout = np.diag(np.array([omega*(float(i)+0.5) for i in range(n)]))
    if kappa!=0.0:
        if not isinstance(q,np.ndarray):
            q = make_ho_q(n)
        hout += kappa*q
    return hout

def construct_sys():

    # dimensions
    nc = 20
    nt = 20
    # energies
    E1 = 3.94
    E2 = 4.84
    # frequencies
    omegac = 0.118
    omegat = 0.074
    # holstein couplings
    kt1 = -0.105
    kt2 = 0.149
    # peierls coupling
    lamda = 0.262
    
    # make position operators
    qc = make_ho_q(nc)
    qt = make_ho_q(nt)
    
    # make single mode hamiltonians
    # c
    hc = make_ho_h(nc,omegac)
    # t
    ht1 = make_ho_h(nt,omegat,kappa=kt1,q=qt)
    ht2 = make_ho_h(nt,omegat,kappa=kt2,q=qt)
    
    # make full hamiltonian
    print('making full hamiltonian')
    # energy shift
    p1 = kron(phi(0),eye(nc),eye(nt))
    p2 = kron(phi(1),eye(nc),eye(nt))
    H  = E1*p1
    H += E2*p2
    # single mode hamiltonians
    # c
    H += kron(eye(2),hc,eye(nt))
    # t1
    H += kron(phi(0),eye(nc),ht1)
    H += kron(phi(1),eye(nc),ht2)
    # Peierls coupling
    H += lamda*kron(phi(0,1),qc,eye(nt))
    H += lamda*kron(phi(1,0),qc,eye(nt))
    
    # initial condition
    # e
    psie = np.zeros((2,1),dtype=complex)
    psie[1,0] = 1.
    # c
    psic = np.zeros((nc,1),dtype=complex)
    psic[0,0] = 1.
    # t
    psit = np.zeros((nt,1),dtype=complex)
    psit[0,0] = 1.
    # full psi
    psi = kron(psie,psic,psit)

    return H,psi,p1,p2

def main():

    # parameters
    dt = 0.1
    times = np.arange(0.0,1000.,dt)
    ntrunc = 500

    # construct system
    H,psi0,p1,p2 = construct_sys()

    # set up qdynos and run
    ham = Hamiltonian(H, nstates=ntrunc, units='ev')
    dynamics = UnitaryWF(ham)
    output = dynamics.solve(psi0, times, eig=False, results=Results(tobs=len(times), e_ops=[p1,p2], print_es=True, es_file='pyr2_exact.txt'))

if __name__ == "__main__":
    main()
