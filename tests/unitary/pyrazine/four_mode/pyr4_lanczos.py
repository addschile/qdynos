import numpy as np
import scipy.sparse as sp

import sys
sys.path.append('/Users/addison/Software/')
from qdynos.hamiltonian import Hamiltonian
from qdynos.unitary import UnitaryWF
from qdynos.results import Results
from qdynos.options import Options

def kron(*mats):
    out = mats[0].copy()
    if isinstance(mat, np.ndarray):
        for mat in mats[1:]:
            out = np.kron(out,mat)
    else:
        for mat in mats[1:]:
            out = sp.kron(out,mat)
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

def eye(n, sparse=False):
    if sparse:
        return sp.eye(n)
    return np.eye(n)

def make_ho_q(n, sparse=False):
    if sparse:
        qout = sp.csr_matrix((n,n))
    else:
        qout = np.zeros((n,n))
    for i in range(n-1):
        qout[i,i+1] = np.sqrt(float(i+1)*0.5)
        qout[i+1,i] = np.sqrt(float(i+1)*0.5)
    return qout

def make_ho_h(n,omega,kappa=0.0,q=None):
    hout = np.diag(np.array([omega*(float(i)+0.5) for i in range(n)]))
    if sparse:
        hout = sp.csr_matrix(hout)
    if kappa != 0.0:
        if not q is None:
            q = make_ho_q(n, sparse=sparse)
        hout += kappa*q
    return hout

def construct_sys():

    # dimensions
    nv10a = 25
    nv6a  = 35
    nv1   = 25
    nv9a  = 20
    nstates = nv10a*nv6a*nv1*nv9a
    # energies
    delta = 0.46165
    # frequencies
    w10a = 0.09357
    w6a  = 0.0740
    w1   = 0.1273
    w9a  = 0.1568
    # holstein couplings
    # H_11
    k6a_1 = -0.0964
    k1_1  = 0.0470
    k9a_1 = 0.1594
    # H_22
    k6a_2 = 0.1194
    k1_2  = 0.2012
    k9a_2 = 0.0484
    # peierls coupling
    lamda = 0.1825
    
    # make position operators
    q10a = make_ho_q(n10a,sparse=True)
    q10a = make_ho_q(n10a,sparse=True)
    q10a = make_ho_q(n10a,sparse=True)
    q10a = make_ho_q(n10a,sparse=True)
    
    # make single mode hamiltonians
    # 10a
    h10a_1 = make_ho_h(nv10a, w10a, kappa=k10a_1, q=q10a, sparse=True)
    h10a_2 = make_ho_h(nv10a, w10a, kappa=k10a_2, q=q10a, sparse=True)
    # 6a
    h6a_1 = make_ho_h(nv6a, w6a, kappa=k6a_1, q=q6a, sparse=True)
    h6a_2 = make_ho_h(nv6a, w6a, kappa=k6a_2, q=q6a, sparse=True)
    # 1
    h1_1 = make_ho_h(nv1, w1, kappa=k1_1, q=q1, sparse=True)
    h1_2 = make_ho_h(nv1, w1, kappa=k1_2, q=q1, sparse=True)
    # 9a
    h9a_1 = make_ho_h(nv9a, w9a, kappa=k9a_1, q=q9a, sparse=True)
    h9a_2 = make_ho_h(nv9a, w9a, kappa=k9a_2, q=q9a, sparse=True)
   
    # make full hamiltonian
    print('making full hamiltonian')
    # energy shift
    p1 = kron(phi(0),eye(nc),eye(nt))
    p2 = kron(phi(1),eye(nc),eye(nt))
    H  = sp.lil_matrix((nstates,nstates))
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
    dt = 0.5
    times = np.arange(0.0,1000.0,dt)
    #times = np.arange(0.0,0.2,dt)

    # construct system
    H,psi0,p1,p2 = construct_sys()

    # set up qdynos and run
    ham = Hamiltonian(H, units='ev')
    dynamics = UnitaryWF(ham)
    output = dynamics.solve(psi0, times, options=Options(method='lanczos'), results=Results(tobs=len(times), e_ops=[p1,p2,p1+p2], print_es=True, es_file='pyr2_lanczos.txt'))

if __name__ == "__main__":
    main()
