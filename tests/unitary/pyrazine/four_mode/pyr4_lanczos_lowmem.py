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
    for mat in mats[1:]:
        out = sp.kron(out,mat)
    return out

def phi(n1,n2=None,nel=None):
    if nel==None:
        phiout = sp.lil_matrix((2,2))
    else:
        phiout = sp.lil_matrix((nel,nel))
    if n2==None:
        phiout[n1,n1] = 1.
    else:
        phiout[n1,n2] = 1.
    return phiout

def eye(n, sparse=False):
    return sp.eye(n, format='lil')

def make_ho_q(n):
    qout = sp.lil_matrix((n,n))
    for i in range(n-1):
        qout[i,i+1] = np.sqrt(float(i+1)*0.5)
        qout[i+1,i] = np.sqrt(float(i+1)*0.5)
    return qout

def make_ho_h(n,omega,kappa=0.0,q=None):
    hout = np.diag(np.array([omega*(float(i)+0.5) for i in range(n)]))
    hout = sp.lil_matrix(hout)
    if kappa != 0.0:
        if not q is None:
            q = make_ho_q(n)
        hout += kappa*q
    return hout

def construct_sys():

    # dimensions
    n10a = 25
    n6a  = 35
    n1   = 25
    n9a  = 20
    nstates = 2*n10a*n6a*n1*n9a
    # energies
    delta = 0.46165
    # frequencies
    w10a = 0.09357
    w6a  = 0.0740
    w1   = 0.1273
    w9a  = 0.1568
    # holstein couplings
    # H_11
    k10a_1 = 0.0
    k6a_1  = -0.0964
    k1_1   = 0.0470
    k9a_1  = 0.1594
    # H_22
    k10a_2 = 0.0
    k6a_2  = 0.1194
    k1_2   = 0.2012
    k9a_2  = 0.0484
    # peierls coupling
    lamda = 0.1825
    
    # make position operators
    q10a = make_ho_q(n10a)
    q6a  = make_ho_q(n6a)
    q1   = make_ho_q(n1)
    q9a  = make_ho_q(n9a)
    
    # make single mode hamiltonians
    # 10a
    h10a_1 = make_ho_h(n10a, w10a, kappa=k10a_1, q=q10a)
    h10a_2 = make_ho_h(n10a, w10a, kappa=k10a_2, q=q10a)
    # 6a
    h6a_1 = make_ho_h(n6a, w6a, kappa=k6a_1, q=q6a)
    h6a_2 = make_ho_h(n6a, w6a, kappa=k6a_2, q=q6a)
    # 1
    h1_1 = make_ho_h(n1, w1, kappa=k1_1, q=q1)
    h1_2 = make_ho_h(n1, w1, kappa=k1_2, q=q1)
    # 9a
    h9a_1 = make_ho_h(n9a, w9a, kappa=k9a_1, q=q9a)
    h9a_2 = make_ho_h(n9a, w9a, kappa=k9a_2, q=q9a)
   
    # make full hamiltonian
    print('making full hamiltonian')
    # energy shift
    print('diag energy')
    p1 = kron(phi(0),eye(n10a),eye(n6a),eye(n1),eye(n9a))
    p2 = kron(phi(1),eye(n10a),eye(n6a),eye(n1),eye(n9a))
    H  = sp.lil_matrix((nstates,nstates))
    H += -delta*p1
    H += delta*p2
    print('mode 10a')
    H += kron(phi(0), h10a_1   , eye(n6a), eye(n1), eye(n9a))
    H += kron(phi(1), h10a_2   , eye(n6a), eye(n1), eye(n9a))
    print('mode 6a')
    H += kron(phi(0), eye(n10a), h6a_1   , eye(n1), eye(n9a))
    H += kron(phi(1), eye(n10a), h6a_2   , eye(n1), eye(n9a))
    print('mode 1')
    H += kron(phi(0), eye(n10a), eye(n6a), h1_1   , eye(n9a))
    H += kron(phi(1), eye(n10a), eye(n6a), h1_2   , eye(n9a))
    print('mode 9a')
    H += kron(phi(0), eye(n10a), eye(n6a), eye(n1), h9a_1)
    H += kron(phi(1), eye(n10a), eye(n6a), eye(n1), h9a_1)
    print('Peierls coupling')
    H += lamda*kron(phi(0,1), q10a, eye(n6a), eye(n1), eye(n9a))
    H += lamda*kron(phi(1,0), q10a, eye(n6a), eye(n1), eye(n9a))

    print('making initial wavefunction')
    # initial condition
    # e
    psie = sp.lil_matrix((2,1),dtype=complex)
    psie[1,0] = 1.
    # 10a
    psi10a = sp.lil_matrix((n10a,1),dtype=complex)
    psi10a[0,0] = 1.
    # 6a
    psi6a = sp.lil_matrix((n6a,1),dtype=complex)
    psi6a[0,0] = 1.
    # 10a
    psi1 = sp.lil_matrix((n1,1),dtype=complex)
    psi1[0,0] = 1.
    # 10a
    psi9a = sp.lil_matrix((n9a,1),dtype=complex)
    psi9a[0,0] = 1.
    # full psi
    psi = kron(psie,psi10a,psi6a,psi1,psi9a)

    # convert to csr_matrix
    H   = sp.csr_matrix(H)
    psi = sp.csr_matrix(psi)
    p1  = sp.csr_matrix(p1)
    p2  = sp.csr_matrix(p2)

    return H,psi,p1,p2

def main():

    # parameters
    dt = 0.5
    times = np.arange(0.0,120.0,dt)

    # construct system
    H,psi0,p1,p2 = construct_sys()

    # set up qdynos and run
    ham = Hamiltonian(H, units='ev')
    dynamics = UnitaryWF(ham)
    output = dynamics.solve(psi0, times, options=Options(method='lanczos',lanczos_lowmem=True), results=Results(tobs=len(times), e_ops=[p1,p2], print_es=True, es_file='pyr4_lanczos_lowmem.txt'))

if __name__ == "__main__":
    main()
