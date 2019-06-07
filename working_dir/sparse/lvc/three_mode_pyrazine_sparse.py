import numpy as np
import scipy.sparse as sp

def kron(*mats):
    out = mats[0].copy()
    for mat in mats[1:]:
        out = sp.kron(out,mat)
    return out

def dot(*mats):
    out = mats[0].copy()
    for mat in mats[1:]:
        out = out.dot(mat)
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
    return sp.csr_matrix(phiout)

def eye(n):
    return sp.eye(n)

def make_ho_q(n):
    qout = np.zeros((n,n))
    for i in range(n-1):
        qout[i,i+1] = np.sqrt(float(i+1)*0.5)
        qout[i+1,i] = np.sqrt(float(i+1)*0.5)
    return sp.csr_matrix(qout)

def make_ho_h(n,omega,kappa=0.0,q=None):
    hout = sp.csr_matrix(np.diag(np.array([omega*(float(i)+0.5) for i in range(n)])))
    if kappa==0.0:
        if q==None:
            q = make_ho_q(n)
        hout += kappa*q
    return hout

hbar = 0.658229
# dimensions
nc = 20
nt1 = 16
nt2 = 20
ntrunc = 1315
# energies
E1 = 3.94
E2 = 4.84
# frequencies
omegac  = 0.118
omegat1 = 0.126
omegat2 = 0.074
# holstein couplings
kt11 = 0.037
kt12 = -0.254
kt21 = -0.105
kt22 = 0.149
# peierls coupling
lamda = 0.262
# times
dt = 0.5
times = np.arange(0.0,500.,dt)

# make position operators
qc  = make_ho_q(nc)
qt1 = make_ho_q(nt1)
qt2 = make_ho_q(nt2)

# make single mode hamiltonians
# c
hc = make_ho_h(nc,omegac)
# t1
ht11 = make_ho_h(nt1,omegat1,kt11,qt1)
ht12 = make_ho_h(nt1,omegat1,kt12,qt1)
# t2
ht21 = make_ho_h(nt2,omegat2,kt21,qt2)
ht22 = make_ho_h(nt2,omegat2,kt22,qt2)

# make full hamiltonian
print('making full hamiltonian')
# energy shift
p1 = kron(phi(0),eye(nc),eye(nt1),eye(nt2))
p2 = kron(phi(1),eye(nc),eye(nt1),eye(nt2))
H  = E1*p1
H += E2*p2
# single mode hamiltonians
# c
H += kron(eye(2),hc,eye(nt1),eye(nt2))
# t1
H += kron(phi(0),eye(nc),ht11,eye(nt2))
H += kron(phi(1),eye(nc),ht12,eye(nt2))
# t2
H += kron(phi(0),eye(nc),eye(nt1),ht21)
H += kron(phi(1),eye(nc),eye(nt1),ht22)
# Peierls coupling
H += lamda*kron(phi(0,1),qc,eye(nt1),eye(nt2))
H += lamda*kron(phi(1,0),qc,eye(nt1),eye(nt2))

# initial condition
# e
psie = np.zeros((2,1),dtype=complex)
psie[1,0] = 1.
psie = sp.csr_matrix(psie)
# c
psic = np.zeros((nc,1),dtype=complex)
psic[0,0] = 1.
psic = sp.csr_matrix(psic)
# t1
psit1 = np.zeros((nt1,1),dtype=complex)
psit1[0,0] = 1.
psit1 = sp.csr_matrix(psit1)
# t2
psit2 = np.zeros((nt2,1),dtype=complex)
psit2[0,0] = 1.
psit2 = sp.csr_matrix(psit2)
# full psi
psi = kron(psie,psic,psit1,psit2)

expects = np.zeros((len(times),2))
#for i,time in enumerate(times):
for i in range(1):
    # compute expectation values
    try:
        expects[i,0] = dot(psi.conj().T, p1, psi).data[0].real
    except:
        expects[i,0] = 0.0
    try:
        expects[i,1] = dot(psi.conj().T, p2, psi).data[0].real
    except:
        expects[i,1] = 0.0
    # build krylov subspace
    tri = np.zeros((20,20))
    #v = psi.copy()
    #tri[0,0] = dot(v.conj().T, H, v).data[0].real
    #for i in range(19):
    #    w = dot(H,v) - tri[i,i]*v
    #    v.append( dot(H,v[i]) - tri[i,i]*v[i] )
    #    tri[i,i+1] = np.sqrt(dot(v[i+1].conj().T,v[i+1]).data[0].real)
    #    tri[i+1,i] = tri[i,i+1]
    #    tri[i+1,i+1] = dot(v[i].conj().T, H, v[i]).data[0].real
    #print(tri)
    #w,v = np.linalg.eigh(tri)
    #print(w)
    #w = np.exp(-1.j*dt*w)
    v = [psi.copy()]
    tri[0,0] = dot(v[0].conj().T, H, v[0]).data[0].real
    for i in range(19):
        v.append( dot(H,v[i]) - tri[i,i]*v[i] )
        tri[i,i+1] = np.sqrt(dot(v[i+1].conj().T,v[i+1]).data[0].real)
        tri[i+1,i] = tri[i,i+1]
        v[i+1] /= tri[i,i+1]
        tri[i+1,i+1] = dot(v[i].conj().T, H, v[i]).data[0].real
    print(tri)
    w,v = np.linalg.eigh(tri)
    w = np.exp(-1.j*dt*w)
    print(w)
    #v = sp.csr_matrix(np.zeros((psi.shape[0],20),dtype=complex))
    #v[:,0] = psi[:,0]
    #tri[0,0] = dot(v[:,0].conj().T, H, v[:,0]).data[0].real
    #for i in range(19):
    #    v.append( dot(H,v[i]) - tri[i,i]*v[i] )
    #    tri[i,i+1] = np.sqrt(dot(v[i+1],v[i+1]).data[0].real)
    #    tri[i+1,i+1] = dot(v[i].conj().T, H, v[i]).data[0].real
#np.savetxt("pyr_3_mode_sparse.txt",expects)
