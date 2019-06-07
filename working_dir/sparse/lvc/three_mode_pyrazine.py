import numpy as np

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
wc,vc = np.linalg.eigh(hc)
# t1
ht11 = make_ho_h(nt1,omegat1,kt11,qt1)
ht12 = make_ho_h(nt1,omegat1,kt12,qt1)
wt11,vt11 = np.linalg.eigh(ht11)
wt12,vt12 = np.linalg.eigh(ht12)
ovt1 = np.zeros((nt1,nt1))
for i in range(nt1):
    for j in range(nt1):
        ovt1[i,j] = np.sum((vt11[:,i]*vt12[:,j])[:])
# t2
ht21 = make_ho_h(nt2,omegat2,kt21,qt2)
ht22 = make_ho_h(nt2,omegat2,kt22,qt2)
wt21,vt21 = np.linalg.eigh(ht21)
wt22,vt22 = np.linalg.eigh(ht22)
ovt2 = np.zeros((nt2,nt2))
for i in range(nt2):
    for j in range(nt2):
        ovt2[i,j] = np.sum((vt21[:,i]*vt22[:,j])[:])

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
H += lamda*kron(phi(0,1),qc,ovt1,ovt2)
H += lamda*kron(phi(1,0),qc,ovt1.T,ovt2.T)
# diagonalize hamiltonian
print('diagonalizing full hamiltonian')
w,v = np.linalg.eigh(H)
prop = np.diag(np.exp(-1.j*dt*w[:ntrunc]/hbar))
p1 = np.dot(v.conj().T, np.dot(p1,v))[:ntrunc,:ntrunc]
p2 = np.dot(v.conj().T, np.dot(p2,v))[:ntrunc,:ntrunc]

# initial condition
# e
psie = np.zeros((2,1),dtype=complex)
psie[1,0] = 1.
# c
psic = np.zeros((nc,1),dtype=complex)
psic[0,0] = 1.
# t1
psit1 = np.zeros((nt1,1),dtype=complex)
for i in range(nt1):
    psit1[i,0] = vt12[i,0]
# t2
psit2 = np.zeros((nt2,1),dtype=complex)
for i in range(nt2):
    psit2[i,0] = vt22[i,0]
# full psi
psi = kron(psie,psic,psit1,psit2)
psi = np.dot(v.conj().T, psi)[:ntrunc,:]

expects = np.zeros((len(times),2))
for i,time in enumerate(times):
    # compute expectation values
    expects[i,0] = np.dot(psi.conj().T, np.dot(p1, psi))[0,0].real
    expects[i,1] = np.dot(psi.conj().T, np.dot(p2, psi))[0,0].real
    # propagate
    psi = np.dot(prop,psi)
np.savetxt("pyr_3_mode.txt",expects)
