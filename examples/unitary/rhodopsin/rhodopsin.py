import sys
import numpy as np
import scipy.sparse as sp
import qdynos as qd

#sys.path.append('/Users/addison/Software/')
#from qdynos.hamiltonian import Hamiltonian
#from qdynos.unitary import UnitaryWF
#from qdynos.results import Results
#from qdynos.options import Options

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

def construct_ops():

    # number of states per mode
    nc = 24
    nphi  = (int(2*150))+1
    nsite = 2*nphi*nc

    # relevant stuff
    eye_e   = eye(2)
    eye_c   = eye(nc)
    eye_phi = eye(nphi)
    phi0    = phi(0)
    phi1    = phi(1)
    phi01   = phi(0,1)
    phi10   = phi(1,0)

    # coupling mode parameters
    sys.stdout.write('Making coupling mode hamiltonian\n')
    sys.stdout.flush()
    omega  = 0.19
    kappa0 = 0.0
    kappa1 = 0.095
    qc = make_ho_q(nc)
    hc0 = make_ho_h(nc, omega, kappa=kappa0, q=qc)
    hc1 = make_ho_h(nc, omega, kappa=kappa1, q=qc)
    wc0,vc0 = np.linalg.eigh(hc0.toarray())
    wc1,vc1 = np.linalg.eigh(hc1.toarray())
    wc0 = sp.lil_matrix(np.diag(wc0))
    wc1 = sp.lil_matrix(np.diag(wc1))
    # make unitary transformations for c
    Vc0 = sp.lil_matrix(vc0)
    Vc1 = sp.lil_matrix(vc1)
    Vc  = kron(phi0, eye_phi, Vc0)
    Vc += kron(phi1, eye_phi, Vc1)
    Vc = sp.csr_matrix(Vc)

    # rotor mode parameters
    sys.stdout.write('Making rotor mode hamiltonian\n')
    sys.stdout.flush()
    nm    = int((nphi-1)/2)
    minv  = 1.43e-3
    E0    = 0.0
    E1    = 2.00
    W0    = 2.3
    W1    = 1.50
    lamda = 0.19

    # hphi
    cosphi = sp.lil_matrix((nphi,nphi))
    for i in range(nphi-1):
        cosphi[i,i+1] = 0.5
        cosphi[i+1,i] = 0.5
    tphi = sp.lil_matrix((nphi,nphi))
    for i in range(-nm,nm+1):
        tphi[i+nm,i+nm] = -float(i)**2.
    V0 = E0*eye_phi + 0.5*W0*(eye_phi-cosphi)
    V1 = E1*eye_phi - 0.5*W1*(eye_phi-cosphi)
    hphi0 = -0.5*minv*tphi + V0
    hphi1 = -0.5*minv*tphi + V1
    wphi0,vphi0 = np.linalg.eigh(hphi0.toarray())
    wphi1,vphi1 = np.linalg.eigh(hphi1.toarray())
    wphi0 = sp.lil_matrix(np.diag(wphi0))
    wphi1 = sp.lil_matrix(np.diag(wphi1))
    # make unitary transformations for phi
    Vphi0 = sp.lil_matrix(vphi0)
    Vphi1 = sp.lil_matrix(vphi1)
    Vphi  = kron(phi0, Vphi0, eye_c)
    Vphi += kron(phi1, Vphi1, eye_c)
    Vphi = sp.csr_matrix(Vphi)

    # FC overlap matrix
    sys.stdout.write('Making overlap matrices\n')
    sys.stdout.flush()
    overlaps = np.zeros((nphi,nphi),dtype=complex)
    for i in range(nphi):
        for j in range(nphi):
            overlaps[i,j] = np.sum((np.conj(vphi0[:,i])*vphi1[:,j])[:])
    coverlaps = np.zeros((nc,nc),dtype=complex)
    for i in range(nc):
        for j in range(nc):
            coverlaps[i,j] = np.sum((np.conj(vc0[:,i])*vc1[:,j])[:])
    # Hamiltonian in full space
    sys.stdout.write('Making full hamiltonian\n')
    sys.stdout.flush()
    # diabatic hamiltonian stuff
    H  = kron(phi0, hphi0, eye_c) # energy of state rotor mode 0
    H += kron(phi1, hphi1, eye_c) # energy of state rotor mode 1
    H += kron(phi0, eye_phi, hc0) # energy of state coupling mode 0
    H += kron(phi1, eye_phi, hc1) # energy of state coupling mode 0
    # electronic coupling
    H += lamda*kron(phi01, eye_phi, qc)
    H += lamda*kron(phi10, eye_phi, qc)
    H = sp.csr_matrix(H)

    sys.stdout.write('Making projection operators\n')
    sys.stdout.flush()
    ptrans = np.zeros((nphi,nphi))
    for i in range(-nm,nm+1):
        for j in range(-nm,nm+1):
            if (i-j)%2!=0: #difference is odd
                diff = float(abs(i-j))
                expo = ((diff-1.)/2.)+1.
                ptrans[i+nm,j+nm] = (-1.)**(expo)/(diff*np.pi)
            elif i==j:
                ptrans[i+nm,j+nm] = 0.5
    pcis = eye_phi-ptrans
    ptrans = sp.lil_matrix(ptrans)
    pcis = sp.lil_matrix(pcis)

    # full Ptrans and Pcis
    Ptrans = kron(eye_e, ptrans, eye_c)
    Pcis   = kron(eye_e, pcis, eye_c)
    # diabatic projectors
    p0 = kron(phi0, eye_phi, eye_c)
    p1 = kron(phi1, eye_phi, eye_c)
    # convert to csr matrix
    Ptrans = sp.csr_matrix(Ptrans)
    Pcis = sp.csr_matrix(Pcis)
    p0 = sp.csr_matrix(p0)
    p1 = sp.csr_matrix(p1)
    # diabatic projected Ptrans
    Ptrans0 = Ptrans.dot(p0)
    Ptrans1 = Ptrans.dot(p1)
    # diabatic projected Pcis
    Pcis0 = Pcis.dot(p0)
    Pcis1 = Pcis.dot(p1)

    ### initial condition
    sys.stdout.write('Creating initial condition\n')
    sys.stdout.flush()
    # cis excitation
    # e
    psie = sp.lil_matrix((2,1),dtype=complex)
    psie[1,0] = 1.
    # phi
    psiphi = sp.lil_matrix((nphi,1),dtype=complex)
    #psiphi[0,0] = 1.
    for j in range(nphi):
        psiphi[j,0] = np.conj(overlaps[0,j])
    # c
    psic = sp.lil_matrix((nc,1),dtype=complex)
    #psic[0,0] = 1.
    for j in range(nc):
        psic[j,0] = np.conj(coverlaps[0,j])
    # full psi
    psi0 = kron(psie,psiphi,psic)
    # conver to csr
    psi0 = sp.csr_matrix(psi0)
    # unitary transformation
    psi0 = Vphi.dot(psi0)
    psi0 = Vc.dot(psi0)

    return psi0, H, p0, p1, Ptrans0, Ptrans1, Pcis0, Pcis1

if __name__ == '__main__':

    # construct operators
    psi0, H, p0, p1, Ptrans0, Ptrans1, Pcis0, Pcis1 = construct_ops()

    # set up qdynos and run
    times = np.arange(0.0,4000.,1.0)
    ham = qd.Hamiltonian(H, eig=False, units='ev')
    dynamics = qd.UnitaryWF(ham)
    output = dynamics.solve(psi0, times, options=qd.Options(method='lanczos'), results=qd.Results(tobs=len(times), e_ops=[p0,p1,Pcis0,Pcis1,Ptrans0,Ptrans1], print_es=True, es_file='rhodopsin.txt'))
