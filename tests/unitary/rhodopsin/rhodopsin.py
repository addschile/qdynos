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

def construct_ops():

    # number of states per mode
    nc = 24
    nphi  = (int(2*150))+1
    nsite = 2*nphi*nc

    # relevant stuff
    eye_e   = sp.eye(2,format='lil')
    eye_c   = sp.eye(nc,format='lil')
    eye_phi = sp.eye(nphi,format='lil')
    phi0    = sp.lil_matrix(np.array([[1.,0.],[0.,0.]]))
    phi1    = sp.lil_matrix(np.array([[0.,0.],[0.,1.]]))
    phi01   = sp.lil_matrix(np.array([[0.,1.],[0.,0.]]))
    phi10   = sp.lil_matrix(np.array([[0.,0.],[1.,0.]]))

    # coupling mode parameters
    sys.stdout.write('Making coupling mode hamiltonian\n')
    sys.stdout.flush()
    omega  = 0.19
    kappa0 = 0.0
    kappa1 = 0.095
    qc = sp.lil_matrix((nc,nc),dtype=complex)
    for i in range(nc-1):
        qc[i,i+1] = np.sqrt(float(i+1)/2.)
        qc[i+1,i] = np.sqrt(float(i+1)/2.)
    hc = sp.lil_matrix((nc,nc),dtype=complex)
    for i in range(nc):
        hc[i,i] = omega*(float(i)+0.5)
    hc0 = hc + kappa0*qc
    hc1 = hc + kappa1*qc

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

    qphi = eye_phi-cosphi
    V0 = E0*eye_phi + 0.5*W0*(eye_phi-cosphi)
    V1 = E1*eye_phi - 0.5*W1*(eye_phi-cosphi)
    hphi0 = -0.5*minv*tphi + V0
    hphi1 = -0.5*minv*tphi + V1

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
    ptrans = sp.lil_matrix((nphi,nphi))
    for i in range(-nm,nm+1):
        for j in range(-nm,nm+1):
            if (i-j)%2!=0: #difference is odd
                diff = float(abs(i-j))
                expo = ((diff-1.)/2.)+1.
                ptrans[i+nm,j+nm] = (-1.)**(expo)/(diff*np.pi)
            elif i==j:
                ptrans[i+nm,j+nm] = 0.5
    pcis = eye_phi-ptrans

    Ptrans = kron(eye_e, ptrans, eye_c)
    Pcis   = kron(eye_e, pcis, eye_c)
    p0 = kron(phi0, eye_phi, eye_c)
    p1 = kron(phi1, eye_phi, eye_c)
    Ptrans0 = Ptrans.dot(p0)
    Ptrans0 = sp.csr_matrix(Ptrans0)
    Ptrans1 = Ptrans.dot(p1)
    Ptrans1 = sp.csr_matrix(Ptrans1)
    Pcis0 = Pcis.dot(p0)
    Pcis0 = sp.csr_matrix(Pcis0)
    Pcis1 = Pcis.dot(p1)
    Pcis1 = sp.csr_matrix(Pcis1)

    #sys.stdout.write('Making system-bath coupling operators\n')
    #sys.stdout.flush()
    ## c
    #Qc = np.kron(phi0, np.kron(eye_phi, np.dot(vc0.conj().T,np.dot(qc,vc0))))
    #Qc += np.kron(phi1, np.kron(eye_phi, np.dot(vc1.conj().T,np.dot(qc,vc1))))
    ## phi
    #Qphi  = np.kron(phi0, np.kron(np.dot(vphi0.conj().T,np.dot(qphi,vphi0)), eye_c))
    #Qphi += np.kron(phi1, np.kron(np.dot(vphi1.conj().T,np.dot(qphi,vphi1)), eye_c))
    #Qphi = np.dot(v.conj().T,np.dot(Qphi,v))
    #Cosphi  = np.kron(phi0, np.kron(np.dot(vphi0.conj().T,np.dot(cosphi,vphi0)), eye_c))
    #Cosphi += np.kron(phi1, np.kron(np.dot(vphi1.conj().T,np.dot(cosphi,vphi1)), eye_c))
    #Cosphi = np.dot(v.conj().T,np.dot(Cosphi,v))

    ### initial condition
    sys.stdout.write('Creating initial condition\n')
    sys.stdout.flush()
    # cis excitation
    psie = sp.lil_matrix((2,1),dtype=complex)
    psie[1,0] = 1.
    psiphi = sp.lil_matrix((nphi,1),dtype=complex)
    psiphi[0,0] = 1.
    psic = sp.lil_matrix((nc,1),dtype=complex)
    psic[0,0] = 1.
    psi0 = kron(psie,psiphi,psic)
    psi0 = sp.csr_matrix(psi0)

    return psi0, H, Ptrans0, Ptrans1, Pcis0, Pcis1

if __name__ == '__main__':

    # construct operators
    psi0,H,pt0,pt1,pc0,pc1 = construct_ops()

    # set up qdynos and run
    times = np.arange(0.0,4000.,1.0)
    ham = Hamiltonian(H, units='ev')
    dynamics = UnitaryWF(ham)
    output = dynamics.solve(psi0, times, options=Options(method='lanczos'), results=Results(tobs=len(times), e_ops=[pt0,pt1,pc0,pc1], print_es=True, es_file='rhodopsin.txt'))
