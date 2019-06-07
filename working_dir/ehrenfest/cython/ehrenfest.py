import numpy as np
from cyeom import eom_pq

def comm(op1,op2):
    return np.dot(op1,op2) - np.dot(op2,op1)

def anticomm(op1,op2):
    return np.dot(op1,op2) + np.dot(op2,op1)

def Jw(w, lamda, wc):
    return 2.*lamda*wc*w/(w**2. + wc**2.)

def _sample_modes(omega, kT, sample):
    if sample=="Boltzmann":
        Q = np.random.normal(0.0, np.sqrt(kT)/omega)
        P = np.random.normal(0.0, np.sqrt(kT))
        return Q , P
    elif sample=="Wigner":
        Q = np.random.normal(0.0, np.sqrt(1/(2*omega*np.tanh(omega/(2*kT)))))
        P = np.random.normal(0.0, np.sqrt(omega/(2*np.tanh(omega/(2*kT)))))
        return Q , P
sample_modes = np.vectorize(_sample_modes)

def compute_omegas(nmodes, wc, lamda):
    omegas = np.array([wc*np.tan(0.5*np.pi*(float(i)+0.5)/nmodes) for i in range(nmodes)])
    rho = (nmodes/np.pi/wc)*2./(1.+(omegas/wc)**2.)
    c_ns = np.array([ np.sqrt((2./np.pi)*omegas[i]*Jw(omegas[i], lamda, wc)/rho[i]) for i in range(nmodes) ])
    return omegas, c_ns

def eom_rho(H,sigz,cs,rho,Qs):
    Htmp = H + sigz*np.sum((cs*Qs)[:])
    drho = -1.j*comm(Htmp, rho)
    return drho

def rk4(rho,Qs,Ps,dt,H,sigz,omegas,cs,nmodes):
    # containers for dq and dp
    dQs = np.zeros_like(Qs)
    dPs = np.zeros_like(Ps)
    # 1
    coup = np.trace(np.dot(sigz,rho)).real
    krho1 = eom_rho(H,sigz,cs,rho,Qs)
    kq1, kp1 = eom_pq(nmodes,coup,omegas,cs,Qs,Ps,dQs,dPs)
    krho1 *= dt
    kq1   *= dt
    kp1   *= dt
    # 2
    coup = np.trace(np.dot(sigz,rho+0.5*krho1)).real
    krho2 = eom_rho(H,sigz,cs,rho+0.5*krho1,Qs+0.5*kq1)
    kq2, kp2 = eom_pq(nmodes,coup,omegas,cs,Qs+0.5*kq1,Ps+0.5*kp1,dQs,dPs)
    krho2 *= dt
    kq2   *= dt
    kp2   *= dt
    # 3
    coup = np.trace(np.dot(sigz,rho+0.5*krho2)).real
    krho3 = eom_rho(H,sigz,cs,rho+0.5*krho2,Qs+0.5*kq2)
    kq3, kp3 = eom_pq(nmodes,coup,omegas,cs,Qs+0.5*kq2,Ps+0.5*kp2,dQs,dPs)
    krho3 *= dt
    kq3   *= dt
    kp3   *= dt
    # 4
    coup = np.trace(np.dot(sigz,rho+krho3)).real
    krho4 = eom_rho(H,sigz,cs,rho+krho3,Qs+kq3)
    kq4, kp4 = eom_pq(nmodes,coup,omegas,cs,Qs+kq3,Ps+kp3,dQs,dPs)
    krho4 *= dt
    kq4   *= dt
    kp4   *= dt
    return (rho + (krho1/6.) + (krho2/3.) + (krho3/3.) + (krho4/6.)) , (Qs + (kq1/6.) + (kq2/3.) + (kq3/3.) + (kq4/6.)) , (Ps + (kp1/6.) + (kp2/3.) + (kp3/3.) + (kp4/6.))

def solve():

    # parameters #
    eps   = 0.0
    delta = 1.0
    lamda = 2.5*delta
    kT    = 2.0*delta
    ntraj = 100
    dt    = 0.01
    times = np.arange(0.0,12.0,dt)
    nmodes = 300

    # operators #
    sigz = np.array([[1.0,0.0],[0.0,-1.]])
    sigx = np.array([[0.0,1.0],[1.0,0.0]])
    H = eps*sigz + delta*sigx
    rho_0 = np.array([[1.0,0.0],[0.0,0.0]], dtype=complex)

    wcs   = [0.025]#, 0.25]#, 5.0, 10.0]
    for wc in wcs:
        # container for expectation values #
        sigz_avg = np.zeros(len(times))
        # discretize spectral density #
        ws , cs = compute_omegas(nmodes, wc, lamda)
        # loop over trajectories #
        for traj in range(ntraj):
            #print(traj)
            #if traj%100==0:
            #    print("%d of %d"%(traj,ntraj))
            # initialize density matrix #
            rho = rho_0.copy()
            # initialize classical bath modes #
            Qs , Ps = sample_modes(ws, kT, "Wigner")
            for i,time in enumerate(times):
                sigz_avg[i] += (rho[0,0] - rho[1,1]).real
                rho , Qs , Ps = rk4(rho,Qs,Ps,dt,H,sigz,ws,cs,nmodes)
        sigz_avg /= float(ntraj)
        np.savetxt("sigz_new_wc_%.3f.txt"%(wc), sigz_avg)
