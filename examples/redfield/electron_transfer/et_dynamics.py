import numpy as np
import qdynos as qd

def construct_ops(eta,omega_c):

    n = 50
    nsite = 2*n
    omega = 0.05*qd.qdconst.ev2au
    delta1 = 1.0
    delta2 = -1.0
    kappa1 = -delta1*omega
    kappa2 = -delta2*omega
    V = omega/10.0
    shift = 0.0*qd.qdconst.ev2au

    Q = np.zeros((n,n))
    for i in range(n-1):
        Q[i,i+1] = np.sqrt(0.5*float(i+1))
        Q[i+1,i] = np.sqrt(0.5*float(i+1))

    # h0
    h0 = omega*np.diag(np.array([0.5+i for i in range(n)]))
    # h1
    h1 = h0 + kappa1*Q + Q.dot(Q)*(eta*omega_c/np.pi)
    w1,v1 = np.linalg.eigh(h1)
    q1 = v1.transpose().dot(Q.dot(v1))
    # h2
    h2 = h0 + kappa2*Q + Q.dot(Q)*(eta*omega_c/np.pi)
    w2,v2 = np.linalg.eigh(h2)
    q2 = v2.transpose().dot(Q.dot(v2))

    # compute frank-condon overlap matrix
    fc_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            fc_mat[i,j] = (np.dot(v1[:,i].transpose(),v2[:,j]))
            fc_mat[j,i] = fc_mat[i,j]

    # H
    H = np.kron(np.array([[1.,0.],[0.,0.]]),np.diag(w1)) # diagonal portion for ho1
    H += np.kron(np.array([[0.,0.],[0.,1.]]),np.diag(w2)) # diagonal portion for ho2
    H += V*np.kron(np.array([[0.,1.],[1.,0.]]),fc_mat)

    # Q in full space
    Qfull = np.kron(np.array([[1.,0.],[0.,0.]]),q1)
    Qfull += np.kron(np.array([[0.,0.],[0.,1.]]),q2)

    # initial condition
    rho_0 =  np.zeros((2*n,2*n),dtype=complex)
    for i in range(n):
        for j in range(n):
            rho_0[n+i,n+j] = v2[i,0]*v2[j,0]

    return nsite,H,Qfull,rho_0

def main():

    kB = qd.qdconst.kbs['au']
    T = 1.e-20
    kT = kB*T

    t_init = 0.
    dt = 0.5*qd.qdconst.fs2au # au
    t_final = 2000.*qd.qdconst.fs2au # au
    times = np.arange(t_init, t_final, dt)
    omega_c = 0.05*qd.qdconst.ev2au
    eta = 0.1

    nsite,H,Q,rho_0=construct_ops(eta,omega_c)

    e_ops = []
    e = np.zeros((nsite,nsite))
    for i in range(int(nsite/2)):
        e[i,i] = 1.
    e_ops.append( e.copy() )
    e = np.zeros((nsite,nsite))
    for i in range(int(nsite/2)):
        e[int(nsite/2)+i,int(nsite/2)+i] = 1.
    e_ops.append( e.copy() )

    bath = [qd.OhmicExp(eta, omega_c, kT, op=Q)]
    ham = qd.Hamiltonian(H, baths=bath)

    # run redfield dynamics
    dynamics = qd.Redfield(ham)
    results = qd.Results(tobs=len(times), e_ops=e_ops, print_es=True, es_file='db_pops_rf.txt')
    output = dynamics.solve(rho_0, times, results=results)

    # run TCL2 dynamics
    dynamics = qd.Redfield(ham, time_dependent=True)
    results = qd.Results(tobs=len(times), e_ops=e_ops, print_es=True, es_file='db_pops_tcl2.txt')
    output = dynamics.solve(rho_0, times, results=results)

if __name__ == '__main__':
    main()
