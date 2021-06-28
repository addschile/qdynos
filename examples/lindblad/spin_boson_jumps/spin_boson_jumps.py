import numpy as np
import qdynos as qd

def main():

    # parameters
    eps   = 0.0
    delta = 1.0
    # strong coupling
    Gam   = 1.0
    dt    = 0.01
    times = np.arange(0.0,10.0,dt)

    # operators
    sigx = np.array([[0.,1.],[1.,0.]])
    sigz = np.array([[1.,0.],[0.,-1.]])
    H = 0.5*delta*sigx + 0.5*eps*sigz
    L = sigz.copy()
    p1 = np.array([[1.,0.],[0.,0.]])
    p2 = np.array([[0.,0.],[0.,1.]])

    # make initial condition
    rho = np.zeros((2,1),dtype=complex)
    rho[0,0] = 1.

    ham = qd.Hamiltonian(H)
    dynamics = qd.Lindblad(ham)
    results = qd.Results(tobs=len(times), e_ops=[L.copy(),p1,p2])
    options = qd.Options(unraveling=True)
    output = dynamics.solve(rho, times, Gam, L.copy(), ntraj=1000, options=options, results=results)
    es_file = 'sb_pops_jumps.txt'
    output.print_expectation(es_file=es_file)

if __name__=="__main__":
    main()
