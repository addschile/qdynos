import sys
sys.path.append('/Users/addisonschile/Software')
import numpy as np
from qdynos.hamiltonian import Hamiltonian
from qdynos.results import Results
from qdynos.lindblad import Lindblad

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
    H = -0.5*delta*sigx + 0.5*eps*sigz
    L = sigz.copy()

    # make initial condition
    rho = np.zeros((2,2),dtype=complex)
    rho[0,0] = 1.

    ham = Hamiltonian(H)
    dynamics = Lindblad(ham)
    es_file = 'sb_qdynos.txt'
    results = Results(tobs=len(times), e_ops=[L.copy()], print_es=True, es_file=es_file)
    output = dynamics.solve(rho, times, Gam, L.copy(), results=results)

if __name__=="__main__":
    main()
