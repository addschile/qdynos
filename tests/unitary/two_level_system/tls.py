import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/addisonschile/Software')
from qdynos.hamiltonian import Hamiltonian
from qdynos.unitary import UnitaryWF
from qdynos.results import Results
from qdynos.options import Options

eps   = 0.0
delta = 1.0
sigz = np.array([[1.,0.],[0.,-1.]])
sigx = np.array([[0.,1.],[1.,0.]])
H = eps*sigz + delta*sigx

p1 = np.array([[1.,0.],[0.,0.]])
p2 = np.array([[0.,0.],[0.,1.]])

psi0 = np.array([[1.],[0.]],dtype=complex)
psi = psi0.copy()
times = np.arange(0.0, 14.0, 0.05)
prop = expm(-1.j*H*0.05)
expect = np.zeros((len(times),3))
f = open('es_me.txt','w')
for i in range(len(times)):
    expect[i,0] = times[i]
    expect[i,1] = (np.conj(psi[0,0])*psi[0,0]).real
    expect[i,2] = (np.conj(psi[1,0])*psi[1,0]).real
    f.write('%.8f %.8f %.8f\n'%(times[i],expect[i,1],expect[i,2]))
    psi = np.dot(prop,psi)
f.close()

ham = Hamiltonian(H)
dynamics = UnitaryWF(ham)
output = dynamics.solve(psi0, times, eig=False, options=Options(method='rk4'), results=Results(tobs=len(times), e_ops=[p1,p2], print_es=True, es_file='tls.txt'))

plt.plot(times,output.expect.T[:,0])#,output.expect[:,1])
plt.plot(expect[:,0],expect[:,1])
plt.show()
