import numpy as np
import scipy.sparse as sp
from operator import Operator

# make numpy matrix
x = np.random.rand(3,3)
x = Operator(x)
print(x)
print(x[1,2])
x[1,2] = 3.0
print(x[1,2])
# make scipy sparse matrix
x = sp.csr_matrix(np.random.rand(3,3))
x = Operator(x)
print(x)
print(x[1,2])
x[1,2] = 3.0
print(x[1,2])
# make numpy vector 
x = np.random.rand(3)
x = Operator(x)
print(x)
print(x[1])
x[1] = 3.0
print(x[1])
# make numpy vector 
x = sp.csr_matrix(np.random.rand(3)[:,np.newaxis])
x = Operator(x)
print(x)
print(x.op.toarray())
print(x[1])
x[1] = 3.0
print(x[1])
