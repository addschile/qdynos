import numpy as np
import scipy.sparse as sp
from operator import Operator

rng = np.random.RandomState(seed=1)

print('op numpy * op numpy')
x = rng.rand(3,3)
x = Operator(x)
y = x + 3.
print(x)
print(y)
x += 3.
print(x)
#y = rng.rand(3,3)
#y = Operator(y)
#z = x.dot(y)
#print(z)
#
#print('op numpy * numpy')
#x = rng.rand(3,3)
#x = Operator(x)
#y = rng.rand(3,3)
#z = x.dot(y)
#print(z)
#
#print('op scipy * op scipy')
#x = sp.csr_matrix(rng.rand(3,3))
#x = Operator(x)
#y = sp.csr_matrix(rng.rand(3,3))
#y = Operator(y)
#z = x.dot(y)
#print(z)
#
#print('op scipy * scipy')
#x = sp.csr_matrix(rng.rand(3,3))
#x = Operator(x)
#y = sp.csr_matrix(rng.rand(3,3))
#z = x.dot(y)
#print(z)
#
#print('op scipy * op numpy')
#x = sp.csr_matrix(rng.rand(3,3))
#x = Operator(x)
#y = rng.rand(3,3)
#y = Operator(y)
#z = x.dot(y)
#print(z)
#
#print('op numpy * op scipy')
#x = rng.rand(3,3)
#x = Operator(x)
#y = sp.csr_matrix(rng.rand(3,3))
#y = Operator(y)
#z = x.dot(y)
#print(z)
#
##print('left is operator numpy mat, right is numpy mat')
##x = sp.csr_matrix(rng.rand(3,3))
##x = Operator(x)
##y = sp.csr_matrix(rng.rand(3,3))
##z = x.dot(y)
##print(z)
##
### This should break, but need to figure out how to make it chill
#### numpy matrix * Operator numpy matrix
###print('left is numpy mat, right is operator numpy mat')
###x = rng.rand(3,3)
###y = rng.rand(3,3)
###y = Operator(y)
###z = x.dot(y)
###print(z)
