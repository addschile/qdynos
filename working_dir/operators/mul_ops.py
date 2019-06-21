import numpy as np
import scipy.sparse as sp
from operator import Operator
from numbers import Number

rng = np.random.RandomState(seed=1)

# testing operator with numpy array times a number
x = rng.rand(3,3)
x = Operator(x)
print(x)
y = x * 3.
print('op numpy * float')
print(y)
print('float * op numpy')
y = 3. * x
print(y)
x *= 3.
print(x)

# testing operator with sparse matrix times a number
x = sp.csr_matrix(rng.rand(3,3))
x = Operator(x)
print(x)
y = x * 3.
print('op scipy * float')
print(y)
print('float * op scipy')
y = 3. * x
print(y)

# testing operator with numpy array times a number
x = rng.rand(3,3)
x = Operator(x)
y = rng.rand(3,3)
print(x)
print('')
print(y)
print('')
print('numpy * op numpy')
z = x * y
print(z)
print('op numpy * numpy')
z = y * x
print(z)
