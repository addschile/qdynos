import numpy as np

class Operator(object):
  """
  """

  def __init__(self, op_mat=None, basis=None, op_function=None, herm=True):
    """
    """
    if op_mat==None:
      self.operator = create_operator()
    else:
      self.operator = op_mat
    self.is_herm = self.is_hermitian()
    if herm != self.is_herm:
      raise ValueError('Operator does not match hermiticity requirement')

  def __add__(self, other):
#  # TODO make operator from basis
#  def create_operator():
#    for i in range():
#      for j in range():
#    return op

  # TODO understand this
  @property
  def is_hermitian(self):
    return np.allclose(self.operator.conj().T == self.operator)

  def eigensystem(self):
    self.ev,ek = numpy.linalg.eigh(self.H)
    self.ek = Operator(ek)

  def to_eigenbasis(self, op)
    if op.is_vector():
      return np.dot(self.ek.dagger(), op)
    elif op.is_matrix():
      return utils.matrix_multiply(self.ek.dagger(), op, self.ek)
    # TODO tensor transformation?
    else:
      raise AttributeError('Not a valid operator')

  def from_eigenbasis(self, op)
    if op.is_vector():
      return np.dot(self.ek, op)
    elif op.is_matrix():
      return matrix_multiply(self.ek, op, self.ek.dagger())
    else:
      raise AttributeError('Not a valid operator')

  def expectation_value(self, state):
    return np.trace( np.dot(self.operator , state) )
#    # TODO 
#    if :
#      return np.trace( np.dot(self.operator , state) )
#    elif :
#      return 
