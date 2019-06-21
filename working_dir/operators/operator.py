import numpy as np
import scipy.sparse as sp
from types import FunctionType
from numbers import Number

# TODO with the whole dot thang
#from utils import is_hermitian

class Operator(object):
    """
    """

    __array_priority__ = 10000

    # TODO I'll need to go back through and set all the attributes of the
    # class consistently when I init things
    def __init__(self, op_mat=None, nstates=None):
        """
        """
        self.op = op_mat
        self.type = type(op_mat)

        # TODO
        #assert(self.type in [FunctionType,np.ndarray,sp.spmatrix,NoneType])
        #assert(self.type in [np.ndarray,sp.spmatrix,NoneType])

        if isinstance(self.op, (np.ndarray, sp.spmatrix)):
            self.dims = self.op.shape
            if len(self.dims) == 1:
                self.op = self.op[:, np.newaxis]
                self.dims = self.op.shape
            if nstates is None:
                self.nstates = max(self.dims)
            else:
                self.nstates = nstates
            # TODO
            #self.herm = is_hermitian(self.op)

    def __str__(self):
        s = "Qdynos Operator of type %s\n"%(str(self.type))
        if self.op is None:
            pass
        elif isinstance(self.op, (np.ndarray, sp.spmatrix)):
            s += "Dimensions %d x %d\n"%(self.dims[0],self.dims[1])
            s += str(self.op)
            s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

    # TODO
    #def __eq__(self, op):

    def __add__(self, op):
        """Add with Operator on the left
        Operator + op
        """
        if isinstance(op, Operator):
            if isinstance(self.op, (np.ndarray, sp.spmatrix)):
                if self.type == op.type:
                    # no type conversion needed
                    opout = self.op + op.op
                else:
                    # default is to convert dense matrix to sparse because
                    # likely the sparse matrix is too big to be dense
                    if isinstance(self.op, sp.spmatrix):
                        opout = self.op + self.convert(op.op)
                    elif isinstance(op.op, sp.spmatrix):
                        opout = op.convert(self.op) + op.op
        elif isinstance(op, (np.ndarray, sp.spmatrix)):
            return self.__add__(Operator(op))
        elif isinstance(op, Number):
            return self.__add__(op*eye(self.dims[0],op_type=self.type))
        else:
            raise AttributeError('Invalid type for added op')

        return Operator(opout)

    def __radd__(self, op):
        """Add with Operator on the right
        Operator + op
        """
        return self.__add__(op)

    def __sub__(self, op):
        """Subtract with Operator on the left
        Operator - op
        """
        return self.__add__(-op)

    def __rsub__(self, op):
        """Subtract with Operator on the right
        Operator - op
        """
        return (-self).__add__(op)

    def __mul__(self, op):
        """Multiply with Operator on the left
        Operator * op
        """
        if isinstance(op, Operator):
            return self.dot(op)
        elif isinstance(op, (np.ndarray, sp.spmatrix)):
            op = Operator(op)
            if self.type == op.type:
                # no type conversion needed
                opout = self.op*op.op
            else:
                # default is to keep operator the same type
                opout = self.op*self.convert(op).op
        elif isinstance(op, Number):
            opout = self.op*op
        else:
            raise AttributeError('Invalid type for multiplied op')
        return Operator(opout)

    def __rmul__(self, op):
        """Multiply with Operator on the right
        op * Operator
        """
        return self.__mul__(op)

    def __truediv__(self, op):
        return self.__div__(op)

    def __div__(self, op):
        """Division of Operator by number or element-wise by matrix"""
        if isinstance(op, Operator):
            raise NotImplementedError
        elif isinstance(op, (np.ndarray, sp.spmatrix)):
            op = Operator(op)
            if self.type == op.type:
                # no type conversion needed
                opout = self.op/op.op
            else:
                # default is to keep operator the same type
                opout = self.op/self.convert(op).op
        elif isinstance(op, Number):
            opout = self.op/op
        else:
            raise AttributeError('Invalid type for multiplied op')
        return Operator(opout)

    def __neg__(self):
        """Negative operator"""
        return Operator(-self.op)

    def __getitem__(self, ind):
        """Get the element of Operator"""
        if isinstance(ind, Number):
            if self.is_vector:
                if self.dims[1] == 1:
                    return self.op[ind,0]
                elif self.dims[0] == 1:
                    return self.op[0,ind]
            else:
                return self.op[ind]
        elif isinstance(ind, list):
            return self.op[ind]
        else:
            return AttributeError('Invalid type for ind')

    def __setitem__(self, ind, op):
        """Set the element of Operator"""
        if isinstance(op, Number):
            self.op[ind] = op
        else:
            raise AttributeError('Invalid type for setting op')

    # TODO
    #@property
    #def is_hermitian(self):
    #    if not self.is_herm is None:
    #        return self.is_herm

    #    self.is_herm = is_hermitian(self.op)
    #    return is_herm

    # TODO decorate with property
    @property
    def is_vector(self):
        """Check whether operator is vector"""
        axes = self.op.shape
        if axes[0] != axes[1]:
            if axes[0] == 1 or axes[1] == 1:
                return True
            else:
                raise AttributeError('Invalid shape for Operator class: '+str(axes))
        else:
            return False

    @is_vector.setter
    def is_vector(self):
        """Check whether operator is vector"""
        axes = self.op.shape
        if axes[0] != axes[1]:
            if axes[0] == 1 or axes[1] == 1:
                return True
            else:
                raise AttributeError('Invalid shape for Operator class: '+str(axes))
        else:
            return False

    # TODO decorate with property
    def is_matrix(self):
        """Check whether operator is matrix"""
        axes = self.op.shape
        if axes[0] == axes[1]:
            return True
        else:
            if not (axes[0] == 1 or axes[1] == 1):
                raise AttributeError('Invalid shape for Operator class: '+str(axes))
            else:
                return False

    def dag(self):
        """Return conjugate transpose"""
        return Operator(self.op.conj().T)

    def convert(self, op):
        """Convert types for consistency"""
        if isinstance(op, (np.ndarray, sp.spmatrix)):
            if isinstance(self.op, np.ndarray):
                return Operator(op.toarray())
            elif isinstance(self.op, sp.csr_matrix):
                return Operator(sp.csr_matrix(op))
            elif isinstance(self.op, sp.csc_matrix):
                return Operator(sp.csc_matrix(op))
            elif isinstance(self.op, sp.lil_matrix):
                return Operator(sp.csr_matrix(op))
            else:
                raise AttributeError('Invalid type for converting operator: '+type(self.op))
        else:
            raise AttributeError('Invalid type for converted operator: '+type(op))

    def dot(self, op):
        """Matrix-matrix and Matrix-Vector multiplcation.

        Note: some things in here will obviously be slow if repeated a bunch, 
        but the idea is that in the computationally heavy parts of the code they
        won't ever be called
        """
        # convert op to Operator class
        if not isinstance(op, Operator):
            op = Operator(op)

        # check type of operator being dotted
        if not isinstance(op.op, (np.ndarray, sp.spmatrix)):
            raise AttributeError('Invalid type for dotted operator')

        # TODO FunctionType
        #if self.type == FunctionType:
        #    if isinstance(op, Operator):
        if isinstance(self.op, (np.ndarray, sp.spmatrix)):
            if self.type == op.type:
                # no type conversion needed
                opout = self.op.dot(op.op)
            else:
                # default is to convert dense matrix to sparse because
                # likely the sparse matrix is too big to be dense
                if isinstance(self.op, sp.spmatrix):
                    return self.dot(self.convert(op.op))
                elif isinstance(op.op, sp.spmatrix):
                    return op.convert(self.op).dot(op.op)
        else:
            raise AttributeError('Invalid type for dotting operator: '+type(self.op))

        return Operator(opout)

    def eigensystem(self):
        """Compute eigenvalues and eigenvectors of operator"""
        if isinstance(self.op, np.ndarray):
            self.ev,ek = numpy.linalg.eigh(self.op)
            self.ek = Operator(ek)
        else:
            raise AttributeError('Operator must be np.ndarray for full eigensystem')

    def to_eigenbasis(self, op):
        """Transform another operator to the eigenbasis of this operator."""
        if isinstance(self.op, np.ndarray):
            # convert op to Operator class
            if not isinstance(op, Operator):
                op = Operator(op)
            # check whether converting vector or matrix
            if op.is_vector():
                return self.ek.dag().dot(op)[:self.nstates,:]
            elif op.is_matrix():
                return self.ek.dag().dot(op.dot(self.ek))[:self.nstates,:self.nstates]
        else:
            raise AttributeError('Operator must be np.ndarray for full eigensystem')

    def from_eigenbasis(self, op):
        """Transform another operator from the eigenbasis of this operator."""
        if isinstance(self.op, np.ndarray):
            # convert op to Operator class
            if not isinstance(op, Operator):
                op = Operator(op)
            # check whether converting vector or matrix
            if op.is_vector():
                return self.ek.dot(op)
            elif op.is_matrix():
                return self.ek.dot(op.dot(self.ek.dag()))
        else:
            raise AttributeError('Operator must be np.ndarray for full eigensystem')

            return self.__add__(op*eye(self.dims[0],op_type=self.type))

    def commutator(self, op):
        """Compute commutator with another operator."""
        return self.dot(op) - op.dot(self)

def eye(n, op_type=np.ndarray, dtype=float):
    if op_type == np.ndarray:
        eyeout = np.eye(n, dtype=dtype)
    elif op_type == sp.csr_matrix:
        eyeout = sp.eye(n, dtype=dtype, fmt='csr')
    elif op_type == sp.csc_matrix:
        eyeout = sp.eye(n, dtype=dtype, fmt='csc')
    elif op_type == sp.lil_matrix:
        eyeout = sp.eye(n, dtype=dtype, fmt='lil')
    return Operator(eyeout)
