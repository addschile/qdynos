import numpy as np
from utils import is_vector,is_matrix

class Results(object):

    def __init__(self, e_ops=None, map_ops=None, store_states=False, jump_stats=False):
        """
        """
        # expectation value containers #
        self.e_ops = e_ops
        self.expect = None
        if self.e_ops != None:
            self.expect = list( list() for i in range(len(self.e_ops)) )
        # mapping expectation value containers #
        self.map_ops = map_ops
        if self.map_ops != None:
            self.maps = list( list() for i in range(len(self.map_ops)) )
        # states containers #
        self.store_states = store_states
        self.states = None
        if self.store_states:
            self.states = list()
        # jump statistics containers #
        self.jumps = None
        if jump_stats:
            self.jumps = list()

    def compute_expectation(self, state):
        """
        """
        if is_vector(state):
            for i,e_op in enumerate(self.e_ops):
                self.expect[i].append( np.dot(state.conj().T, np.dot(e_op,state))[0,0].real )
        elif is_matrix(state):
            for i,e_op in enumerate(self.e_ops):
                self.expect[i].append( np.trace( np.dot(e_op,state) ).real )
        else:
            raise ValueError("State needs to be either a vector or matrix")

    def analyze_state(self, state):
        if self.store_states:
            self.states.append( state.copy() )
        self.compute_expectation(state)

    def store_jumps(self, jumps):
        self.jumps.append( jumps.copy() )
