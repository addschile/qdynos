import numpy as np
from qdynos.utils import dag,is_vector,is_matrix

class Results(object):

    def __init__(self, e_ops=None, expects_every=1, map_ops=None, maps_every=1, 
        store_states=False, store_every=1, print_states=False, states_file=None, 
        print_every=None, jump_stats=False):
        """
        """
        # expectation value containers #
        self.e_ops = e_ops
        self.expect = None
        self.expects_every = expects_every
        if self.e_ops != None:
            self.expect = list( list() for i in range(len(self.e_ops)) )
        # mapping expectation value containers #
        self.map_ops = map_ops
        self.maps_every = maps_every
        if self.map_ops != None:
            self.maps = list( list() for i in range(len(self.map_ops)) )
        # states containers #
        self.store_states = store_states
        self.states = None
        if self.store_states:
            self.states = list()
        # print states info #
        self.print_states = print_states
        self.states_file = None
        self.print_every = None
        if self.print_states:
            self.states_file = states_file
            self.print_every = print_every
        # jump statistics containers #
        self.jumps = None
        self.jump_times = None
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

    def mapping_expect(self, state):
        n = state.shape[0]
        if is_vector(state):
            for i,u in enumerate(self.map_ops):
                map_expect = np.zeros(n)
                for j in range(n):
                    ket = u[:,j]
                    da = np.dot(dag(ket),state)[0]
                    map_expect[j] = (np.conj(da)*da).real
                self.maps[i].append( map_expect.copy() )
        elif is_matrix(state):
            for i,u in enumerate(self.map_ops):
                map_expect = np.zeros(n)
                for j in range(n):
                    ket = u[:,j]
                    map_expect[j] = np.dot(dag(ket),np.dot(state,ket))[0,0].real
                self.maps[i].append( map_expect.copy() )

    def print_state(self, state):
        n = state.shape[0]
        if is_vector(state):
        elif is_matrix(state):

    def analyze_state(self, ind, time, state):
        if self.store_states:
            if ind%self.store_every==0:
                self.states.append( state.copy() )
        if self.print_states:
            if ind%self.print_every==0:
                self.print_state(time,state)
        if self.e_ops != None:
            if ind%self.expects_every==0:
                self.compute_expectation(state)
        if self.map_ops != None:
            if ind%self.maps_every==0:
                self.mapping_expect(state)

    def store_jumps(self, jumps):
        self.jumps.append( jumps.copy() )
