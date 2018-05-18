import numpy as np
from .utils import dag,is_vector,is_matrix
#import mpi4py

# TODO: add mpi functionality

# TODO
#def add_results():
#    """
#    Function that adds results from a results class to another results class.
#    Used for any dynamics that has to sample over trajectory realizations.
#
#    Parameters
#    ----------
#    c_results: Results class
#        Container results for averaging.
#    a_results: Results class
#        Results to add to the container.
#    """
#    return

# TODO
#def avg_results():
#    """
#    Function that averages results from a sum over other results. Used for any 
#    dynamics that has to sample over trajectory realizations.
#
#    Parameters
#    ----------
#    c_results: Results class
#        Container results for averaging.
#    """
#    return

class Results(object):
    """
    Results class that helps organize and print out relevant results.
    """

    def __init__(self, e_ops=None, map_ops=False, store_states=False, 
        print_states=False, states_file=None, jump_stats=False, every=1):
        """
        Initialize results class.

        Parameters
        ----------
        e_ops: list of np.ndarrays
        map_ops: bool
        store_states: bool
        print_states: bool
        states_file: string
        jump_stats: bool
        every: int
        """
        # how often do we compute results #
        self.every = every
        # expectation value containers #
        self.e_ops = e_ops
        self.expect = None
        if self.e_ops != None:
            self.expect = list( list() for i in range(len(self.e_ops)) )
        # mapping expectation value containers #
        self.map_ops = map_ops
        if self.map_ops:
            self.maps = list()
        # states containers #
        self.store_states = store_states
        self.states = None
        if self.store_states:
            self.states = list()
        # print states info #
        self.print_states = print_states
        self.states_file = None
        if self.print_states:
            self.states_file = states_file
        # jump statistics containers #
        self.jumps = None
        self.jump_times = None
        if jump_stats:
            self.jumps = list()

    def compute_expectation(self, state):
        """
        Computes expectation values.
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
        self.maps.append( self.map_function(state) )

    def print_state(self, ind, time, state):
        n = state.shape[0]
        if is_vector(state):
            if ind%self.print_every==0:
                self.state_file.write('%.8f '%(time))
                for i in range(n):
                    self.state_file.write('%s '%(state[i,0]))
                self.state_file.write('\n')
        elif is_matrix(state):
            if ind%self.print_every==0:
                self.state_file.write('%.8f\n'%(time))
                for i in range(n):
                    for j in range(n):
                        self.state_file.write('%s '%(state[i,j]))
                    self.state_file.write('\n')

    def analyze_state(self, ind, time, state):
        """
        Functional interface between dynamics and results class
        """
        if self.store_states:
            self.states.append( state.copy() )
        if self.print_states:
            self.print_state(time,state)
        if self.e_ops != None:
            self.compute_expectation(state)
        if self.map_ops:
            self.mapping_expect(state)

    def store_jumps(self, jumps):
        self.jumps.append( jumps.copy() )
