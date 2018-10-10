from __future__ import print_function,absolute_import
import numpy as np
from .utils import dag,is_vector,is_matrix

def add_results(results1, results2, weight=None):
    """
    Function that adds results from a results class to another results class.
    Used for any dynamics that has to sample over trajectory realizations.

    Parameters
    ----------
    results1: Results class
        Container results for averaging.
    results2: Results class
        Results to add to the container.
    weight: TODO
    """
    if results1==None:
        from copy import deepcopy
        return deepcopy(results2)
    else:
        if results1.store_states:
            results1.states += results2.states
        if results1.e_ops != None:
            results1.expect += results2.expect
        if results1.map_ops:
            results1.maps += results2.maps
        if results1.jumps != None:
            results1.jumps += results2.jumps
        return results1

def avg_results(ntraj, results):
    #if results.store_states:
    #    results.states /= float(ntraj)
    if results.e_ops != None:
        results.expect /= float(ntraj)
    if results.map_ops:
        results.maps /= float(ntraj)
    return results

class Results(object):
    """
    Results class that helps organize and print out relevant results.
    """

    def __init__(self, tobs=None, e_ops=None, print_es=False, es_file=None, map_ops=False, store_states=False, 
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
        self.tobs = tobs
        # how often do we compute results #
        self.every = every
        # expectation value containers #
        self.e_ops = e_ops
        self.expect = None
        self.print_es = print_es
        self.fes = None
        if self.e_ops != None:
            if self.print_es:
                if es_file==None:
                    self.fes = open('output.dat','w', buffering=1)
                else:
                    self.fes = open(es_file,'w', buffering=1)
            else:
                self.expect = np.zeros((len(self.e_ops),tobs))
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
            self.states_file = open(states_file, 'w', buffering=1)
        # jump statistics containers #
        self.jumps = None
        self.jump_times = None
        if jump_stats:
            self.jumps = list()

    def close_down(self):
        if self.fes != None and self.fes.closed == False:
            self.fes.close()
        if self.states_file != None and self.states_file.closed == False:
            self.states_file.close()

    def compute_expectation(self, ind, state):
        """
        Computes expectation values.
        """
        if self.print_es:
            if is_vector(state):
                for i,e_op in enumerate(self.e_ops):
                    self.fes.write('%.8f '%(np.dot(state.conj().T, np.dot(e_op,state))[0,0].real))
                self.fes.write('\n')
            elif is_matrix(state):
                for i,e_op in enumerate(self.e_ops):
                    self.fes.write('%.8f '%(np.trace( np.dot(e_op,state) ).real))
                self.fes.write('\n')
            else:
                raise ValueError("State needs to be either a vector or matrix")
        else:
            if is_vector(state):
                for i,e_op in enumerate(self.e_ops):
                    self.expect[i,ind] = np.dot(state.conj().T, np.dot(e_op,state))[0,0].real
            elif is_matrix(state):
                for i,e_op in enumerate(self.e_ops):
                    self.expect[i,ind] = np.trace( np.dot(e_op,state) ).real
            else:
                raise ValueError("State needs to be either a vector or matrix")

    def mapping_expect(self, state):
        self.maps.append( self.map_function(state) )

    def print_state(self, ind, time, state):
        n = state.shape[0]
        if is_vector(state):
            self.states_file.write('%.8f '%(time))
            for i in range(n):
                self.states_file.write('%s '%(state[i,0]))
            self.states_file.write('\n')
        elif is_matrix(state):
            self.states_file.write('%.8f\n'%(time))
            for i in range(n):
                for j in range(n):
                    self.states_file.write('%s '%(state[i,j]))
                self.states_file.write('\n')

    def analyze_state(self, ind, time, state):
        """
        Functional interface between dynamics and results class
        """
        if self.store_states:
            self.states.append( state.copy() )
        if self.print_states:
            self.print_state(ind, time,state)
        if self.e_ops != None:
            if self.print_es: self.fes.write('%.8f '%(time))
            self.compute_expectation(ind, state)
        if self.map_ops:
            self.mapping_expect(state)
        if ind==(self.tobs-1):
            self.close_down()

    def store_jumps(self, njumps, jumps):
        self.jumps.append( [njumps, jumps.copy()] )
