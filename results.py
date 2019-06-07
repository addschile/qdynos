from __future__ import print_function,absolute_import
import numpy as np
from .utils import dag,is_vector,is_matrix
from .log import print_basic

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
    if results.e_ops != None:
        results.expect /= float(ntraj)
    if results.map_ops:
        results.maps /= float(ntraj)
    return results

class Results(object):
    """
    Results class that helps organize and print out relevant results.
    """

    def __init__(self, tobs=None, e_ops=None, print_es=False, es_file=None, 
                 map_ops=False, store_states=False, print_final=False, 
                 final_file=None, final_every=1, print_states=False, 
                 states_file=None, states_every=1, jump_stats=False, every=1):
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
        self.expect = None
        self.print_es = print_es
        self.fes = None
        if e_ops != None:
            if isinstance(e_ops, list):
                self.e_ops = e_ops
            else:
                self.e_ops = [e_ops]
            if self.print_es:
                if es_file==None:
                    self.fes = open("output.dat","w")
                else:
                    self.fes = open(es_file,"w")
            self.expect = np.zeros((len(self.e_ops),tobs))
        else:
            self.e_ops = e_ops
        # mapping expectation value containers #
        self.map_ops = map_ops
        if self.map_ops:
            self.maps = list()
        # states containers #
        self.store_states = store_states
        self.states = None
        if self.store_states:
            self.states = list()
        # print dynamic state info #
        self.print_final = print_final
        self.final_file = None
        if self.print_final:
            self.final_file = final_file
        self.final_every = final_every
        # print states info #
        self.print_states = print_states
        self.states_file = None
        if self.print_states:
            self.states_file = states_file
        self.states_every = states_every
        # jump statistics containers #
        self.jump_stats = jump_stats
        self.jumps = None
        self.jump_times = None
        if self.jump_stats:
            self.jumps = list()

    def close_down(self):
        if self.fes != None and self.fes.closed == False:
            self.fes.close()

    def compute_expectation(self, ind, state):
        """
        Computes expectation values.
        """
        if is_vector(state):
            for i,e_op in enumerate(self.e_ops):
                self.expect[i,ind] = np.dot(state.conj().T, np.dot(e_op,state))[0,0].real
                if self.print_es:
                    self.fes.write('%.8f '%(self.expect[i,ind]))
        elif is_matrix(state):
            for i,e_op in enumerate(self.e_ops):
                self.expect[i,ind] = np.trace( np.dot(e_op,state) ).real
                if self.print_es:
                    self.fes.write('%.8f '%(self.expect[i,ind]))
        else:
            raise ValueError("State needs to be either a vector or matrix")

    def mapping_expect(self, state):
        self.maps.append( self.map_function(state) )

    def print_final_state(self, state):
        np.save(self.final_file, state)

    def print_state(self, ind, time, state):
        np.save(self.states_file+"_"+str(ind), state)

    def analyze_state(self, ind, time, state):
        """
        Functional interface between dynamics and results class
        """
        if self.store_states:
            self.states.append( state.copy() )
        if self.print_final:
            if ind%self.final_every==0:
                self.print_final_state(state)
                print_basic("last state printed: %d, %.8f"%(ind,time))
        if self.print_states:
            if ind%self.states_every==0:
                self.print_state(ind, time, state)
        if self.e_ops != None:
            if self.print_es: 
                self.fes.write('%.8f '%(time))
            self.compute_expectation(ind, state)
            if self.print_es: 
                self.fes.write('\n')
                self.fes.flush()
        if self.map_ops:
            self.mapping_expect(state)
        if ind==(self.tobs-1):
            self.close_down()

    def store_jumps(self, njumps, jumps):
        self.jumps.append( [njumps, jumps.copy()] )

    def print_expectation(self, es_file=None):
        if es_file==None:
            np.savetxt('expectation_values', self.expect.T)
        else:
            np.savetxt(es_file, self.expect.T)
