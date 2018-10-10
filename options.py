import numpy as np
from .log import Restart

class Options(object):
    """
    Options class for specific non-default options for dynamics classes.
    """

    def __init__(self, verbose=True, really_verbose=False, progress=True,
                 method='rk4', ntraj=1000, block_avg=False, nblocks=10,
                 norm_steps=1000, norm_tol=1.e-3, seed=None, markov_time=np.inf,
                 restart=False, restart_file=None, from_restart=False):

        # program run options #
        self.verbose = verbose
        self.really_verbose = really_verbose
        self.progress = progress 

        # integrator options #
        assert(method in ['rk4','exact'])
        self.method = method

        # TCL2 options #
        self.markov_time = markov_time

        # unraveling options #
        self.ntraj = ntraj
        self.block_avg = block_avg
        if block_avg:
            self.nblocks = nblocks
        self.norm_steps = norm_steps
        self.norm_tol = norm_tol
        self.seed = seed

        # restart options #
        self.restart = restart
        self.restart_file = None
        if self.restart: 
            if restart_file == None:
                # instantiate generic restart class
                self.restart_file = Restart("")
            else:
                # instantiate restart class with specific file name
                self.restart_file = Restart(restart_file)

        self.from_restart = from_restart
        if restart_file == None:
            self.restart_file = 
        else:
            self.restart_file = 
