import numpy as np

class Options(object):
    """
    Options class for specific non-default options for dynamics classes.
    """

    def __init__(self, verbose=True, really_verbose=False, progress=True,
                 method="rk4", space="hilbert", ntraj=1000, block_avg=False, 
                 nblocks=10, norm_steps=1000, norm_tol=1.e-3, seed=None, 
                 markov_time=np.inf, restart_file=None, restart=False):

        # program run options #
        self.verbose = verbose
        self.really_verbose = really_verbose
        self.progress = progress 

        # integrator options #
        assert(method in ['rk4','exact'])
        self.method = method

        # liouville or hilbert space #
        if self.method != "exact":
            self.space = space
        else:
            self.space = "liouville"

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

        # unraveling options #
        self.restart_file = restart_file
        self.restart = restart
