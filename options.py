class Options(object):
    """
    Options class for specific non-default options for dynamics classes.
    """

    def __init__(self, verbose=False, method='rk4', ntraj=1000, block_avg=False, 
        nblocks=10, norm_steps=1000, norm_tol=1.e-3, seed=None):

        # program run options #
        self.verbose = verbose

        # integrator options #
        assert(method in ['rk4','exact'])
        self.method = method

        # unraveling options #
        self.ntraj = ntraj
        self.block_avg = block_avg
        if block_avg:
            self.nblocks = nblocks
        self.norm_steps = norm_steps
        self.norm_tol = norm_tol
        self.seed = seed
