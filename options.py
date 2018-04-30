class Options(object):

    def __init__(self, method='rk4', 
        print_states=False, print_every=None, store_states=False, 
        ntraj=1000, block_avg=False, nblocks=10, norm_steps=1000, norm_tol=1.e-3, seed=None):
        # integrator options #
        assert(method in ['rk4','exact'])
        self.method = method

        # redfield options #
        self.markov_tol = markov_tol
        self.markov_time = markov_time

        # unraveling options #
        self.ntraj = ntraj
        self.block_avg = block_avg
        if block_avg:
            self.nblocks = nblocks
        self.norm_steps = norm_steps
        self.norm_tol = norm_tol
        self.print_states = print_states
        self.store_states = store_states
        self.seed = seed
