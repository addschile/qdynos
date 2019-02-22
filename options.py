import numpy as np

class Options(object):
    """
    Options class for specific non-default options for dynamics classes.
    """

    def __init__(self, verbose=True, really_verbose=False, progress=True,
                 method="rk4", space="hilbert", print_coup_ops=False, 
                 coup_ops_file=None, print_decomp=False, decomp_file=None, 
                 ham_file=None, ntraj=1000, traj_results=False, 
                 traj_results_file=None, traj_states=False, traj_states_file=None,
                 traj_states_every=1, block_avg=False, nblocks=10, norm_steps=1000, 
                 norm_tol=1.e-3, seed=None, markov_time=np.inf, restart_file=None, 
                 restart=False):

        # program run options #
        self.verbose = verbose
        self.really_verbose = really_verbose
        self.progress = progress 

        # integrator options #
        assert(method in ['rk4','exact'])
        self.method = method

        # liouville or hilbert space #
        self.space = space

        # Redfield/TCL2 options #
        self.print_coup_ops = print_coup_ops
        self.coup_ops_file = coup_ops_file
        # TCL2 options #
        self.markov_time = markov_time

        # frozen mode options #
        self.print_decomp = print_decomp
        self._decomp_file = decomp_file
        self.decomp_file = None
        self.ham_file = ham_file

        # sampling options #
        self.ntraj = ntraj
        self.traj_results = traj_results
        self.traj_results_file = traj_results_file
        self.traj_states = traj_states
        self.traj_states_file = traj_states_file
        self.traj_states_every = traj_states_every
        self.block_avg = block_avg
        if block_avg:
            self.nblocks = nblocks
        self.seed = seed

        # unraveling options #
        self.norm_steps = norm_steps
        self.norm_tol = norm_tol

        # NOTE in progress: restart options #
        self.restart_file = restart_file
        self.restart = restart
