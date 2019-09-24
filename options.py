import numpy as np

class Options(object):
    """
    Options class for specific non-default options for dynamics classes.
    """

    def __init__(self, verbose=True, really_verbose=False, progress=True,
                 method="rk4", space="hilbert", norm_tol=0.99, nlanczos=20,
                 lanczos_lowmem=False, print_coup_ops=False, coup_ops_file=None, 
                 print_decomp=False, decomp_file=None, ham_file=None, ntraj=1000, 
                 traj_results=False, traj_results_file=None, traj_states=False, 
                 traj_states_file=None,traj_states_every=1, block_avg=False, nblocks=10,
                 jump_time_steps=1000, jump_time_tol=1.e-3, seed=None, 
                 markov_time=np.inf, unraveling=False, which_unraveling='jump', 
                 restart_file=None, restart=False):

        # program run options #
        self.verbose = verbose
        self.really_verbose = really_verbose
        self.progress = progress 

        # integrator options #
        assert(method in ['euler','rk4','exact','lanczos','arnoldi'])
        self.method = method

        # unitary evolution options #
        # TODO check default, that might be a bit ridiculous
        self.norm_tol = norm_tol
        self.nlanczos = nlanczos
        self.lanczos_lowmem = lanczos_lowmem

        # liouville or hilbert space #
        self.space = space

        # Redfield/TCL2 options #
        self.print_coup_ops = print_coup_ops
        if self.print_coup_ops:
            if coup_ops_file == None:
                self.coup_ops_file = ''
            else:
                self.coup_ops_file = coup_ops_file
        else:
            self.coup_ops_file = coup_ops_file
        # TCL2 options #
        self.markov_time = markov_time

        # frozen mode options #
        self.print_decomp = print_decomp
        self._decomp_file = decomp_file
        self.decomp_file = None
        self.ham_file = ham_file

        # trajectory sampling options #
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
        self.unraveling = unraveling
        self.which_unraveling = which_unraveling
        if self.unraveling:
            if self.which_unraveling=='jump':
                if self.method == 'rk4':
                    self.method = 'exact'
        self.jump_time_steps = jump_time_steps
        self.jump_time_tol = jump_time_tol

        # NOTE in progress: restart options #
        self.restart_file = restart_file
        self.restart = restart
