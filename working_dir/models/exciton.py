from __future__ import print_function,absolute_import

import numpy as np
import .bath
from ..hamiltonian import Hamiltonian
from .options import Options
from .results import Results
from .log import *

class SiteExciton(model):

    def __init__(self, ham, bath_params=None):
        """
        """
        ### system parameters ###
        self.nsite = ham.shape[0]
        ### set up baths ###
        if bath_params!=None:
            baths = []
            self.nbath = bath_params['nbath']
            if bath_params['same']:
                # TODO how to import the right bath type
            for i in range(self.nsite):
                
        ### set up Hamiltonian class ###
        self.ham = Hamiltonian(ham, baths=baths, hbar=)

    def run_dynamics()
        return self.dynamics.solve()
