from abc import ABCMeta, abstractmethod

class Dynamics(metaclass=ABCMeta):
    """
    Abstract base class for dynamics.
    """

    def __init__(self, hamiltonian):
        """Initialize the Dynamics class.

        Parameters
        ----------
        hamiltonian : Hamiltonian
        """
        self.ham = hamiltonian

    @abstractmethod
    def setup(self, options, results):
        """
        Sets up options class and results class for dynamics.
        """

    @abstractmethod
    def eom(self, state):
        """
        Define equation of motion for the dynamics class.
        """
