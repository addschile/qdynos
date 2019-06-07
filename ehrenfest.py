from abc import ABCMeta, abstractmethod

class Dynamics():
class Ehrenfest(Dynamics):
    """
    Abstract base class for dynamics.
    """
    __metaclass__=ABCMeta
    def __init__(self, hamiltonian):
        """Initialize the Dynamics class.

        Parameters
        ----------
        hamiltonian : Hamiltonian
        """
        super(Ehrenfest, self).__init__(ham)
        self.ham = hamiltonian

    @abstractmethod
    def setup(self, options, results):
        """
        Sets up options class and results class for dynamics.
        """

    @abstractmethod
    def eom(self, state, order):
        """
        Define equation of motion for the dynamics class.
        """

    @abstractmethod
    def solve(self, state):
        """
        Solve the equations of motion for the dynamics class.
        """
