from abc import ABC, abstractmethod

class Dynamics(ABC):
  """Abstract base class for dynamics.
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
    """Sets up options class and results class for dynamics."""

  @abstractmethod
  def eom(self, state, order):
    """Define equation of motion for the dynamics class."""

  @abstractmethod
  def solve(self, state, times):
    """Solve the equations of motion for the dynamics class."""
