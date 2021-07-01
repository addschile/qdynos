QDYNOS - A not so prehistoric way to do quantum dynamics in python!

This is a small package that can perform quantum dynamics on model systems that are represented as vectors or matrices. For a code that can perform MCTDH calculations in python see my [pymctdh](https://github.com/addschile/pymctdh) code!

The code is built around a Dynamics abstract class. This class serves as the structure for defining new dynamics methods for both open and closed quantum systems.

Additional generic classes (though not abstract) take care of the remaining parts of the calculations:

- Hamiltonian - stores the Hamiltonian of the system and takes care of eigenspectrum.

- Bath - for open systems problems, this class defines the spectral density and calculates relevant quantities like the bath correlation function. This is a generic class and subclasses of this class (e.g. DebyeBath) implement the specific spectral density required for the calculation.

- Integrator - performs some basic propagation, but calls equations of motion from the dynamics class.

- Results - handles calculation of expectation values and output of states. Written in a manner such that results can be added and averaged for stochastic propagation methods.

- Options - simple class that contains options flags and parameters for dynamics.

Currently, the only dynamics methods that are implemented are the following:

- Wavefunction propagation via time-dependent Schrodinger equation

- Density matrix propagation via Liouville-von Neumann equation

- Redfield theory / TCL2

- Lindblad master equation (can also be set up using Redfield theory class for a microscopic model)

- Ehrenfest dynamics

- Frozen modes

## Installation
```
git clone https://github.com/addschile/qdynos.git
cd qdynos
python setup.py install
```
See the examples in the examples folder for testing installation and usage.

##Extending qdynos
```
```
I have worked at various times on thinking about ways to generalize parts of the code. There are some sparse implementations and Krylov subspace methods have been implemented for a number of clases, though these can likely be refactored. Other useful extensions would be implementing higher-order integrators with adaptive time-stepping (e.g. scipy.odeintegrate), adding different dynamics methods (e.g. HEOM), and adding new sets of calculations like spectroscopy calcualtions. Check out the working_dir if you want to check out any of my half-baked attempts at some generalizations.
