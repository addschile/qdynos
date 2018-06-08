import numpy as np
import qdynos.constants as const

from scipy.integrate import quad
from numba import vectorize,float32,float64

def switch(w, wstar):
    """A smooth switching function for spectral density decompositions.
    """
    if abs(w) < wstar:
        return (1. - (w/wstar)**2.)**2.
    else:
        return 0.0

def _sample_modes(omega, kT, sample):
    if sample=="Boltzmann":
        Q = np.random.normal(0.0, np.sqrt(kT)/omega)
        P = np.random.normal(0.0, np.sqrt(kT))
        return Q , P
    elif sample=="Wigner":
        Q = np.random.normal(0.0, np.sqrt(1/(2*omega*np.tanh(omega/(2*kT)))))
        P = np.random.normal(0.0, np.sqrt(omega/(2*np.tanh(omega/(2*kT)))))
        return Q , P
sample_modes = np.vectorize(_sample_modes)

class Bath(object):
    """
    Base bath class.

    Methods
    -------
    """

    def bose(self, w):
        """
        Fucntion for Bose-Einstein distribution.
        """
        return 1./np.expm1(w*const.hbar/self.kT)

    def real_bath_corr(self, w):
        """
        Real part of the bath correlation function.
        """
        if w==0:
            return self.J0*self.kT
        else:
            return const.hbar*(self.bose(w)+1.)*((w>0)*self.J_omega(w) - (w<0)*self.J_omega(abs(w)))

    def spectral_density_func(self, w):
        """Spectral density function in functional form.

        Notes
        -----
        J(w) = frac{\pi}{2} \sum_n  frac{c_n^2}{m_n \omega_n^2} \delta(w-w_n)
        """
        raise NotImplementedError

    def zero_T_bcf_t(self, t):
        """
        """
        raise NotImplementedError

    def spectral_density_function_at_0(self, w):
        """Limit of the spectral density function at 0.
        """
        raise NotImplementedError

    def renormalization_integral(self):
        """Renormaliztion term added to system Hamiltonian.

        Notes
        -----
        (1/\pi)\int_0^{\infty} dw J(w)/w
        """
        raise NotImplementedError

    def ft_bath_corr(self, omega):
        """Compute Fourier-Laplace transform of the bath correlation function.

        Uses clever code from pyrho - https://github.com/berkelbach-group/pyrho

        Notes
        -----
        \int_0^{\infty} ds e^{i omega s} C(s)
        """
        ppv = quad(self.real_bath_corr, 
                             -self.omega_inf, self.omega_inf,
                             limit=1000, weight='cauchy', wvar=omega)
        ppv = -ppv[0]
        return self.real_bath_corr(omega) + (1.j/np.pi)*ppv

    def bath_corr_t(self, t):
        """Compute Fourier-Laplace integral of bath correlation function up to 
        some time t.

        Uses clever code from pyrho - https://github.com/berkelbach-group/pyrho

        Notes
        -----
        \int_0^t ds e^{i omega s} C(s)
        """
        # NOTE: integration when kT != 0 is bad, need to fix
        if self.kT == 0:
            return (1.0/np.pi)*self.zero_T_bcf_t(t)
        else:
            re_Ct = quad(self.real_bath_corr,
                                    -self.omega_inf, self.omega_inf,
                                    limit=1000, weight='cos', wvar=t)
            im_Ct = quad(self.real_bath_corr, 
                                    -self.omega_inf, self.omega_inf,
                                    limit=1000, weight='sin', wvar=t)
            re_Ct, im_Ct = re_Ct[0], -im_Ct[0]
            return (0.5/np.pi)*(re_Ct - 1.j*im_Ct)

    def compute_omegas(self, nmodes):
        """
        """
        raise NotImplementedError

    def frozen_mode_decomp(self, omega_star, PD=True):
        """
        Compute the frozen modes for sampling

        Parameters
        ----------
        omega_star: float
        PD: bool

        References
        ----------
        JCP 147, 244109 (2017).
        JCP 143, 194108 (2015).
        JCP 136, 034113 (2012).
        """
        self.Jslow = lambda w: switch(w,omega_star)*self.J_omega(w)
        self.Jfast = lambda w: (1.-switch(w,omega_star))*self.J_omega(w) +\
                        float(PD)*float(abs(w) < 1.e-4)*self.J_omega(w)

    def sample_modes(self, nmodes, sample):
        omegas, c_ns = self.compute_omegas(nmodes)
        Qs, Ps = sample_modes(omegas, self.kT, sample)
        return omegas , c_ns , Ps , Qs


class OhmicExp(Bath):
    """
    Class for Ohmic bath with exponential cuttoff

    Notes
    -----
    J(\omega) = \eta \omega e^{- \omega / \omega_c}
    """
    def __init__(self, eta, wc, kT, op=None):
        self.eta = eta
        self.wc = wc
        self.omega_inf = 20.*wc
        self.kT = kT
        self.c_op = op
        self.J_omega = self.spectral_density_func
        self.J0 = self.spectral_density_limit_at_zero

    def spectral_density_func(self, w):
        return self.eta*w*np.exp(-w/self.wc)

    def zero_T_bcf_t(self, t):
        re_bcf_t = self.eta*self.wc**2.*(1.-self.wc**2.*t**2.)/(1.+self.wc**2.*t**2.)**2.
        im_bcf_t = 2.*self.eta*self.wc**3.*t/(1.+self.wc**2.*t**2.)**2.
        return re_bcf_t - 1.j*im_bcf_t

    @property
    def spectral_density_limit_at_zero(self):
        return self.eta

    @property
    def renormalization_integral(self):
        return self.eta*self.wc/np.pi

    def compute_omegas(self, nmodes):
        omegas = np.array([self.wc*(-np.log((nmodes-i-0.5)/nmodes)) for i in range(nmodes)])
        rho_slow = (nmodes/self.wc)*np.exp(-omegas/self.wc)/self.wc
        c_ns = np.array([np.sqrt((2./np.pi)*omegas[i]*self.J_omega(omegas[i]))/rho_slow[i] for i in range(nmodes)])
        return omegas, c_ns

class DebyeBath(Bath):
    """
    Class for Ohmic bath with Lorentz cuttoff (Debye)

    Notes
    -----
    J(\omega) = \eta \omega / (\omega^2 + \omega_c^2)
    """
    def __init__(self, eta, wc, kT, op=None):
        self.eta = eta
        self.wc = wc
        self.omega_inf = 20.*wc
        self.kT = kT
        self.c_op = op
        self.J_omega = self.spectral_density_func
        self.J0 = self.spectral_density_limit_at_zero

    def spectral_density_func(self, w):
        return self.eta*w/(w**2.+self.wc**2.)

    def zero_T_bcf_t(self, t):
        raise NotImplementedError

    @property
    def spectral_density_limit_at_zero(self):
        return self.eta/self.wc**2. 

    @property
    def renormalization_integral(self):
        return 0.5*self.eta/self.wc

    def compute_omegas(self, nmodes):
        omegas = np.array([self.wc*np.tan(0.5*np.pi*(float(i)+0.5)/nmodes) for i in range(nmodes)])
        rho_slow = 2.*(nmodes/np.pi)/(1.+(omegas/self.wc)**2.)/self.wc
        c_ns = np.array([ np.sqrt((2./np.pi)*omegas[i]*self.J_omega(omegas[i])/rho_slow[i]) for i in range(nmodes) ])
        return omegas, c_ns

# TODO
class SuperOhmic(Bath):
    """
    Class for Ohmic bath with exponential cuttoff

    Notes
    -----
    TODO
    J(\omega) = \eta \omega e^{- \omega / \omega_c} / (\omega_c )
    """
    def __init__(self, eta, wc, kT, op=None):
        self.eta = eta
        self.wc = wc
        self.omega_inf = 20.*wc
        self.kT = kT
        self.op = op

    def spectral_density_func(self, w):
        return self.eta*w*np.exp(-w/self.wc)/self.wc/6.

    @property
    def spectral_density_limit_at_zero(self):
        raise NotImplementedError

    @property
    def renormalization_integral(self):
        raise NotImplementedError

