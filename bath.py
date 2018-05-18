import numpy as np
import .constants as const

# TODO figure out where i need to put hbars

class Bath(object):
    """
    """

    def bose(self, w):
        """
        Fucntion for Bose-Einstein distribution.
        """
        return 1./np.expm1(self.kT*w)

    def real_bath_corr(self, w):
        """
        Real part of the bath correlation function.
        """
        if w==0:
            return self.J0()*self.kT
        else:
            return (w>0)*self.J_omega(w) - (w<0)*self.J_omega(-w)

    def spectral_density_func(self, w):
        """
        Spectral density function in functional form.

        Notes
        -----
        J(w) = \pi/2 \sum_n  frac{c_n^2}{m_n \omega_n^2} \delta(w-w_n)
        """
        raise NotImplementedError

    def spectral_density_function_at_0(self, w):
        """
        Limit of the spectral density function at 0.
        """
        raise NotImplementedError

    def renormalization_integral(self):
        """
        Renormaliztion term added to system Hamiltonian.

        Notes
        -----
        (1/\pi)\int_0^{\infty} dw J(w)/w
        """
        raise NotImplementedError

    def ft_bath_corr(self, omega):
        """
        Compute Fourier-Laplace transform of the bath correlation function.

        Uses clever code from pyrho - https://github.com/berkelbach-group/pyrho

        Notes
        -----
        \int_0^{\infty} ds e^{i omega s} C(s)
        """
        ppv = integrate.quad(self.real_bath_corr, 
                             -self.omega_inf, self.omega_inf,
                             limit=200, weight='cauchy', wvar=omega)
        ppv = -ppv[0]
        return self.real_bath_corr(omega) + (1.j/np.pi)*ppv

    def bath_corr_t(self, omega, t):
        """
        Compute Fourier-Laplace integral of bath correlation function up to 
        some time t.

        Uses clever code from pyrho - https://github.com/berkelbach-group/pyrho

        Notes
        -----
        \int_0^t ds e^{i omega s} C(s)
        """
        re_Ct = integrate.quad(self.real_bath_corr,
                                -self.omega_inf, self.omega_inf,
                                limit=1000, weight='cos', wvar=t)
        im_Ct = integrate.quad(self.real_bath_corr, 
                                -self.omega_inf, self.omega_inf,
                                limit=1000, weight='sin', wvar=t)
        re_Ct, im_Ct = re_Ct[0], -im_Ct[0]
        return (1.0/np.pi)*(re_Ct + 1j*im_Ct)

    def frozen_mode_decomp(self):
        """
        Compute the frozen modes for sampling

        References
        ----------
        JCP 147, 244109 (2017).
        JCP 143, 194108 (2015).
        JCP 136, 034113 (2012).
        """
        raise NotImplementedError

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
        self.op = op

    def spectral_density_func(self, w):
        return self.eta*w*np.exp(-w/self.wc)

    @property
    def spectral_density_func_at_0(self, w):
        return self.eta

    @property
    def renormalization_integral(self):
        return self.eta*self.wc/np.pi

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
        self.op = op

    def spectral_density_func(self, w):
        return self.eta*w/(w**2.+self.wc**2.)

    @property
    def spectral_density_limit_at_zero(self):
        return 2.*self.eta/self.wc**2. 

    @property
    def renormalization_integral(self):
        return 0.5*self.eta/self.wc

# TODO
#class SuperOhmic(Bath):
#    """
#    Class for Ohmic bath with exponential cuttoff
#
#    Notes
#    -----
#    TODO
#    J(\omega) = \eta \omega e^{- \omega / \omega_c} / (\omega_c )
#    """
#    def __init__(self, eta, wc, kT, op=None):
#        self.eta = eta
#        self.wc = wc
#        self.omega_inf = 20.*wc
#        self.kT = kT
#        self.op = op
#
#    def spectral_density_func(self, w):
#        return self.eta*w*np.exp(-w/self.wc)/self.wc/6.
#
#    @property
#    def spectral_density_limit_at_zero(self):
#        return 2.*self.eta/self.wc**2. 
#
#    @property
#    def renormalization_integral(self):
#        # TODO
#        return self.eta*self.wc/np.pi

# TODO
#class PsuedomodeBath(Bath):
