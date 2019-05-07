import numpy as np
import qdynos.constants as const

from scipy.integrate import quad
from scipy.special import jn

from numba import jit,double,complex128

@jit(double(double, double, double),nopython=True)
def bose(w, kT, hbar):
    """
    Fucntion for Bose-Einstein distribution.
    """
    return 1./np.expm1(w*hbar/kT)

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

    def real_bath_corr(self, w):
        """
        Real part of the bath correlation function.
        """
        if w==0:
            return self.J0*self.kT
        else:
            return const.hbar*(bose(w,self.kT,const.hbar)+1.)*((w>0)*self.J_omega(w) - (w<0)*self.J_omega(abs(w)))

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

        Uses inspiration from pyrho - https://github.com/berkelbach-group/pyrho

        Notes
        -----
        \int_0^t ds e^{i omega s} C(s)
        """
        def bath_corr_bose(w):
            if w==0:
                return self.J0*self.kT
            else:
                return const.hbar*(2.*bose(w,self.kT,const.hbar)+1.)*self.J_omega(w)
        if self.kT == 0:
            return (1.0/np.pi)*self.zero_T_bcf_t(t)
        else:
            #NOTE: don't integrate to infinity for numerical stability
            re_Ct = quad(bath_corr_bose, 0.0, self.omega_inf, limit=1000, weight='cos', wvar=t)
            im_Ct = self.im_bath_corr_bose(t)
            re_Ct = re_Ct[0]
            return (1.0/np.pi)*(re_Ct - 1.j*im_Ct)

    def compute_omegas(self, nmodes):
        """
        """
        raise NotImplementedError

    def frozen_mode_decomp(self, omega_star, PD=False):
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
        self.omega_star = omega_star
        self.Jslow = lambda w: switch(w,omega_star)*self.spectral_density_func(w)
        if PD:
            self.Jfast = lambda w: (1.-switch(w,omega_star))*self.spectral_density_func(w) +\
                            float(PD)*float(abs(w) < 1.e-8)*self.spectral_density_func(w)
        else:
            self.Jfast = lambda w: (1.-switch(w,omega_star))*self.spectral_density_func(w)
            self.J0 = 0.0
        self.J_omega = self.Jfast

    def sample_modes(self, nmodes, sample):
        omegas, c_ns = self.compute_omegas(nmodes)
        Qs, Ps = sample_modes(omegas, self.kT, sample)
        return omegas , c_ns , Ps , Qs


@jit(double(double, double, double),nopython=True)
def ohmic_exp(w, eta, wc):
    return eta*w*np.exp(-w/wc)

@jit(complex128(double, double, double),nopython=True)
def ohmic_exp_zero_T_bcf_t(t, eta, wc):
    re_bcf_t = eta*wc**2.*(1.-wc**2.*t**2.)/(1.+wc**2.*t**2.)**2.
    im_bcf_t = 2.*eta*wc**3.*t/(1.+wc**2.*t**2.)**2.
    return re_bcf_t - 1.j*im_bcf_t

@jit(double(double, double, double, double),nopython=True)
def ohmic_exp_im_bath_corr_bose(t, eta, wc, hbar):
    im_bcf_t = 2.*eta*wc**3.*t/(1.+wc**2.*t**2.)**2.
    return hbar*im_bcf_t

class OhmicExp(Bath):
    """
    Class for Ohmic bath with exponential cuttoff

    Notes
    -----
    J(\omega) = \eta \omega e^{- \omega / \omega_c}
    """
    def __init__(self, eta, wc, kT, op=None, disc='log'):
        self.type = "ohmic exponential"
        self.eta = eta
        self.wc = wc
        self.omega_inf = 50.*wc
        self.kT = kT
        self.c_op = op
        self.J_omega = self.spectral_density_func
        self.J0 = self.spectral_density_limit_at_zero
        assert(disc in ['log', 'uniform'])
        self.disc = disc

    def spectral_density_func(self, w):
        return ohmic_exp(w,self.eta,self.wc)

    def zero_T_bcf_t(self, t):
        return ohmic_exp_zero_T_bcf_t(t,self.eta,self.wc)

    def im_bath_corr_bose(self, t):
        return ohmic_exp_im_bath_corr_bose(t,self.eta,self.wc,const.hbar)

    @property
    def spectral_density_limit_at_zero(self):
        return self.eta

    @property
    def renormalization_integral(self):
        return self.eta*self.wc/np.pi

    def compute_omegas(self, nmodes):
        if self.disc == 'log':
            # smart discretization
            omegas = np.array([self.wc*(-np.log((nmodes-i-0.5)/nmodes)) for i in range(nmodes)])
            rho_slow = (nmodes/self.wc)*np.exp(-omegas/self.wc)/self.wc
            c_ns = np.array([np.sqrt((2./np.pi)*omegas[i]*self.Jslow(omegas[i])/rho_slow[i]) for i in range(nmodes)])
        elif self.dis == 'uniform':
            # naive discretization
            dw = self.omega_star/nmodes
            omegas = np.array([(i+0.5)*dw for i in range(nmodes)])
            c_ns = np.array([np.sqrt((2./np.pi)*omegas[i]*self.Jslow(omegas[i])*dw) for i in range(nmodes)])
        return omegas, c_ns

@jit(double(double, double, double),nopython=True)
def debye(w, eta, wc):
    return eta*w/(w**2. + wc**2.)

@jit(double(double, double, double, double),nopython=True)
def debye_im_bath_corr_bose(t, eta, wc, hbar):
    im_bcf_t = 0.5*np.pi*eta*np.exp(-wc*t)
    return hbar*im_bcf_t

class DebyeBath(Bath):
    """
    Class for Ohmic bath with Lorentz cuttoff (Debye)

    Notes
    -----
    J(\omega) = \eta \omega / (\omega^2 + \omega_c^2)
    """
    def __init__(self, eta, wc, kT, op=None, disc='log'):
        self.type = "debye"
        self.eta = eta
        self.wc = wc
        self.omega_inf = 50.*wc
        self.kT = kT
        self.c_op = op
        self.J_omega = self.spectral_density_func
        self.J0 = self.spectral_density_limit_at_zero
        assert(disc in ['log'])
        self.disc = disc

    def spectral_density_func(self, w):
        return debye(w,self.eta,self.wc)

    def zero_T_bcf_t(self, t):
        raise NotImplementedError

    def im_bath_corr_bose(self, t):
        return debye_im_bath_corr_bose(t,self.eta,self.wc,const.hbar)

    @property
    def spectral_density_limit_at_zero(self):
        return self.eta/self.wc**2. 

    @property
    def renormalization_integral(self):
        return 0.5*self.eta/self.wc

    def compute_omegas(self, nmodes):
        if self.disc == 'log':
            omegas = np.array([self.wc*np.tan(0.5*np.pi*(float(i)+0.5)/nmodes) for i in range(nmodes)])
            rho_slow = 2.*(nmodes/np.pi)/(1.+(omegas/self.wc)**2.)/self.wc
            c_ns = np.array([ np.sqrt((2./np.pi)*omegas[i]*self.Jslow(omegas[i])/rho_slow[i]) for i in range(nmodes) ])
        return omegas, c_ns

@jit(double(double,double[:],double[:],double[:]),nopython=True)
def mt_decomp(w,pk,omk,gamk):
    return 0.5*np.pi*np.sum((pk*w/( ((w+omk)**2. + gamk**2.)*((w-omk)**2. + gamk**2.) ))[:])

@jit(double(double,double[:],double[:],double[:],double),nopython=True)
def mt_decomp_im_bath_corr_bose(t,pk,omk,gamk,hbar):
    return (np.pi**2./8.)*np.sum((np.exp(-gamk*t)*pk*np.sin(omk*t)/(gamk*omk))[:])

class MeierTannor(Bath):
    """
    Class for Meier-Tannor spectral density that approximates an arbitrary
    spectral density.

    Notes
    -----
    J(\omega) = \pi/2 \sum_k p_k w / ( ((w + wk)^2 + gamk^2) * ((w-wk)^2 + gamk^2) )
    """
    def __init__(self, pk, omk, gamk, kT, op=None):
        self.type = "meier-tannor"
        self.pk = pk
        self.omk = omk
        self.gamk = gamk
        self.omega_inf = np.amax(omk) + 20.*np.amax(gamk)
        self.kT = kT
        self.c_op = op
        self.J_omega = self.spectral_density_func
        self.J0 = self.spectral_density_limit_at_zero

    def spectral_density_func(self, w):
        return mt_decomp(w,self.pk,self.omk,self.gamk)

    def zero_T_bcf_t(self, t):
        raise NotImplementedError
        # TODO
        #return mt_decomp_zero_T_bcf_t(t,self.pk,self.omk,self.gamk)

    def im_bath_corr_bose(self, t):
        # NOTE: need to test my analytical to this numerical
        #im_Ct = quad(self.J_omega, 0.0, self.omega_inf, limit=1000, weight='sin', wvar=t)
        #return im_Ct[0]
        # analytical integration
        return mt_decomp_im_bath_corr_bose(t,self.pk,self.omk,self.gamk,const.hbar)

    @property
    def spectral_density_limit_at_zero(self):
        return np.sum((self.pk/(self.omk**2.+self.gamk**2.)**2.)[:])

@jit(double(double, double, double),nopython=True)
def rubin(w, eta, wr):
    return eta*w*wr*np.sqrt(1.-(w/wr)**2.)

#TODO
#@jit(complex128(double, double),nopython=True)
#def rubin_zero_T_bcf_t(t, wr):
#    re_bcf_t = eta*wc**2.*(1.-wc**2.*t**2.)/(1.+wc**2.*t**2.)**2.
#    im_bcf_t = 2.*eta*wc**3.*t/(1.+wc**2.*t**2.)**2.
#    return re_bcf_t - 1.j*im_bcf_t

#@jit(double(double, double, double),nopython=True)
def rubin_im_bath_corr_bose(t, eta, wr, hbar):
    if t==0:
        return 0.0
    else:
        im_bcf_t = 0.25*np.pi*wr**2.*jn(2,wr*t)/t
        return hbar*im_bcf_t

class RubinBath(Bath):
    """
    Class for Rubin (quasi-Ohmic) bath

    Notes
    -----
    J(\omega) = 0.5 \omega \omega_R \sqrt{ 1- (\omega/\omega_R)^2}
    """
    def __init__(self, eta, wr, kT, op=None):
        self.type = "rubin"
        self.eta = eta
        self.wr = wr
        self.omega_inf = wr
        self.kT = kT
        self.c_op = op
        self.J_omega = self.spectral_density_func
        self.J0 = self.spectral_density_limit_at_zero

    def spectral_density_func(self, w):
        return rubin(w,self.eta,self.wr)

    def zero_T_bcf_t(self, t):
        raise NotImplementedError

    def im_bath_corr_bose(self, t):
        return rubin_im_bath_corr_bose(t,self.eta,self.wr,const.hbar)

    @property
    def spectral_density_limit_at_zero(self):
        return self.wr

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
#        self.type = "super ohmic exponential"
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
#        raise NotImplementedError
#
#    @property
#    def renormalization_integral(self):
#        raise NotImplementedError
