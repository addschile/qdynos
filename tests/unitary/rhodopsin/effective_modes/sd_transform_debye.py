import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def spec_dens(w,eta,wc):
    return eta*w*np.exp(-w/wc)

def rubin(w,eta,wR):
    return eta*w*wR*np.sqrt(1.-(w/wR)**2.)

def ohmic(w,eta,wc):
    return eta*w*np.exp(-w/wc)

def linear(w,eta):
    return eta*w

# spectral density parameters
wc  = 0.2
eta = 2.5

# discretize spectral density
wcs = np.linspace(1.e-20,50.*wc,2000)
ns = len(wcs)
dw = wcs[1]-wcs[0]
cs = np.sqrt(2.*spec_dens(wcs,eta,wc)*dw*wcs/np.pi)
# og sd
Jw = 0.5*np.pi*(cs)**2./dw/wcs
plt.subplot(131)
plt.plot(wcs,Jw)

# wang-thoss
kappa = np.sqrt(np.sum((cs**2.)[:]))
omega = np.sqrt(kappa**2./np.sum((cs**2./wcs**2.)[:]))

# transform to mass-weighted coordinates
omega_s = list()
om_ind = list()
kappas = list()
kap_ind = list()
nmodes = 4
for i in range(nmodes):
    om_ind.append(i+1)
    kap_ind.append(i)
    # compute quantities for transform
    kappa = np.sqrt(np.sum((cs**2.)[:]))
    kappas.append(kappa)
    # make reaction coordinate mode
    zn = np.zeros((ns,1))
    zn[:,0] = cs[:]/kappa
    # construct Householder reflector
    Qn = np.zeros((ns,1))
    Qn[-1,0] = 1.
    wn = zn-Qn
    wn /= np.sqrt(np.dot(wn.T,wn)[0,0])
    P = np.identity(ns) - 2.*np.dot(wn,wn.T)
    # make matrix of omegas
    omegas = np.diag(wcs**2.)
    # transform it using P
    K = np.dot(P,np.dot(omegas,P))
    ws = K[-1,-1]
    Ksub = K[:ns-1,:ns-1].copy()
    w,v = np.linalg.eigh(Ksub)
    d = np.zeros((ns-1,1))
    d[:,0] = K[:ns-1,-1]
    d = np.dot(v.T,d)
    pos_w = np.sqrt(np.abs(w))
    # compute frequency and new kappa
    omega = np.sqrt(kappa**2./np.sum((cs**2./wcs**2.)[:]))
    omega_s.append( omega )
    dww = pos_w[1]-pos_w[0]
    # transformed sd
    plt.subplot(132)
    Jeff = 0.5*np.pi*d[:,0]**2./dww/pos_w[:]
    plt.plot(pos_w[:],Jeff)
    if i==(nmodes-1):
        #popt,pocv = curve_fit(rubin,pos_w,Jeff,p0=[0.2,100.*wc])
        #print(popt[0],popt[1])
        #plt.plot(pos_w[:],rubin(pos_w[:],popt[0],popt[1]),'--k',lw=3)
        popt,pocv = curve_fit(linear,pos_w[:500],Jeff[:500])
        print(popt[0],omega**2./wc)
        plt.plot(pos_w[:1000],linear(pos_w[:1000],popt[0]),'--b',lw=3)
    # discretize spectral density
    cs = d[:,0]
    wcs = pos_w
    ns = len(cs)
plt.subplot(133)
plt.plot(om_ind,omega_s,'-ok')
plt.plot(kap_ind,kappas,'-sg')
#plt.savefig('sd_transform_debye.png')
plt.tight_layout()
plt.show()

print('new frequencies')
print(omega_s)
print('new kappas')
print(kappas)
