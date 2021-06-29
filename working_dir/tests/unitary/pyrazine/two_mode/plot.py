from pylab import *

da = loadtxt('pyr2_exact.txt')
plot(da[:,0],da[:,2])
#da = loadtxt('pyr2_exact_eig.txt')
#plot(da[:,0],da[:,2])
da = loadtxt('pyr2_lanczos.txt')
plot(da[:,0],da[:,2])
#da = loadtxt('pyr2_lanczos_lowmem.txt')
#plot(da[:,0],da[:,2])
#da = loadtxt('pyr2_arnoldi.txt')
#plot(da[:,0],da[:,2])
#da = loadtxt('pyr2_rk4.txt')
#plot(da[:,0],da[:,2])
ylim(0.,1.)
xlim(0.,1000.)
show()

