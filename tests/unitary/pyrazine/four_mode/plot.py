from pylab import *

da = loadtxt('pyr4_lanczos.txt')
plot(da[:,0],da[:,2],'o')
da = loadtxt('pyr4_lanczos_lowmem.txt')
plot(da[:,0],da[:,2])
da = loadtxt('chk.pl')
plot(da[:,0],da[:,2])
ylim(0.,1.)
xlim(0.,120.)
show()

