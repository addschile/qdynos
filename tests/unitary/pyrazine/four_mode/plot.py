from pylab import *

da = loadtxt('pyr4_lanczos.txt')
plot(da[:,0],da[:,2])
#da = loadtxt('pyr4_lanczos_lowmem.txt')
#plot(da[:,0],da[:,2])
da = loadtxt('chk.pl')
plot(da[:,0],da[:,2])
ylim(0.,1.)
xlim(0.,120.)
xlabel(r'$t$ / fs')
ylabel(r'$P_2 (t)$')
savefig('pyr4.png')
show()
