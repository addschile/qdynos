from pylab import *

da = loadtxt('pyr_3_mode.txt')
times = linspace(0.0,500.,len(da[:,0]))
plot(times,da[:,0])
plot(times,da[:,1])
da = loadtxt('pyr_3_mode_sparse.txt')
times = linspace(0.0,500.,len(da[:,0]))
plot(times,da[:,0])
plot(times,da[:,1])
show()
