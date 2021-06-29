from pylab import *

da = loadtxt('spin_boson.txt')
plot(da[:,0],da[:,1])
da2 = loadtxt('spin_boson_jumps.txt')
plot(da[:,0],da2[:,0])
ylim(-1,1)
show()
