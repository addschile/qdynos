from pylab import *

fs2au = 41.3413745758
da = loadtxt('benchmark.dat')
plot(da[:,0]/fs2au,da[:,-1]+da[:,-2])
#plot(da[:,0]/fs2au,da[:,1],'-b')
#plot(da[:,0]/fs2au,da[:,2],'-r')
da = loadtxt('rhodopsin.txt')
plot(da[:,0],da[:,-1]+da[:,-2])
#plot(da[:,0],da[:,1],'ob')
#plot(da[:,0],da[:,2],'or')
xlim(0,4000)
ylim(0,0.8)
show()
