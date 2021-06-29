from pylab import *

fs2au = 41.3413745758
#da = loadtxt('thoss_model_ci_no_bath.dat')
#plot(da[:,0]/fs2au,da[:,-1]+da[:,-2],'-')
#plot(da[:,0]/fs2au,da[:,1],'-b')
#plot(da[:,0]/fs2au,da[:,2],'-r')
da = loadtxt('../two_mode/rhodopsin.txt')
plot(da[:,0],da[:,-1]+da[:,-2])
da = loadtxt('rhodopsin_eta_2.50.txt')
plot(da[:,0],da[:,-1]+da[:,-2])
#plot(da[:,0],da[:,1],'ob')
#plot(da[:,0],da[:,2],'or')
xlim(0,2000)
ylim(0,0.8)
xlabel(r'$t$ / fs')
ylabel(r'$P_{\mathrm{trans}} (t)$')
savefig('rhodopsin.png')
show()
