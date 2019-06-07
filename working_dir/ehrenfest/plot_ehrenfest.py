from pylab import *

dts = [0.01, 0.01, 0.001, 0.001]
#dts = [0.01, 0.01, 0.01, 0.01]
wcs = [0.025, 0.25, 5.0, 10.0]
for i in range(3):
    times = np.arange(0.0,12.0,dts[i])
    subplot(2,2,i+1)
    da = loadtxt('sigz_wc_%.3f.txt'%(wcs[i]))
    plot(times,da)
    xlim(0,12)
    ylim(0,1)
show()
