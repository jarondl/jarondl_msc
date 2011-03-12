#!/usr/bin/python
import matplotlib
matplotlib.use('agg')
import pylab
import sparsedl

A = sparsedl.NN_dense(200)
rho0 = pylab.zeros(200)
rho0[100] = 1
surv = list()
pylab.hold(False)
upper_ylim = max(rho0)
def survival(t):
    return sparsedl.rho(t,rho0,A,100)
vecsurvival = pylab.np.vectorize(survival)
t= pylab.linspace(0,1000,100)
y= vecsurvival(t)
a,b = sparsedl.cvfit(sparsedl.strexp,t,y,[1,1])
#for t in range(100):
#    rhot = sparsedl.rho(t,rho0,A)
#    surv.append(rhot[100])
#    pylab.plot(rhot)
#    pylab.ylim(0,upper_ylim)
#    pylab.savefig('figs/rho{0:02}.png'.format(t),format="png")
pylab.semilogy(t,y)
pylab.savefig('figs/surv.png')
pylab.np.savez( 'survival',t=t,y=y)


