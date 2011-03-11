#!/usr/bin/python
import matplotlib
matplotlib.use('agg')
import pylab
import sparsedl

A = sparsedl.NN_sparse(200).todense()
rho0 = pylab.zeros(200)
rho0[100] = 1
pylab.hold(False)
#for t in range(100):
#    pylab.plot(sparsedl.rho(A,rho0,t))
#    pylab.savefig('figs/rho{0:02}.png'.format(t),format="png")


