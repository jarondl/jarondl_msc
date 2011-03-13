#!/usr/bin/python
import matplotlib
matplotlib.use('agg')
import pylab
import sparsedl
import subprocess

A = sparsedl.NN_matrix(200)
rho0 = pylab.zeros(200)
rho0[100] = 1
pylab.hold(False)
upper_ylim = max(rho0)

for t in range(100):
    rhot = sparsedl.rho(t,rho0,A)
    pylab.plot(rhot)
    pylab.ylim(0,upper_ylim)
    pylab.savefig('figs/rho{0:02}.png'.format(t))

#make a movie
command = ("mencoder","mf://figs/rho*.png","-ovc","lavc","-o","rho.mpg")
subprocess.check_call(command)
