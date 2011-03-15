#!/usr/bin/python
import matplotlib
matplotlib.use('agg')
import pylab
import sparsedl
import subprocess

nodes = 200
A = sparsedl.NN_matrix(nodes)
rho0 = pylab.zeros(nodes)
rho0[nodes//2] = 1
pylab.hold(False)
upper_ylim = max(rho0)

m2 = list()  # The second moments list
surv  = list()
xlocation = pylab.linspace(-1,1,nodes)

for t in range(500):
    rhot = sparsedl.rho(t,rho0,A)
    m2.append(sparsedl.var(xlocation,rhot))
    surv.append(pylab.dot(rho0,rhot))
    pylab.plot(rhot)
    pylab.ylim(0,upper_ylim)
    pylab.savefig('figs/rho{0:03}.png'.format(t))

# Plot the second moment
pylab.subplot(2,1,1)
pylab.plot(range(500),m2,'r.',label="second moment")
pylab.legend()
pylab.subplot(2,1,2)
pylab.plot(range(500),surv,'r.',label="Survival")
pylab.legend()
pylab.savefig("second_moments.png")


# make a movie
command = ("mencoder","mf://figs/rho*.png","-ovc","lavc","-speed","0.2","-o","rho.mpg")
try: 
    subprocess.check_call(command)
except OSError:
    print( "Movie creation failed. Make sure you have mencoder installed")
    
