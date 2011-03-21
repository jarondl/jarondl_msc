#!/usr/bin/python
""" Input:  filename of the data set (should be something with a npz suffix, 
                created by create_data.py)
            e.g.:
                ./mk_figs.py  data/mat200.1.npz
    Output: - 500 figures of rho as a function of t (called figs/rhoxxx.png)
            - 1 movie of rho as a function of t (called rho.mpg)
            - 1 figure with 3 subplots: second moment as a function of t  ("second_moment_and_survival.eps")
                                        survival as a function of t
                                        logarithmic survival as a function of t
"""
import matplotlib
matplotlib.use('agg')
import pylab
import subprocess
import numpy
import sys

import sparsedl

def mk_figs(filename):
    data = numpy.load(filename)
    A = data['A']
    rho0 = data['rho0']
    nodes = len(rho0)

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

    # Plot the second moments and the survival
    pylab.subplot(3,1,1)
    pylab.plot(range(500),m2,'r.',label="second moment")
    pylab.legend()
    pylab.subplot(3,1,2)
    pylab.plot(range(500),surv,'r.',label="Survival")
    pylab.legend()
    pylab.subplot(3,1,3)
    pylab.semilogy(range(500),surv,'r.',label="Survival")
    pylab.legend()
    pylab.savefig("second_moment_and_survival.eps")


    # make a movie
    command = ("mencoder","mf://figs/rho*.png","-ovc","lavc","-speed","0.2","-o","rho.mpg")
    try: 
        subprocess.check_call(command)
    except OSError:
        print( "Movie creation failed. Make sure you have mencoder installed")
        
if __name__ == "__main__":
    mk_figs(sys.argv[1])
