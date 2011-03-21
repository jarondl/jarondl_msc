#!/usr/bin/python
""" 
    Input:  filename of the data set (should be something with a npz suffix, 
                created by create_data.py)
            e.g.:
                ./survival_figs.py  data/mat200.1.npz 
    Output: - Plot of survival at t=500 for 30 values of sigma
"""
import matplotlib
matplotlib.use('agg')
import pylab
import subprocess
import numpy
import scipy
import sys

import sparsedl

def survival_figs(filename):
    # get the hop values from the file:
    data = numpy.load(filename)
    hop_values = data['A'].diagonal(1)

    nodes=len(hop_values)+1
    time = 500
    rho0 = numpy.zeros(nodes)
    rho0[nodes//2] = 1
    pylab.hold(True)
    m2 = list()  # The second moments list
    surv  = list()
    sigmarange = numpy.linspace(0.2,8,50)
    xlocation = pylab.linspace(-1,1,nodes)
    for sigma in sigmarange:
        # rescale the hop values:
        re_hv= hop_values**sigma
        A = sparsedl.NN_matrix(nodes,re_hv)
        rhot = sparsedl.rho(time,rho0,A)
        m2.append(sparsedl.var(xlocation,rhot))
        surv.append(pylab.dot(rho0,rhot))   
     
    # Plot the second moments and the survival
    pylab.subplot(2,1,1)
    pylab.plot(sigmarange,m2,'r.',label="second moment at t=500")
    pylab.xlabel("sigma")
    pylab.legend()
    pylab.subplot(2,1,2)
    pylab.semilogy(sigmarange,surv,'r.',label="Survival at t= 500")
    pylab.xlabel("sigma")
    pylab.legend()

    pylab.savefig("m2_and_surv_of_sigma.eps")

if __name__ == "__main__":
    survival_figs(sys.argv[1])


