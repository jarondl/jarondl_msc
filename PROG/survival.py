#!/usr/bin/python
import matplotlib
matplotlib.use('agg')
import pylab
import sparsedl
from scipy import stats, linalg
import scipy

nodes = 200

rho0 = scipy.zeros(200)
rho0[nodes//2] = 1
pylab.hold(False)
t = scipy.linspace(0,1000,100)


A = sparsedl.NN_matrix(nodes)
eigvals, eigvecs = linalg.eig(A)
decr = scipy.dot(eigvals,rho0)  # rho0 in the eigenvector basis

def survival1(t):
    return sparsedl.rho(t,rho0,A,nodes//2)
def survival2(t): #Using decomposition
    return scipy.dot(decr**2, scipy.exp(eigvals*t))

vecsurvival1 = pylab.np.vectorize(survival1)
vecsurvival2 = pylab.np.vectorize(survival2)
y1 = vecsurvival1(t)
#y2 = vecsurvival2(t)
#a,b = sparsedl.cvfit(sparsedl.strexp,t,y,[1,1])
#m1 = stats.moment(y,1)
pylab.semilogy(t,y1)#,t,y2)
pylab.savefig('survival.png')




