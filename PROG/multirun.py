#!/usr/bin/python
import matplotlib
matplotlib.use('agg')
import pylab
import sparsedl
from scipy import stats

nodes = 200

rho0 = pylab.zeros(200)
rho0[nodes//2] = 1
pylab.hold(False)
t = pylab.linspace(0,1000,100)
fits_moments = list()

'''
for i in range(10):
    print i
    A = sparsedl.NN_dense(nodes)
    def survival(t):
        return sparsedl.rho(t,rho0,A,nodes//2)
    vecsurvival = pylab.np.vectorize(survival)
    y = vecsurvival(t)
    #a,b = sparsedl.cvfit(sparsedl.strexp,t,y,[1,1])
    #m1 = stats.moment(y,1)
    m2 = stats.moment(y,2)
    fits_moments.append(  m2  )
'''
# now only moments
m2s = list()
x = range(10,500)
xlocation = range(nodes)
for t in x:
    #print t
    A = sparsedl.NN_dense(nodes)
    m2s.append(sparsedl.var(xlocation,sparsedl.rho(t,rho0,A)))
    

pylab.np.save('m2.npy',m2s)
pylab.plot(x,m2s,'r.')
pylab.xlabel("time - Arb. Units")
pylab.ylabel("spreading second moment")
pylab.savefig('m2s.png')
#print(m2s)




