#!/usr/bin/env python
"""  

"""
# make python2 behave more like python3
from __future__ import division, print_function, absolute_import


import numpy as np
import logging
import h5py

from numpy import pi
from scipy.optimize import brentq


### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)

#set up logging:
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug



def theor_banded_ev(b,k = np.linspace(0,pi,2000)):
    n = np.arange(1,b+1)
    km,nm = np.meshgrid(k,n)
    return -2*(np.cos(km*nm)).sum(axis=0)
    
def theor_banded_dev(b,k = np.linspace(0,pi,2000)):
    n = np.arange(1,b+1)
    km,nm = np.meshgrid(k,n)
    return -2*(nm*np.sin(km*nm)).sum(axis=0)
    
def theor_banded_ev_k_1(k,b,lam):
    # only for the brentq below
    return theor_banded_ev(b, (k,))-lam
    
def theor_banded_dossum_k(k,b):
    #debug("dossum, k.shape ={}, k= {}, b= {}".format(k.shape,k,b))
    n = np.arange(1,b+1)
    km,nm = np.meshgrid(k,n)
    #debug("dossum, km*nm.shape ={}".format((km*nm).shape))
    return 2*(np.reciprocal((np.reciprocal(abs((nm*np.sin(km*nm)).sum(axis=0)))).sum()))


def find_ks(b, lam):
    # find the relevant k's for this lambda
    k = np.linspace(0,pi,3000)
    evs = theor_banded_ev(b, k) -lam
    sign_changes, = np.nonzero(np.diff(np.sign(evs)))
    intervals = zip(k[sign_changes], k[sign_changes+1])
    ks = [brentq(theor_banded_ev_k_1, start,end, args=(b,lam)) for ((start),(end)) in intervals]
    return np.array(ks)

def get_sum_dos(b):
    k = np.linspace(0,pi,2000)
    evs = theor_banded_ev(b,k)
    lams = np.linspace(evs.min(), evs.max(),2000)
    k_of_ev = [find_ks(b,ev) for ev in lams]
    #debug("k_of_ev.shape = {} ".format(k_of_ev.shape))
    doses = np.array([theor_banded_dossum_k(k_of,b) for k_of in k_of_ev])
    return (lams, doses )
    
def cached_get_sum_dos(b):
    with h5py.File('banded_dos.hdf5') as f:
        if str(b) in f:
            lams = np.array(f[str(b)]['eig_vals'])
            dos = np.array(f[str(b)]['dos'])
        if str(b) not in f:
            fb = f.create_group(str(b))
            lams, dos= get_sum_dos(b)
            f[str(b)].create_dataset('eig_vals', data=lams)
            f[str(b)].create_dataset('dos', data=dos)
    return lams, dos


