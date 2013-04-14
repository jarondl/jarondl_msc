#!/usr/bin/env python
"""  

"""
# make python2 behave more like python3
from __future__ import division, print_function, absolute_import


import numpy as np
import logging
#import h5py

from numpy import pi
from scipy.optimize import brentq

from .libdl.tools import cached_get_key


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
    """ k array can be padded by nans """
    n = np.arange(1,b+1)
    km,nm = np.meshgrid(k,n)
    #debug("dossum, km*nm.shape ={}".format((km*nm).shape))
    return 2*np.reciprocal(np.nansum(np.reciprocal(abs(np.sum(nm*np.sin(km*nm),axis=0)))))


def find_ks(b, lam):
    # find the relevant k's for this lambda
    k = np.linspace(0,pi,3000)
    evs = theor_banded_ev(b, k) -lam
    sign_changes, = np.nonzero(np.diff(np.sign(evs)))
    intervals = zip(k[sign_changes], k[sign_changes+1])
    ks = [brentq(theor_banded_ev_k_1, start,end, args=(b,lam)) for ((start),(end)) in intervals]
    return ks


def sum_dos_type_by_b_N(b,N):
	return np.dtype([   ("eig_vals", (np.float64, N)),
						("inverse_dos", (np.float64, N)),
						("k_for_ev", (np.float64, (N,b)))])

def get_sum_dos(b,N):
    k = np.linspace(0,pi,N)
    evs = theor_banded_ev(b,k)
    lams = np.linspace(evs.min(), evs.max(),N)
    result_k = np.empty([N,b])
    result_k.fill(np.nan)
    for n, ev in enumerate(lams):
		ks = find_ks(b, ev)
		result_k[n,:len(ks)] = ks
    #k_of_ev = [find_ks(b,ev) for ev in lams]
    #debug("k_of_ev.shape = {} ".format(k_of_ev.shape))
    doses = np.array([theor_banded_dossum_k(k_of_ev,b) for k_of_ev in result_k])
    return np.array((lams, doses, result_k), dtype = sum_dos_type_by_b_N(b,N) )
    
def cached_get_sum_dos(b, N=2000):
    return cached_get_key(get_sum_dos, "banded_b_{}.npz".format(N), str(b), b,N)



