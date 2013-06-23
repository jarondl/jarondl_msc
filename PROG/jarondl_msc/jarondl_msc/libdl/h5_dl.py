#!/usr/bin/env python
"""  
This is my abstraction layer over python-tables for dealing with hdf5 files
"""
from __future__ import division  # makes true division instead of integer division


import numpy as np
import logging
import hashlib
import pdb

from collections import OrderedDict
import itertools

import tables

from . import tools
from .tools import (h5_create_if_missing, h5_get_first_rownum_by_args)


### Warn about all float errors
np.seterr(all='warn')

#set up logging:
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug
########################################################################
################  data factory classes. These are functions that accept 
################  input and can create h5 lines 
########################################################################


def input_iterator(args):
    """ This function iterates over all possible combinations of input args
    :param args: The keys must match 
                 the function input keys, and the values must be lists
                 of argument values.
    :type args: dictionary of lists
    :rtype: iterator"""
    ovar_args = OrderedDict(args)
    for val_set in itertools.product(*ovar_args.values()):
        yield dict( zip(ovar_args.keys(), val_set))

def input_iterator_size(args):
    """  Return the expected size of the input iterator without running
            the entire iteration """
    return np.prod([len(vals) for vals in args.values()])
    
def args_hash(args):
    """ create a hash of args """
    fset = frozenset((key, frozenset(val)) for key,val in args.items())
    #return fset
    md = hashlib.md5()
    md.update(str(fset))
    return md.hexdigest()

class DataFactory(object):
    dtype = None
    def __init__(self, npz_fname):
        self.npz_fname = npz_fname
        try:
            self.npz = np.load(npz_fname)
            self.data = self.npz['nums']
            self.prev_hash = self.npz['args_hash']
        except (OSError, IOError, KeyError):
            self.data = None
    def write_npz(self, args):
        new_hash = args_hash(args)
        self.npz = np.savez(self.npz_fname, nums=self.data, args_hash=new_hash)
        
    def create_if_missing(self, args):
        if (self.data is not None) and (self.prev_hash == args_hash(args)):
            return self.data
        self.data = np.zeros(input_iterator_size(args), self.dtype)
        for n, val_set in enumerate(input_iterator(args)):
            info("creating new data for {}, {}".format(n, val_set))
            res = self.calculate(**val_set)
            newdata =dict(**val_set)
            newdata.update(**res)
            #temp and ugly
            for key, val in newdata.items():
                #pdb.set_trace()
                # weird but works. settle this later
                if isinstance(val, np.ndarray) and (val.size>1):
                    self.data[n][key][:] = val
                else:
                    self.data[n][key] = val
        self.write_npz(args)
        return self.data 
        

            
            
            


        


