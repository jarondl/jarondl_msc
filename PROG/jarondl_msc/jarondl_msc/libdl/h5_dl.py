#!/usr/bin/env python
"""  
This is my abstraction layer over python-tables for dealing with hdf5 files
"""
from __future__ import division  # makes true division instead of integer division


import numpy as np
import logging

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

class DataFactory(object):
    def __init__(self, h5file):
        try:
            self.h5table = h5file.getNode(self.table_group + "/" + self.table_name)
        except tables.exceptions.NoSuchNodeError:
            self.h5table = h5file.createTable(self.table_group, 
                           self.table_name, self.table_class, 
                           self.table_description, createparents=True)
    


    def create_if_missing(self, args):
        nrows = []
        for val_set in input_iterator(args):
            h5_create_if_missing(self.h5table, self.calculate, val_set)
            nrows.append(h5_get_first_rownum_by_args(self.h5table, val_set))
        return self.h5table[nrows]    
        

            
            
            


        


