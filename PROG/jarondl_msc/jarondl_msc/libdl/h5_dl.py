#!/usr/bin/env python
"""  
This is my abstraction layer over python-tables for dealing with hdf5 files
"""
from __future__ import division  # makes true division instead of integer division


import numpy as np
import logging
import time
from collections import OrderedDict
import itertools

import tables

from .tools import ev_and_pn_class, ev_pn_g_class, c_k_g_class, h5_create_if_missing
from jarondl_msc import models
from . import sparsedl
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



class DataFactory(object):
    def __init__(self, h5file):
        try:
            self.h5table = h5file.getNode(self.table_group + "/" + self.table_name)
        except tables.exceptions.NoSuchNodeError:
            self.h5table = h5file.createTable(self.table_group, 
                           self.table_name, self.table_class, 
                           self.table_description, createparents=True)
    
    def input_iterator(self, args):
        ovar_args = OrderedDict(args)
        for val_set in itertools.product(*ovar_args.values()):
            yield dict( zip(ovar_args.keys(), val_set))

    def create_if_missing(self, args):
        nrows = []
        for val_set in self.input_iterator(args):
            h5_create_if_missing(self.h5table, self.calculate, val_set)
            nrows.append(h5_get_first_rownum_by_args(self.h5table, val_set))
        return self.h5table[nrows]    
        
        
class Factory_Transmission_g(DataFactory):
    table_group   = '/ckg'
    table_name = 'ckg'
    table_class = c_k_g_class()
    table_description = ' ckg ' 
    def calculate(self, model_name, number_of_points, bandwidth,dis_param, c,k):
        if model_name != "Anderson":
            raise Error("NotImpelmented")
        m = models.Model_Anderson_DD_1d(number_of_points=number_of_points,
                 bandwidth=bandwidth, dis_param=dis_param, periodic=False)
        g = sparsedl.A_matrix_inv(m.rate_matrix,c,k)
        return {'g': g}
            
            
            


def ev_and_pn_class(_number_of_points):
    class Ev_and_PN(tables.IsDescription):
        
        model_name      = tables.StringCol(100)
        date            = tables.StringCol(20)
              
        number_of_points= tables.Int64Col()
        bandwidth       = tables.Int64Col()
        dis_param       = tables.Float64Col()
        eig_vals        = tables.Float64Col(_number_of_points)
        PN              = tables.Float64Col(_number_of_points)
    return Ev_and_PN

def ev_pn_g_class(_number_of_points):
    class Ev_PN_g(tables.IsDescription):
        
        model_name      = tables.StringCol(100)
        date            = tables.StringCol(20)
              
        number_of_points= tables.Int64Col()
        bandwidth       = tables.Int64Col()
        dis_param       = tables.Float64Col()
        eig_vals        = tables.Float64Col(_number_of_points)
        PN              = tables.Float64Col(_number_of_points)
        g               = tables.Float64Col(_number_of_points)
        precision       = tables.Float64Col(_number_of_points)
    return Ev_PN_g

    

def ev_pn_gggg_class(_number_of_points):
    class Ev_PN_g(tables.IsDescription):
        
        model_name      = tables.StringCol(100)
        date            = tables.StringCol(20)
              
        number_of_points= tables.Int64Col()
        bandwidth       = tables.Int64Col()
        dis_param       = tables.Float64Col()
        eig_vals        = tables.Float64Col(_number_of_points)
        PN              = tables.Float64Col(_number_of_points)
        g               = tables.Float64Col(_number_of_points)
        dcg             = tables.Float64Col(_number_of_points)
        precision       = tables.Float64Col(_number_of_points)
    return Ev_PN_g

def c_k_g_class():
    class CKG(tables.IsDescription):
        model_name      = tables.StringCol(100)
        date            = tables.StringCol(20)
              
        number_of_points= tables.Int64Col()
        bandwidth       = tables.Int64Col()
        dis_param       = tables.Float64Col()
        c               = tables.Float64Col()
        k               = tables.Float64Col()
        g               = tables.ComplexCol(16)
        
    return CKG
        


