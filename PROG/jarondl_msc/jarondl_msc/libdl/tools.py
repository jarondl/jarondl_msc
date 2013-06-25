#!/usr/bin/env python
"""  

"""
from __future__ import division  # makes true division instead of integer division


import numpy as np
import logging
import time
import decimal

import tables


### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)

#set up logging:
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug
     



def lazyprop_old(fn):
    """ based on http://stackoverflow.com/questions/3012421/python-lazy-property-decorator"""
    attr_name = '_lazy_' + fn.__name__
    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    @_lazyprop.setter
    def _lazyprop(self, value):
        setattr(self, attr_name, value)
    return _lazyprop
    
class lazyprop(object):
    '''
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    '''

    def __init__(self,fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self,obj,cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj,self.func_name,value)
        return value
    



def cached_get_key(function, filename, key='numerics', *args, **kwargs):
    """ Cached data getter.
        The idea is that we have a function producing numpy arrays
        that can be stored in a npz file. 
        If the file exists and the key exists, this data is returned.
        If the file exists but key doesn't, create data, add to file, and return data.
        If both file doesn't exist it will be created.
    """

    try:
        f = np.load(filename)
        fd = dict(f)
        f.close()
        
    except (OSError, IOError):
        fd = dict()
    debug("current keys in {} are {}".format(filename, fd.keys()))
    debug("We seek {}".format(key))
    if key not in fd:
        fd[key]  = function(*args, **kwargs)
        np.savez(filename, **fd)
    return fd[key]

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
        seed            = tables.Int64Col()
        
    return CKG
    
ckg_dtype = np.dtype([
                            ("model_name", (np.unicode_, 100)),
                            ("date", (np.unicode_, 20)),
                            ("number_of_points", (np.int64)),
                            ("bandwidth", (np.int64)),
                            ("dis_param", (np.float64)),
                            ("c", (np.float64)),
                            ("k", (np.float64)),
                            ("g", (np.complex128)),
                            ("heat_g", (np.complex128)),
                            ("seed", (np.int64))])

def ckg_psis_dtyper(N): 
    """ extend the dtype to have psi's, by length N """
    return  np.dtype(  ckg_dtype.descr + [
                            ("psi_1", (np.float64, N)),
                            ("psi_N", (np.float64, N)),
                            ("thouless_g", (np.float64, N)),
                            ("thouless_sum", np.float64),
                            ("eig_vals", (np.float64,N)),
                            ("phi", np.float64)])


def check_args_in_row(row, args_dict):
    return all(row[key]==val for (key, val) in args_dict.items() if not key.startswith("_"))
    
def old_h5_get_first_rownum_by_args(h5table, args_dict):
    for row in h5table.iterrows():
        if check_args_in_row(row,args_dict):
            return row.nrow

def h5_get_all_rownums_by_args(h5table, args_dict):
    #return h5table.get_where_list(build_silly_condition(args_dict))
    return h5table.getWhereList(build_silly_condition(args_dict))


def h5_get_first_rownum_by_args(h5table, args_dict):
    return h5_get_all_rownums_by_args(h5table, args_dict)[0]


def build_silly_condition(args_dict):
    cond = []
    for (key, val) in args_dict.items():
        if not key.startswith("_"):
            if isinstance(val, str):
                pval = '"{}"'.format(val)
            elif isinstance(val,np.float64) :
                pval = repr(val)
            else:
                pval = val
            cond.append('( {} == {} ) '.format(key,pval))
    #print("&".join(cond))
    #debug("&".join(cond))
    return "&".join(cond)
    
def fill_args_in_row(row,args_dict):
    for (key,val) in args_dict.items():
        if not key.startswith("_"):
            row[key] = val
        
    
def h5_create_if_missing(h5table, data_factory, factory_args):
    """ data_factory must be a function accepting factory_args,
    and returning a dictionary of data to fill..
    factory args are also filled in the table, so the must be compatible 
    with the table definition
    """
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    exists =  (len(h5_get_all_rownums_by_args(h5table, factory_args)) != 0)
    if not exists:
        info("creating new data for {}".format(factory_args))
        data = data_factory(**factory_args)
        nr = h5table.row
        fill_args_in_row(nr, factory_args)
        fill_args_in_row(nr, data)
        nr['date'] = t
        nr.append()
        h5table.flush()

