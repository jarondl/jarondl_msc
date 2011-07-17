#!/usr/bin/python
"""
    A module to create a npz file with sparsedl.NN_matrix.
    INPUT: number of nodes, and filename.
    OUTPUT: a npz file with matrix A and initial condition rho0
        It can be imported in another python file:
            import create_data
            create_data.mk_NN_file(300, 'examplefile')
        And it can be used as a standalone script:
            ./create_data.py 300 examplefile

    If no arguments are given, the default values are: 200 nodes, and 'data/mat200.1.npz' as filename.
"""
import sparsedl
import numpy
import sys


def mk_NN_file(nodes=200, filename='data/mat200.1.npz'):
    """ Crates a file with a NxN Nearest Neighbor matrix called A, and an initial rho0.
        If no values are supplied, the defaults are used.  """
    A = sparsedl.NN_matrix(nodes)
    rho0 = numpy.zeros(nodes)
    rho0[nodes//2] = 1
    numpy.savez(filename, A=A, rho0=rho0)

# This allows us to use this module as a script.
if __name__ == "__main__":
    if len(sys.argv)==3:
        mk_NN_file(int(sys.argv[1]), sys.argv[2])
    else:
        mk_NN_file()
