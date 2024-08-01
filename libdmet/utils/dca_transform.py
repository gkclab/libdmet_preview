#! /usr/bin/env python

from functools import reduce
import numpy as np
import scipy.linalg as la
import itertools as it
from cmath import *

def dca_rot1d(nc, nsc):
    """
    nc - number of cells
    nsc - number of sites in a cell
    """

    X = np.arange(nsc)
    x1 = nsc * np.arange(nc)

    K = 1. / nsc * np.arange(nsc)
    k1 = 1. / (nc * nsc) * np.arange(nc)
    k1 -= np.mean(k1)

    rotA = np.sum([np.exp(2.j*pi*k1[m] * (x1.reshape(-1,1) - X)) for m in \
            range(nc)], axis=0)

    rotB = np.exp(-2.j*pi * K.reshape(-1,1) * X)

    rotC = np.exp(2.j*pi * X.reshape(-1,1) * K)

    # human-readable notation
    #rot = np.einsum("ik,nk,jn->ijk", rotA, rotB, rotC)
    rot = rotA[:, np.newaxis, :] * np.dot(rotC, rotB)[np.newaxis, :, :]
    if np.allclose(rot.imag, 0.):
        rot = rot.real
    rot /= nc * nsc
    return rot

def to_sparse(H):
    idx = np.asarray(np.nonzero(abs(H) > 1e-12)).T
    return [(tuple(i), H[tuple(i)]) for i in idx]

def to_full(nc, nsc, Hsparse):
    H = np.zeros((nc, nsc))
    for idx, val in Hsparse:
        H[idx] = val
    return H

def rotate_term(rot1d, H):
    # rotate a single 1d Hamiltonian term
    nc, nsc = H.shape
    assert(rot1d.shape == (nc, nsc, nsc))
    
    H1 = np.zeros_like(H, dtype = rot1d.dtype)
    
    for i in range(nc):
        for l in range(nc):
            H1[i] += np.dot(rot1d[i-l % nc], H[l])

    return H1

def transform_terms(nc, nsc, terms):
    # map rotate_term over a few terms
    rot = dca_rot1d(nc, nsc)
    def rot_term(t):
        if isinstance(t, tuple):
            return rotate_term(rot, to_full(nc, nsc, [(t, 1.)]))
        else:
            return rotate_term(rot, t)
    return [rot_term(x) for x in terms]

def transformHam(lattice, cell, H, sparse = True):
    r"""
    takes multi-dimensional Hamiltonian
    Hamiltonian is rewritten as
    
        H = \sum_i outer(H_i^x, H_i^y)
    
    therefore
    
        H' = \sum_i outer(dot(rotX, H_i^x), dot(rotY, H_i^y))
    
    gives a cheaper algorithm

    instead of doing

        H' = dot(outer(rotX, rotY), \sum_i outer(H_i^x, H_i^y))
    """

    dim = len(lattice)
    if isinstance(H, np.ndarray):
        Hsparse = to_sparse(H)
    else:
        Hsparse = H
    
    nterms = len(Hsparse)
    vals = [v for (idx, v) in Hsparse]
    terms = [idx for (idx, v) in Hsparse]

    dca_terms_by_dim = []
    for d in range(dim):
        term_d = [idx[d::dim] for idx in terms]
        dca_terms_by_dim.append(transform_terms(lattice[d], cell[d], term_d))
    
    if sparse:
        def comb_idx(indices):
            s = [None] * (dim * 2)
            for d in range(dim):
                s[d::dim] = indices[d]
            return tuple(s)
        
        nonzeros = []
        for i in range(nterms):
            v = vals[i]
            base = map(to_sparse, [dca_terms_by_dim[d][i] for d in range(dim)])
            for c in it.product(*base):
                pos = comb_idx([idx for (idx, val) in c])
                factor = np.prod([val for (idx, val) in c])
                nonzeros.append((pos, v * factor))

        return sorted(nonzeros)

    else:
        def reshape_dim(d):
            s = [1] * (dim * 2)
            s[d] = lattice[d]
            s[d+dim] = cell[d]
            return tuple(s)

        Hnew = np.zeros(list(lattice) + list(cell))
        for i in range(nterms):
            v = vals[i]
            Hnew += v * reduce(np.multiply, \
                    [dca_terms_by_dim[d][i].reshape(reshape_dim(d)) for d in range(dim)])
        return Hnew


if __name__ == "__main__":

    lattice = (33, 31)
    cell = (8, 8)
    H = np.zeros(lattice + cell)
    # because of translational invariance
    # indices are H(x2-x1, y2-y1, z2-z1, X2-X1, Y2-Y1, Z2-Z1)
    # where ai is supercell index and Ai is site index
    H[0, 0, 0, 1] = -1
    H[0, 0, 1, 0] = -1
    H[-1, 0, -1, 0] = -1
    H[0, -1, 0, -1] = -1
    #t1 = 0.2
    #H[0, 0, 1, 1] = t1
    #H[0, -1, 1, -1] = t1
    #H[-1, 0, -1, 1] = t1
    #H[-1, -1, -1, -1] = t1

    Hnew = transformHam(lattice, cell, H)
    
    for idx, v in Hnew:
        print (idx, v)
