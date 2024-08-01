#! /usr/bin/env python

"""
Miscellaneous helper functions:
    NumPy helper
    text processing

Author:
    Zhi-Hao Cui
"""

import os
import subprocess as sub
from functools import reduce
import itertools as it
import numpy as np
import scipy.linalg as la
import h5py

from pyscf import lib
from pyscf.lib import pack_tril, unpack_tril
from libdmet.utils import logger as log
from libdmet.utils.iotools import read_poscar, write_poscar

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

# ******************************************************************
# NumPy helper
# ******************************************************************

def max_abs(x):
    """
    Equivalent to np.max(np.abs(x)), but faster.
    """
    if np.iscomplexobj(x):
        return np.abs(x).max()
    else:
        return max(np.max(x), abs(np.min(x)))

def mdot(*args):
    """
    Reduced matrix dot.
    """
    return reduce(np.dot, args)

def kdot(a, b):
    """
    Matrix dot with kpoints.
    """
    ka, s1_a, s2_a = a.shape
    kb, s1_b, s2_b = b.shape
    assert ka == kb
    res = np.zeros((ka, s1_a, s2_b), dtype=np.result_type(a.dtype, b.dtype))
    for k in range(ka):
        np.dot(a[k], b[k], out=res[k])
    return res

def get_spin_dim(arrays, non_spin_dim=3):
    """
    Get spin dimension for a list of arrays.
    """
    spin = 1
    for a in arrays:
        a = np.asarray(a)
        if a.ndim == non_spin_dim:
            continue
        elif a.ndim == non_spin_dim + 1:
            spin = max(spin, a.shape[0])
        else:
            raise ValueError
    return spin

def add_spin_dim(H, spin, non_spin_dim=3):
    """
    Add an additional dimension to array H.
    """
    H = np.asarray(H)
    if H.ndim == non_spin_dim:
        H = H[None]
    assert H.ndim == (non_spin_dim + 1)
    if H.shape[0] < spin:
        H = np.asarray((H[0],) * spin)
    return H

def save_h5(fname, arr, arr_name=None):
    """
    Save array like object to a .h5 file.

    Args:
        fname: file name.
        arr: array like object.
        arr_name: name for data set, default is the basename of fname.
    """
    if not fname.endswith(".h5"):
        fname = fname + ".h5"
    if arr_name is None:
        arr_name = os.path.basename(fname)[:-3]
    f = h5py.File(fname, 'w')
    f[arr_name] = np.asarray(arr)
    f.close()

def load_h5(fname, arr_name=None):
    """
    Load array like object from a .h5 file.

    Args:
        fname: file name.
        arr_name: name for data set.
    
    Returns:
        arr: array like object.
    """
    if not fname.endswith(".h5"):
        fname = fname + ".h5"
    if arr_name is None:
        arr_name = os.path.basename(fname)[:-3]
    f = h5py.File(fname, 'r')
    arr = np.asarray(f[arr_name])
    f.close()
    return arr

def flatten_list_of_array(lst):
    """
    flatten a list of ndarray:
    e.g.
    [[0, 1, 2], [[3, 4], [5, 6]]] -> [0, 1, 2, 3, 4, 5, 6], [(3,), (2, 2)]
    """
    shape_lst = [np.shape(x) for x in lst]
    res = [np.ravel(x) for x in lst]
    return np.hstack(res), shape_lst

def reshape_list_of_array(arr, shape_lst):
    """
    inverse function of flatten_list_of_array.
    """
    size_lst = [np.prod(x) for x in shape_lst]
    step_lst = np.cumsum(size_lst)[:-1]
    res = np.split(arr, step_lst)
    res = [np.reshape(res[i], shape_lst[i]) for i in range(len(shape_lst))]
    return res

def flatten_list_list_array(lst):
    lst1, lst1_shape = zip(*[flatten_list_of_array(lst[s]) for s in range(len(lst))])
    lst2, lst2_shape = flatten_list_of_array(lst1)
    return lst2, lst1_shape, lst2_shape

def reshape_list_list_array(lst2, lst1_shape, lst2_shape):
    lst1 = reshape_list_of_array(lst2, lst2_shape)
    lst = [reshape_list_of_array(lst1[s], lst1_shape[s]) for s in range(len(lst1_shape))]
    return lst

def search_idx1d(a, b):
    """
    For each element in a, find its (first) index in b.
    e.g.,
    a = [1, 5, 2, 3, 6, 7]
    b = [9, 2, 3, 5, 1, 4, 7, 8, 0, 1, 1, 2, 6]
    would return 
        [4, 3, 1, 2, 12, 6]
    if not exists, would return length of b.

    Args:
        a: int array
        b: int array
    
    Returns:
        idx: same length of a, its index in b.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    idx_sort_b = np.argsort(b, kind='mergesort')
    b_sorted = b[idx_sort_b]
    idx = np.searchsorted(b_sorted, a, side='left')
    idx = np.take(idx_sort_b, idx, mode="clip")
    idx[b[idx] != a] = len(b)
    return idx

"""
triu and tril indexing.
"""

def triu_mat2arr(mat):
    norb = mat.shape[0]
    return mat[np.triu_indices(norb)]

def triu_arr2mat(arr):
    norb = int(np.sqrt(len(arr) * 2))
    mat = np.zeros((norb, norb), dtype=arr.dtype)
    mat[np.triu_indices(norb)] = arr
    mat = mat + mat.conj().T
    if mat.dtype == int:
        mat[np.arange(norb), np.arange(norb)] //= 2
    else:
        mat[np.arange(norb), np.arange(norb)] *= 0.5
    return mat

def triu_diag_indices(n):
    """
    Get diagonal indices (of the unpacked matrix) in a compact triu arr.
    """
    return np.cumsum([0] + list(range(n, 1, -1)))

tril_mat2arr = pack_tril

tril_arr2mat = unpack_tril

def tril_diag_indices(n):
    """
    Get diagonal indices (of the unpacked matrix) in a compact tril arr.
    
    Args:
        n: length of matrix.

    Returns:
        diagonal indices.
    """
    return np.cumsum([0] + list(range(2, n + 1)))

def tril_idx(i, j):
    """
    For a pair / list of tril matrix indices i, j,
    find the corresponding compound indices ij in the tril array.
    
    Args:
        i, j

    Returns:
        ij: compound indices.
    """
    ij  = np.maximum(i, j)
    ij *= (ij + 1)
    ij //= 2
    ij += np.minimum(i, j)
    return ij

def tril_take_idx(idx_list1, idx_list2=None, compact=False):
    """
    Take a submatrix from tril array, 

    If one list is provide:
    return the corresponding compound indices in the tril array.
        e.g. idx_list = [1, 3]
              X     X
          00 01 02 03
        X 10 11 12 13
          20 21 22 23
        X 30 31 32 33
              X     X
          0   *  *  *
        X 1   2  *  *
          3   4  5  *
        X 6   7  8  9
        will return 2, 7, 9 (if compact), else 2, 7, 7, 9. 
        i.e. the indices of [(1, 1), (3, 1), (3, 3)].

    If two lists are provide:
    will return a set of indices for generating a 2D matrix.
        e.g. idx_list1 = [1, 3], idx_list2 = [1, 2]
              X  X   
          00 01 02 03
        X 10 11 12 13
          20 21 22 23
        X 30 31 32 33
              X  X   
          0   *  *  *
        X 1   2  *  *
          3   4  5  *
        X 6   7  8  9
        will return 2, 4, 7, 8,
        i.e. the indices of [(1, 1), (1, 2), (3, 1), (3, 2)].
    """
    if idx_list2 is None:
        idx_list2 = idx_list1
    if compact:
        l = len(idx_list1)
        x = np.tri(l, l, dtype=bool).ravel()
        idx = tril_idx(*lib.cartesian_prod((idx_list1, idx_list1))[x].T)
    else:
        idx = tril_idx(*lib.cartesian_prod((idx_list1, idx_list2)).T)
    return idx

def take2d_tril(a, idx_list1, idx_list2=None, compact=False):
    if idx_list2 is None:
        idx_list2 = idx_list1
    idx = tril_take_idx(idx_list1, idx_list2=idx_list2, compact=compact)
    if compact:
        return a[idx]
    else:
        return a[idx].reshape(len(idx_list1), len(idx_list2))

"""
ERI indexing, take sub-ERI.
"""

def eri_idx(i, j, k, l, nao=None, eri_format='s1'):
    if eri_format == 's1':
        return s1_idx(i, j, k, l, nao=nao)
    elif eri_format == 's4':
        return s4_idx(i, j, k, l, nao=nao)
    elif eri_format == 's8':
        return s8_idx(i, j, k, l, nao=nao)
    else:
        raise ValueError

def s1_idx(i, j, k, l, nao=None):
    return (i, j, k, l)

def s4_idx(i, j, k, l, nao=None):
    """
    Find the compound indices pair (ij, kl)
    for 4-fold symmetrized ERI with indices (i, j, k, l).
    
    Args:
        i, j, k, l: can be integer or a list of integers
        nao: if provide, i, j, k, l can be negative.
    
    Returns:
        (ij, kl)
    """
    i = np.asarray(i)
    j = np.asarray(j)
    k = np.asarray(k)
    l = np.asarray(l)
    if nao is None:
        assert (i >= 0).all()
        assert (j >= 0).all()
        assert (k >= 0).all()
        assert (l >= 0).all()
        ij = tril_idx(i, j)
        kl = tril_idx(k, l)
    else:
        ij = tril_idx(i % nao, j % nao)
        kl = tril_idx(k % nao, l % nao)
    return (ij, kl)

def s8_idx(i, j, k, l, nao=None):
    """
    Find the compound indices ijkl
    for 8-fold symmetrized ERI with indices (i, j, k, l).
    
    Args:
        i, j, k, l: can be integer or a list of integers
        nao: if provide, i, j, k, l can be negative.
    
    Returns:
        ijkl
    """
    i = np.asarray(i)
    j = np.asarray(j)
    k = np.asarray(k)
    l = np.asarray(l)
    if nao is None:
        assert (i >= 0).all()
        assert (j >= 0).all()
        assert (k >= 0).all()
        assert (l >= 0).all()
        ij = tril_idx(i, j)
        kl = tril_idx(k, l)
    else:
        ij = tril_idx(i % nao, j % nao)
        kl = tril_idx(k % nao, l % nao)
    return tril_idx(ij, kl)

def take_eri(eri, list1, list2, list3, list4, compact=False):
    """
    Take sub block of ERI.
    
    Args:
        eri: 1-fold symmetrized ERI, (nao, nao, nao, nao)
          or 4-fold symmetrized ERI, (nao_pair, nao_pair)
          or 8-fold symmetrized ERI, (nao_pair_pair,) 
        list1, list2, list3, list4: list of indices, can be negative.
        compact: only return the compact form of eri, only valid when lists
                 obey the permutation symmetry (only list1, list3 are used)

    Returns:
        res: (len(list1), len(list2), len(list3), len(list4)) if not compact
             else: compact shape depend only on list1 and list3.
    """
    if eri.ndim == 2: # 4-fold
        nao_a = int(np.sqrt(eri.shape[-2] * 2))
        nao_b = int(np.sqrt(eri.shape[-1] * 2))
        list1 = np.asarray(list1) % nao_a
        list2 = np.asarray(list2) % nao_a
        list3 = np.asarray(list3) % nao_b
        list4 = np.asarray(list4) % nao_b
        idx1 = tril_take_idx(list1, list2, compact=compact)
        idx2 = tril_take_idx(list3, list4, compact=compact)
        if compact:
            res = eri[np.ix_(idx1, idx2)]
        else:
            res = eri[np.ix_(idx1, idx2)].reshape(len(list1), len(list2), 
                                                  len(list3), len(list4))
    elif eri.ndim == 1: # 8-fold
        nao = int(np.sqrt(int(np.sqrt(eri.shape[-1] * 2)) * 2))
        list1 = np.asarray(list1) % nao
        list2 = np.asarray(list2) % nao
        list3 = np.asarray(list3) % nao
        list4 = np.asarray(list4) % nao
        idx1 = tril_take_idx(list1, list2, compact=compact)
        idx2 = tril_take_idx(list3, list4, compact=compact)
        if compact:
            res = eri[tril_take_idx(idx1, idx2, compact=compact)]
        else:
            res = eri[tril_take_idx(idx1, idx2)].reshape(len(list1), len(list2), 
                                                         len(list3), len(list4))
    else: # 1-fold
        res = eri[np.ix_(list1, list2, list3, list4)]
    return res

def tile_eri(eri_aa, eri_bb, eri_ab):
    assert eri_aa.shape == eri_bb.shape == eri_ab.shape
    if eri_aa.ndim == 2:
        nao_pair = eri_aa.shape[-1]
        nao = int(np.sqrt(nao_pair * 2))
        nso = nao * 2
        nso_pair = nso * (nso + 1) // 2
        aa_idx = tril_take_idx(np.arange(nao), np.arange(nao), compact=True)
        bb_idx = tril_take_idx(np.arange(nao, nso), np.arange(nao, nso), compact=True)
        eri = np.zeros((nso_pair, nso_pair), dtype=np.result_type(eri_aa, eri_bb, eri_ab))
        eri[np.ix_(aa_idx, aa_idx)] = eri_aa
        eri[np.ix_(bb_idx, bb_idx)] = eri_bb
        eri[np.ix_(aa_idx, bb_idx)] = eri_ab
        eri[np.ix_(bb_idx, aa_idx)] = eri_ab.conj().T
    elif eri_aa.ndim == 4:
        nao = eri_aa.shape[-1]
        nso = nao * 2
        eri = np.zeros((nso, nso, nso, nso), dtype=np.result_type(eri_aa, eri_bb, eri_ab))
        eri[:nao, :nao, :nao, :nao] = eri_aa
        eri[nao:, nao:, nao:, nao:] = eri_bb
        eri[:nao, :nao, nao:, nao:] = eri_ab
        eri[nao:, nao:, :nao, :nao] = eri_ab.conj().transpose(3, 2, 1, 0)
    else:
        raise ValueError("eri_aa has wrong shape: %s"%(str(eri_aa.shape)))
    return eri

def untile_eri(eri):
    if eri.ndim == 1:
        from pyscf import ao2mo
        nao = int(np.sqrt(int(np.sqrt(eri.shape[-1] * 2)) * 2))
        eri = ao2mo.restore(4, eri, nao)
    
    if eri.ndim == 2:
        nso_pair = eri.shape[-1]
        nso = int(np.sqrt(nso_pair * 2))
        nao = nso // 2
        aa_idx = tril_take_idx(np.arange(nao), np.arange(nao), compact=True)
        bb_idx = tril_take_idx(np.arange(nao, nso), np.arange(nao, nso), compact=True)
        eri_aa = eri[np.ix_(aa_idx, aa_idx)]
        eri_bb = eri[np.ix_(bb_idx, bb_idx)]
        eri_ab = eri[np.ix_(aa_idx, bb_idx)]
    elif eri.ndim == 4:
        nso = eri.shape[-1]
        nao = nso // 2
        eri_aa = eri[:nao, :nao, :nao, :nao]
        eri_bb = eri[nao:, nao:, nao:, nao:]
        eri_ab = eri[:nao, :nao, nao:, nao:]
    else:
        raise ValueError("eri shape (%s) is not supported" % (str(eri.shape)))
    return eri_aa, eri_bb, eri_ab

def cart2sph(x, y, z):
    """
    Conversion from cartisian to spherical coordinates. 

    sph coord convention:
        theta: measured from z axis
        phi: measured from x axis 
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    return r, theta, phi

def sph2cart(r, theta, phi):
    rsin_theta = r * np.sin(theta)
    x = np.cos(phi) * rsin_theta
    y = np.sin(phi) * rsin_theta
    z = r * np.cos(theta)
    return x, y, z

def cartesian_prod(arrays, out=None, order='C'):
    """
    Generate a cartesian product of input arrays.
    Support different order.
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = np.result_type(*arrays)
    nd = len(arrays)
    dims = [nd] + [len(x) for x in arrays]
    out = np.ndarray(dims, dtype, buffer=out)

    shape = [-1] + [1] * nd
    for i, arr in enumerate(arrays):
        out[i] = arr.reshape(shape[:nd-i])
    return out.reshape((nd, -1), order=order).T

def get_cart_prod_idx(cart_prod, mesh, order='C'):
    """
    Get the index of a set of cartesian product e.g. (0, 0, 1), (2, 0, 1),
    to their indices.

    Args:
        cart_prod: one cartesian product, e.g. (2, 0, 1)
                   or a list of cartesian products.
        mesh: a tuple of max value along each axis.
        order: 'C' or 'F'
    
    Returns:
        res: a / a list of indices.
    """
    cart_prod = np.asarray(cart_prod)
    if order == 'C':
        mesh = np.cumprod(mesh[::-1])[::-1]
        res = (np.dot(cart_prod[..., :-1], mesh[1:]) + cart_prod[..., -1]).astype(int)
    else:
        mesh = np.cumprod(mesh)
        res = (np.dot(cart_prod[..., 1:], mesh[:-1]) + cart_prod[..., 0]).astype(int)
    return res

# ******************************************************************
# text file processing
# ******************************************************************

def grep(string, f, A=None, B=None):
    cmd = """grep "%s" %s""" % (string, f)
    if A is not None:
        cmd += " -A %d" % A
    if B is not None:
        cmd += " -B %d" % B
    cmd += "; exit 0"
    return sub.check_output(cmd, shell=True).decode()[:-1]

def readlines_find(string, filename):
    with open("%s"%(filename), "r") as f:
        lines = f.readlines()
        line_num = [i for i, line in enumerate(lines) if string in line]
    return lines, line_num

def find(x, l):
    return [i for i, v in enumerate(l) if v == x]

def counted(fn):
    def wrapper(*args, **kwargs):
        wrapper.count+= 1
        return fn(*args, **kwargs)
    wrapper.count= 0
    wrapper.__name__= fn.__name__
    return wrapper

def format_idx(idx_list):
    """
    Format a list of integers, consecutive numbers are grouped.
    e.g. [1, 2, 3, 6, 7, 9, 5] -> '1-3, 6-7, 9, 5'
    https://docs.python.org/2.6/library/itertools.html#examples
    """
    string = ''
    for k, g in it.groupby(enumerate(idx_list), lambda ix: ix[0] - ix[1]):
        g = list(g)
        if len(g) > 1:
            string += '%d-%d, '%(g[0][1], g[-1][1])
        else:
            string += '%d, '%(g[0][1])
    return string[:-2]

if __name__ == '__main__':
    a = [1, 2, 3, 6, 7, 9, 5]
    print (format_idx(a))
    
    a = [np.arange(6).reshape(2,3), np.arange(3), np.arange(12).reshape(3, 4)] 
    arr, shape_lst = flatten_list_of_array(a)
    print (arr)
    print (shape_lst)
    a_re = reshape_list_of_array(arr, shape_lst)
    print (a_re)
    
    b = [-np.arange(6).reshape(2,3), -np.arange(3), -np.arange(12).reshape(4, 3)] 
    c = [a, b]
    lst2, lst1_shape, lst2_shape = flatten_list_list_array(c)
    print ("c")
    print (c)
    print ("")
    print (lst2)
    print (lst1_shape)
    print (lst2_shape)
    print ("")
    print (reshape_list_list_array(lst2, lst1_shape, lst2_shape))
    string = "XXX"
    #filename = "./test.dat"
    #lines, line_num = readlines_find(string, filename)
    #print (lines)
    #print (line_num)
    #for num in line_num:
    #    print (lines[num])
