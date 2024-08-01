#! /usr/bin/env python

"""
Test misc.
"""
import pytest

def test_mdot():
    import numpy as np
    from functools import reduce
    from libdmet.utils import misc
    a = np.random.random((2, 3))
    b = np.random.random((3, 3))
    c = np.random.random((3, 5))
    d = np.random.random((5,))
    e = np.random.random((2, 1))
    f = misc.mdot(a, b, c, d, e)
    f_ref = reduce(np.dot, (a, b, c, d, e))
    assert misc.max_abs(f - f_ref) < 1e-10
    assert f.shape == (1,)

def test_h5():
    import numpy as np
    from libdmet.utils import misc
    a = np.random.random((2, 3))
    misc.save_h5("a", a)
    a_re = misc.load_h5("a")
    assert np.allclose(a, a_re)

def test_reshape():
    import numpy as np
    from libdmet.utils.misc import flatten_list_of_array, \
            reshape_list_of_array, flatten_list_list_array, \
            reshape_list_list_array, readlines_find
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

def test_spin_dim():
    import numpy as np
    from libdmet.utils.misc import get_spin_dim
    a = np.random.random((2, 2, 3))
    b = np.random.random((2, 2, 3))
    c = np.random.random((2, 2, 3))
    spin = get_spin_dim((a, b, c))
    assert spin == 1
    
    a = np.random.random((2, 2, 2, 3))
    spin = get_spin_dim((a, b, c))
    assert spin == 2

    b = np.random.random((1, 2, 2, 3))
    spin = get_spin_dim((a, b, c))
    assert spin == 2
    
    c = np.random.random((5, 2, 2, 3))
    spin = get_spin_dim((a, b, c))
    assert spin == 5
    
    c = np.random.random((2, 2, 2, 3))
    spin = get_spin_dim((a, b, c))
    assert spin == 2
    
    a = np.random.random((1, 3, 2, 3))
    b = np.random.random((1, 5, 2, 3))
    c = np.random.random((4, 2, 3))
    d = np.random.random((3, 2, 3))
    spin = get_spin_dim((a, b, c, d))
    assert spin == 1
    
    c = np.random.random((2, 4, 2, 3))
    spin = get_spin_dim((c,))
    assert spin == 2

def test_search_idx1d():
    import numpy as np
    from libdmet.utils import search_idx1d
    a = [1, 5, 2, 3, 6, 7]
    b = [9, 2, 3, 5, 1, 4, 7, 8, 0, 1, 1, 2, 6]
    ref = [4, 3, 1, 2, 12, 6]
    idx = search_idx1d(a, b)
    assert np.allclose(idx, ref)

    a = [1, 1, 5, 2, 2, 3, 6, 7]
    b = [9, 2, 5, 1, 4, 7, 8, 0, 1, 1, 2, 6]
    ref = [3, 3, 2, 1, 1, 12, 11, 5]
    idx = search_idx1d(a, b)
    assert np.allclose(idx, ref)

@pytest.mark.parametrize(
    "norb", [1, 2, 5, 8]
)
@pytest.mark.parametrize(
    "dtype", [int, float, complex]
)
def test_tril_triu(norb, dtype):
    import numpy as np
    from libdmet.utils import misc
    mat = np.arange(norb * norb).reshape(norb, norb).astype(dtype)
    mat = mat + mat.T
    
    arr = misc.tril_mat2arr(mat)
    mat_re = misc.tril_arr2mat(arr)
    assert np.allclose(mat, mat_re)
    
    mat[np.diag_indices(norb)] += 5
    arr[misc.tril_diag_indices(norb)] += 5
    mat_re = misc.tril_arr2mat(arr)
    assert np.allclose(mat, mat_re)

    arr = misc.triu_mat2arr(mat)
    mat_re = misc.triu_arr2mat(arr)
    assert np.allclose(mat, mat_re)
    
    mat[np.diag_indices(norb)] -= 3
    arr[misc.triu_diag_indices(norb)] -= 3
    mat_re = misc.triu_arr2mat(arr)
    assert np.allclose(mat, mat_re)

def test_tril_idx():
    import numpy as np
    from libdmet.utils import misc
    from libdmet.utils.misc import tril_idx, max_abs
    
    norb = 7
    dtype = int
    mat = np.arange(norb * norb).reshape(norb, norb).astype(dtype)
    mat = mat + mat.T
    arr = misc.tril_mat2arr(mat)
    
    # one 1 indices
    i = 5
    j = 2
    mat[i, j] += 10
    mat[j, i] += 10
    arr[tril_idx(i, j)] += 10
    mat_re = misc.tril_arr2mat(arr)
    assert np.allclose(mat, mat_re)
    
    i = 2
    j = 5
    mat[i, j] += 10
    mat[j, i] += 10
    arr[tril_idx(i, j)] += 10
    mat_re = misc.tril_arr2mat(arr)
    assert np.allclose(mat, mat_re)
    
    i = 5
    j = 5
    mat[i, j] += 10
    #mat[j, i] += 10
    arr[tril_idx(i, j)] += 10
    mat_re = misc.tril_arr2mat(arr)
    assert np.allclose(mat, mat_re)
    
    # a list of indices
    i = [3, 5, 6, 1]
    j = [5, 5, 0, 2]
    for x, y in zip(i, j):
        mat[x, y] += 20
        if x != y:
            mat[y, x] += 20
    
    arr[tril_idx(i, j)] += 20
    mat_re = misc.tril_arr2mat(arr)
    assert np.allclose(mat, mat_re)
    
    i = [1]
    j = [2]
    for x, y in zip(i, j):
        mat[x, y] += 20
        if x != y:
            mat[y, x] += 20
    
    arr[tril_idx(i, j)] += 20
    mat_re = misc.tril_arr2mat(arr)
    assert np.allclose(mat, mat_re)
    
    # should be consistent with tril_diag_indices
    diag_idx_ref = misc.tril_diag_indices(norb)
    diag_idx = tril_idx(range(norb), np.arange(norb))
    assert np.allclose(diag_idx, diag_idx_ref)

def test_tril_take_idx():
    import numpy as np
    from pyscf import ao2mo
    from libdmet.utils import misc
    from libdmet.utils.misc import tril_take_idx, take2d_tril, max_abs
    
    norb = 7
    dtype = np.double
    mat = np.arange(norb * norb).reshape(norb, norb).astype(dtype)
    mat = mat + mat.T
    arr = misc.tril_mat2arr(mat)
    
    # 1. compact == True
    idx = tril_take_idx([1, 3], compact=True)
    assert (idx == np.array((2, 7, 9))).all()
    
    # can be used to reorder the compact array.
    perm_idx = np.random.permutation(norb)
    arr_permed = arr[tril_take_idx(perm_idx, compact=True)]
    mat_permed = mat[np.ix_(perm_idx, perm_idx)]
    mat_re = misc.unpack_tril(arr_permed)
    assert max_abs(mat_re - mat_permed) < 1e-10

    # 2. compact == False
    idx = tril_take_idx([1, 3], compact=False)
    assert (idx == np.array((2, 7, 7, 9))).all()
    
    idx = tril_take_idx([1, 3], [1, 2])
    assert (idx == np.array((2, 4, 7, 8))).all()
    
    idx = tril_take_idx([3, 1], [2, 1])
    assert (idx == np.array((8, 7, 4, 2))).all()
    
    idx_list1 = [3, 2, 6, 5]
    idx_list2 = [3, 0, 1]
    idx = tril_take_idx(idx_list1, idx_list2)
    res = arr[idx].reshape(len(idx_list1), len(idx_list2))
    res_ref = mat[np.ix_(idx_list1, idx_list2)]
    assert np.allclose(res, res_ref)
    
    res2 = take2d_tril(arr, idx_list1, idx_list2)
    assert np.allclose(res2, res_ref)
    
    idx_list1 = [3, 2, 6, 5]
    idx_list2 = [4, 1, 5, 5]
    idx = tril_take_idx(idx_list1, idx_list2)
    res = arr[idx].reshape(len(idx_list1), len(idx_list2))
    res_ref = mat[np.ix_(idx_list1, idx_list2)]
    assert np.allclose(res, res_ref)
    
    res2 = take2d_tril(arr, idx_list1, idx_list2)
    assert np.allclose(res2, res_ref)

def test_s4_idx():
    import itertools as it
    import numpy as np
    from pyscf import ao2mo
    from libdmet.utils import misc
    from libdmet.utils.misc import s4_idx, max_abs
    from libdmet.system.integral import check_perm_symm
    
    norb = 7
    dtype = np.double
    eri = np.arange(norb * norb * norb * norb)\
            .reshape(norb, norb, norb, norb).astype(dtype)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    #eri = eri + eri.transpose(2, 3, 0, 1)
    
    eri_s4 = ao2mo.restore(4, eri, norb)
    i = 5
    j = 2
    k = 3
    l = 6
    assert np.allclose(eri[i, j, k, l], eri_s4[s4_idx(i, j, k, l)])
    
    i = 2
    j = 2
    k = 0
    l = 0
    assert np.allclose(eri[i, j, k, l], eri_s4[s4_idx(i, j, k, l)])
    
    i = np.random.randint(0, norb)
    j = np.random.randint(0, norb)
    k = np.random.randint(0, norb)
    l = np.random.randint(0, norb)
    assert np.allclose(eri[i, j, k, l], eri_s4[s4_idx(i, j, k, l)])
    
    i = -5
    j = 2
    k = -3
    l = 0
    assert np.allclose(eri[i, j, k, l], eri_s4[s4_idx(i, j, k, l, nao=norb)])
    
    i = -1
    j = -3
    k = -3
    l = -3
    assert np.allclose(eri[i, j, k, l], eri_s4[s4_idx(i, j, k, l, nao=norb)])

    # a list of indices
    i = [3, 5, 6, 1]
    j = [5, 5, 0, 2]
    k = [5, 5, 0, 2]
    l = [1, 5, 0, 2]
    
    def idx_symm(i, j, k, l):
        idx = set()
        for x, y, z, w in zip(i, j, k, l):
            idx.add((x, y, z, w))
            idx.add((y, x, z, w))
            idx.add((x, y, w, z))
            idx.add((y, x, w, z))
        idx = tuple(zip(*list(idx)))
        return idx 
    
    eri[idx_symm(i, j, k, l)] += 20
    eri_s4[s4_idx(i, j, k, l)] += 20
    eri_re = ao2mo.restore(1, eri_s4, norb)
    assert np.allclose(eri, eri_re)
    
    i = [-1, -5]
    j = [-3, -5]
    k = [-3, -5]
    l = [-5, -3]
    
    eri[idx_symm(i, j, k, l)] += 20
    eri_s4[s4_idx(i, j, k, l, norb)] += 20
    eri_re = ao2mo.restore(1, eri_s4, norb)
    assert np.allclose(eri, eri_re)

def test_s8_idx():
    import numpy as np
    from pyscf import ao2mo
    from libdmet.utils import misc
    from libdmet.utils.misc import s8_idx, max_abs
    
    norb = 7
    dtype = np.double
    eri = np.arange(norb * norb * norb * norb)\
            .reshape(norb, norb, norb, norb).astype(dtype)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)
    
    eri_s8 = ao2mo.restore(8, eri, norb)
    i = 5
    j = 2
    k = 3
    l = 6
    assert np.allclose(eri[i, j, k, l], eri_s8[s8_idx(i, j, k, l)])

    i = np.random.randint(0, norb)
    j = np.random.randint(0, norb)
    k = np.random.randint(0, norb)
    l = np.random.randint(0, norb)
    assert np.allclose(eri[i, j, k, l], eri_s8[s8_idx(i, j, k, l)])
    
    i = -5
    j = 2
    k = -3
    l = 0
    assert np.allclose(eri[i, j, k, l], eri_s8[s8_idx(i, j, k, l, nao=norb)])
    
    i = -1
    j = -3
    k = -3
    l = -3
    assert np.allclose(eri[i, j, k, l], eri_s8[s8_idx(i, j, k, l, nao=norb)])

    # a list of indices
    i = [3, 5, 6, 1]
    j = [5, 5, 0, 2]
    k = [5, 5, 0, 2]
    l = [1, 5, 0, 2]
    
    def idx_symm(i, j, k, l):
        idx = set()
        for x, y, z, w in zip(i, j, k, l):
            idx.add((x, y, z, w))
            idx.add((y, x, z, w))
            idx.add((x, y, w, z))
            idx.add((y, x, w, z))
            idx.add((z, w, x, y))
            idx.add((z, w, y, x))
            idx.add((w, z, x, y))
            idx.add((w, z, y, x))
        idx = tuple(zip(*list(idx)))
        return idx 
    
    eri[idx_symm(i, j, k, l)] += 20
    eri_s8[s8_idx(i, j, k, l)] += 20
    eri_re = ao2mo.restore(1, eri_s8, norb)
    assert np.allclose(eri, eri_re)

    i = [-1, -5]
    j = [-3, -5]
    k = [-3, -5]
    l = [-5, -3]
    
    eri[idx_symm(i, j, k, l)] += 20
    eri_s8[s8_idx(i, j, k, l, norb)] += 20
    eri_re = ao2mo.restore(1, eri_s8, norb)
    assert np.allclose(eri, eri_re)

def test_eri_idx():
    import numpy as np
    from pyscf import ao2mo
    from libdmet.utils import misc
    from libdmet.utils.misc import eri_idx, max_abs
    
    norb = 7
    dtype = np.double
    eri = np.arange(norb * norb * norb * norb)\
            .reshape(norb, norb, norb, norb).astype(dtype)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)
    
    eri_s4 = ao2mo.restore(4, eri, norb)
    eri_s8 = ao2mo.restore(8, eri, norb)

    i = np.random.randint(0, norb)
    j = np.random.randint(0, norb)
    k = np.random.randint(0, norb)
    l = np.random.randint(0, norb)
    assert np.allclose(eri[eri_idx(i, j, k, l, None, 's1')], \
            eri_s4[eri_idx(i, j, k, l, norb, 's4')])
    assert np.allclose(eri[eri_idx(i, j, k, l, None, 's1')], \
            eri_s8[eri_idx(i, j, k, l, norb, 's8')])
    
    i = -5
    j = 2
    k = -3
    l = 0
    assert np.allclose(eri[eri_idx(i, j, k, l, None, 's1')], \
            eri_s4[eri_idx(i, j, k, l, norb, 's4')])
    assert np.allclose(eri[eri_idx(i, j, k, l, None, 's1')], \
            eri_s8[eri_idx(i, j, k, l, norb, 's8')])
    
    # a list of indices
    i = [3, 5, 6, 1]
    j = [5, 5, 0, 2]
    k = [5, 5, 0, 2]
    l = [1, 5, 0, 2]
    assert np.allclose(eri[eri_idx(i, j, k, l, norb, 's1')], \
            eri_s4[eri_idx(i, j, k, l, norb, 's4')])
    assert np.allclose(eri[eri_idx(i, j, k, l, norb, 's1')], \
            eri_s8[eri_idx(i, j, k, l, norb, 's8')])
    
    i = [-1, -5]
    j = [-3, -5]
    k = [-3, -5]
    l = [-5, -3]
    assert np.allclose(eri[eri_idx(i, j, k, l, norb, 's1')], \
            eri_s4[eri_idx(i, j, k, l, norb, 's4')])
    assert np.allclose(eri[eri_idx(i, j, k, l, norb, 's1')], \
            eri_s8[eri_idx(i, j, k, l, norb, 's8')])

def test_take_eri():
    import numpy as np
    from pyscf import ao2mo
    from libdmet.utils import misc
    from libdmet.utils.misc import take_eri, max_abs
    
    norb = 7
    dtype = np.double
    eri = np.arange(norb * norb * norb * norb)\
            .reshape(norb, norb, norb, norb).astype(dtype)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)

    eri_s4 = ao2mo.restore(4, eri, norb)
    eri_s8 = ao2mo.restore(8, eri, norb)
    
    list1 = [0, 2]
    list2 = [5, 3, 6]
    list3 = [-2]
    list4 = [5, -3, 1, 1]
    eri_1 = take_eri(eri, list1, list2, list3, list4)
    eri_4 = take_eri(eri_s4, list1, list2, list3, list4)
    eri_8 = take_eri(eri_s8, list1, list2, list3, list4)
    assert max_abs(eri_4 - eri_1) < 1e-12
    assert max_abs(eri_8 - eri_1) < 1e-12
    
    list1 = range(norb)
    list2 = [5, -4, 6]
    list3 = [4, 3]
    list4 = [4, 3]

    eri_1 = take_eri(eri, list1, list2, list3, list4)
    eri_4 = take_eri(eri_s4, list1, list2, list3, list4)
    eri_8 = take_eri(eri_s8, list1, list2, list3, list4)
    
    assert max_abs(eri_4 - eri_1) < 1e-12
    assert max_abs(eri_8 - eri_1) < 1e-12
    
    # can be used to reorder the eri.
    perm_idx = np.random.permutation(norb)
    eri = eri[perm_idx, :, :, :]
    eri = eri[:, perm_idx, :, :]
    eri = eri[:, :, perm_idx, :]
    eri = eri[:, :, :, perm_idx]

    eri_4 = take_eri(eri_s4, perm_idx, perm_idx, perm_idx, perm_idx, \
            compact=True)
    eri_re = ao2mo.restore(1, eri_4, norb)
    assert max_abs(eri_re - eri) < 1e-12
    
    eri_8 = take_eri(eri_s8, perm_idx, perm_idx, perm_idx, perm_idx, \
            compact=True)
    eri_re = ao2mo.restore(1, eri_8, norb)
    assert max_abs(eri_re - eri) < 1e-12

def test_tile_eri():
    import numpy as np
    from pyscf import ao2mo
    from libdmet.utils import misc
    from libdmet.utils.misc import tile_eri, untile_eri
    
    norb = 7
    def make_eri(norb):
        eri = np.random.random((norb, norb, norb, norb))
        eri = eri + eri.transpose(1, 0, 2, 3)
        eri = eri + eri.transpose(0, 1, 3, 2)
        eri = eri + eri.transpose(2, 3, 0, 1)
        eri_s4 = ao2mo.restore(4, eri, norb)
        return eri, eri_s4 
    
    eri_aa, eri_aa_s4 = make_eri(norb)
    eri_bb, eri_bb_s4 = make_eri(norb)
    eri_ab, eri_ab_s4 = make_eri(norb)
    
    eri_ref = tile_eri(eri_aa, eri_bb, eri_ab)
    eri_s4  = tile_eri(eri_aa_s4, eri_bb_s4, eri_ab_s4)
    eri_re  = ao2mo.restore(1, eri_s4, norb*2)
    assert misc.max_abs(eri_re - eri_ref) < 1e-12
    
    eri_aa_re, eri_bb_re, eri_ab_re = untile_eri(eri_s4)
    assert misc.max_abs(eri_aa_re - eri_aa_s4) < 1e-12
    assert misc.max_abs(eri_bb_re - eri_bb_s4) < 1e-12
    assert misc.max_abs(eri_ab_re - eri_ab_s4) < 1e-12
    
    eri_aa_re, eri_bb_re, eri_ab_re = untile_eri(eri_ref)
    assert misc.max_abs(eri_aa_re - eri_aa) < 1e-12
    assert misc.max_abs(eri_bb_re - eri_bb) < 1e-12
    assert misc.max_abs(eri_ab_re - eri_ab) < 1e-12

def test_cart_sph():
    import numpy as np
    import scipy.linalg as la
    from libdmet.utils.misc import cart2sph, sph2cart
    x, y, z = 0.0, 0.0, 1.0
    r, theta, phi = cart2sph(x, y, z)
    assert abs(r - 1.0) < 1e-12
    assert abs(theta) < 1e-12
    assert abs(phi) < 1e-12
    
    x_re, y_re, z_re = sph2cart(r, theta, phi)
    assert abs(x_re - x) < 1e-12 
    assert abs(y_re - y) < 1e-12 
    assert abs(z_re - z) < 1e-12 

@pytest.mark.parametrize(
    "order", ['C', 'F']
)
def test_cartesian_prod(order):
    import numpy as np
    from libdmet.utils.misc import cartesian_prod, get_cart_prod_idx
    
    mesh = [5,]
    a = [range(x) for x in mesh]
    res = cartesian_prod(a, out=None, order=order)
    i = np.random.randint(0, len(res))
    j = np.random.randint(0, len(res))
    k = i
    idx = get_cart_prod_idx([res[i], res[j], res[k]], mesh=mesh, order=order)
    assert idx[0] == i
    assert idx[1] == j
    assert idx[2] == k
    
    mesh = [5, 4]
    a = [range(x) for x in mesh]
    res = cartesian_prod(a, out=None, order=order)
    i = np.random.randint(0, len(res))
    j = np.random.randint(0, len(res))
    k = i
    idx = get_cart_prod_idx([res[i], res[j], res[k]], mesh=mesh, order=order)
    assert idx[0] == i
    assert idx[1] == j
    assert idx[2] == k
    
    mesh = [3, 4, 5]
    a = [range(x) for x in mesh]
    res = cartesian_prod(a, out=None, order=order)
    i = np.random.randint(0, len(res))
    j = np.random.randint(0, len(res))
    idx = get_cart_prod_idx([res[i], res[j]], mesh=mesh, order=order)
    assert idx[0] == i
    assert idx[1] == j
    
    mesh = [8, 3, 5, 1, 9]
    a = [range(x) for x in mesh]
    res = cartesian_prod(a, out=None, order=order)
    i = np.random.randint(0, len(res))
    idx = get_cart_prod_idx(res[i], mesh=mesh, order=order)
    assert idx == i

def test_format_idx():
    from libdmet.utils.misc import format_idx
    data = [4, 5, 6, 5, 2, 1,  4,5,6, 10, 15,16,17,18, 22, 25,26,27,28]
    string = format_idx(data)
    assert string == "4-6, 5, 2, 1, 4-6, 10, 15-18, 22, 25-28"

if __name__ == "__main__":
    test_tile_eri()
    test_search_idx1d()
    test_format_idx()
    test_cartesian_prod(order='C')
    #test_cartesian_prod(order='F')
    test_mdot()

    test_s4_idx()
    test_s8_idx()
    test_eri_idx()
    test_take_eri()
    
    test_tril_idx()
    test_tril_triu(5, int)
    test_tril_take_idx()

    test_spin_dim()
    test_reshape()
    test_h5()
    test_cart_sph()
    
