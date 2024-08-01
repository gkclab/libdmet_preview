#! /usr/bin/env python

"""
Test scf routines.
"""

def test_get_veff():
    import numpy as np
    import scipy.linalg as la

    from pyscf import ao2mo
    from pyscf.scf import hf

    from libdmet.solver import scf
    from libdmet.system import integral
    from libdmet.utils.misc import max_abs, mdot

    nao = 5
    eri = np.random.random((nao, nao, nao, nao))
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(2, 3, 0, 1)
    integral.check_perm_symm(eri)

    rdm1 = np.random.random((nao, nao))
    rdm1 = rdm1 + rdm1.T

    vj_ref = np.einsum('ijkl, lk -> ij', eri, rdm1)
    vk_ref = np.einsum('ijkl, jk -> il', eri, rdm1)

    vj, vk = scf._get_jk(rdm1, eri)
    assert max_abs(vj - vj_ref) < 1e-12
    assert max_abs(vk - vk_ref) < 1e-12

    eri_s4 = ao2mo.restore(4, eri, nao)
    vj, vk = scf._get_jk(rdm1, eri_s4)
    assert max_abs(vj - vj_ref) < 1e-12
    assert max_abs(vk - vk_ref) < 1e-12

    vj, vk = scf._get_jk(rdm1[None], eri_s4[None])
    assert vj.shape == (1,) + vj_ref.shape
    assert max_abs(vj - vj_ref) < 1e-12
    assert max_abs(vk - vk_ref) < 1e-12

    rdm1_b = np.random.random((nao, nao))
    rdm1_b = rdm1_b + rdm1_b.T
    rdm1 = np.asarray((rdm1, rdm1_b))

    vj_ref, vk_ref = hf.dot_eri_dm(eri_s4, rdm1, hermi=0)

    # UHF
    vj, vk = scf._get_jk(rdm1, eri_s4[None])
    assert max_abs(vj - vj_ref) < 1e-12
    assert max_abs(vk - vk_ref) < 1e-12

    # UIHF
    mo_a = la.qr(np.random.random((nao, nao)))[0]
    mo_b = la.qr(np.random.random((nao, nao)))[0]

    eri_aa = ao2mo.general(eri_s4, (mo_a, mo_a, mo_a, mo_a))
    eri_ab = ao2mo.general(eri_s4, (mo_a, mo_a, mo_b, mo_b))
    eri_bb = ao2mo.general(eri_s4, (mo_b, mo_b, mo_b, mo_b))
    #eri_mo = np.asarray((eri_aa, eri_ab, eri_bb))
    eri_mo = np.asarray((eri_aa, eri_bb, eri_ab))
    rdm1_mo = np.array((mdot(mo_a.T, rdm1[0], mo_a), \
                        mdot(mo_b.T, rdm1[1], mo_b)))

    vj_mo, vk_mo = scf._get_jk(rdm1_mo, eri_mo)
    assert vj_mo.shape == (2, 2, nao, nao)
    assert vk_mo.shape == (2, nao, nao)

    vj_ref = vj_ref[0] + vj_ref[1]
    vj_ref = np.array((vj_ref, vj_ref))

    vj_mo = vj_mo[0] + vj_mo[1]
    vj = np.array([mdot(mo_a, vj_mo[0], mo_a.T), \
                   mdot(mo_b, vj_mo[1], mo_b.T)])
    vk = np.array([mdot(mo_a, vk_mo[0], mo_a.T), \
                   mdot(mo_b, vk_mo[1], mo_b.T)])
    assert max_abs(vj - vj_ref) < 1e-11
    assert max_abs(vk - vk_ref) < 1e-11

    # a list of rdm1
    rdm1 = np.random.random((4, nao, nao))
    rdm1 = rdm1 + rdm1.transpose(0, 2, 1)
    vj_ref, vk_ref = hf.dot_eri_dm(eri_s4, rdm1, hermi=0)
    vj, vk = scf._get_jk(rdm1, eri_s4)
    assert max_abs(vj - vj_ref) < 1e-12
    assert max_abs(vk - vk_ref) < 1e-12

def test_dot_eri_dm():
    import numpy as np
    from pyscf.scf import hf
    from pyscf import ao2mo
    from libdmet.utils import max_abs

    nao = 5
    eri = np.random.random((nao, nao, nao, nao))
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(1, 0, 2, 3)
    #eri = eri + eri.transpose(2, 3, 0, 1)
    eri_s4 = ao2mo.restore(4, eri, nao)

    rdm1 = np.random.random((nao, nao))
    rdm1 = rdm1 + rdm1.T

    vj_ref = np.einsum('ijkl, lk -> ij', eri, rdm1)
    #vj_ref = np.einsum('ijkl, ji -> kl', eri, rdm1)
    vk_ref = np.einsum('ijkl, jk -> il', eri, rdm1)

    vj, vk = hf.dot_eri_dm(eri, rdm1, hermi=0)
    vj_s4, vk_s4 = hf.dot_eri_dm(eri_s4, rdm1, hermi=0)

    print ("s1 diff")
    print (max_abs(vj - vj_ref))
    print (max_abs(vk - vk_ref))

    print ("s4 diff")
    print (max_abs(vj_s4 - vj_ref))
    print (max_abs(vk_s4 - vk_ref))

    assert max_abs(vj_s4 - vj_ref) < 1e-10
    assert max_abs(vk_s4 - vk_ref) < 1e-10

def test_incore_transform():
    import numpy as np
    import scipy.linalg as la

    from pyscf import ao2mo
    from pyscf.scf import hf

    from libdmet.solver import scf
    from libdmet.system import integral
    from libdmet.utils.misc import max_abs, mdot

    nao = 5
    eri = np.random.random((nao, nao, nao, nao))
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(2, 3, 0, 1)

    eri_s4 = ao2mo.restore(4, eri, nao)

    # UIHF
    mo_a = la.qr(np.random.random((nao, nao)))[0]
    mo_b = la.qr(np.random.random((nao, nao)))[0]
    mo = (mo_a, mo_b)

    eri_aa = ao2mo.general(eri_s4, (mo_a, mo_a, mo_a, mo_a))
    eri_bb = ao2mo.general(eri_s4, (mo_b, mo_b, mo_b, mo_b))
    eri_ab = ao2mo.general(eri_s4, (mo_a, mo_a, mo_b, mo_b))
    eri_mo_ref = np.asarray((eri_aa, eri_bb, eri_ab))

    eri_mo = scf.incore_transform(eri_s4, (mo,)*4, compact=True)
    assert max_abs(eri_mo - eri_mo_ref) < 1e-12

if __name__ == "__main__":
    test_dot_eri_dm()
    test_incore_transform()
    test_get_veff()
