#!/usr/bin/env python
    
import numpy as np
from scipy import linalg as la
from pyscf import gto, scf, ao2mo, lib
from libdmet.utils import cholesky

def test_get_cderi_rhf():
    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = gto.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'ccpvdz',
        verbose = 4
    )
    norb = mol.nao_nr()
    myhf = mol.HF()
    myhf.kernel()
    eri = myhf._eri

    eri_s4 = ao2mo.restore(4, eri, norb)
    
    cderi_s4 = cholesky.get_cderi_rhf(eri_s4, norb, tol=1e-8)
    print (cderi_s4.shape)
    
    eri_re = np.einsum('Lpq, Lrs -> pqrs', cderi_s4, cderi_s4)
    eri_ref = ao2mo.restore(1, eri_s4, norb)
    diff = la.norm(eri_re - eri_ref)
    print ("diff", diff)
    assert diff < 1e-7

def test_get_cderi_uhf():
    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = gto.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'ccpvdz',
        verbose = 4,
        spin = 2
    )
    norb = mol.nao_nr()
    myhf = mol.HF()
    myhf.kernel()
    eri = myhf._eri
    eri_s4 = ao2mo.restore(4, eri, norb)

    def get_cderi_uhf_ref(eri, norb, tol=1e-8):
        """
        uhf ERI decomposation, assume 4-fold symmetry.
        assume aa, bb, ab order. reference.
        """
        assert len(eri) == 3
        assert eri[0].ndim == 2
        block_eri = np.block([[eri[0], eri[2]], [eri[2].T, eri[1]]])
        evecs = cholesky.modified_cholesky(block_eri, max_error=tol)
        nchol = evecs.shape[0]
        chol = np.zeros((2, nchol, norb, norb))
        chol[0] = lib.unpack_tril(evecs[:, :evecs.shape[-1]//2])
        chol[1] = lib.unpack_tril(evecs[:, evecs.shape[-1]//2:])
        return chol
    
    # uhf integrals
    cderi_s4 = cholesky.get_cderi_uhf([eri_s4*0.5, eri_s4, eri_s4 * 0.05], norb, tol=1e-8)
    print (cderi_s4.shape)
    
    cderi_s4_ref = get_cderi_uhf_ref([eri_s4*0.5, eri_s4, eri_s4 * 0.05], norb, tol=1e-8)
    print (cderi_s4.shape)
    
    diff = la.norm(cderi_s4 - cderi_s4_ref)
    print ("cderi diff", diff)
    assert diff < 1e-7
    
    eri_aa_ref = ao2mo.restore(1, (eri_s4 * 0.5), norb)
    eri_bb_ref = ao2mo.restore(1, (eri_s4), norb)
    eri_ab_ref = ao2mo.restore(1, (eri_s4 * 0.05), norb)

    eri_aa = np.einsum("Lpq, Lrs -> pqrs", cderi_s4[0], cderi_s4[0])
    eri_bb = np.einsum("Lpq, Lrs -> pqrs", cderi_s4[1], cderi_s4[1])
    eri_ab = np.einsum("Lpq, Lrs -> pqrs", cderi_s4[0], cderi_s4[1])
    
    diff = la.norm(eri_aa_ref - eri_aa)
    print ("eri_aa diff", diff)
    assert diff < 1e-7
    
    diff = la.norm(eri_bb_ref - eri_bb)
    print ("eri_bb diff", diff)
    assert diff < 1e-7
    
    diff = la.norm(eri_ab_ref - eri_ab)
    print ("eri_ab diff", diff)
    assert diff < 1e-7

    # MO integrals
    print ("-" * 79)
    print ("MO integral")
    
    eri_aa_ref = ao2mo.kernel(eri_s4, myhf.mo_coeff[0])
    eri_bb_ref = ao2mo.kernel(eri_s4, myhf.mo_coeff[1])
    eri_ab_ref = ao2mo.general(eri_s4, (myhf.mo_coeff[0], myhf.mo_coeff[0],
                                        myhf.mo_coeff[1], myhf.mo_coeff[1]))
    
    cderi_s4 = cholesky.get_cderi_uhf([eri_aa_ref, eri_bb_ref, eri_ab_ref], norb, tol=1e-8)
    print (cderi_s4.shape)
    eri_aa_ref = ao2mo.restore(1, eri_aa_ref, norb)
    eri_bb_ref = ao2mo.restore(1, eri_bb_ref, norb)
    eri_ab_ref = ao2mo.restore(1, eri_ab_ref, norb)

    eri_aa = np.einsum("Lpq, Lrs -> pqrs", cderi_s4[0], cderi_s4[0])
    eri_bb = np.einsum("Lpq, Lrs -> pqrs", cderi_s4[1], cderi_s4[1])
    eri_ab = np.einsum("Lpq, Lrs -> pqrs", cderi_s4[0], cderi_s4[1])
    
    diff = la.norm(eri_aa_ref - eri_aa)
    print ("eri_aa diff", diff)
    assert diff < 1e-7
    
    diff = la.norm(eri_bb_ref - eri_bb)
    print ("eri_bb diff", diff)
    assert diff < 1e-7
    
    diff = la.norm(eri_ab_ref - eri_ab)
    print ("eri_ab diff", diff)
    assert diff < 1e-7

if __name__ == "__main__":
    test_get_cderi_rhf()
    test_get_cderi_uhf()
