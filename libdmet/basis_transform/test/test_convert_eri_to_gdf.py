#!/usr/bin/env python

def test_convert_eri_to_gdf():
    import numpy as np
    from scipy import linalg as la
    from pyscf import gto, scf, ao2mo, lib
    from libdmet.basis_transform import eri_transform
    
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
    feri = eri_transform.convert_eri_to_gdf(eri, norb, fname=None, tol=1e-8)
    print (feri.keys())
    feri = eri_transform.convert_eri_to_gdf(eri, norb, fname="converted.h5", tol=1e-8)

if __name__ == "__main__":
    test_convert_eri_to_gdf()
