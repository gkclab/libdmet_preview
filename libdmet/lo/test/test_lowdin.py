#! /usr/bin/env python

"""
Test Lowdin routines.
"""

import os, sys
import numpy as np
import scipy.linalg as la

np.set_printoptions(4, linewidth=1000, suppress=True)

def test_lowdin():
    """
    Test check_orthonormal, check_orthogonal, check_span_same_space.
    """
    from pyscf.pbc import scf, gto, df, dft
    from libdmet.system import lattice
    from libdmet.lo import lowdin

    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    exxdiv = None

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()

    # 1. restricted test
    mo_energy = np.asarray(kmf.mo_energy)
    mo_coeff = np.asarray(kmf.mo_coeff)
    ovlp = np.asarray(kmf.get_ovlp())
    # MOs should be orthogonal at each k point
    assert lowdin.check_orthonormal(mo_coeff, ovlp)

    # occupied should be orthogornal to virtual
    assert lowdin.check_orthogonal(mo_coeff[:, :, :2], mo_coeff[:, :, 2:], \
            ovlp)
    # definitely not span the same space
    assert not lowdin.check_span_same_space(mo_coeff[:, :, :2], \
            mo_coeff[:, :, 2:], ovlp)

    # first should be orthogonal to others
    assert lowdin.check_orthogonal(mo_coeff[:, :, :1], mo_coeff[:, :, 3:], \
            ovlp)

    # Lowdin orbitals should span the same space
    C_ao_lo = lowdin.lowdin_k(kmf)
    assert lowdin.check_span_same_space(mo_coeff, C_ao_lo, ovlp)

    # check postive definite
    assert lowdin.check_positive_definite(ovlp)

    # get_labels, should be just ao_labels
    labels = lowdin.give_labels_to_lo(kmf, C_ao_lo)
    ao_labels = np.asarray(cell.ao_labels())
    assert (ao_labels == labels).all()

    # 2. unrestricted test
    kmf = scf.addons.convert_to_uhf(kmf)
    mo_energy = np.asarray(kmf.mo_energy)
    mo_coeff = np.asarray(kmf.mo_coeff)
    ovlp = np.asarray(kmf.get_ovlp())
    # MOs should be orthogonal at each k point
    assert lowdin.check_orthonormal(mo_coeff, ovlp)

    # occupied should be orthogornal to virtual
    assert lowdin.check_orthogonal(mo_coeff[:, :, :, :2], mo_coeff[:, :, :, 2:], \
            ovlp)
    # definitely not span the same space
    assert not lowdin.check_span_same_space(mo_coeff[:, :, :, :2], \
            mo_coeff[:, :, :, 2:], ovlp)

    # first should be orthogonal to others
    assert lowdin.check_orthogonal(mo_coeff[:, :, :, :1], mo_coeff[:, :, :, 3:], \
            ovlp)

    # check postive definite
    assert lowdin.check_positive_definite(np.array((ovlp, ovlp)))

    # Lowdin orbitals should span the same space
    C_ao_lo = lowdin.lowdin_k(kmf)
    assert lowdin.check_span_same_space(mo_coeff, C_ao_lo, ovlp)

def test_lowdin_mol():
    """
    Test check_orthonormal, check_orthogonal, check_span_same_space.
    """
    import numpy as np
    import pyscf
    from pyscf import scf
    from libdmet.lo import lowdin

    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'ccpvdz')
    myhf = mol.HF()
    myhf.kernel()

    # 1. restricted test
    mo_energy = np.asarray(myhf.mo_energy)
    mo_coeff = np.asarray(myhf.mo_coeff)
    ovlp = np.asarray(myhf.get_ovlp())
    # MOs should be orthogonal
    assert lowdin.check_orthonormal(mo_coeff, ovlp)

    # occupied should be orthogornal to others
    assert lowdin.check_orthogonal(mo_coeff[:, :2], mo_coeff[:, 2:], \
            ovlp)
    # definitely not span the same space
    assert not lowdin.check_span_same_space(mo_coeff[:, :2], \
            mo_coeff[:, -2:], ovlp)
    # first should be orthogonal to others
    assert lowdin.check_orthogonal(mo_coeff[:, :1], mo_coeff[:, 3:], \
            ovlp)

    # Lowdin orbitals should span the same space
    C_ao_lo = lowdin.lowdin(myhf)
    assert lowdin.check_span_same_space(mo_coeff, C_ao_lo, ovlp)

    # check postive definite
    assert lowdin.check_positive_definite(ovlp)

    # 2. unrestricted test
    myhf = scf.addons.convert_to_uhf(myhf)
    mo_energy = np.asarray(myhf.mo_energy)
    mo_coeff = np.asarray(myhf.mo_coeff)
    ovlp = np.asarray(myhf.get_ovlp())
    # MOs should be orthogonal at each k point
    assert lowdin.check_orthonormal(mo_coeff, ovlp)

    # occupied should be orthogornal to virtual
    assert lowdin.check_orthogonal(mo_coeff[:, :, :2], mo_coeff[:, :, -2:], \
            ovlp)
    # definitely not span the same space
    assert not lowdin.check_span_same_space(mo_coeff[:, :, :2], \
            mo_coeff[:, :, -2:], ovlp)

    # first should be orthogonal to others
    assert lowdin.check_orthogonal(mo_coeff[:, :, :1], mo_coeff[:, :, 3:], \
            ovlp)

    # Lowdin orbitals should span the same space
    C_ao_lo = lowdin.lowdin_k(myhf)
    assert lowdin.check_span_same_space(mo_coeff, C_ao_lo, ovlp)

    # check postive definite
    assert lowdin.check_positive_definite(np.array((ovlp, ovlp)))

def test_lowdin_2():
    from libdmet.lo import lowdin
    import numpy as np
    a = -np.eye(4)
    assert not lowdin.check_positive_definite(a)
    assert not lowdin.check_positive_definite(a[None])

    a_orth = lowdin.vec_lowdin(a[None], np.eye(4))
    assert lowdin.check_orthonormal(a, np.eye(4))

if __name__ == "__main__":
    test_lowdin_2()
    test_lowdin_mol()
    test_lowdin()

