#! /usr/bin/env python

"""
Test make_basis.py
"""

def test_tile_u_matrix():
    import numpy as np
    from libdmet.basis_transform import make_basis
    
    u_core = np.eye(2)[None] 
    u_val = np.arange(3, 7).reshape(1, 2, 2)
    u_virt = np.array(7).reshape(1, 1, 1)
    u_tiled = make_basis.tile_u_matrix(u_val, u_virt, u_core=u_core)
    u_ref = np.array([[[1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 3, 4, 0],
                       [0, 0, 5, 6, 0],
                       [0, 0, 0, 0, 7]]])
    assert (u_tiled.shape == u_ref.shape) and np.allclose(u_tiled, u_ref)
    
    # different dimension.
    u_tiled = make_basis.tile_u_matrix(u_val[None], u_virt, u_core=u_core)
    assert (u_tiled.shape == (1,) + u_ref.shape) and np.allclose(u_tiled, u_ref)

def test_exclude_string():
    from libdmet.basis_transform import make_basis
    
    # 5 valence, 1 virtual, 4 cores
    exclude_string = make_basis._get_exclude_bands_strings(5, 1, 4)
    print (exclude_string)

def test_get_proj_string():
    import numpy as np
    from pyscf.pbc import gto 
    from libdmet.basis_transform import make_basis
    from libdmet.utils.misc import max_abs
    from libdmet.lo.iao import get_idx_each

    cell = gto.Cell(
        atom = 'H 0 0 0; F 0 0 1.1',
        basis = '321g')
    cell.build(a = np.eye(3) * 5.0)

    idx_all = get_idx_each(cell, minao='minao', kind='all')
    string = make_basis.get_proj_string(cell, idx_all) 
    string = "\n".join(string)
    print (string)

def test_get_mo_ovlp():
    import numpy as np
    import pyscf
    from pyscf import scf
    from libdmet.basis_transform import make_basis
    from libdmet.utils.misc import max_abs

    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',
        basis = '321g')
    myhf = mol.HF()
    myhf.kernel()
    mo_coeff = myhf.mo_coeff
    ovlp = myhf.get_ovlp()
    
    # MO is normalized
    mo_ovlp = make_basis.get_mo_ovlp(mo_coeff, mo_coeff, ovlp)
    assert max_abs(mo_ovlp - np.eye(mo_ovlp.shape[-1])) < 1e-10
    
    # occupied is orthogonal to virtual
    nocc = np.sum(myhf.mo_occ > 0)
    mo_ovlp = make_basis.get_mo_ovlp(mo_coeff[:, :nocc], mo_coeff[:, nocc:], \
            ovlp)
    assert max_abs(mo_ovlp) < 1e-10
    
    # UHF test
    myhf = scf.addons.convert_to_uhf(myhf)
    mo_coeff = myhf.mo_coeff
    # MO is normalized
    mo_ovlp = make_basis.get_mo_ovlp(mo_coeff, mo_coeff, ovlp)
    for s in range(mo_ovlp.shape[0]):
        assert max_abs(mo_ovlp - np.eye(mo_ovlp.shape[-1])) < 1e-10
    
    # occupied is orthogonal to virtual
    nocc = np.sum(myhf.mo_occ[0] > 0)
    mo_coeff = np.asarray(mo_coeff)
    mo_ovlp = make_basis.get_mo_ovlp(mo_coeff[:, :, :nocc], \
            mo_coeff[:, :, nocc:], ovlp)
    assert max_abs(mo_ovlp) < 1e-10

def test_transform_h1():
    import numpy as np
    from pyscf import lo
    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.utils.misc import max_abs

    cell = lattice.HChain()
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')
    
    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nkpts = Lat.nkpts

    hcore = np.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpts))
    ovlp = np.asarray(cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
    C_ao_lo = np.asarray([lo.orth_ao(cell, s=ovlp[k]) for k in range(nkpts)])
    
    hcore_lo = make_basis.transform_h1_to_lo(hcore, C_ao_lo)
    hcore_ao = make_basis.transform_h1_to_ao(hcore_lo, C_ao_lo, ovlp)
    assert max_abs(hcore_ao - hcore) < 1e-12
    
    hcore_lo = make_basis.transform_h1_to_lo(hcore[None], C_ao_lo)
    hcore_ao = make_basis.transform_h1_to_ao(hcore_lo, C_ao_lo, ovlp)
    assert max_abs(hcore_ao - hcore) < 1e-12
    
    hcore_lo = make_basis.transform_h1_to_lo(hcore, C_ao_lo[None])
    hcore_ao = make_basis.transform_h1_to_ao(hcore_lo, C_ao_lo, ovlp)
    assert max_abs(hcore_ao - hcore) < 1e-12
    
    hcore_lo = make_basis.transform_h1_to_lo(hcore[None], C_ao_lo[None])
    hcore_ao = make_basis.transform_h1_to_ao(hcore_lo, C_ao_lo[None], ovlp)
    hcore_ao = make_basis.transform_h1_to_ao(hcore_lo, C_ao_lo, ovlp)
    assert max_abs(hcore_ao - hcore) < 1e-12
    
    hcore_lo = make_basis.transform_h1_to_lo(hcore[None], np.asarray([C_ao_lo] * 2))
    hcore_ao = make_basis.transform_h1_to_ao(hcore_lo, C_ao_lo, ovlp)
    assert max_abs(hcore_ao[0] - hcore) < 1e-12
    
    hcore_lo = make_basis.transform_h1_to_lo(np.asarray([hcore]*2), C_ao_lo)
    hcore_ao = make_basis.transform_h1_to_ao(hcore_lo, C_ao_lo, ovlp)
    assert max_abs(hcore_ao[0] - hcore) < 1e-12
    
    hcore_lo = make_basis.transform_h1_to_lo(np.asarray([hcore]*2), C_ao_lo[None])
    hcore_ao = make_basis.transform_h1_to_ao(hcore_lo, C_ao_lo[None], ovlp)
    assert max_abs(hcore_ao[0] - hcore) < 1e-12
    
    # molecular transform
    hcore_mol = hcore[0]
    ovlp_mol = ovlp[0]
    C_ao_lo = C_ao_lo[0]
    
    hcore_lo_mol = make_basis.transform_h1_to_mo_mol(hcore_mol, C_ao_lo)
    hcore_ao_mol = make_basis.transform_h1_to_ao_mol(hcore_lo_mol, C_ao_lo, \
            ovlp_mol)
    assert max_abs(hcore_ao_mol - hcore_mol) < 1e-12
    
    hcore_lo_mol = make_basis.transform_h1_to_mo_mol(np.asarray((hcore_mol,
        hcore_mol+1.0)), C_ao_lo)
    hcore_ao_mol = make_basis.transform_h1_to_ao_mol(hcore_lo_mol, C_ao_lo, \
            ovlp_mol)
    assert max_abs(hcore_ao_mol[0] - hcore_mol) < 1e-12
    assert max_abs(hcore_ao_mol[1] - hcore_mol - 1.0) < 1e-12

if __name__ == "__main__":
    test_get_proj_string()
    test_transform_h1()
    test_get_mo_ovlp()
    test_tile_u_matrix()
    test_exclude_string()
