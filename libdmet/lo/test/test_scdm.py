#! /usr/bin/env python

def test_scdm_mol():
    import numpy as np
    from pyscf import gto, scf, tools, lo
    from pyscf.tools import molden
    from libdmet.lo import scdm
    from libdmet.utils import logger as log

    np.set_printoptions(3, linewidth=1000) 
    log.verbose = "DEBUG2"
    atom_coords = np.array([[3.17500000, 3.17500000, 3.17500000],
                            [2.54626556, 2.54626556, 2.54626556],
                            [3.80373444, 3.80373444, 2.54626556],
                            [2.54626556, 3.80373444, 3.80373444],
                            [3.80373444, 2.54626556, 3.80373444]]) \
                 - np.array([3.17500000, 3.17500000, 3.17500000])
    mol = gto.Mole()
    mol.build(
        atom = [['C', atom_coords[0]], 
                ['H', atom_coords[1]], 
                ['H', atom_coords[2]],
                ['H', atom_coords[3]], 
                ['H', atom_coords[4]]], 
        basis = 'ccpvdz')

    mf = scf.RHF(mol)
    mf.kernel()

    mo_coeff = mf.mo_coeff
    orb_list = range(1, 5)
    mo = mo_coeff[:, orb_list]

    # SCDM
    C_ao_lo = scdm.scdm_mol(mol, mo, grid='U', mesh=[101, 101, 101])[0]
    C_ao_lo = scdm.scdm_mol(mol, mo, grid='B', level=5)[0]
    C_mo_lo = mo.conj().T.dot(mf.get_ovlp()).dot(C_ao_lo)
    molden.from_mo(mol, 'CH4_SCDM.molden', C_ao_lo)

    # Boys and compare cf values
    loc = lo.Boys(mol, mo)
    log.info("Dipole cf values:")
    log.info("cf (MO): %s", loc.cost_function())
    log.info("cf (SCDM): %s", loc.cost_function(u=C_mo_lo))
    
    loc_orb = loc.kernel()
    log.info("cf (Boys): %s", loc.cost_function())
    molden.from_mo(mol, 'CH4_Boys.molden', loc_orb)

def test_scdm_k():
    import os, sys
    import numpy as np
    import scipy.linalg as la

    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft

    from libdmet.system import lattice
    from libdmet.lo import scdm
    from libdmet.utils.plot import plot_orb_k_all
    import libdmet.dmet.Hubbard as dmet
    from libdmet.utils import logger as log

    log.verbose = "DEBUG2"
    np.set_printoptions(4, linewidth=1000, suppress=True)
    
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

    mo_energy = np.asarray(kmf.mo_energy)
    mo_coeff = np.asarray(kmf.mo_coeff)


    orb_list = range(0, 2) # val
    #orb_list = range(2, 4) # virt
    mo = mo_coeff[:, :, orb_list]

    # SCDM-k
    C_ao_lo, C_mo_lo = scdm.scdm_k(cell, mo, kpts, grid='U', level=5, \
            use_gamma_perm=True, return_C_mo_lo=True)
    
    C_ao_lo, C_mo_lo = scdm.scdm_k(cell, mo, kpts, grid='U', level=5, \
            use_gamma_perm=True, return_C_mo_lo=True, order='F', mesh=[30, 30, 30])

    plot_orb_k_all(cell, 'scdm-H-chain', C_ao_lo[0], kpts, nx=100, ny=100, nz=100, margin=5.0)
    
    # SCDM-k entangled
    nlo = 2
    mo_energy_sorted = np.sort(mo_energy, axis=None, kind='mergesort')
    mu = 0.5 * (mo_energy_sorted[cell.nelectron*nkpts] + mo_energy_sorted[cell.nelectron*nkpts-1])
    sigma = 0.3
    smear_func = scdm.smear_func(mo_energy, mu, sigma)
    print ("smear_func")
    print (smear_func)
    C_ao_lo, C_mo_lo = scdm.scdm_k(cell, mo_coeff, kpts, grid='B', level=5, \
            use_gamma_perm=True, return_C_mo_lo=True, nlo=nlo, smear_func=smear_func)
    
    for method in ["erfc", "erf", "gaussian", "fermi"]:
        smear_func = scdm.smear_func(mo_energy, mu, sigma, method=method)

if __name__ == "__main__":
    test_scdm_mol()
    test_scdm_k()
