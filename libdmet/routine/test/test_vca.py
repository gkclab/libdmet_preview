#!/usr/bin/env python

import os, sys
import numpy as np

def test_vca_all_elec():
    """
    Test vca potential, all electron.
    """
    from pyscf.pbc import gto, scf, df
    from libdmet.system import lattice
    from libdmet.utils.misc import max_abs, read_poscar
    from libdmet.routine import pbc_helper as pbc_hp
    import libdmet.utils.logger as log
    log.verbose = "DEBUG1"
    np.set_printoptions(3, linewidth=1000, suppress=True)
    
    pos = os.path.dirname(os.path.realpath(__file__)) + "/CCO.pos"
    pos_ghost = os.path.dirname(os.path.realpath(__file__)) + "/CCO-ghost.pos"

    cell = read_poscar(fname=pos)
    cell.basis = 'minao'

    kmesh = [1, 1, 3]
    cell.spin = 0
    cell.verbose = 5
    cell.precision = 1e-9
    cell.build()

    nelec_dop = 0.6
    nkpts = np.prod(kmesh)
    nelec0 = int(cell.nelectron)

    n_Ca = 2
    cell.nelectron = (nelec0 * nkpts - nelec_dop) / nkpts
    charges_old = np.array(cell.atom_charges(), dtype=float)
    portion = nelec_dop / (n_Ca * charges_old[-1] * nkpts)
    
    occ = 1.0 - (nelec_dop / (n_Ca * charges_old[-1] * nkpts))
    atom_idx = np.arange(len(charges_old) - n_Ca, len(charges_old))


    charges_old[-n_Ca:] -= (nelec_dop / nkpts / n_Ca)
    cell.atom_charges = lambda *args: charges_old
    def atom_charge(atm_id):
        return charges_old[atm_id]

    cell.atom_charge = atom_charge
    cell.build()

    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    gdf = df.GDF(cell, kpts)

    kmf = scf.KUHF(cell, kpts).density_fit()
    hcore_ref = kmf.get_hcore()

    # original cell
    cell = read_poscar(fname=pos)
    cell.basis = 'minao'
    cell.spin = 0
    cell.verbose = 5
    cell.max_memory = 40000
    cell.precision = 1e-9
    cell.build()

    nkpts = np.prod(kmesh)
    nelec0 = int(cell.nelectron)
    cell.nelectron = (nelec0 * nkpts - nelec_dop) / nkpts
    cell.build()
    gdf = df.GDF(cell, kpts)

    kmf = scf.KUHF(cell, kpts).density_fit()
    hcore_full = kmf.get_hcore()
    hcore_new = hcore_full + pbc_hp.get_veff_vca(gdf, atom_idx, occ, kpts_symm=None)

    diff = max_abs(hcore_ref - hcore_new)
    print ("diff to reference")
    print (diff)
    assert diff < 1e-11

def test_vca_pseudo():
    """
    Test vca potential, with pp.
    """
    from pyscf.pbc import gto, scf, df
    from libdmet.system import lattice
    from libdmet.utils.misc import max_abs, read_poscar
    from libdmet.routine import pbc_helper as pbc_hp
    import libdmet.utils.logger as log
    log.verbose = "DEBUG1"
    np.set_printoptions(3, linewidth=1000, suppress=True)
    
    pos = os.path.dirname(os.path.realpath(__file__)) + "/CCO.pos"
    pos_ghost = os.path.dirname(os.path.realpath(__file__)) + "/CCO-ghost.pos"

    cell = read_poscar(fname=pos)
    cell.basis = 'gth-szv-molopt-sr'
    cell.pseudo = 'gth-pbe'

    kmesh = [1, 1, 3]
    cell.spin = 0
    cell.verbose = 5
    cell.max_memory = 40000
    cell.precision = 1e-9
    cell.build()

    nelec_dop = 0.6
    nkpts = np.prod(kmesh)
    nelec0 = int(cell.nelectron)

    n_Ca = 2
    cell.nelectron = (nelec0 * nkpts - nelec_dop) / nkpts
    charges_old = np.array(cell.atom_charges(), dtype=float)
    portion = nelec_dop / (n_Ca * charges_old[-1] * nkpts)
    
    occ = 1.0 - (nelec_dop / (n_Ca * charges_old[-1] * nkpts))
    atom_idx = np.arange(len(charges_old) - n_Ca, len(charges_old))

    cell.build()

    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    gdf = df.GDF(cell, kpts)

    kmf = scf.KUHF(cell, kpts).density_fit()
    hcore_full = kmf.get_hcore()
    hcore_new = hcore_full + pbc_hp.get_veff_vca(gdf, atom_idx, occ, kpts_symm=None)
    
    # ghost cell reference
    cell_ghost = read_poscar(fname=pos_ghost)
    cell_ghost.basis = cell.basis
    cell_ghost.pseudo = cell.pseudo
    cell_ghost.precision = 1e-9
    cell_ghost.build()

    gdf_ghost = df.GDF(cell_ghost, kpts)

    vnuc_Ca = np.asarray(gdf_ghost.get_pp(kpts))
    hcore_ref = hcore_full - vnuc_Ca * portion
    
    diff = max_abs(hcore_ref - hcore_new)
    print ("diff to reference")
    print (diff)
    assert diff < 1e-11

if __name__ == "__main__":
    test_vca_pseudo()
    test_vca_all_elec()

