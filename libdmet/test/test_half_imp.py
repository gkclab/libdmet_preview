#!/usr/bin/env python

'''
Test half of impurity size is exact.
'''

import numpy as np
import scipy
import scipy.linalg as la
import os, sys

import pytest

def test_half_imp():
    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, tools, cc

    from libdmet.lo import iao, pywannier90
    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.basis_transform import eri_transform
    from libdmet.utils.misc import mdot
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet

    log.verbose = "DEBUG2"
    np.set_printoptions(3, linewidth=1000, suppress=True)
    int_bath = True
    
    with pytest.warns(UserWarning):
        cell = gto.Cell()
        cell.a = ''' 10.0    0.0     0.0
                     0.0     10.0    0.0
                     0.0     0.0     4.5 '''
        cell.atom = ''' H 5.0      5.0      0.75
                        H 5.0      5.0      2.25
                        H 5.0      5.0      3.75 '''
        cell.basis = 'sto3g'
        cell.verbose = 5
        cell.precision = 1e-12
        cell.build(unit='Angstrom')
        

        kmesh = [1, 1, 2]
        Lat = lattice.Lattice(cell, kmesh)
        kpts = Lat.kpts
        nao = Lat.nao
        nkpts = Lat.nkpts

        gdf_fname = 'gdf_ints.h5'
        gdf = df.GDF(cell, kpts)
        gdf._cderi_to_save = gdf_fname
        #if not os.path.isfile(gdf_fname):
        if True:
            gdf.build()

        chkfname = 'hchain.chk'
        #if os.path.isfile(chkfname):
        if False:
            kmf = scf.KRHF(cell, kpts, exxdiv=None)
            kmf.with_df = gdf
            kmf.with_df._cderi = 'gdf_ints.h5'
            kmf.conv_tol = 1e-12
            data = chkfile.load(chkfname, 'scf')
            kmf.__dict__.update(data)
        else:
            kmf = scf.KRHF(cell, kpts, exxdiv=None)
            kmf.with_df = gdf
            kmf.with_df._cderi = 'gdf_ints.h5'
            kmf.conv_tol = 1e-12
            kmf.chkfile = chkfname
            kmf.kernel()

        S = kmf.get_ovlp()
        # IAO guess
        C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True)

        # Wannier orbitals
        ncore = 0
        nval = C_ao_iao_val.shape[-1]
        nvirt = cell.nao_nr() - ncore - nval
        Lat.set_val_virt_core(nval, nvirt, ncore)
        C_ao_lo = C_ao_iao
        symm = 1
        Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=symm)

        from libdmet.routine.mfd import HF
        from libdmet.routine import slater
        from libdmet.dmet import Hubbard

        restricted = True
        bogoliubov = False
        nscsites = Lat.nscsites

        vcor = Hubbard.VcorLocal(restricted, bogoliubov, nscsites)
        z_mat = np.zeros((2, nscsites, nscsites))
        vcor.assign(z_mat)
        occ = cell.nelectron / (2.0 * nscsites)

        rhoT, mu, E = HF(Lat, vcor, occ, restricted, mu0 = 0., beta = np.inf, ires = False)

        E_diff = np.abs(E - (kmf.e_tot - kmf.energy_nuc())) 
        rdm_diff = np.max(np.abs(rhoT - Lat.rdm1_lo_R*0.5))

        print ("diff E between HF and pyscf")
        print (E_diff)
        print ("diff Latice rdm between HF and pyscf")
        print (rdm_diff)
        assert E_diff < 1e-10
        assert rdm_diff < 1e-8

        Lat.update_Ham(rhoT*2.0)

        rdm1_lo_R_full = Lat.expand(rhoT)[0]
        diff_idem = np.max(np.abs(rdm1_lo_R_full - rdm1_lo_R_full.dot(rdm1_lo_R_full)))
        print ("check idempotent, diff: %s" %diff_idem)
        assert(diff_idem < 1e-10)

        basis = slater.get_emb_basis(Lat, rhoT)
        basis_k = Lat.R2k_basis(basis)

        ImpHam, _ = slater.get_emb_Ham(Lat, basis, vcor, eri_symmetry=symm, \
                int_bath=int_bath)
        # RHF solver
        from libdmet.solver.scf import SCF
        solver = SCF(newton_ah=True)
        solver.set_system((Lat.ncore+Lat.nval)*2, 0, False, True)
        solver.set_integral(ImpHam)
        rdm1_fold = slater.foldRho_k(Lat.rdm1_lo_k, basis_k)
        E, rdm1_emb = solver.HF(MaxIter=100, tol=1e-15, InitGuess=rdm1_fold)
        rdm1_emb = (rdm1_emb*2.0)[0]
        rdm2_emb =  np.einsum('qp, sr -> pqrs', rdm1_emb, rdm1_emb) - \
                0.5*np.einsum('sp, qr -> pqrs', rdm1_emb, rdm1_emb)
        diff_emb_rot = np.max(np.abs(rdm1_emb - rdm1_fold))
        print ("rdm1_emb")
        print (rdm1_emb)
        print ("rdm1_fold") 
        print (rdm1_fold)
        print ("diff emb rdm between HF solver and rotated rdm")
        print (diff_emb_rot)
        assert diff_emb_rot < 1e-8

        from libdmet.routine.slater_helper import get_rho_glob_R
        rho_glob = get_rho_glob_R(basis[0], Lat, rdm1_emb, symmetric=False)
        print ("rho_glob diff")
        print (np.max(np.abs(rho_glob - Lat.rdm1_lo_R)))

        from libdmet.routine.slater import get_H_dmet
        ImpHam_scaled = get_H_dmet(basis, Lat, ImpHam, 0.0, compact=False)
        h1_scaled = ImpHam_scaled.H1["cd"][0]
        h2_scaled = ImpHam_scaled.H2["ccdd"][0]
        Efrag = np.einsum('pq,qp', h1_scaled, rdm1_emb) + 0.5*np.einsum('pqrs,pqrs', h2_scaled, rdm2_emb)
        print ("Efrag from HF solver")
        print (Efrag)
        print ("E from kmf")
        print ((kmf.e_tot - kmf.energy_nuc()))
        print ("diff Efrag and E from pyscf")
        print (np.abs(Efrag - (kmf.e_tot - kmf.energy_nuc())))
        assert np.abs(Efrag - (kmf.e_tot - kmf.energy_nuc())) < 1e-8

        # test E_mf from get_E_dmet_HF
        from libdmet.routine.slater import get_E_dmet_HF
        last_dmu = 0.0
        Efrag2 = get_E_dmet_HF(basis, Lat, ImpHam, last_dmu, solver)
        print ("diff of 2 diffrent way to evualate Efrag from HF solver")
        print (np.abs(Efrag - Efrag2))
        assert np.abs(Efrag - Efrag2) < 1e-10

        from libdmet.solver.impurity_solver import FCI, FCI_AO
        from libdmet.dmet.Hubbard import transformResults, SolveImpHam_with_fitting, apply_dmu
        ImpHam, _ = slater.get_emb_Ham(Lat, basis, vcor, eri_symmetry=symm,
                int_bath=int_bath)
        Filling = cell.nelectron / float(Lat.nscsites*2.0)
        solver = FCI_AO(restricted=True)
        last_dmu = 0.0
        H1e = 0.0

        cisolver = dmet.impurity_solver.CCSD(restricted=True, tol=1e-9, tol_normt=1e-6)
        solver = cisolver
        rhoEmb, EnergyEmb, ImpHam, dmu = \
            SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args={"nelec": Lat.nval*2, \
            "dm0": dmet.foldRho_k(Lat.R2k(rhoT), basis_k)*2.0}, thrnelec = 1e-5)

        last_dmu += dmu
        rhoImp, Efrag, nelecImp = \
            transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
            lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, solver=solver)
        E_from_dmet = Efrag*nscsites
        print ("Efrag from CC solver: %s" %E_from_dmet)

        # compare energy with KCCSD
        mycc = cc.KCCSD(kmf)
        mycc.kernel()
        E_kcc = mycc.e_tot - cell.energy_nuc()
        print ("KRCCSD energy (per unit cell): %s" %E_kcc)
        
        print ("diff: %s" %(E_from_dmet - E_kcc))
        assert abs(E_from_dmet - E_kcc) < 1e-6

if __name__ == "__main__":
    test_half_imp()

