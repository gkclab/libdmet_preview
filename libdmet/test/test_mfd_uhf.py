#!/usr/bin/env python

'''
Test UHF mean-field solver in mfd.py (no self-consistency).
Using GDF.
'''

import numpy as np
import scipy
import scipy.linalg as la
import os, sys

def test_mfd_uhf():
    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, tools, cc

    from libdmet.lo import iao, pywannier90
    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.basis_transform import eri_transform
    from libdmet.utils.misc import mdot, max_abs
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet

    log.verbose = "DEBUG2"
    np.set_printoptions(3, linewidth=1000, suppress=True)

    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    cell.basis = '321g'
    cell.verbose = 5
    cell.precision = 1e-16
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
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

    #import h5py
    #f = h5py.File('gdf_ints.h5', 'r+')
    #for k1 in f[u'j3c'].keys():
    #    for k2 in f[u'j3c'][k1].keys():
    #        print (f[u'j3c'][k1][k2])
    ##        f[u'j3c'][k1][k2][...] = np.zeros(f[u'j3c'][k1][k2].shape, dtype=f[u'j3c'][k1][k2].dtype)

    dm0_a = np.diag([0.5, 0.5, 0.0, 0.0 ])
    dm0_b = np.diag([0.0, 0.0, 0.5, 0.5 ])
    dm0 = np.asarray([[dm0_a]*3, [dm0_b]*3])

    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KUHF(cell, kpts, exxdiv=None)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-16
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KUHF(cell, kpts, exxdiv=None)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-16
        kmf.chkfile = chkfname
        kmf.kernel(dm0=dm0)

    rho_kmf = kmf.make_rdm1()
    rhoT_kmf = Lat.k2R(rho_kmf)

    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True)
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

    restricted = False
    bogoliubov = False
    nscsites = Lat.nscsites

    vcor = Hubbard.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((2, nscsites, nscsites))
    vcor.assign(z_mat)
    occ = cell.nelectron / (2.0 * nscsites)

    ### Check the mean-field from LO is the same as kmf

    rhoT, mu, E = HF(Lat, vcor, occ, restricted, mu0 = 0., beta = np.inf, ires = False)
    Lat.update_Ham(rhoT)

    E_diff = np.abs(E - (kmf.e_tot - kmf.energy_nuc()))
    rdm_diff = max_abs(rhoT - Lat.rdm1_lo_R)

    print ("diff E between HF and pyscf", E_diff)
    print ("diff lattice rdm between HF and pyscf", rdm_diff)
    assert E_diff < 1e-9
    assert rdm_diff < 1e-8

    ### Check the rdm1 in LO is idempotent

    rdm1_lo_R_full = Lat.expand(rhoT)[0]
    diff_idem = max_abs(rdm1_lo_R_full - rdm1_lo_R_full.dot(rdm1_lo_R_full))
    print ("check idempotent", diff_idem)
    assert diff_idem < 1e-10


    basis = slater.get_emb_basis(Lat, rhoT)
    basis_k = Lat.R2k_basis(basis)
    ImpHam, _ = slater.get_emb_Ham(Lat, basis, vcor, eri_symmetry=symm)

    ### Check the embedding H1 alpha and beta has the same eigvals

    ea = la.eigh(ImpHam.H1["cd"][0])[0]
    eb = la.eigh(ImpHam.H1["cd"][1])[0]

    diff_h1 = max_abs(ea - eb)
    print ("eigval diff with 1-body ImpHam alpha and beta", diff_h1)
    assert diff_h1 < 1e-9

    # UHF solver
    from libdmet.solver.scf import SCF
    solver = SCF(newton_ah=True)
    # nelec, spin, bcs, restricted
    solver.set_system((Lat.ncore+Lat.nval)*2, 0, False, False)
    solver.set_integral(ImpHam)
    rdm1_fold = slater.foldRho_k(Lat.rdm1_lo_k, basis_k)
    E, rdm1_emb = solver.HF(MaxIter=100, tol=1e-15, InitGuess=(rdm1_fold))

    ### Check the rdm1 from UHF solver is the same as folded one from kmf

    diff_emb_rot = max_abs(rdm1_emb - rdm1_fold)
    print ("rdm1_emb")
    print (rdm1_emb)
    print ("rdm1_fold")
    print (rdm1_fold)
    print ("diff emb rdm between HF solver and rotated rdm", diff_emb_rot)
    assert diff_emb_rot < 1e-8

    ### Check the rho_glob is the same as rdm1 in LO
    from libdmet.routine.slater_helper import get_rho_glob_R
    rho_glob = np.asarray([get_rho_glob_R(basis[s], Lat, rdm1_emb[s], symmetric=False) for s in range(2)])
    print ("rho_glob diff", max_abs(rho_glob - Lat.rdm1_lo_R))

    # fragment energy evaluation

    rdm2_emb_aa =  np.einsum('qp, sr -> pqrs', rdm1_emb[0], rdm1_emb[0]) - \
            1.0*np.einsum('sp, qr -> pqrs', rdm1_emb[0], rdm1_emb[0])
    rdm2_emb_bb =  np.einsum('qp, sr -> pqrs', rdm1_emb[1], rdm1_emb[1]) - \
            1.0*np.einsum('sp, qr -> pqrs', rdm1_emb[1], rdm1_emb[1])
    rdm2_emb_ab = np.einsum('qp, sr -> pqrs', rdm1_emb[0], rdm1_emb[1])

    ### Check the Edmet is the same as kmf energy

    r1 = rdm1_emb
    # r2 is in aa, bb, ab order
    r2 = rdm2_emb = np.asarray([rdm2_emb_aa, rdm2_emb_bb, rdm2_emb_ab])

    from libdmet.routine.slater import get_H_dmet
    ImpHam_scaled = get_H_dmet(basis, Lat, ImpHam, 0.0, compact=False)
    h1 = h1_scaled = ImpHam_scaled.H1["cd"]
    h2 = h2_scaled = ImpHam_scaled.H2["ccdd"]

    E1 = np.tensordot(h1, r1, axes=((0,1,2), (0,2,1)))
    E2_aa = 0.5 * np.sum(r2[0]*h2[0])
    E2_bb = 0.5 * np.sum(r2[1]*h2[1])
    E2_ab = np.sum(r2[2]*h2[2])
    Efrag = E1 + E2_aa + E2_bb + E2_ab + ImpHam_scaled.H0

    print ("Efrag from HF solver", Efrag)
    print ("E from kmf", (kmf.e_tot - kmf.energy_nuc()))
    print ("diff Efrag and E from pyscf", abs(Efrag - \
            (kmf.e_tot - kmf.energy_nuc())))
    assert abs(Efrag - (kmf.e_tot - kmf.energy_nuc())) < 1e-8

    # test E_mf from get_E_dmet_HF
    from libdmet.routine.slater import get_E_dmet_HF
    last_dmu = 0.0
    Efrag2 = get_E_dmet_HF(basis, Lat, ImpHam, last_dmu, solver)
    print ("diff of 2 diffrent way to evualate Efrag from HF solver", \
            abs(Efrag - Efrag2))
    assert abs(Efrag - Efrag2) < 1e-10

    from libdmet.solver.impurity_solver import FCI
    from libdmet.dmet.Hubbard import transformResults, SolveImpHam_with_fitting, apply_dmu
    ImpHam, _ = slater.get_emb_Ham(Lat, basis, vcor, eri_symmetry=symm)
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    solver = FCI(restricted=ImpHam.restricted)
    last_dmu = 0.0
    H1e = 0.0
    rhoEmb, EnergyEmb, ImpHam, dmu = \
        SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
        solver_args={"nelec": Lat.nval*2, \
        "dm0": dmet.foldRho_k(Lat.R2k(rhoT), basis_k)}, thrnelec = 1e-5)
    last_dmu += dmu
    rhoImp, Efrag, nelecImp = \
        transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
        lattice=Lat, last_dmu=last_dmu, int_bath=True, solver=solver)
    print ("Efrag from FCI solver", Efrag*nscsites)

    # FCI in the LMO basis
    ncas = ImpHam.norb
    nelecas = (Lat.ncore+Lat.nval)*2
    #ImpHam, _ = slater.get_emb_Ham(Lat, basis, vcor)
    #last_dmu = 0.0

    fcisolver = FCI(restricted=False)
    solver = dmet.impurity_solver.CASCI(ncas=ncas, nelecas=nelecas, \
                splitloc=False, cisolver=fcisolver, mom_reorder=False, tmpDir="./tmp")
    solver_args = {"guess": dmet.foldRho_k(Lat.R2k(rhoT), basis_k), \
            "basis": basis, "nelec":(Lat.ncore+Lat.nval)*2}
    rhoEmb2, EnergyEmb2, ImpHam, dmu = \
        SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
        solver_args=solver_args, thrnelec = 1e-5)
    last_dmu += dmu

    rhoImp, Efrag2, nelecImp = \
        transformResults(rhoEmb2, EnergyEmb2, basis, ImpHam, H1e, \
        lattice=Lat, last_dmu=last_dmu, int_bath=True, \
        solver=solver, solver_args=solver_args)
    print ("Efrag from FCI solver in LMO basis", Efrag2*nscsites)
    print ("diff", Efrag2 - Efrag)
    assert Efrag2 - Efrag < 1e-13
    assert max_abs(rhoEmb2 - rhoEmb) < 1e-12

    # compare energy with KCCSD
    mycc = cc.KUCCSD(kmf)
    mycc.kernel()
    print ("KUCCSD energy (per unit cell)", mycc.e_tot - cell.energy_nuc())

if __name__ == "__main__":
    test_mfd_uhf()
