#!/usr/bin/env python

import os, sys
import numpy as np
import scipy.linalg as la

np.set_printoptions(4, linewidth=1000)

def test_neighbor():
    from libdmet.system.lattice import ChainLattice, SquareLattice, ham
    from libdmet.utils import logger as log
    
    chain = ChainLattice(240, 4)
    log.result("%s", chain)
    log.result("kpoints: %s", chain.kpoints())
    log.result("neighbors: %s", chain.neighbor(sitesA = range(4)))
    log.result("")

    square = SquareLattice(72, 72, 2, 2)
    log.result("%s", square)
    log.result("1st neigbors: %s", square.neighbor(dis = 1., sitesA = range(4)))
    log.result("2nd neigbors: %s", square.neighbor(dis = 2**0.5, sitesA = range(4)))

    Ham = ham.HubbardHamiltonian(square, 4, [1, 0.])
    square.setHam(Ham)
    log.result("%s", square.getH1(kspace = True))
    log.result("%s", square.getFock(kspace = True))
    log.result("%s", square.getH2())

def test_analyze():
    """
    Test analyze.
    """
    from pyscf.pbc import gto as gto
    import pyscf.pbc.scf as scf
    import pyscf.pbc.df as df
    from pyscf.pbc.lib import chkfile

    from libdmet.lo import iao
    from libdmet.basis_transform import make_basis
    from libdmet.system import lattice
    from libdmet.utils.misc import max_abs
    from libdmet.system.lattice import analyze_kmo, analyze_cas, translate_labels

    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''

    cell.atom = ''' H 0.      0.      0.
                    H 0.      0.      1.5 '''
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-10
    cell.build(unit='Angstrom')
   
    # lattice class
    kmesh = [1, 1, 3]  
    Lat = lattice.Lattice(cell, kmesh)
    nscsites = Lat.nscsites
    kpts = Lat.kpts_abs
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
        kmf.conv_tol = 1e-10
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-10
        kmf.chkfile = chkfname
        kmf.kernel()
    
    ovlp = np.asarray(kmf.get_ovlp())
    # test RHF with meta Lowdin
    print ("\nTest meta-lowdin RHF analysis")
    pop_rhf, chg_rhf = Lat.analyze(kmf)
    
    
    # test RHF with IAOs
    print ("\nTest IAO+PAO RHF analysis")
    minao = 'minao'
    C_ao_lo = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao)
    labels = iao.get_labels(cell, minao)[0]
    idx_to_ao_labels = iao.get_idx_to_ao_labels(cell, minao)
    Lat.analyze(kmf, C_ao_lo=C_ao_lo, labels=labels)
    
    # analyze each MO
    analyze_kmo(kmf, C_ao_lo, labels)
    idx_dic = iao.get_idx_each(cell, minao=minao, kind='atom nl')
    analyze_kmo(kmf, C_ao_lo, idx_dic, k_loop_first=False)
    Lat.analyze_kmo(kmf, num_max=1, k_loop_first=False)

    rdm1_ao = np.asarray(kmf.make_rdm1())
    rdm1_ao_R0 = Lat.k2R(rdm1_ao)[0]
    rdm1_lo_R0 = Lat.k2R(make_basis.transform_rdm1_to_lo(rdm1_ao, C_ao_lo, \
            ovlp))[0]
    print ("\nTest IAO+PAO RHF compare density")
    Lat.mulliken_lo_R0(rdm1_lo_R0=rdm1_ao_R0, rdm1_lo_R0_2=rdm1_lo_R0, \
            labels=labels)
    
    # test UHF with meta Lowdin
    print ("\nTest IAO+PAO UHF analysis")
    kmf = scf.addons.convert_to_uhf(kmf) 
    pop_uhf, chg_uhf = Lat.analyze(kmf)
    assert max_abs(pop_uhf[0] + pop_uhf[1] - pop_rhf) < 1e-10
    assert max_abs(chg_uhf - chg_rhf) < 1e-10
    
    # test UHF with IAOs, 1 valence and 1 virtual
    print ("\nTest IAO+PAO UHF analysis 1 valence 1 virtual")
    C_ao_lo = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao)
    idx = [0, 3]
    labels = [labels[i] for i in idx]
    Lat.analyze(kmf, C_ao_lo=C_ao_lo[:, :, :, idx], labels=labels)

    print ("\nTest IAO+PAO UHF compare density")
    labels = iao.get_labels(cell, minao)[0]
    rdm1_lo_R0 = np.asarray((rdm1_lo_R0, rdm1_lo_R0)) * 0.5
    Lat.mulliken_lo_R0(rdm1_lo_R0=rdm1_lo_R0, rdm1_lo_R0_2=rdm1_lo_R0, \
            labels=labels)
    Lat.mulliken_lo_R0(rdm1_lo_R0=rdm1_lo_R0, rdm1_lo_R0_2=rdm1_lo_R0, \
            labels=labels)
    
    import libdmet.dmet.Hubbard as dmet 
    kmf = scf.addons.convert_to_rhf(kmf) 
    C_ao_lo = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao)
    nval, nvirt, ncore = 2, 2, 0
    Lat.set_val_virt_core(nval, nvirt, ncore)
    Lat.set_Ham(kmf, gdf, C_ao_lo)
    vcor = dmet.vcor_zeros(True, False, Lat.nao)
    Filling = cell.nelectron / (Lat.nao * 2.0)
    Mu = 0.0
    beta = np.inf
    rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
    
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, int_bath=True)
    basis_k = Lat.R2k_basis(basis)
    
    solver = dmet.impurity_solver.SCFSolver(restricted=True, tol=1e-10)
    solver_args = {"nelec": min((Lat.ncore+Lat.nval)*2, \
            cell.nelectron*nkpts), \
            "dm0": dmet.foldRho_k(res["rho_k"], basis_k)*2.0}
    rhoEmb, EnergyEmb, ImpHam, dmu = \
        dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
        solver_args=solver_args, thrnelec=1e+6)
    
    mf = solver.scfsolver.mf
    analyze_cas(mf, basis, labels, num_max=4, \
                mo_print_list=None, nmo_print=None)
    
    order, C_lo_mo_abs = analyze_cas(mf, basis, idx_dic, num_max=4, 
                                     mo_print_list=None, nmo_print=100)
    
    scell = Lat.bigcell
    labels_sc = translate_labels(labels, Lat.nkpts, cell.natm)
    keep_id = [0, 1, 2]
    coords = np.array([atm[1] for atm in scell._atom])
    coords_frac = lattice.Real2Frac(scell.lattice_vectors(), coords)
    keep_id = np.where(coords_frac[:, -1] < 0.5)[0]
    
    symbols = ["imp" if int(label.split()[0]) in keep_id else "other" for label in labels_sc]
    idx_dic = iao.get_idx_each(scell, symbols=symbols)
    print (idx_dic)

def test_bond_analysis():
    from pyscf import gto
    from libdmet.system import lattice
    from libdmet.utils.misc import max_abs
    np.set_printoptions(3, linewidth=1000, suppress=True)
    
    benzene = [[ 'C'  , ( 4.673795 ,   6.280948 , 0.00  ) ],
               [ 'C'  , ( 5.901190 ,   5.572311 , 0.00  ) ],
               [ 'C'  , ( 5.901190 ,   4.155037 , 0.00  ) ],
               [ 'C'  , ( 4.673795 ,   3.446400 , 0.00  ) ],
               [ 'C'  , ( 3.446400 ,   4.155037 , 0.00  ) ],
               [ 'C'  , ( 3.446400 ,   5.572311 , 0.00  ) ],
               [ 'H'  , ( 4.673795 ,   7.376888 , 0.00  ) ],
               [ 'H'  , ( 6.850301 ,   6.120281 , 0.00  ) ],
               [ 'H'  , ( 6.850301 ,   3.607068 , 0.00  ) ],
               [ 'H'  , ( 4.673795 ,   2.350461 , 0.00  ) ],
               [ 'H'  , ( 2.497289 ,   3.607068 , 0.00  ) ],
               [ 'H'  , ( 2.497289 ,   6.120281 , 0.00  ) ]]

    mol = gto.M(atom=benzene,
                basis='631g')
    
    myhf = mol.RHF()
    myhf.kernel()
    ovlp = myhf.get_ovlp()
    rdm1 = myhf.make_rdm1()
    pairs_r, dis_r, orders_r = lattice.get_bond_order_all(mol, rdm1, ovlp, 
                               length_range=[0.5, 3.0], bond_type=[("C", "C")])

    myhf = mol.UHF()
    myhf.kernel()
    rdm1 = myhf.make_rdm1()
    pairs, dis, orders = lattice.get_bond_order_all(mol, rdm1, ovlp, 
                         length_range=[0.5, 3.0], bond_type=set([("C", "C")]))
    
    assert max_abs(pairs - pairs_r) < 1e-7
    assert max_abs(dis - dis_r) < 1e-7
    assert max_abs(orders - orders_r) < 1e-7

def test_bond_analysis_pbc():
    from pyscf.pbc import gto, df, scf
    from libdmet.system import lattice
    from libdmet.utils.misc import mdot
    np.set_printoptions(3, linewidth=1000, suppress=True)
    
    cell = lattice.HChain(nH=6, R=1.5)
    cell.basis = '321G'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')
    
    kmesh = [1, 1, 1]
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
        kmf = scf.RHF(cell, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.RHF(cell, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()
    
    ovlp = kmf.get_ovlp()
    rdm1 = kmf.make_rdm1()
    pairs_r, dis_r, orders_r = lattice.get_bond_order_all(cell, rdm1, ovlp, length_range=[0.5, 2.0])
    
    from libdmet.basis_transform import make_basis
    from libdmet.lo import iao, ibo
    kmf.mo_coeff = kmf.mo_coeff[None]
    kmf.mo_occ = kmf.mo_occ[None]
    minao = 'minao'
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True)
    C_ao_iao = C_ao_iao[0].real
    lo_labels = iao.get_labels(cell, minao=minao)[0]
    idx_re = iao.get_idx_to_ao_labels(cell, minao=None, labels=lo_labels)
    lo_labels = np.asarray(lo_labels)[idx_re]
    C_ao_iao = np.asarray(C_ao_iao)[:, idx_re]
    
    rdm1 = make_basis.transform_rdm1_to_mo_mol(rdm1, C_ao_iao, ovlp)
    ovlp = make_basis.transform_h1_to_mo_mol(ovlp, C_ao_iao)
    pairs_r, dis_r, orders_r = lattice.get_bond_order_all(cell, rdm1, ovlp, length_range=[0.5, 2.0], labels=lo_labels)

if __name__ == '__main__':
    test_bond_analysis()
    test_bond_analysis_pbc()
    test_neighbor()
    test_analyze()
