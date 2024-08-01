#! /usr/bin/env python

"""
Test CC Solver.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

def test_rccsd():
    import numpy as np
    import pyscf
    from pyscf import ao2mo
    from libdmet.system.integral import Integral
    from libdmet.utils.misc import mdot, max_abs
    from libdmet.solver import impurity_solver 

    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'ccpvdz',
    )

    myhf = mol.HF()
    myhf.kernel()
    E_ref = myhf.e_tot
    rdm1_mf = myhf.make_rdm1()
    
    # CC reference
    mycc = myhf.CCSD().set(frozen=2).run()
    E_cc_ref = mycc.e_tot

    restricted = True
    bogoliubov = False
    norb = mol.nao_nr()
    H1 = myhf.get_hcore()[None]
    ovlp = myhf.get_ovlp()
    H2 = myhf._eri[None]
    H0 = myhf.energy_nuc()

    Ham = Integral(norb, restricted, bogoliubov, H0, {"cd": H1}, \
            {"ccdd": H2}, ovlp=ovlp)

    solver = impurity_solver.CCSD(restricted=restricted, tol=1e-10, frozen=2)
    rdm1, E = solver.run(Ham, nelec=mol.nelectron, dm0=rdm1_mf)
    assert abs(E - E_cc_ref) < 1e-8

    rdm2 = solver.make_rdm2(ao_repr=True)
    E_from_rdm = np.einsum('pq, qp ->', H1[0], rdm1[0] * 2.0) + \
            0.5 * np.einsum('pqrs, pqrs ->', ao2mo.restore(1, H2[0], norb), \
            rdm2[0]) + H0
    assert abs(E_from_rdm - E_cc_ref) < 1e-8
    
    # customized MO
    mo_coeff_custom = solver.scfsolver.mf.mo_coeff
    rdm1, E = solver.run(Ham, nelec=mol.nelectron, dm0=rdm1_mf, 
                         mo_coeff_custom=mo_coeff_custom)
    assert abs(E - E_cc_ref) < 1e-8

def test_uccsd_t():
    import numpy as np
    import pyscf
    from pyscf import ao2mo
    from libdmet.system.integral import Integral
    from libdmet.solver import impurity_solver 
    from libdmet.basis_transform import make_basis
    from libdmet.utils.misc import mdot
    from libdmet.utils import logger as log

    log.verbose = "DEBUG1"

    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = '321g',
        spin = 2,
        verbose = 5)
    
    myhf = mol.UHF()
    myhf.conv_tol = 1e-12
    myhf.kernel()
    
    nao = mol.nao_nr()
    C_ao_lo = myhf.mo_coeff
    hcore = make_basis.transform_h1_to_mo_mol(myhf.get_hcore(), C_ao_lo)
    ovlp = make_basis.transform_h1_to_mo_mol(myhf.get_ovlp(), C_ao_lo)
    rdm1 = make_basis.transform_rdm1_to_mo_mol(myhf.make_rdm1(), C_ao_lo, myhf.get_ovlp())
    
    eri_aa = ao2mo.general(myhf._eri, (C_ao_lo[0], C_ao_lo[0], \
            C_ao_lo[0], C_ao_lo[0]))
    eri_bb = ao2mo.general(myhf._eri, (C_ao_lo[1], C_ao_lo[1], \
            C_ao_lo[1], C_ao_lo[1]))
    eri_ab = ao2mo.general(myhf._eri, (C_ao_lo[0], C_ao_lo[0], \
            C_ao_lo[1], C_ao_lo[1]))
    eri = np.asarray((eri_aa, eri_bb, eri_ab))
    eri_s1 = np.zeros((3, nao, nao, nao, nao))
    for s in range(3):
        eri_s1[s] = ao2mo.restore(1, eri[s], nao)
    
    mycc = myhf.CCSD().set(frozen=0)
    eris = mycc.ao2mo()
    E_corr = mycc.kernel(eris=eris)[0]
    E_t = mycc.ccsd_t(eris=eris)
    E_ccsdt = mycc.e_tot + E_t

    Ham = Integral(nao, False, False, myhf.energy_nuc(), {"cd": hcore}, {"ccdd": eri}, ovlp=ovlp)
    solver = impurity_solver.CCSD(restricted=False, tol=1e-10, Sz=mol.spin)
    rdm1, E = solver.run(Ham, nelec=mol.nelectron, dm0=rdm1, ccsdt=True, ccsdt_energy=True) 
    assert abs(E - E_ccsdt) < 1e-6
    
    E_re = solver.run_dmet_ham(Ham, ccsdt=True)
    assert abs(E_re - E_ccsdt) < 1e-6

def test_bccd():
    import numpy as np
    import pyscf
    from pyscf import ao2mo
    from libdmet.system.integral import Integral
    from libdmet.utils.misc import mdot, max_abs
    from libdmet.solver import impurity_solver 

    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'ccpvdz',
    )

    myhf = mol.HF()
    myhf.kernel()
    E_ref = myhf.e_tot
    rdm1_mf = myhf.make_rdm1()
    
    restricted = True
    bogoliubov = False
    norb = mol.nao_nr()
    H1 = myhf.get_hcore()[None]
    ovlp = myhf.get_ovlp()
    H2 = myhf._eri[None]
    H0 = myhf.energy_nuc()

    Ham = Integral(norb, restricted, bogoliubov, H0, {"cd": H1}, 
                   {"ccdd": H2}, ovlp=ovlp)
    solver = impurity_solver.CCSD(restricted=restricted, tol=1e-10)
    rdm1, E = solver.run(Ham, nelec=mol.nelectron, dm0=rdm1_mf, bcc=True,
                         bcc_tol=1e-8)

    rdm2 = solver.make_rdm2(ao_repr=True)
    E_from_rdm = np.einsum('pq, qp ->', H1[0], rdm1[0] * 2.0) + \
            0.5 * np.einsum('pqrs, pqrs ->', ao2mo.restore(1, H2[0], norb), \
            rdm2[0]) + H0
    assert abs(E_from_rdm - E) < 1e-8
    assert max_abs(solver.cisolver.t1) < 1e-6

def test_exp_val_rccsd():
    import numpy as np
    import pyscf
    from pyscf import ao2mo
    from libdmet.utils.misc import mdot
    from libdmet.solver.cc import exp_val_rccsd
    
    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = '321g',
        verbose = 5)
    
    myhf = mol.HF()
    myhf.conv_tol = 1e-12
    myhf.kernel()
    E_ref = myhf.e_tot
    rdm1_mf = myhf.make_rdm1()
    
    mycc = myhf.CCSD().set(frozen=0).run()
    hcore_mo = mdot(mycc.mo_coeff.T, myhf.get_hcore(), mycc.mo_coeff)
    eri_mo = ao2mo.restore(4, ao2mo.full(myhf._eri, mycc.mo_coeff), mol.nao_nr())
    eri_mo_s1 = ao2mo.restore(1, ao2mo.full(myhf._eri, mycc.mo_coeff), mol.nao_nr())

    mycc.solve_lambda()
    rdm1 = mycc.make_rdm1(ao_repr=False)
    rdm2 = mycc.make_rdm2(ao_repr=False)
    E1_ref = np.einsum('pq, qp ->', hcore_mo, rdm1)
    E2_ref = 0.5 * np.einsum('pqrs, pqrs ->', eri_mo_s1, rdm2)
    E_ref = E1_ref + E2_ref
    
    E = exp_val_rccsd(mycc, hcore_mo, eri_mo, blksize=1)
    assert abs(E - E_ref) < 1e-10
    
    mycc = myhf.CCSD().set(frozen=[0, 2, 9, 10]).run()
    hcore_mo = mdot(mycc.mo_coeff.T, myhf.get_hcore(), mycc.mo_coeff)
    eri_mo = ao2mo.restore(4, ao2mo.full(myhf._eri, mycc.mo_coeff), mol.nao_nr())
    eri_mo_s1 = ao2mo.restore(1, ao2mo.full(myhf._eri, mycc.mo_coeff), mol.nao_nr())

    mycc.solve_lambda()
    rdm1 = mycc.make_rdm1(ao_repr=False)
    rdm2 = mycc.make_rdm2(ao_repr=False)
    E1_ref = np.einsum('pq, qp ->', hcore_mo, rdm1)
    E2_ref = 0.5 * np.einsum('pqrs, pqrs ->', eri_mo_s1, rdm2)
    E_ref = E1_ref + E2_ref
    
    E = exp_val_rccsd(mycc, hcore_mo, eri_mo, blksize=1)
    assert abs(E - E_ref) < 1e-10
    
def test_exp_val_uccsd():
    import numpy as np
    import pyscf
    from pyscf import ao2mo
    from libdmet.basis_transform import make_basis
    from libdmet.solver.cc import exp_val_uccsd
    from libdmet.utils.misc import mdot
    from libdmet.utils import logger as log

    log.verbose = "DEBUG1"

    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = '321g',
        spin = 2,
        verbose = 5)
    
    myhf = mol.UHF()
    myhf.conv_tol = 1e-12
    myhf.kernel()
    
    nao = mol.nao_nr()
    C_ao_lo = myhf.mo_coeff
    hcore = make_basis.transform_h1_to_mo_mol(myhf.get_hcore(), C_ao_lo)
    ovlp = make_basis.transform_h1_to_mo_mol(myhf.get_ovlp(), C_ao_lo)
    rdm1 = make_basis.transform_rdm1_to_mo_mol(myhf.make_rdm1(), C_ao_lo, myhf.get_ovlp())
    
    eri_aa = ao2mo.general(myhf._eri, (C_ao_lo[0], C_ao_lo[0], \
            C_ao_lo[0], C_ao_lo[0]))
    eri_bb = ao2mo.general(myhf._eri, (C_ao_lo[1], C_ao_lo[1], \
            C_ao_lo[1], C_ao_lo[1]))
    eri_ab = ao2mo.general(myhf._eri, (C_ao_lo[0], C_ao_lo[0], \
            C_ao_lo[1], C_ao_lo[1]))
    eri = np.asarray((eri_aa, eri_bb, eri_ab))
    eri_s1 = np.zeros((3, nao, nao, nao, nao))
    for s in range(3):
        eri_s1[s] = ao2mo.restore(1, eri[s], nao)

    mycc = myhf.CCSD().set(frozen=0)
    mycc.kernel()
    mycc.solve_lambda()
    rdm1 = np.asarray(mycc.make_rdm1(ao_repr=False))
    rdm2 = np.asarray(mycc.make_rdm2(ao_repr=False))
    
    E1_ref  = np.einsum('spq, sqp ->', hcore, rdm1)
    E2_aa_ref = 0.5 * np.einsum('pqrs, pqrs ->', eri_s1[0], rdm2[0])
    E2_bb_ref = 0.5 * np.einsum('pqrs, pqrs ->', eri_s1[1], rdm2[2])
    E2_ab_ref = np.einsum('pqrs, pqrs ->', eri_s1[2], rdm2[1])
    E2_ref = E2_aa_ref + E2_bb_ref + E2_ab_ref
    E_ref = E1_ref + E2_ref
    print ("E1_ref")
    print (E1_ref)
    print ("E2_ref")
    print (E2_ref)
    print ("E_ref")
    print (E_ref)
    E_ref2 = mycc.e_tot - myhf.energy_nuc()

    E = exp_val_uccsd(mycc, hcore, eri, blksize=3)
    print ("E")
    print (E)
    print ("diff")
    print (E_ref - E_ref2)
    assert abs(E_ref - E_ref2) < 1e-7
    assert abs(E - E_ref) < 1e-10
    
    mycc = myhf.CCSD().set(frozen=[[2, 8, 10], [0, 8, 10]])
    mycc.kernel()
    mycc.solve_lambda()
    rdm1 = np.asarray(mycc.make_rdm1(ao_repr=False))
    rdm2 = np.asarray(mycc.make_rdm2(ao_repr=False))
    
    E1_ref  = np.einsum('spq, sqp ->', hcore, rdm1)
    E2_aa_ref = 0.5 * np.einsum('pqrs, pqrs ->', eri_s1[0], rdm2[0])
    E2_bb_ref = 0.5 * np.einsum('pqrs, pqrs ->', eri_s1[1], rdm2[2])
    E2_ab_ref = np.einsum('pqrs, pqrs ->', eri_s1[2], rdm2[1])
    E2_ref = E2_aa_ref + E2_bb_ref + E2_ab_ref
    E_ref = E1_ref + E2_ref
    E_ref2 = mycc.e_tot - myhf.energy_nuc()
    assert abs(E_ref - E_ref2) < 1e-8

    E = exp_val_uccsd(mycc, hcore, eri, blksize=3)
    print ("E")
    print (E)
    print ("diff")
    print (abs(E - E_ref))
    assert abs(E_ref - E_ref2) < 1e-8
    assert abs(E - E_ref) < 1e-10
    
    mycc = myhf.CCSD().set(frozen=[[1, 2], [0, 1, 2, 9]])
    mycc.kernel()
    mycc.solve_lambda()
    rdm1 = np.asarray(mycc.make_rdm1(ao_repr=False))
    rdm2 = np.asarray(mycc.make_rdm2(ao_repr=False))
    
    E1_ref  = np.einsum('spq, sqp ->', hcore, rdm1)
    E2_aa_ref = 0.5 * np.einsum('pqrs, pqrs ->', eri_s1[0], rdm2[0])
    E2_bb_ref = 0.5 * np.einsum('pqrs, pqrs ->', eri_s1[1], rdm2[2])
    E2_ab_ref = np.einsum('pqrs, pqrs ->', eri_s1[2], rdm2[1])
    E2_ref = E2_aa_ref + E2_bb_ref + E2_ab_ref
    E_ref = E1_ref + E2_ref
    E_ref2 = mycc.e_tot - myhf.energy_nuc()
    assert abs(E_ref - E_ref2) < 1e-8
    
    E = exp_val_uccsd(mycc, hcore, eri, blksize=3)
        
    print ("E")
    print (E)
    print ("diff")
    print (abs(E - E_ref))
    assert abs(E - E_ref) < 1e-10

def test_exp_val_gccsd():
    import numpy as np
    import pyscf
    from pyscf import ao2mo
    from libdmet.basis_transform import make_basis
    from libdmet.solver.cc import exp_val_gccsd
    from libdmet.utils.misc import mdot
    from libdmet.utils import logger as log

    log.verbose = "DEBUG1"
    
    np.random.seed(1)
    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = '321g',
        spin = 2,
        verbose = 5)
    mol.incore_anyway = True

    myhf = mol.GHF()
    hcoreX = myhf.get_hcore()
    pot = (np.random.random(hcoreX.shape) - 0.5) * 0.0 #* 1e-1
    pot = pot + pot.T
    hcoreX += pot
    myhf.get_hcore = lambda *args: hcoreX

    myhf.conv_tol = 1e-12
    myhf.kernel()
    
    nao = mol.nao_nr()
    C_ao_lo = myhf.mo_coeff

    hcore = make_basis.transform_h1_to_mo_mol(myhf.get_hcore(), C_ao_lo)
    #ovlp = make_basis.transform_h1_to_mo_mol(myhf.get_ovlp(), C_ao_lo)
    
    mo_a = C_ao_lo[:nao]
    mo_b = C_ao_lo[nao:]
    eri  = ao2mo.kernel(myhf._eri, mo_a)
    eri += ao2mo.kernel(myhf._eri, mo_b)
    eri1 = ao2mo.kernel(myhf._eri, (mo_a, mo_a, mo_b, mo_b))
    eri += eri1
    eri += eri1.T
    eri1 = None
    eri_s1 = ao2mo.restore(1, eri, nao*2)

    mycc = myhf.CCSD().set(frozen=0)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-8
    mycc.kernel()
    mycc.solve_lambda()
    rdm1 = np.asarray(mycc.make_rdm1(ao_repr=False))
    rdm2 = np.asarray(mycc.make_rdm2(ao_repr=False))
    
    E1_ref  = np.einsum('pq, qp ->', hcore, rdm1)
    E2_ref = 0.5 * np.einsum('pqrs, pqrs ->', eri_s1, rdm2)
    E_ref = E1_ref + E2_ref
    print ("E1_ref")
    print (E1_ref)
    print ("E2_ref")
    print (E2_ref)
    print ("E_ref")
    print (E_ref)
    E_ref2 = mycc.e_tot - myhf.energy_nuc()
    print (abs(E_ref - E_ref2))
    assert abs(E_ref - E_ref2) < 1e-7
    
    E = exp_val_gccsd(mycc, hcore, eri, blksize=2)
    print ("E")
    print (E)
    print ("diff")
    print (abs(E - E_ref))
    assert abs(E - E_ref) < 1e-10
    
    mycc = myhf.CCSD().set(frozen=[0, 2, 7, 8, 9])
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-8
    mycc.kernel()
    mycc.solve_lambda()
    rdm1 = np.asarray(mycc.make_rdm1(ao_repr=False))
    rdm2 = np.asarray(mycc.make_rdm2(ao_repr=False))
    
    E1_ref  = np.einsum('pq, qp ->', hcore, rdm1)
    E2_ref = 0.5 * np.einsum('pqrs, pqrs ->', eri_s1, rdm2)
    E_ref = E1_ref + E2_ref
    print (E_ref)
    E_ref2 = mycc.e_tot - myhf.energy_nuc()
    print (abs(E_ref - E_ref2))
    assert abs(E_ref - E_ref2) < 1e-7
    
    E = exp_val_gccsd(mycc, hcore, eri, blksize=2)
    print ("E")
    print (E)
    print ("diff")
    print (abs(E - E_ref))
    assert abs(E - E_ref) < 1e-10
    
def test_cc_restart():
    import numpy as np
    import pyscf
    from pyscf import ao2mo
    from libdmet.system.integral import Integral
    from libdmet.utils.misc import mdot, max_abs
    from libdmet.solver import impurity_solver 
    from libdmet.utils import logger as log
    log.verbose = "DEBUG1"

    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'ccpvdz',
    )

    myhf = mol.HF()
    myhf.kernel()
    E_ref = myhf.e_tot
    rdm1_mf = myhf.make_rdm1()
    
    # CC reference
    mycc = myhf.CCSD().set(frozen=2).run()
    E_cc_ref = mycc.e_tot
    
    restricted = True
    bogoliubov = False
    norb = mol.nao_nr()
    H1 = myhf.get_hcore()[None]
    ovlp = myhf.get_ovlp()
    H2 = myhf._eri[None]
    H0 = myhf.energy_nuc()

    Ham = Integral(norb, restricted, bogoliubov, H0, {"cd": H1}, \
            {"ccdd": H2}, ovlp=ovlp)
    
    # RCC
    solver = impurity_solver.CCSD(restricted=restricted, tol=1e-10, frozen=2)
    basis = np.eye(mol.nao_nr())
    rdm1, E = solver.run(Ham, nelec=mol.nelectron, dm0=rdm1_mf, restart=True, basis=basis)
    assert abs(E - E_cc_ref) < 1e-8
    rdm1_2, E_2 = solver.run(Ham, nelec=mol.nelectron, dm0=rdm1_mf, restart=True, basis=basis)
    assert abs(E_2 - E) < 1e-8
    
    # UCC
    solver = impurity_solver.CCSD(restricted=False, tol=1e-10, frozen=[[0, 1], [0, 1]])
    basis = np.asarray((np.eye(mol.nao_nr()), np.eye(mol.nao_nr())))
    rdm1, E = solver.run(Ham, nelec=mol.nelectron, dm0=rdm1_mf, restart=True, basis=basis)
    assert abs(E - E_cc_ref) < 1e-8
    rdm1_2, E_2 = solver.run(Ham, nelec=mol.nelectron, dm0=rdm1_mf, restart=True, basis=basis)
    assert abs(E_2 - E) < 1e-8

def test_ao2mo_uhf():
    import numpy as np
    from pyscf import gto, ao2mo
    from libdmet.solver import scf as scf_hp
    from libdmet.solver import cc
    from libdmet.utils import max_abs
    nao = 14
    eri_aa = np.random.random((nao, nao, nao, nao)) 
    eri_aa = eri_aa + eri_aa.transpose(1, 0, 2, 3)
    eri_aa = eri_aa + eri_aa.transpose(0, 1, 3, 2)
    #eri_aa = eri_aa + eri_aa.transpose(2, 3, 0, 1)
    eri_aa = ao2mo.restore(4, eri_aa, nao)

    eri_bb = np.random.random((nao, nao, nao, nao)) 
    eri_bb = eri_bb + eri_bb.transpose(1, 0, 2, 3)
    eri_bb = eri_bb + eri_bb.transpose(0, 1, 3, 2)
    #eri_bb = eri_bb + eri_bb.transpose(2, 3, 0, 1)
    eri_bb = ao2mo.restore(4, eri_bb, nao)
    
    eri_ab = np.random.random((nao, nao, nao, nao)) 
    eri_ab = eri_ab + eri_ab.transpose(1, 0, 2, 3)
    eri_ab = eri_ab + eri_ab.transpose(0, 1, 3, 2)
    eri_ab = ao2mo.restore(4, eri_ab, nao)
    
    eri = np.array((eri_aa, eri_bb, eri_ab))
    
    hcore_a = np.random.random((nao, nao))
    hcore_a = hcore_a + hcore_a.T
    hcore_b = np.random.random((nao, nao))
    hcore_b = hcore_b + hcore_b.T
    hcore = np.asarray((hcore_a, hcore_b))

    mol = gto.M(verbose=4)
    n = nao
    mol.nelectron = 8
    mol.spin = 2
    mol.incore_anyway = True

    mf = scf_hp.UIHF(mol)
    mf.get_hcore = lambda *args: hcore
    mf.get_ovlp = lambda *args: np.eye(n)
    mf._eri = eri
    mf.max_cycle = 2
    mf.kernel()

    mycc = cc.UICCSD(mf)
    mycc.frozen = [[0, 2], [1]]
    eris_ref = cc._make_eris_incore_uhf_ref(mycc)
    eris = mycc.ao2mo()
    
    for key in eris.__dict__.keys():
        print ("key", key)
        if isinstance(eris.__dict__[key], np.ndarray):
            diff = max_abs(eris.__dict__[key] - eris_ref.__dict__[key])
            print (eris.__dict__[key].shape)
            print (diff)   
            assert diff < 1e-12 

def test_ao2mo_ghf():
    import numpy as np
    from pyscf import gto, ao2mo
    from libdmet.solver import scf as scf_hp
    from libdmet.solver import cc
    from libdmet.utils import max_abs
    nao = 10
    eri = np.random.random((nao, nao, nao, nao)) 
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)
    hcore = np.random.random((nao, nao))
    hcore = hcore + hcore.T
    
    mol = gto.M(verbose=4)
    n = nao
    mol.nelectron = 4
    mol.incore_anyway = True

    mf = scf_hp.GGHF(mol)
    mf.get_hcore = lambda *args: hcore
    mf.get_ovlp = lambda *args: np.eye(n)
    mf._eri = ao2mo.restore(4, eri, n)
    mf.max_cycle = 2
    mf.kernel()
    
    mycc = cc.GGCCSD(mf)
    eris_ref = cc._make_eris_incore_ghf_ref(mycc)
    eris = mycc.ao2mo()
    
    for key in eris.__dict__.keys():
        print ("key", key)
        if isinstance(eris.__dict__[key], np.ndarray):
            diff = max_abs(eris.__dict__[key] - eris_ref.__dict__[key])
            print (diff)        
            assert diff < 1e-12 

def t_utccsd_dmrg():
    import numpy as np
    import pyscf
    from pyscf import ao2mo
    from libdmet.system.integral import Integral
    from libdmet.solver import impurity_solver 
    from libdmet.basis_transform import make_basis
    from libdmet.utils.misc import mdot
    from libdmet.utils import logger as log

    log.verbose = "DEBUG1"

    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = '321g',
        spin = 2,
        verbose = 5)
    
    myhf = mol.UHF()
    myhf.conv_tol = 1e-12
    myhf.kernel()
    
    nao = mol.nao_nr()
    C_ao_lo = myhf.mo_coeff
    hcore = make_basis.transform_h1_to_mo_mol(myhf.get_hcore(), C_ao_lo)
    ovlp = make_basis.transform_h1_to_mo_mol(myhf.get_ovlp(), C_ao_lo)
    rdm1 = make_basis.transform_rdm1_to_mo_mol(myhf.make_rdm1(), C_ao_lo, myhf.get_ovlp())
    
    eri_aa = ao2mo.general(myhf._eri, (C_ao_lo[0], C_ao_lo[0], \
            C_ao_lo[0], C_ao_lo[0]))
    eri_bb = ao2mo.general(myhf._eri, (C_ao_lo[1], C_ao_lo[1], \
            C_ao_lo[1], C_ao_lo[1]))
    eri_ab = ao2mo.general(myhf._eri, (C_ao_lo[0], C_ao_lo[0], \
            C_ao_lo[1], C_ao_lo[1]))
    eri = np.asarray((eri_aa, eri_bb, eri_ab))
    eri_s1 = np.zeros((3, nao, nao, nao, nao))
    for s in range(3):
        eri_s1[s] = ao2mo.restore(1, eri[s], nao)
    
    mycc = myhf.CCSD().set(frozen=0)
    eris = mycc.ao2mo()
    E_corr = mycc.kernel(eris=eris)[0]
    E_t = mycc.ccsd_t(eris=eris)
    E_ccsdt = mycc.e_tot + E_t

    Ham = Integral(nao, False, False, myhf.energy_nuc(), {"cd": hcore}, {"ccdd": eri}, ovlp=ovlp)
    solver = impurity_solver.UTCCSD(restricted=False, tol=1e-10, Sz=mol.spin)
    rdm1, E = solver.run(Ham, nelec=mol.nelectron, ncas=6, nelecas=(4, 2))

if __name__ == "__main__":
    test_ao2mo_ghf()
    t_utccsd_dmrg()
    test_ao2mo_uhf()
    test_exp_val_uccsd()
    test_exp_val_gccsd()
    test_cc_restart()
    test_exp_val_rccsd()
    test_uccsd_t()
    test_rccsd()
    test_bccd()
