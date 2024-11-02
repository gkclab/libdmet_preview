#!/usr/bin/env python

def test_gtccsd_with_rtccsd():
    from pyscf import gto, scf, ao2mo
    from pyscf import cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.build(verbose=4)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.run()
    mf = scf.addons.convert_to_ghf(mf)
    rdm1_mf_ref = mf.make_rdm1()
    E_mf_ref = mf.e_tot

    #mycc = cc.GCCSD(mf)
    #e_cc_ref, t1, t2 = mycc.kernel()

    from libdmet.system import integral
    from libdmet.solver import scf as scf_hp
    from libdmet.solver import scf_solver
    from libdmet.solver import gtccsd
    from libdmet.utils import tile_eri

    nelec = mol.nelectron
    nao = mol.nao_nr()
    nso = nao * 2

    # core
    # O 1s O 2s -> 2 core orbitals (4 spin orbitals), 4 core electrons
    # val
    # O 2p H 1s x 2 -> 5 val orbitals (10 spin orbitals), 6 val electrons
    # virt
    # 34 spin orbitals

    ncas = 10
    nelecas = 6

    e_nuc = mf.energy_nuc()
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri_ao = ao2mo.restore(4, mf._eri, nao)
    eri = tile_eri(eri_ao, eri_ao, eri_ao)
    eri_ao = None

    print (hcore.shape)
    print (ovlp.shape)
    print (eri.shape)

    Ham = integral.Integral(hcore.shape[-1], True, False, e_nuc, {"cd": hcore[None]},
                            {"ccdd": eri[None]}, ovlp=ovlp)


    solver = scf_solver.SCFSolver(ghf=True, tol=1e-10, max_cycle=200,
                                  oomp2=False, tol_normt=1e-6, ci_conv_tol=1e-8,
                                  level_shift=0.1, restart=True, mc_conv_tol=1e-6)

    rdm1_mf, E_mf = solver.run(Ham, nelec=mol.nelectron, dm0=rdm1_mf_ref)

    diff_mf = abs(E_mf - E_mf_ref)
    print ("E_mf : ", E_mf)
    print ("diff to ref : ", diff_mf)
    assert diff_mf < 1e-8

    mf = solver.scfsolver.mf

    mycc = gtccsd.GGTCCSD(mf, ncas=ncas, nelecas=nelecas)
    mycc.conv_tol   = 1e-10
    e_cc, t1, t2 = mycc.kernel()

    print (e_cc)
    e_cc_ref = -0.213484111125395
    diff_cc = abs(e_cc - e_cc_ref)
    print ("e_cc : ", e_cc)
    print ("diff to ref : ", diff_cc)
    assert diff_cc < 1e-8

def t_gtccsd_with_utccsd():
    from pyscf import gto, scf, ao2mo
    from pyscf import cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [8 , (0. , 0. ,  1.5)]]

    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.build(verbose=4)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-13
    mf.run()
    mf.analyze()
    mf = scf.addons.convert_to_ghf(mf)
    rdm1_mf_ref = mf.make_rdm1()
    E_mf_ref = mf.e_tot

    #mycc = cc.GCCSD(mf)
    #e_cc_ref, t1, t2 = mycc.kernel()

    from libdmet.system import integral
    from libdmet.solver import scf as scf_hp
    from libdmet.solver import scf_solver
    from libdmet.solver import gtccsd
    from libdmet.utils import tile_eri

    nelec = mol.nelectron
    nao = mol.nao_nr()
    nso = nao * 2

    # core
    # O 1s O 2s -> 4 core orbitals (8 spin orbitals), 8 core electrons
    # val
    # O 2p -> 6 val orbitals (12 spin orbitals), 8 val electrons
    # virt
    # XX spin orbitals

    ncas = 12
    nelecas = 8

    e_nuc = mf.energy_nuc()
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri_ao = ao2mo.restore(4, mf._eri, nao)
    eri = tile_eri(eri_ao, eri_ao, eri_ao)
    eri_ao = None

    print (hcore.shape)
    print (ovlp.shape)
    print (eri.shape)

    Ham = integral.Integral(hcore.shape[-1], True, False, e_nuc, {"cd": hcore[None]},
                            {"ccdd": eri[None]}, ovlp=ovlp)


    solver = scf_solver.SCFSolver(ghf=True, tol=1e-10, max_cycle=200,
                                  oomp2=False, tol_normt=1e-6, ci_conv_tol=1e-8,
                                  level_shift=0.1, restart=True, mc_conv_tol=1e-6)

    rdm1_mf, E_mf = solver.run(Ham, nelec=mol.nelectron, dm0=rdm1_mf_ref)

    diff_mf = abs(E_mf - E_mf_ref)
    print ("E_mf : ", E_mf)
    print ("diff to ref : ", diff_mf)
    assert diff_mf < 1e-8

    mf = solver.scfsolver.mf

    mycc = gtccsd.GGTCCSD(mf, ncas=ncas, nelecas=nelecas)
    mycc.conv_tol   = 1e-10
    e_cc, t1, t2 = mycc.kernel()

    e_cc_ref = -0.4753733832031933
    diff_cc = abs(e_cc - e_cc_ref)
    print ("e_cc : ", e_cc)
    print ("diff to ref : ", diff_cc)
    assert diff_cc < 1e-6

def test_N2():
    from pyscf import gto, scf, ao2mo
    from pyscf import cc

    mol = gto.M(
        atom  = '''
        N    0.0000000    0.0000000    0.5600041
        N    0.0000000    0.0000000   -0.5600041
        ''',
        basis = 'cc-pvdz',
        spin  = 0
        )
    mol.incore_anyway = True
    mol.build()

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-13
    mf.run()

    mf = scf.addons.convert_to_ghf(mf)
    rdm1_mf_ref = mf.make_rdm1()
    E_mf_ref = mf.e_tot

    from libdmet.system import integral
    from libdmet.solver import scf as scf_hp
    from libdmet.solver import scf_solver
    from libdmet.solver import gtccsd
    from libdmet.utils import tile_eri

    nelec = mol.nelectron
    nao = mol.nao_nr()
    nso = nao * 2

    # core
    # O 1s O 2s -> 2 core orbitals (4 spin orbitals), 4 core electrons
    # val
    # O 2p H 1s x 2 -> 5 val orbitals (10 spin orbitals), 6 val electrons
    # virt
    # 34 spin orbitals

    ncas = 12
    nelecas = 6

    e_nuc = mf.energy_nuc()
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri_ao = ao2mo.restore(4, mf._eri, nao)
    eri = tile_eri(eri_ao, eri_ao, eri_ao)
    eri_ao = None

    print (hcore.shape)
    print (ovlp.shape)
    print (eri.shape)

    Ham = integral.Integral(hcore.shape[-1], True, False, e_nuc, {"cd": hcore[None]},
                            {"ccdd": eri[None]}, ovlp=ovlp)


    solver = scf_solver.SCFSolver(ghf=True, tol=1e-10, max_cycle=200,
                                  oomp2=False, tol_normt=1e-6, ci_conv_tol=1e-8,
                                  level_shift=0.1, restart=True, mc_conv_tol=1e-6)

    rdm1_mf, E_mf = solver.run(Ham, nelec=mol.nelectron, dm0=rdm1_mf_ref)

    diff_mf = abs(E_mf - E_mf_ref)
    print ("E_mf : ", E_mf)
    print ("diff to ref : ", diff_mf)
    assert diff_mf < 1e-8

    mf = solver.scfsolver.mf

    mycc = gtccsd.GGTCCSD(mf, ncas=ncas, nelecas=nelecas)
    mycc.conv_tol   = 1e-10
    e_cc, t1, t2 = mycc.kernel()

    e_cc_ref = -0.328404952959594
    diff_cc = abs(e_cc - e_cc_ref)
    print ("e_cc : ", e_cc)
    print ("diff to ref : ", diff_cc)
    assert diff_cc < 1e-6

if __name__ == "__main__":
    t_gtccsd_with_utccsd()
    test_N2()
    test_gtccsd_with_rtccsd()
