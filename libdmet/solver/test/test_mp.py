#! /usr/bin/env python

"""
Test mp routines.
"""

def test_mp2():
    import numpy as np
    import scipy.linalg as la

    from pyscf import gto, scf, mp
    from libdmet.solver import scf as scf_hp
    from libdmet.solver import mp as mp_hp
    from libdmet.system import integral
    from libdmet.utils.misc import max_abs, mdot
    
    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = gto.M(
        atom = 'H 0 0 0; F 0 0 1.1',
        basis = '321G')
    mf = mol.RHF()
    mf.conv_tol = 1e-12
    mf.run()
    E_mf_ref = mf.e_tot
    rdm1_mf_ref = mf.make_rdm1()
    mypt = mf.MP2()
    E_mp2_ref, t2_ref = mypt.kernel()
    rdm1_mp2_ref = mypt.make_rdm1(ao_repr=True)

    restricted = True
    bogoliubov = False
    nelec = mol.nelectron
    spin = mol.spin
    
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    nao = mol.nao_nr()
    eri = mf._eri
    H0 = mf.energy_nuc()
    ints = integral.Integral(nao, restricted, bogoliubov, H0, {"cd": hcore[None]},
            {"ccdd": eri[None]}, ovlp=ovlp)
    
    myscf = scf_hp.SCF()
    myscf.set_system(nelec, spin, bogoliubov=bogoliubov, \
            spinRestricted=restricted)
    myscf.set_integral(ints)
    E_mf, rdm1_mf = myscf.HF(InitGuess=rdm1_mf_ref)
    assert abs(E_mf - E_mf_ref) < 1e-11
    assert max_abs(rdm1_mf * 2.0 - rdm1_mf_ref) < 1e-7
    
    E_mp2, rdm1_mp2 = myscf.MP2()
    assert abs(E_mp2 - E_mp2_ref) < 1e-9
    assert max_abs(rdm1_mp2 * 2.0 - rdm1_mp2_ref) < 1e-7

def test_ump2():
    import numpy as np
    import scipy.linalg as la

    from pyscf import gto, scf, mp
    from libdmet.solver import scf as scf_hp
    from libdmet.solver import mp as mp_hp
    from libdmet.system import integral
    from libdmet.basis_transform import make_basis
    from libdmet.utils.misc import max_abs, mdot
    
    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = gto.M(
        atom = 'H 0 0 0; O 0 0 1.1',
        basis = '631G',
        spin = 1,
        verbose = 4)
    mf = mol.UHF()
    mf.conv_tol = 1e-12
    mf.run()
    E_mf_ref = mf.e_tot
    rdm1_mf_ref = mf.make_rdm1()
    mypt = mf.MP2()
    E_mp2_ref, t2_ref = mypt.kernel()
    rdm1_mp2_ref = mypt.make_rdm1(ao_repr=True)

    restricted = False
    bogoliubov = False
    nelec = mol.nelectron
    spin = mol.spin
    
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    nao = mol.nao_nr()
    eri = mf._eri
    H0 = mf.energy_nuc()
    ints = integral.Integral(nao, restricted, bogoliubov, H0, {"cd": hcore[None]},
            {"ccdd": eri[None]}, ovlp=ovlp)
    
    myscf = scf_hp.SCF()
    myscf.set_system(nelec, spin, bogoliubov=bogoliubov, \
            spinRestricted=restricted)
    myscf.set_integral(ints)
    E_mf, rdm1_mf = myscf.HF(InitGuess=rdm1_mf_ref)
    assert abs(E_mf - E_mf_ref) < 1e-11
    assert max_abs(rdm1_mf - rdm1_mf_ref) < 1e-6
    
    E_mp2, rdm1_mp2 = myscf.MP2()
    assert abs(E_mp2 - E_mp2_ref) < 1e-8
    assert max_abs(np.array(rdm1_mp2) - np.array(rdm1_mp2_ref)) < 1e-6
    
    # MO based UIHF and UIMP2
    ints_mo = scf_hp.ao2mo_Ham(ints, mf.mo_coeff)
    ints_mo.ovlp = np.eye(nao) 
    myscf = scf_hp.SCF()
    myscf.set_system(nelec, spin, bogoliubov=bogoliubov, \
            spinRestricted=restricted)
    myscf.set_integral(ints_mo)
    dm0 = make_basis.transform_rdm1_to_mo_mol(rdm1_mf_ref, mf.mo_coeff, ovlp)
    E_mf, rdm1_mf = myscf.HF(InitGuess=dm0)
    rdm1_mf = make_basis.transform_rdm1_to_ao_mol(rdm1_mf, mf.mo_coeff)
    assert abs(E_mf - E_mf_ref) < 1e-11
    assert max_abs(rdm1_mf - rdm1_mf_ref) < 1e-6
    
    E_mp2, rdm1_mp2 = myscf.MP2()
    assert abs(E_mp2 - E_mp2_ref) < 1e-8
    rdm1_mp2 = make_basis.transform_rdm1_to_ao_mol(rdm1_mp2, mf.mo_coeff)
    assert max_abs(np.array(rdm1_mp2) - np.array(rdm1_mp2_ref)) < 1e-6

def test_gmp2():
    import numpy as np
    import scipy.linalg as la

    from pyscf import gto, scf, ao2mo, mp
    from libdmet.solver import scf as scf_hp
    from libdmet.solver import mp as mp_hp
    from libdmet.system import integral
    from libdmet.basis_transform import make_basis
    from libdmet.utils.misc import max_abs, mdot
    
    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = gto.M(
        atom = 'H 0 0 0; O 0 0 1.1',
        basis = '631G',
        spin = 1,
        verbose = 4)
    mf = mol.UHF()
    mf.conv_tol = 1e-12
    mf.run()
    E_mf_ref = mf.e_tot
    rdm1_mf_ref = mf.make_rdm1()
    mypt = mf.MP2()
    E_mp2_ref, t2_ref = mypt.kernel()
    rdm1_mp2_ref = mypt.make_rdm1(ao_repr=True)
    
    restricted = False
    bogoliubov = False
    nelec = mol.nelectron
    spin = mol.spin
    
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    nao = mol.nao_nr()
    nso = nao * 2
    eri = ao2mo.restore(1, mf._eri, nao)
    H0 = mf.energy_nuc()
    
    hcore_g = la.block_diag(hcore, hcore)
    ovlp_g = la.block_diag(ovlp, ovlp)
    eri_g = np.zeros((nso, nso, nso, nso))
    eri_g[:nao, :nao, :nao, :nao] = eri
    eri_g[:nao, :nao, nao:, nao:] = eri
    eri_g[nao:, nao:, :nao, :nao] = eri
    eri_g[nao:, nao:, nao:, nao:] = eri
    eri_g = ao2mo.restore(8, eri_g, nso)
    ints = integral.Integral(nso, restricted, bogoliubov, H0, {"cd":
        hcore_g[None]}, {"ccdd": eri_g[None]}, ovlp=ovlp_g)

    myscf = scf_hp.SCF()
    myscf.set_system(nelec, spin, bogoliubov=bogoliubov, \
            spinRestricted=restricted)
    myscf.set_integral(ints)

    dm0 = la.block_diag(*rdm1_mf_ref)

    E_mf, rdm1_mf = myscf.GGHF(InitGuess=dm0)
    assert abs(E_mf - E_mf_ref) < 1e-11
    assert max_abs(rdm1_mf - dm0) < 1e-6
    
    E_mp2, rdm1_mp2 = myscf.GMP2()
    assert abs(E_mp2 - E_mp2_ref) < 1e-8
    rdm1_mp2 = np.array((rdm1_mp2[:nao, :nao], rdm1_mp2[nao:, nao:]))
    assert max_abs(np.array(rdm1_mp2) - np.array(rdm1_mp2_ref)) < 1e-6

def test_oomp2():
    import numpy as np
    from pyscf import gto, scf, ao2mo
    from libdmet_solid.system import integral
    from libdmet_solid.solver import scf as scf_hp
    from libdmet_solid.utils.misc import tile_eri
    from libdmet.utils import logger as log 
    from libdmet.solver.mp import MP2AsFCISolver
    from libdmet.solver.gmc1step import GCASSCF
    np.set_printoptions(3, linewidth=1000)
    log.verbose = "DEBUG2"
    
    mol = gto.M(atom = 'H 0 0 0; F 0 0 1.2',
                basis = 'ccpvdz',
                verbose = 4)
    mf = scf.RHF(mol).run()
    
    mf = scf.addons.convert_to_ghf(mf)
    dm0 = mf.make_rdm1()
    mf.kernel(dm0=dm0)
    
    ovlp = mf.get_ovlp()
    H0 = mf.energy_nuc()
    H1 = mf.get_hcore()
    H2 = mf._eri
    H2 = ao2mo.restore(4, H2, mol.nao_nr())
    
    H2 = tile_eri(H2, H2, H2)
    dm0 = mf.make_rdm1()
    
    Ham = integral.Integral(H1.shape[-1], True, False, H0, {"cd": H1[None]},
                            {"ccdd": H2[None]}, ovlp=ovlp)

    scfsolver = scf_hp.SCF(newton_ah=True)
    scfsolver.set_system(mol.nelectron, 0, False, True, max_memory=mol.max_memory)
    scfsolver.set_integral(Ham)
    ehf, rhoHF = scfsolver.GGHF(tol=1e-12, InitGuess=dm0)
    mf = scfsolver.mf

    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelectron
    
    mc = GCASSCF(mf, norb//2, nelec)
    mc.fcisolver = MP2AsFCISolver(ghf=True, level_shift=0.1)
    mc.internal_rotation = True
    E = mc.kernel()[0]
    
    print ("diff to ROOMP2", abs(E - -100.176641888785))
    assert abs(E - -100.176641888785) < 1e-9

if __name__ == "__main__":
    test_oomp2()
    test_mp2()
    test_ump2()
    test_gmp2()
