#!/usr/bin/env python

def test_GGMP2():
    import numpy as np
    from pyscf import gto, scf, ao2mo
    from libdmet_solid.system import integral
    from libdmet_solid.solver import scf as scf_hp
    from libdmet_solid.utils.misc import tile_eri
    from libdmet.solver import gmp2
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.verbose = 4
    mol.build()
    mf = scf.RHF(mol).run()

    mf = scf.addons.convert_to_ghf(mf)
    dm0 = mf.make_rdm1()
    dm_guess = mf.get_init_guess()
    mf.kernel(dm0=dm0)

    ovlp = mf.get_ovlp()
    H0 = mf.energy_nuc()
    H1 = mf.get_hcore()
    H2 = mf._eri
    H2 = ao2mo.restore(4, H2, mol.nao_nr())

    H2 = tile_eri(H2, H2, H2)
    Ham = integral.Integral(H1.shape[-1], True, False, H0, {"cd": H1[None]},
                            {"ccdd": H2[None]}, ovlp=ovlp)

    scfsolver = scf_hp.SCF(newton_ah=True)
    scfsolver.set_system(mol.nelectron, 0, False, True, max_memory=mol.max_memory)
    scfsolver.set_integral(Ham)
    ehf, rhoHF = scfsolver.GGHF(tol=1e-12, InitGuess=dm0)
    emp2_ref, rdm1_mp2_ref = scfsolver.GMP2()

    emp2_ref2 = -0.204019967288338

    mf = scfsolver.mf

    pt = gmp2.GGMP2(mf)
    emp2, t2 = pt.kernel()

    diff_1 = abs(emp2 - emp2_ref)
    print("diff to emp2_ref from pyscf")
    print (diff_1)
    assert diff_1 < 1e-9

    diff_2 = abs(emp2 - emp2_ref2)
    print("diff to RMP2 from pyscf")
    print (diff_2)
    assert diff_2 < 1e-9

    rdm1 = pt.make_rdm1(ao_repr=True)
    rdm2 = pt.make_rdm2(ao_repr=True)

    E_re = np.einsum('pq, qp -> ', mf.get_hcore(), rdm1) + \
           np.einsum('pqrs, pqrs ->', ao2mo.restore(1, mf._eri, mol.nao_nr()*2), rdm2) * 0.5 + \
           mf.energy_nuc()

    diff_3 = abs(E_re - pt.e_tot)
    print ("diff E from rdm12")
    print (diff_3)
    assert diff_3 < 1e-10

    diff_rdm1 = np.linalg.norm(rdm1 - rdm1_mp2_ref)
    print ("diff rdm1")
    print (diff_rdm1)
    assert diff_rdm1 < 1e-10

    # non canonical MP2
    mf = scf.RHF(mol).run(max_cycle=1)
    mf = scf.addons.convert_to_ghf(mf)
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy

    scfsolver = scf_hp.SCF(newton_ah=False, no_kernel=True)
    scfsolver.set_system(mol.nelectron, 0, False, True, max_memory=mol.max_memory)
    scfsolver.set_integral(Ham)
    ehf, rhoHF = scfsolver.GGHF(tol=1e-12, InitGuess=dm_guess, MaxIter=0)
    mf = scfsolver.mf
    mf.mo_coeff = mo_coeff
    mf.mo_occ = mo_occ
    mf.mo_energy = mo_energy

    pt = gmp2.GGMP2(mf)
    pt.conv_tol = 1e-10
    pt.conv_tol_normt = 1e-8
    E, t2 = pt.kernel()

    E_ref = -0.204479916653321
    diff_4 = abs(E - E_ref)
    print ("diff non canonical")
    print (diff_4)
    assert diff_4 < 1e-9

if __name__ == "__main__":
    test_GGMP2()
