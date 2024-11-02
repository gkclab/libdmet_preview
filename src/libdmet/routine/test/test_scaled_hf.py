#!/usr/bin/env python

def test_scaled_hf():
    import copy
    import numpy as np
    from scipy import linalg as la

    import pyscf
    from pyscf import gto, scf, ao2mo, lib
    from pyscf.lib import logger

    from libdmet.system import lattice, integral
    from libdmet.basis_transform import make_basis
    from libdmet.routine import mfd, slater, spinless
    from libdmet.solver import scf as scf_hp
    from libdmet.solver import scf_solver
    from libdmet.utils import max_abs, mdot
    from libdmet.utils import logger as log
    from libdmet import lo
    from libdmet.lo import iao
    log.verbose = 'DEBUG1'

    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = pyscf.M(
        atom = 'O 0 0 0; O 0 0 1.2',
        basis = '321g',
        verbose=4)
    mol.incore_anyway = True

    alpha = 0.5
    beta = np.inf
    mf = scf_hp.RIHF(mol, alpha=alpha)
    E_rhf = mf.kernel()

    ## 6 orbitals, 8 electrons
    #mc = mf.CASCI(6, 8)
    #mc.run()

    nao = mol.nao_nr()
    nelec = mol.nelectron
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri = ao2mo.restore(4, mf._eri, nao)
    rdm1 = mf.make_rdm1()
    E_nuc = mf.energy_nuc()

    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff

    ncore = 4
    ncas = 6
    nvirt = nao - ncore - ncas

    print ("ncore", ncore)
    print ("ncas", ncas)
    print ("nvirt", nvirt)

    # first construct a set of LOs that froze F 1s and F 2s.
    minao = 'sto3g'
    pmol = iao.reference_mol(mol, minao=minao)
    basis = pmol._basis
    basis_val = {}
    basis_core = {}

    # O core is 1s and 2s
    basis_val["O"] = copy.deepcopy(basis["O"])
    basis_core["O"] = copy.deepcopy(basis["O"])
    basis_val["O"] = basis_val["O"][2:]
    basis_core["O"] = basis_core["O"][:2]

    pmol_val = pmol.copy()
    pmol_val.basis = basis_val
    pmol_val.build()

    print ("Valence:")
    val_labels = pmol_val.ao_labels()
    for i in range(len(val_labels)):
        val_labels[i] = val_labels[i].replace("O 1s", "O 2s")
    pmol_val.ao_labels = lambda *args: val_labels
    print (len(pmol_val.ao_labels()))
    print (pmol_val.ao_labels())
    pmol_core = pmol.copy()
    pmol_core.basis = basis_core
    pmol_core.build()
    core_labels = pmol_core.ao_labels()

    print ("Core:")
    print (len(pmol_core.ao_labels()))
    print (pmol_core.ao_labels())
    ncore = len(pmol_core.ao_labels())
    nval = pmol_val.nao_nr()
    nvirt = mol.nao_nr() - ncore - nval

    C_ao_lo, C_ao_lo_val, C_ao_lo_virt, C_ao_lo_core = \
            make_basis.get_C_ao_lo_iao_mol(mf, minao=minao, orth_virt=True,
                                           full_virt=False, full_return=True,
                                           pmol_val=pmol_val,
                                           pmol_core=pmol_core, tol=1e-10)
    C_ao_lo_xcore = make_basis.tile_C_ao_iao(C_ao_lo_val, C_virt=C_ao_lo_virt, C_core=None)

    lo_labels_xcore, val_labels, virt_labels = lo.get_labels(mol, minao=minao, full_virt=False,
                                                       B2_labels=val_labels,
                                                       core_labels=core_labels)
    lo_labels = core_labels + lo_labels_xcore

    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:ncore+ncas]
    mo_virt = mo_coeff[ncore+ncas:]
    #mo_xcore = mo_coeff[:, ncore:]
    #mo_xcore = np.dot(mo_xcore, la.qr(np.random.random((mo_xcore.shape[-1],mo_xcore.shape[-1])))[0])
    mo_xcore = C_ao_lo_xcore

    mo_occ_core = mo_occ[:ncore]
    mo_occ_cas = mo_occ[ncore:ncore+ncas]
    mo_occ_virt = mo_occ[ncore+ncas:]


    rdm1_core = mf.make_rdm1(mo_coeff=mo_core, mo_occ=mo_occ_core)
    rdm1_core_lo = make_basis.transform_rdm1_to_lo_mol(rdm1_core, mo_core, ovlp)
    rdm1_xcore = rdm1 - rdm1_core
    rdm1_xcore_lo = make_basis.transform_rdm1_to_lo_mol(rdm1_xcore, mo_xcore, ovlp)

    lattice.mulliken_lo_R0(mol, rdm1_xcore_lo, labels=lo_labels_xcore)

    #print (rdm1_core)
    #
    #hcore_core = mdot(mo_core.T, hcore, mo_core)
    #ovlp_core = mdot(mo_core.T, ovlp, mo_core)

    veff_core = mf.get_veff(dm=rdm1_core)
    E_core =  np.einsum('pq, qp ->', hcore, rdm1_core) + \
              0.5 * np.einsum('pq, qp ->', veff_core, rdm1_core)


    hcore_cas = mdot(mo_cas.T, hcore + veff_core, mo_cas)
    ovlp_cas = mdot(mo_cas.T, ovlp, mo_cas)
    rdm1_cas = make_basis.transform_rdm1_to_mo_mol(rdm1, mo_cas, ovlp)
    eri_cas = ao2mo.kernel(eri, mo_cas)
    nelecas = int(np.round(rdm1_cas.trace()))

    Ham = integral.Integral(ncas, restricted=True, bogoliubov=False,
                            H0=E_nuc+E_core, H1=hcore_cas[None], H2=eri_cas[None], ovlp=ovlp_cas)

    solver = scf_solver.SCFSolver(ghf=False, restricted=True, tol=1e-7, max_cycle=200,
                                  scf_newton=False, alpha=alpha, beta=beta)

    rdm1_re, E_re = solver.run(Ham, nelec=nelecas, scf_max_cycle=200, dm0=rdm1_cas)

    E_diff = abs(E_re - E_rhf)
    rdm1_diff = max_abs(rdm1_re * 2.0 - rdm1_cas)

    print ("RHF cas")
    print ("energy diff: ", E_diff)
    assert E_diff < 1e-10
    print ("rdm1 diff: ", rdm1_diff)
    assert rdm1_diff < 1e-10
    print ()

    ewocc, Mu = mfd.assignocc(mo_energy, nelec*0.5, beta=beta, mu0=0.0)[:2]

    print ("mo_energy")
    print (mo_energy)
    print ("Mu", Mu)
    print ("nelec")
    print (nelec)
    print (ewocc)
    print (np.sum(ewocc))

    # GHF
    nso = nao * 2
    ovlp_ghf = spinless.combine_mo_coeff(ovlp)
    C_sao_slo = spinless.combine_mo_coeff(mo_coeff)
    C_sao_slo_core = spinless.combine_mo_coeff(mo_core)
    C_sao_slo_xcore = spinless.combine_mo_coeff(mo_xcore)

    GH1, GH0 = spinless.transform_H1_local(hcore, ovlp=ovlp, compact=False)
    GV2_hf, GV1_hf, GV0_hf = spinless.transform_H2_local(eri, ovlp, compact=False)
    GV2, GV1, GV0 = spinless.transform_H2_local(eri, ovlp, compact=False, hyb=alpha)
    Grdm1 = spinless.transform_rdm1_local(rdm1*0.5, ovlp=ovlp, compact=False)

    GH1 += GV1
    E0 = E_nuc + GH0 + GV0

    Ham = integral.Integral(nso, restricted=True, bogoliubov=False,
                            H0=E0, H1=GH1[None], H2=GV2[None], ovlp=ovlp_ghf)

    solver = scf_solver.SCFSolver(ghf=True, restricted=True, tol=1e-7, max_cycle=200,
                                  scf_newton=False, alpha=alpha, beta=beta)
    rdm1_re_gso, E_re_gso = solver.run(Ham, nelec=GH1.shape[-1]//2, scf_max_cycle=200,
            dm0=Grdm1, Mu=Mu)

    E_diff = abs(E_re_gso - E_rhf)

    print ("GHF cas")
    print ("energy diff: ", E_diff)
    assert E_diff < 1e-10


    # GHF frozen core

    Grdm1_core_lo = spinless.transform_rdm1_local(rdm1_core_lo * 0.5, compact=False)
    Grdm1_core = make_basis.transform_rdm1_to_ao_mol(Grdm1_core_lo, C_sao_slo_core)
    Grdm1_xcore_lo = spinless.transform_rdm1_local(rdm1_xcore_lo * 0.5, compact=False)

    mf = solver.scfsolver.mf
    Gveff_core = mf.get_veff(dm=Grdm1_core)
    Ghcore = GH1 + Gveff_core
    GH0_core = np.einsum('pq, qp ->', GH1 + 0.5 * Gveff_core, Grdm1_core)

    GH1_lo = make_basis.transform_h1_to_lo_mol(Ghcore, C_sao_slo_xcore)
    GV2_lo = ao2mo.kernel(GV2, C_sao_slo_xcore)
    Govlp_lo = make_basis.transform_h1_to_lo_mol(ovlp_ghf, C_sao_slo_xcore)

    Ham = integral.Integral(GH1_lo.shape[-1], restricted=True, bogoliubov=False,
                            H0=E0+GH0_core, H1=GH1_lo[None], H2=GV2_lo[None],
                            ovlp=Govlp_lo)

    from libdmet.solver import impurity_solver
    solver = scf_solver.SCFSolver(ghf=True, restricted=True, tol=1e-7, max_cycle=200,
                                  scf_newton=False, alpha=alpha, beta=beta)
    rdm1_re_gso_froze, E_re_gso_froze = solver.run(Ham, nelec=GH1_lo.shape[-1]//2, scf_max_cycle=200,
                                                   dm0=Grdm1_xcore_lo, Mu=Mu)

    E_diff = abs(E_re_gso_froze - E_rhf)
    print ("GHF cas frozen core")
    print ("energy diff: ", E_diff)
    assert E_diff < 1e-10

    rdm1_A, rdm1_B, rdm1_D = spinless.extract_rdm1(rdm1_re_gso_froze)
    lattice.mulliken_lo_R0(mol, (rdm1_A, rdm1_B), labels=lo_labels_xcore)

    # GCCSD

    solver = impurity_solver.CCSD(ghf=True, restricted=True, tol=1e-5,
            tol_normt=2e-5, max_cycle=200, scf_newton=False, diis_space=10,
            level_shift=0.05, approx_l=True, alpha=alpha)

    rdm1_re_gso_froze, E_re_gso_froze = solver.run(Ham, nelec=GH1_lo.shape[-1]//2, scf_max_cycle=200,
                                                   dm0=Grdm1_xcore_lo, Mu=Mu)

    print ("GCCSD")
    rdm1_A, rdm1_B, rdm1_D = spinless.extract_rdm1(rdm1_re_gso_froze)
    lattice.mulliken_lo_R0(mol, (rdm1_A, rdm1_B), labels=lo_labels_xcore)

def test_scaled_rhf_ghf():
    import copy
    import numpy as np
    from scipy import linalg as la

    import pyscf
    from pyscf import gto, scf, ao2mo, lib
    from pyscf.lib import logger

    from libdmet.system import lattice, integral
    from libdmet.basis_transform import make_basis
    from libdmet.routine import mfd, slater, spinless
    from libdmet.solver import scf as scf_hp
    from libdmet.solver import scf_solver
    from libdmet.utils import max_abs, mdot
    from libdmet.utils import logger as log
    from libdmet import lo
    from libdmet.lo import iao
    log.verbose = 'DEBUG1'

    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = pyscf.M(
        atom = 'O 0 0 0; O 0 0 1.2',
        basis = '321g',
        verbose=4)
    mol.incore_anyway = True

    alpha = 0.5
    beta = np.inf
    mf = scf_hp.RIHF(mol, alpha=alpha)
    E_rhf = mf.kernel()

    nao = mol.nao_nr()
    nelec = mol.nelectron
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri = ao2mo.restore(4, mf._eri, nao)
    rdm1 = mf.make_rdm1()
    E_nuc = mf.energy_nuc()

    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff

    # O 1s 2s
    ncore = 4
    ncas = nao - ncore
    nvirt = nao - ncore - ncas

    print ("ncore", ncore)
    print ("ncas", ncas)
    print ("nvirt", nvirt)

    # first construct a set of LOs that froze O 1s and O 2s.
    minao = 'sto3g'
    pmol = iao.reference_mol(mol, minao=minao)
    basis = pmol._basis
    basis_val = {}
    basis_core = {}

    # O core is 1s and 2s
    basis_val["O"] = copy.deepcopy(basis["O"])
    basis_core["O"] = copy.deepcopy(basis["O"])
    basis_val["O"] = basis_val["O"][2:]
    basis_core["O"] = basis_core["O"][:2]

    pmol_val = pmol.copy()
    pmol_val.basis = basis_val
    pmol_val.build()

    print ("Valence:")
    val_labels = pmol_val.ao_labels()
    for i in range(len(val_labels)):
        val_labels[i] = val_labels[i].replace("O 1s", "O 2s")
    pmol_val.ao_labels = lambda *args: val_labels
    print (len(pmol_val.ao_labels()))
    print (pmol_val.ao_labels())
    pmol_core = pmol.copy()
    pmol_core.basis = basis_core
    pmol_core.build()
    core_labels = pmol_core.ao_labels()

    print ("Core:")
    print (len(pmol_core.ao_labels()))
    print (pmol_core.ao_labels())
    ncore = len(pmol_core.ao_labels())
    nval = pmol_val.nao_nr()
    nvirt = mol.nao_nr() - ncore - nval

    C_ao_lo, C_ao_lo_val, C_ao_lo_virt, C_ao_lo_core = \
            make_basis.get_C_ao_lo_iao_mol(mf, minao=minao, orth_virt=True,
                                           full_virt=False, full_return=True,
                                           pmol_val=pmol_val,
                                           pmol_core=pmol_core, tol=1e-10)
    C_ao_lo_xcore = make_basis.tile_C_ao_iao(C_ao_lo_val, C_virt=C_ao_lo_virt, C_core=None)

    lo_labels_xcore, val_labels, virt_labels = lo.get_labels(mol, minao=minao, full_virt=False,
                                                             B2_labels=val_labels,
                                                             core_labels=core_labels)
    lo_labels = core_labels + lo_labels_xcore

    mo_core = mo_coeff[:, :ncore]
    mo_occ_core = mo_occ[:ncore]
    mo_cas = mo_xcore = C_ao_lo_xcore

    rdm1_core = mf.make_rdm1(mo_coeff=mo_core, mo_occ=mo_occ_core)
    rdm1_core_lo = make_basis.transform_rdm1_to_lo_mol(rdm1_core, mo_core, ovlp)
    rdm1_xcore = rdm1 - rdm1_core
    rdm1_xcore_lo = make_basis.transform_rdm1_to_lo_mol(rdm1_xcore, mo_xcore, ovlp)

    lattice.mulliken_lo_R0(mol, rdm1_xcore_lo, labels=lo_labels_xcore)

    veff_core = mf.get_veff(dm=rdm1_core)
    E_core =  np.einsum('pq, qp ->', hcore, rdm1_core) + \
              0.5 * np.einsum('pq, qp ->', veff_core, rdm1_core)

    hcore_cas = mdot(mo_cas.T, hcore + veff_core, mo_cas)
    ovlp_cas = mdot(mo_cas.T, ovlp, mo_cas)
    rdm1_cas = make_basis.transform_rdm1_to_mo_mol(rdm1, mo_cas, ovlp)
    eri_cas = ao2mo.kernel(eri, mo_cas)
    nelecas = int(np.round(rdm1_cas.trace()))

    Ham = integral.Integral(ncas, restricted=True, bogoliubov=False,
                            H0=E_nuc+E_core, H1=hcore_cas[None], H2=eri_cas[None], ovlp=ovlp_cas)

    solver = scf_solver.SCFSolver(ghf=False, restricted=True, tol=1e-7, max_cycle=200,
                                  scf_newton=False, alpha=1.0, beta=beta)

    rdm1_re, E_re = solver.run(Ham, nelec=nelecas, scf_max_cycle=200, dm0=rdm1_cas)

    E_diff = abs(E_re - E_rhf)
    rdm1_diff = max_abs(rdm1_re * 2.0 - rdm1_cas)

    print ("RHF cas")
    print ("energy diff: ", E_diff)
    #assert E_diff < 1e-10
    print ("rdm1 diff: ", rdm1_diff)
    #assert rdm1_diff < 1e-10
    print ()

    lattice.mulliken_lo_R0(mol, rdm1_re, labels=lo_labels_xcore)

    # GHF case

    ewocc, Mu = mfd.assignocc(mo_energy, nelec*0.5, beta=beta, mu0=0.0)[:2]

    print ("mo_energy")
    print (mo_energy)
    print ("Mu", Mu)
    print ("nelec")
    print (nelec)
    print (ewocc)
    print (np.sum(ewocc))

    # GHF
    nso = nao * 2
    ovlp_ghf = spinless.combine_mo_coeff(ovlp)
    #C_sao_slo = spinless.combine_mo_coeff(mo_coeff)
    C_sao_slo_core = spinless.combine_mo_coeff(mo_core)
    C_sao_slo_xcore = spinless.combine_mo_coeff(mo_xcore)

    np.random.seed(10086)
    mat = np.eye(C_sao_slo_xcore.shape[-1])
    mat += (np.random.random(mat.shape) - 0.5) * 0.1
    Q = la.qr(mat, pivoting=True)[0]
    order = []
    for i in range(Q.shape[-1]):
        col = Q[:, i]
        idx = np.argmax(np.abs(col))
        order.append(idx)
        if col[idx] < 0.0:
            col *= -1

    Q = Q[order]
    #Q = la.qr(np.random.random((C_sao_slo_xcore.shape[-1], C_sao_slo_xcore.shape[-1])))[0]
    print (Q)
    #Q = np.eye(C_sao_slo_xcore.shape[-1])
    C_sao_slo_xcore = C_sao_slo_xcore @ Q
    #print (C_sao_slo_xcore)

    GH1, GH0 = spinless.transform_H1_local(hcore, ovlp=ovlp, compact=False)
    GV2_hf, GV1_hf, GV0_hf = spinless.transform_H2_local(eri, ovlp, compact=False)
    GV2, GV1, GV0 = spinless.transform_H2_local(eri, ovlp, compact=False, hyb=alpha)
    Grdm1 = spinless.transform_rdm1_local(rdm1*0.5, ovlp=ovlp, compact=False)

    GH1 += GV1

    dGV1 = GV1_hf - GV1

    E0 = E_nuc + GH0 + GV0

    Ham = integral.Integral(nso, restricted=True, bogoliubov=False,
                            H0=E0, H1=GH1[None], H2=GV2[None], ovlp=ovlp_ghf)

    solver = scf_solver.SCFSolver(ghf=True, restricted=True, tol=1e-7, max_cycle=200,
                                  scf_newton=False, alpha=alpha, beta=beta)
    rdm1_re_gso, E_re_gso = solver.run(Ham, nelec=GH1.shape[-1]//2, scf_max_cycle=200,
            dm0=Grdm1, Mu=Mu)

    E_diff = abs(E_re_gso - E_rhf)

    print ("GHF cas")
    print ("energy diff: ", E_diff)
    assert E_diff < 1e-10

    # GHF frozen core

    Grdm1_core_lo = spinless.transform_rdm1_local(rdm1_core_lo * 0.5, compact=False)
    Grdm1_core = make_basis.transform_rdm1_to_ao_mol(Grdm1_core_lo, C_sao_slo_core)
    Grdm1_xcore_ao = spinless.transform_rdm1_local(rdm1_xcore * 0.5, ovlp=ovlp, compact=False)
    Grdm1_xcore_lo = make_basis.transform_rdm1_to_lo_mol(Grdm1_xcore_ao, C_sao_slo_xcore, ovlp_ghf)

    # GV1_add for full HF in the active space
    rdm1_eye_xcore_lo = np.eye(C_ao_lo_xcore.shape[-1])
    rdm1_eye_xcore_ao = make_basis.transform_rdm1_to_ao_mol(rdm1_eye_xcore_lo, C_ao_lo_xcore)
#    GV1_add = np.zeros((nso, nso))
#    vj, vk = scf.hf.dot_eri_dm(eri, rdm1_eye_xcore_ao, hermi=1)
#    vk *= (1.0 - alpha)
#    GV1_add[nao:, nao:] = vk
    GV1_add, GV0_add = spinless.transform_H2_local(eri, ovlp=None,
                                                   compact=False,
                                                   C_ao_lo=C_ao_lo_xcore,
                                                   hyb=(1.0-alpha),
                                                   hyb_j=0.0,
                                                   ao_repr=True)[1:]
    E0 += GV0_add

    mf = solver.scfsolver.mf
    Gveff_core = mf.get_veff(dm=Grdm1_core)
    Ghcore = GH1 + Gveff_core + GV1_add
    GH0_core = np.einsum('pq, qp ->', GH1 + 0.5 * Gveff_core, Grdm1_core)

    GH1_lo = make_basis.transform_h1_to_lo_mol(Ghcore, C_sao_slo_xcore)

    GV2_lo = ao2mo.kernel(GV2, C_sao_slo_xcore)
    Govlp_lo = make_basis.transform_h1_to_lo_mol(ovlp_ghf, C_sao_slo_xcore)

    Grdm1_eye_xcore_ao = np.zeros((nso, nso))
    Grdm1_eye_xcore_ao[nao:, nao:] = rdm1_eye_xcore_ao
    Grdm1_eye_xcore_lo = \
            make_basis.transform_rdm1_to_lo_mol(Grdm1_eye_xcore_ao, C_sao_slo_xcore,
                                                ovlp_ghf)
    vj, vk = scf.hf.dot_eri_dm(GV2_lo, Grdm1_eye_xcore_lo, hermi=1)
    vk *= (1.0 - alpha)
    GV1_add_re = vk
    GV1_add_ref = make_basis.transform_h1_to_lo_mol(GV1_add, C_sao_slo_xcore)
    diff_GV1_add = max_abs(GV1_add_re - GV1_add_ref)

    print (GV1_add_re)

    print ("diff GV1_add: ", diff_GV1_add)
    assert diff_GV1_add < 1e-10

    Ham = integral.Integral(GH1_lo.shape[-1], restricted=True, bogoliubov=False,
                            H0=E0+GH0_core, H1=GH1_lo[None], H2=GV2_lo[None],
                            ovlp=Govlp_lo)

    from libdmet.solver import impurity_solver
    #Mu = 0.0
    solver = scf_solver.SCFSolver(ghf=True, restricted=True, tol=1e-7, max_cycle=200,
                                  scf_newton=False, alpha=1.0, beta=beta)
    rdm1_re_gso_froze, E_re_gso_froze = solver.run(Ham,
            nelec=GH1_lo.shape[-1]//2, scf_max_cycle=100,
                                                   dm0=Grdm1_xcore_lo,
                                                   Mu=Mu, fit_mu=False,
                                                   nelec_target=(mol.nelectron - ncore * 2))

    rdm1_re_gso_froze = make_basis.transform_rdm1_to_ao_mol(rdm1_re_gso_froze, Q)

    E_diff = abs(E_re_gso_froze - E_rhf)
    print ("GHF cas frozen core")
    print ("energy diff: ", E_diff)
    #assert E_diff < 1e-10

    rdm1_A, rdm1_B, rdm1_D = spinless.extract_rdm1(rdm1_re_gso_froze)
    lattice.mulliken_lo_R0(mol, (rdm1_A, rdm1_B), labels=lo_labels_xcore)

    diff_E_rhf_ghf = abs(E_re_gso_froze - E_re)
    diff_rdm1_rhf_ghf = max_abs(rdm1_A - rdm1_re)

    print ("E diff to rhf: ", diff_E_rhf_ghf)
    assert diff_E_rhf_ghf < 1e-10
    print ("rdm1 diff to rhf: ", diff_rdm1_rhf_ghf)
    assert diff_rdm1_rhf_ghf < 1e-10

    # GCCSD
    solver = impurity_solver.CCSD(ghf=True, restricted=True, tol=1e-5,
            tol_normt=2e-5, max_cycle=200, scf_newton=False, diis_space=10,
            level_shift=0.05, approx_l=True, alpha=1.0)

    rdm1_re_gso_froze, E_re_gso_froze = solver.run(Ham, nelec=GH1_lo.shape[-1]//2, scf_max_cycle=200,
                                                   dm0=Grdm1_xcore_lo, Mu=Mu)
    rdm1_re_gso_froze = make_basis.transform_rdm1_to_ao_mol(rdm1_re_gso_froze, Q)

    print ("GCCSD")
    rdm1_A, rdm1_B, rdm1_D = spinless.extract_rdm1(rdm1_re_gso_froze)
    lattice.mulliken_lo_R0(mol, (rdm1_A, rdm1_B), labels=lo_labels_xcore)

if __name__ == "__main__":
    test_scaled_rhf_ghf()
    test_scaled_hf()
