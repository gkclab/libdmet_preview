#! /usr/bin/env python

#def run_dmrg_uhf():
#    import numpy as np
#    import pyscf
#    from pyscf import ao2mo
#    from pyscf import fci
#    from libdmet.solver.scf import restore_Ham
#    from libdmet.system.integral import Integral
#    from libdmet.solver import impurity_solver
#    from libdmet.basis_transform import make_basis
#    from libdmet.utils.misc import mdot
#    from libdmet.utils import logger as log
#
#    log.verbose = "DEBUG1"
#
#    np.set_printoptions(3, linewidth=1000, suppress=True)
#    mol = pyscf.M(
#        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
#        basis = '321g',
#        spin = 2,
#        verbose = 5)
#
#    myhf = mol.UHF()
#    myhf.conv_tol = 1e-12
#    myhf.kernel()
#
#    nao = mol.nao_nr()
#    C_ao_lo = myhf.mo_coeff
#    hcore = make_basis.transform_h1_to_mo_mol(myhf.get_hcore(), C_ao_lo)
#    ovlp = make_basis.transform_h1_to_mo_mol(myhf.get_ovlp(), C_ao_lo)
#    rdm1 = make_basis.transform_rdm1_to_mo_mol(myhf.make_rdm1(), C_ao_lo, myhf.get_ovlp())
#
#    eri_aa = ao2mo.general(myhf._eri, (C_ao_lo[0], C_ao_lo[0], \
#            C_ao_lo[0], C_ao_lo[0]))
#    eri_bb = ao2mo.general(myhf._eri, (C_ao_lo[1], C_ao_lo[1], \
#            C_ao_lo[1], C_ao_lo[1]))
#    eri_ab = ao2mo.general(myhf._eri, (C_ao_lo[0], C_ao_lo[0], \
#            C_ao_lo[1], C_ao_lo[1]))
#    eri = np.asarray((eri_aa, eri_ab, eri_bb))
#
#    cisolver = fci.direct_uhf.FCI()
#    cisolver.max_cycle = 100
#    cisolver.conv_tol = 1e-8
#    nelec_a, nelec_b = (mol.nelectron + mol.spin) // 2, (mol.nelectron - mol.spin) // 2
#    e, fcivec = cisolver.kernel(hcore, eri, nao, (nelec_a, nelec_b), ecore=myhf.energy_nuc())
#    (dm1a, dm1b), (dm2aa,dm2ab,dm2bb) = cisolver.make_rdm12s(fcivec, nao, (nelec_a,nelec_b))
#
#    Ham = Integral(nao, False, False, myhf.energy_nuc(), {"cd": hcore}, {"ccdd": eri[[0, 2, 1]]}, ovlp=ovlp)
#    ncas = nao
#    nelecas = mol.nelectron
#    block = impurity_solver.StackBlock(nproc=1, nthread=28, nnode=1, \
#            bcs=False, tol=1e-9, maxM=800, SharedDir="./shdir", \
#            maxiter_initial=36, maxiter_restart=14, mem=120)
#
#    solver = impurity_solver.CASCI(ncas=ncas, nelecas=nelecas, Sz=mol.spin, \
#                splitloc=True, MP2natorb=True, cisolver=block, \
#                mom_reorder=False, tmpDir="./tmp", loc_method='ciah')
#
#    #Ham_cas, (cas, core, rdm1_core) = solver.run(Ham, ci_args={}, guess=rdm1, nelec=nelecas, ham_only=True)
#    Ham_cas = Ham
#    rdm1_dmrg, E_dmrg = block.run(Ham_cas, nelec=nelecas, spin=mol.spin)
#    rdm2_dmrg = block.twopdm()
#    Ham_cas = restore_Ham(Ham_cas, 1)
#
#    E_re = np.einsum('spq, sqp', Ham_cas.H1["cd"], rdm1_dmrg) + \
#            0.5 * (np.einsum('pqrs, pqrs', Ham_cas.H2["ccdd"][0], rdm2_dmrg[0]) + \
#                   np.einsum('pqrs, pqrs', Ham_cas.H2["ccdd"][1], rdm2_dmrg[1])) + \
#           np.einsum('pqrs, pqrs', Ham_cas.H2["ccdd"][2], rdm2_dmrg[2]) + \
#           Ham_cas.H0
#
#    print (E_re)
#    print (abs(E_re - e))

def test_block_rdm():
    import os
    import numpy as np
    import pyscf
    from libdmet.utils import max_abs
    from libdmet.solver.block import read1pdm, read2pdm, \
            read1pdm_bin, read2pdm_bin
    from libdmet.utils import logger as log

    log.verbose = "DEBUG1"
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = '321g',
        spin = 2,
        verbose = 5)
    norb = mol.nao_nr()
    rdm1 = read1pdm_bin(dir_path + "onepdm.0.0.bin", norb)
    rdm2 = read2pdm_bin(dir_path + "twopdm.0.0.bin", norb)
    rdm1 = read1pdm_bin(dir_path + "onepdm.0.0.bin", norb, raw_data=True)
    rdm2 = read2pdm_bin(dir_path + "twopdm.0.0.bin", norb, raw_data=True)

    rdm1_ref = read1pdm(dir_path + "onepdm.0.0")
    rdm2_ref = read2pdm(dir_path + "twopdm.0.0")
    print (max_abs(rdm1 - rdm1_ref))
    print (max_abs(rdm2 - rdm2_ref))
    assert max_abs(rdm1 - rdm1_ref) < 1e-13
    assert max_abs(rdm1 - rdm1_ref) < 1e-13

def test_schedule():
    from libdmet.solver import block
    from libdmet.utils import logger as log
    log.verbose = "DEBUG2"

    tol = 1e-6
    minM = 250
    maxM = 1200
    my_schedule = block.Schedule(maxiter=35, sweeptol=tol)
    my_schedule.gen_initial(minM=minM, maxM=maxM)
    my_schedule.maxiter = 15
    my_schedule.gen_restart(maxM)

    my_schedule = block.Schedule(maxiter=35, sweeptol=tol, sweep_per_M=6)
    my_schedule.gen_initial(minM=minM, maxM=maxM)
    my_schedule.maxiter = 15
    my_schedule.gen_restart(maxM)

if __name__ == "__main__":
    test_schedule()
    #run_dmrg_uhf()
    test_block_rdm()
