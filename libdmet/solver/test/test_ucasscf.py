#!/usr/bin/env python

def test_uicasscf():
    import numpy as np
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf.mcscf import addons
    from libdmet.solver.umc1step import UCASSCF
    from libdmet.utils import max_abs, mdot

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g'}
    mol.charge = 1
    mol.spin = 1
    mol.verbose = 4
    mol.build()

    m = scf.UHF(mol)
    ehf = m.scf()
    mc = UCASSCF(m, 4, (2,1))
    #mo = m.mo_coeff
    #mo = addons.sort_mo(mc, m.mo_coeff, [(3,4,5,6),(3,4,6,7)], 1)
    emc = mc.kernel()[0]
    print ("ehf, emc, emc - ehf")
    print(ehf, emc, emc-ehf)
    print ("diff to ref")
    print(emc - -2.9782774463926618)
    assert abs(emc - -2.9782774463926618) < 5e-7

    C = m.mo_coeff
    h1e = [mdot(C[0].T, m.get_hcore(), C[0]), mdot(C[1].T, m.get_hcore(), C[1])]
    h1e = np.asarray(h1e)
    ovlp = mdot(C[0].T, m.get_ovlp(), C[0])
    g2e = [ao2mo.kernel(m._eri, C[0]), ao2mo.kernel(m._eri, C[1]), 
           ao2mo.general(m._eri, (C[0], C[0], C[1], C[1]))]
    g2e = np.asarray(g2e)
    rdm1 = [np.diag(m.mo_occ[0]), np.diag(m.mo_occ[1])]

    from libdmet.solver.scf import SCF
    
    myscf = SCF(newton_ah=False)

    # UHF
    myscf.set_system(mol.nelectron, mol.spin, False, False)
    myscf.set_integral(mol.nao_nr(), m.energy_nuc(), h1e, g2e)
    E, rhoHF = myscf.HF(MaxIter=100, tol=1e-8, InitGuess=rdm1)
    
    print (E)

    mc = UCASSCF(myscf.mf, 4, (2,1))
    #mo = m.mo_coeff
    #mo = addons.sort_mo(mc, m.mo_coeff, [(3,4,5,6),(3,4,6,7)], 1)
    emc = mc.kernel()[0]
    print ("diff to ref")
    print(emc - -2.9782774463926618)
    assert abs(emc - -2.9782774463926618) < 5e-7


if __name__ == "__main__":
    test_uicasscf()
