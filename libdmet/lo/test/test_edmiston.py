#! /usr/bin/env python

"""
Test Edmiston-Ruedenberg localization
"""

def test_ER():
    import numpy as np
    import scipy.linalg as la
    from pyscf import gto, scf, ao2mo
    from libdmet.lo.edmiston import ER, ER_model
    from libdmet.routine.localizer import Localizer as ER_jac
    from libdmet.solver import scf as scf_hp
    from libdmet.utils.misc import mdot
    
    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = gto.Mole()
    mol.atom = '''
         He   0.    0.     0.2
         H    0.   -0.5   -0.4
         H    0.    0.5   -0.4
      '''
    mol.basis = '631g'
    mol.build(verbose=4)
    mf = scf.RHF(mol).run()

    eri = mf._eri
    eri_s1 = ao2mo.restore(1, eri, mol.nao_nr())
    jk_func = scf_hp._get_jk
    
    nmo = 5
    eri_mo = ao2mo.general(eri_s1, (mf.mo_coeff[:,:nmo], mf.mo_coeff[:,:nmo], \
            mf.mo_coeff[:,:nmo], mf.mo_coeff[:,:nmo]))
    
    localizer2 = ER_jac(eri_mo)
    localizer2.optimize(thr=1e-12)
    f_jac = localizer2.getL()
    C_mo_lo = localizer2.coefs.T
    
    localizer = ER(mol=None, eri=eri, jk_func=jk_func, \
            mo_coeff=mf.mo_coeff[:, :nmo])
    localizer.conv_tol = 1e-12
    #localizer.init_guess = C_mo_lo
    C_ao_lo = localizer.kernel()
    f_ciah = localizer.cost_function(np.eye(nmo))
    assert abs(f_ciah - f_jac) < 1e-10
    
    localizer = ER(mol, mo_coeff=mf.mo_coeff[:, :nmo])
    localizer.conv_tol = 1e-12
    localizer.init_guess = 'scdm'
    C_ao_lo = localizer.kernel()
    f_ciah2 = localizer.cost_function(np.eye(nmo))
    assert abs(f_ciah2 - f_jac) < 1e-10
    
    C_ao_lo, _ = ER_model(mf.mo_coeff[:, :nmo], eri, jk_func=jk_func, \
            num_rand=5, noise=1.0, guess=None, conv_tol=1e-12)
    C_mo_lo = mdot(mf.mo_coeff[:, :nmo].conj().T, mf.get_ovlp(), C_ao_lo)
    localizer = ER(mol=None, eri=eri, jk_func=jk_func, mo_coeff=mf.mo_coeff[:, :nmo])
    f_ciah3 = localizer.cost_function(C_mo_lo)
    assert abs(f_ciah3 - f_jac) < 1e-10

    nmo = 1
    eri_mo = ao2mo.general(eri_s1, (mf.mo_coeff[:,:nmo], mf.mo_coeff[:,:nmo], \
            mf.mo_coeff[:,:nmo], mf.mo_coeff[:,:nmo]))
    
    localizer2 = ER_jac(eri_mo)
    localizer2.optimize(thr=1e-12)
    f_jac = localizer2.getL()
    C_mo_lo = localizer2.coefs.T
    
    localizer = ER(mol=None, eri=eri, jk_func=jk_func, \
            mo_coeff=mf.mo_coeff[:, :nmo])
    localizer.conv_tol = 1e-12
    #localizer.init_guess = C_mo_lo
    C_ao_lo = localizer.kernel()
    f_ciah = localizer.cost_function(np.eye(nmo))
    assert abs(f_ciah - f_jac) < 1e-10

    localizer = ER(mol, mo_coeff=mf.mo_coeff[:, :nmo])
    localizer.conv_tol = 1e-12
    localizer.init_guess = 'scdm'
    C_ao_lo = localizer.kernel()
    f_ciah2 = localizer.cost_function(np.eye(nmo))
    assert abs(f_ciah2 - f_jac) < 1e-10
    
    C_ao_lo, _ = ER_model(mf.mo_coeff[:, :nmo], eri, jk_func=jk_func, \
            num_rand=5, noise=1.0, guess=None, conv_tol=1e-12)
    C_mo_lo = mdot(mf.mo_coeff[:, :nmo].conj().T, mf.get_ovlp(), C_ao_lo)
    localizer = ER(mol=None, eri=eri, jk_func=jk_func, mo_coeff=mf.mo_coeff[:, :nmo])
    f_ciah3 = localizer.cost_function(C_mo_lo)
    assert abs(f_ciah3 - f_jac) < 1e-10

def test_ER_2():
    import numpy as np
    import scipy.linalg as la
    from libdmet.utils import logger as log
    from libdmet.routine.localizer import Localizer
    log.verbose = 'DEBUG1'
    np.random.seed(9)
    norbs = 10
    s = np.random.rand(norbs,norbs,norbs,norbs)
    s = s + np.swapaxes(s, 0, 1)
    s = s + np.swapaxes(s, 2, 3)
    s = s + np.transpose(s, (2, 3, 0, 1))
    loc = Localizer(s)
    loc.optimize()
    R = loc.coefs
    err = loc.Int2e - np.einsum("pi,qj,rk,sl,ijkl->pqrs", R, R, R, R, s)
    log.check(np.allclose(err, 0), "Inconsistent coefficients and integrals,"
              " difference is %.2e", la.norm(err))
    assert np.allclose(err, 0) 

if __name__ == "__main__":
    test_ER_2()
    test_ER()
