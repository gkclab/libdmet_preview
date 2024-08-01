#! /usr/bin/env python

import numpy as np
import scipy.linalg as la
import pytest

def test_ftsystem():
    import numpy as np
    import scipy.linalg as la
    from libdmet.routine import ftsystem as ft
    from libdmet.utils.misc import tril_mat2arr, tril_arr2mat 
    np.set_printoptions(5, linewidth =1000)
    np.random.seed(1)

    norb = 10
    nelec = 4
    beta = 10.0

    #deg_orbs = []
    #deg_energy = []
    deg_orbs = [[0,3], [1,2], [4,5,6], [7]]
    deg_energy = [1.0 , 0.1, 0.8, 3.0]
    #h = get_h_random(norb)
    h = ft.get_h_random_deg(norb, deg_orbs = deg_orbs, deg_energy = deg_energy)
    print ("h: \n%s" %h)

    fix_mu = False
    mo_energy, mo_coeff, mo_occ, mu = ft.kernel(h, nelec, beta)
    grad_ana = ft.get_rho_grad(mo_energy, mo_coeff, mu, beta, fix_mu=fix_mu, compact=False)
    print ("mo_energy: \n%s" % mo_energy)
    print ("mo_occ: \n%s" %mo_occ)
    print ("mu: %s" % mu)
    
    # test dw_dv
    drho = ft.make_rdm1(mo_coeff, mo_occ)
    f0 = (drho * drho).sum()
    mo_energy = mo_energy[None]
    mo_coeff = mo_coeff[None]
    drho = drho[None]

    dw_dv = ft.get_dw_dv(mo_energy, mo_coeff, drho, mu, beta, fix_mu=fix_mu, compact=True)    
    h_arr_ref = tril_mat2arr(h)
    grad = np.zeros_like(h_arr_ref)
    dx = 1e-6
    for i in range(len(h_arr_ref)):
        h_arr = h_arr_ref.copy()
        h_arr[i] += dx
        h_mat = tril_arr2mat(h_arr)
        mo_energy, mo_coeff, mo_occ, _ = ft.kernel(h_mat, nelec, beta, mu0=mu, fix_mu=fix_mu)
        rho = ft.make_rdm1(mo_coeff, mo_occ)
        f = (rho*rho).sum()
        grad[i] = (f - f0) / dx
    
    print ("diff of gradients: %s" %(la.norm(grad - dw_dv)))
    assert (la.norm(grad - dw_dv) < 1e-4)
    
    # test rho_grad
    rho0 = drho
    grad = np.zeros_like(grad_ana)
    dx = 1e-6
    for i in range(len(h_arr_ref)):
        h_arr = h_arr_ref.copy()
        h_arr[i] += dx
        h_mat = tril_arr2mat(h_arr)
        mo_energy, mo_coeff, mo_occ, _ = ft.kernel(h_mat, nelec, beta, mu0=mu, fix_mu=fix_mu)
        rho = ft.make_rdm1(mo_coeff, mo_occ)
        grad[i] = (rho - rho0) / dx
    
    print ("diff of gradients: %s" %(la.norm(grad - grad_ana)))
    assert (la.norm(grad - grad_ana) < 1e-4)

@pytest.mark.parametrize(
    "method",
    ['fermi', 'gaussian'],
)
@pytest.mark.parametrize(
    "mu",
    [-1.0, np.array(1.0), np.array([1.0]), [1.0, 2.0], np.arange(2)],
)
def test_smearing_occ(method, mu):
    import numpy as np
    from libdmet.routine.ftsystem import fermi_smearing_occ, gaussian_smearing_occ
    
    if method == 'fermi':
        f_occ = fermi_smearing_occ
    else:
        f_occ = gaussian_smearing_occ
    
    beta = 50.0
    
    ew = np.arange(120).reshape(2, 3, 5, 4) 
    occ = f_occ(mu, ew, beta)
    
    ew = np.arange(8).reshape(2, 4).astype(np.double)
    occ = f_occ(mu, ew, beta)
    
    if np.array(mu).size == 1: # restricted case
        # first case
        ew = np.arange(6)
        occ = f_occ(mu, ew, beta)

if __name__ == "__main__":
    test_smearing_occ(method='fermi', mu=0.0)
    test_ftsystem()
