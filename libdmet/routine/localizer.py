#! /usr/bin/env python

"""
Edmiston-Ruedenberg localization through Jacobi rotations
following the algorithm by
Raffenetti et al. Theor Chim Acta 86, 149 (1992)

Pipek-Mezey localization of Hubbard model.
modified from pyscf.

SCDM for particle / hole character separation

Author:
    Bo-Xiao Zheng
    Zhi-Hao Cui
"""

import numpy as np

from pyscf import lo
from pyscf import gto

from libdmet.lo import scdm
from libdmet.utils import logger as log
from libdmet.lo.edmiston import Localizer

def localize_bath(B, method, **kwargs):
    """
    Localization on bath orbitals.
    Assume B is in site basis (model).
    """
    if method == "pm":
        B_local = localize_bath_pm(B, **kwargs)
    elif method == "scdm":
        B_local = localize_bath_scdm(B, **kwargs)
    else:
        raise ValueError
    return B_local

class HubbardPM(lo.pipek.PM):
    def __init__(self, *args, **kwargs):
        lo.pipek.PM.__init__(self, *args, **kwargs)
        self.init_guess = 'rand'
    def atomic_pops(self, mol, mo_coeff, method=None):
        return np.einsum('pi,pj->pij', mo_coeff, mo_coeff)
    def get_init_guess(self, key='atomic'):
        '''Generate initial guess for localization.

        Kwargs:
            key : str or bool
                If key is 'atomic', initial guess is based on the projected
                atomic orbitals. False
        '''
        nmo = self.mo_coeff.shape[1]
        if isinstance(key, str) and key.lower() == 'atomic':
            u0 = atomic_init_guess(self.mol, self.mo_coeff)
        else:
            u0 = np.eye(nmo)
        if (isinstance(key, str) and key.lower().startswith('rand')
            or np.linalg.norm(self.get_grad(u0)) < 1e-5):
            # Add noise to kick initial guess out of saddle point
            dr = np.cos(np.arange((nmo-1)*nmo//2)) * np.random.rand()
            u0 = self.extract_rotation(dr)
        return u0

def localize_bath_pm(B, **kwargs):
    B_shape = B.shape
    bath_orbs = B.reshape((-1, B_shape[-1])) # full
    log.info("PM localization of bath orbitals")
    loc_orb_collect = []

    for i in range(5):
        mol = gto.M()
        mol.verbose = 3
        loc_orb_input = bath_orbs
        loc_obj = HubbardPM(mol, loc_orb_input)
        cost_before = loc_obj.cost_function()
        if i == 0:
            log.debug(0, 'cost function of PM localization of bath orb (before): %12.5f', cost_before)
        log.debug(0, "%d th trial: ", i)

        loc_orb = loc_obj.kernel()
        loc_obj = HubbardPM(mol, loc_orb)
        loc_orb = loc_obj.kernel()
        loc_obj = HubbardPM(mol, loc_orb)
        loc_orb = loc_obj.kernel()
        loc_obj = HubbardPM(mol, loc_orb)
        loc_orb = loc_obj.kernel()

        cost_after = loc_obj.cost_function()
        loc_orb_collect.append((cost_after, loc_orb))
        log.debug(0, 'cost function of PM localization of bath orb (after): %12.5f', cost_after)

    loc_orb_collect.sort(key = lambda tup: tup[0])
    bath_orbs = loc_orb_collect[-1][1]
    return bath_orbs.reshape(B_shape)

def localize_bath_scdm(B, cholesky=False, **kwargs):
    """
    Get maximum particle/hole character by mixing the orbitals.
    """
    log.info("SCDM localization of bath orbitals.")
    B_shape = B.shape
    bath_orbs = scdm.scdm_model(B.reshape((-1, B_shape[-1])), cholesky=cholesky)
    return bath_orbs.reshape(B_shape)

def visualize_bath(lattice, LatSize, GRho, localize_bath=False, spin=0, \
        bath_index=None, figname='bath.png'):
    import matplotlib.pyplot as plt
    from libdmet.routine.bcs import embBasis
    nscsites = lattice.nscsites
    lat_coords = np.array(lattice.sites).T
    # make impurity at the center of lattice
    lat_coords[0][lat_coords[0] > LatSize[0] // 2] -= LatSize[0]
    lat_coords[1][lat_coords[1] > LatSize[1] // 2] -= LatSize[1]

    B = embBasis(lattice, GRho, localize_bath=localize_bath, return_bath=True)
    z = np.zeros((nscsites, nscsites*2))
    if spin == 0:
        bath_orbs = B[0, :, :nscsites].reshape(-1, 2*nscsites) # alpha
    else:
        bath_orbs = B[1, :, nscsites:].reshape(-1, 2*nscsites) # beta

    #bath_orbs = np.vstack((z, bath_orbs))
    if bath_index is None:
        ave_orb = np.abs(bath_orbs).sum(axis=1)
    else:
        ave_orb = np.abs(bath_orbs)[:, bath_index]
    ave_orb *= 500.0

    plt.xlim(-LatSize[0] * 0.5 + 1.0, LatSize[0] * 0.5 + 1.0)
    plt.ylim(-LatSize[1] * 0.5 + 1.0, LatSize[1] * 0.5 + 1.0)
    plt.xticks(np.arange(-LatSize[0] * 0.5 , LatSize[0] * 0.5 + 1.0, 1.0))
    plt.yticks(np.arange(-LatSize[1] * 0.5 , LatSize[1] * 0.5 + 1.0, 1.0))
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.grid()
    bath_plot = plt.scatter(lat_coords[0], lat_coords[1], s=ave_orb)
    plt.savefig(figname)

