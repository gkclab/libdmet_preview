#!/usr/bin/env python

"""
SCF class with chemical potential mu fitting.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

from functools import partial
import numpy as np
from scipy import linalg as la

from pyscf import lib
from pyscf.lib import logger
from pyscf import scf
from pyscf.scf import chkfile

from libdmet.routine.spinless_helper import (mono_fit, mono_fit_2, transform_local,
                                             separate_basis)
from libdmet.solver import scf as scf_hp
from libdmet.utils import logger as log
from libdmet.utils import mdot

COUNT_UNIT_CELL = False
    
def trans_mu(mu, basis_Ra, basis_Rb):
    nao = basis_Ra.shape[-2]
    mu_mat = np.zeros((2, nao, nao))
    np.fill_diagonal(mu_mat[0], -mu)
    np.fill_diagonal(mu_mat[1],  mu)
    return transform_local(basis_Ra, basis_Rb, mu_mat)

trans_mu_1 = trans_mu

def _get_nelec(dm0, basis, count_unit_cell=COUNT_UNIT_CELL):
    nelec = 0.0
    if count_unit_cell:
        basis = [basis[0]]
    for i, C in enumerate(basis):
        dm = mdot(C, dm0, C.conj().T)
        norb = dm.shape[-1] // 2
        nelec += dm[range(norb), range(norb)].sum() - \
                 dm[range(norb, norb*2), range(norb, norb*2)].sum() + \
                 norb
    return nelec

_get_nelec_1 = _get_nelec

def trans_mu_2(mu, basis_Ra, basis_Rb, nso=None):
    nao = nso // 2
    v_mu = np.zeros((nso, nso))
    v_mu[range(nao), range(nao)] = -mu
    v_mu[range(nao, nso), range(nao, nso)] = mu
    return v_mu

def _get_nelec_2(dm0, basis, count_unit_cell=COUNT_UNIT_CELL):
    norb = dm0.shape[-1] // 2
    nelec = dm0[range(norb), range(norb)].sum() - \
            dm0[range(norb, norb*2), range(norb, norb*2)].sum() + \
            norb
    return nelec

def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    """
    The SCF driver with chemical potential fitting.
    """

    if 'init_dm' in kwargs:
        raise RuntimeError('''
You see this error message because of the API updates in pyscf v0.11.
Keyword argument "init_dm" is replaced by "dm0"''')
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess)
    else:
        dm = dm0
    
    # ZHC NOTE first compute v_mu0
    nelec_target = mf.nelec_target
    tol_nelec = mf.tol_nelec
    if tol_nelec is None:
        tol_nelec = conv_tol * 0.1

    if mf.basis is not None:
        basis = mf.basis
        basis_Ra, basis_Rb = separate_basis(basis)
        trans_mu = trans_mu_1
        _get_nelec = _get_nelec_1
    else: # no basis is given
        basis = mf.basis
        basis_Ra = basis_Rb = None
        trans_mu = partial(trans_mu_2, nso=dm.shape[-1])
        _get_nelec = _get_nelec_2

    mu_elec = mf.mu_elec
    v_mu0 = trans_mu(mu_elec, basis_Ra, basis_Rb)
    mf.v_mu0 = v_mu0

    def get_nelec(mu):
        v_mu = trans_mu(mu, basis_Ra, basis_Rb)
        fock_1 = fock + v_mu
        mo_energy, mo_coeff = mf.eig(fock_1, s1e)
        with lib.temporary_env(mf, verbose=0):
            mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        nelec_calc = _get_nelec(dm, basis)
        return nelec_calc

    h1e  = np.array(mf.get_hcore(mol), copy=True)
    h1e -= v_mu0
    
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    s1e = mf.get_ovlp(mol)
    cond = lib.cond(s1e)
    logger.debug(mf, 'cond(S) = %s', cond)
    if np.max(cond)*1e-17 > conv_tol:
        logger.warn(mf, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                    'SCF may be inaccurate and hard to converge.', np.max(cond))

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mu_elec = mono_fit(get_nelec, nelec_target, mu_elec, tol_nelec, increase=True, dx=0.1, 
                           verbose=True)
        v_mu = trans_mu(mu_elec, basis_Ra, basis_Rb)
        mf.v_mu = v_mu
        mf.mu_elec = mu_elec
        fock = fock + v_mu
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        mf.fock = fock
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    cput1 = logger.timer(mf, 'initialize scf', *cput0)
    
    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot
        
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        fock0 = np.array(fock, copy=True)
        mu_elec_old = mu_elec
        mu_elec = mono_fit(get_nelec, nelec_target, mu_elec, tol_nelec, 
                           increase=True, dx=0.1, verbose=False)
        v_mu = trans_mu(mu_elec, basis_Ra, basis_Rb)
        fock = fock + v_mu

        if mf_diis:
            # ZHC NOTE PySCF 2.1 use a new interface of get_err_vec
            try:
                errvec = scf.diis.get_err_vec(s1e, dm, fock)
            except TypeError:
                errvec = scf.diis.get_err_vec(s1e, dm, fock, None)

            fock0 = lib.diis.DIIS.update(mf_diis, fock0, xerr=errvec)
            fock = fock0
            mu_elec = mono_fit(get_nelec, nelec_target, mu_elec, tol_nelec,
                               increase=True, dx=0.1, verbose=False)
            v_mu = trans_mu(mu_elec, basis_Ra, basis_Rb)
            fock = fock + v_mu
        dmu = abs(mu_elec - mu_elec_old)
        mf.v_mu = v_mu
        mf.mu_elec = mu_elec
        mf.fock = fock

        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        # attach mo_coeff and mo_occ to dm to improve DFT get_veff efficiency
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot = mf.energy_tot(dm, h1e, vhf)
        e_mu = np.einsum('pq, qp ->', mf.v_mu, dm)
        logger.debug(mf, 'E (mu) = %.15g', e_mu)

        # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
        # instead of the statement "fock = h1e + vhf" because Fock matrix may
        # be modified in some methods.
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        fock = fock + v_mu
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        norm_gorb = norm_gorb / np.sqrt(norm_gorb.size)
        norm_ddm = np.linalg.norm(dm-dm_last)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g  |dmu| = %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm, dmu)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad and dmu < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())
        
        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break
    
    logger.info(mf, "mu (final) = %s", mf.mu_elec)
    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

class GGHFpMu(scf_hp.GGHF):
    """
    A routine for generalized HF with generalized integrals.
    nelec_target is fixed during the SCF iteration.
    """
    def __init__(self, mol, DiisDim=12, MaxIter=500, alpha=None,
                 mu_elec=None, nelec_target=None, basis=None, tol_nelec=None,
                 damp=0.0):
        scf_hp.GGHF.__init__(self, mol, DiisDim=DiisDim, MaxIter=MaxIter, alpha=alpha)
        self._keys = self._keys.union(["mu_elec", "nelec_target", "basis", 
                                       "tol_nelec", "v_mu0", "v_mu"])
        self.mu_elec = mu_elec
        self.nelec_target = nelec_target
        self.basis = basis
        self.tol_nelec = tol_nelec
        self.damp = damp
    
    def dump_flags(self, verbose=None):
        scf_hp.GGHF.dump_flags(self, verbose)
        logger.info(self, 'nelec (target) = %s', self.nelec_target)
        logger.info(self, 'tol_nelec = %s', self.tol_nelec)
        logger.info(self, 'mu (initial) = %s', self.mu_elec)

    def scf(self, dm0=None, **kwargs):
        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()
        self.build(self.mol)

        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, \
                    self.mo_energy, self.mo_coeff, self.mo_occ = \
                    kernel(self, self.conv_tol, self.conv_tol_grad,
                           dm0=dm0, callback=self.callback,
                           conv_check=self.conv_check, **kwargs)
        else:
            # Avoid to update SCF orbitals in the non-SCF initialization
            # (issue #495).  But run regular SCF for initial guess if SCF was
            # not initialized.
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        logger.timer(self, 'SCF', *cput0)
        self._finalize()
        return self.e_tot
    kernel = lib.alias(scf, alias_name='kernel')
