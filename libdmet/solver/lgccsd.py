#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Linearized GCCSD.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import numpy as np
from scipy.sparse import linalg as spla
from functools import reduce

from pyscf import lib
from pyscf import ao2mo
from pyscf import scf
from pyscf.lib import logger
from pyscf.cc import gccsd
from pyscf import __config__

MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

LINEAR_SOLVER = 'lgmres'
#LINEAR_SOLVER = 'gcrotmk'

einsum = lib.einsum

def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None, hermi=True, method=LINEAR_SOLVER,
           precond='finv'):
    import inspect
    from libdmet.utils import max_abs
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(CCSD) = %.15g', eccsd)
    log.info("LCCSD method = %s", method)
    log.info("LCCSD hermi = %s", hermi)

    def aop(x):
        return mycc.hop(x, eris=eris, hermi=hermi)

    if precond == 'finv':
        def mop(x):
            return mycc.precond_finv(x, eris)
    elif precond == 'diag':
        def mop(x):
            return mycc.precond_diag(x, eris)
    elif callable(precond):
        def mop(x):
            return precond(mycc, x, eris)
    elif precond:
        raise ValueError("Unknown precond %s" % precond)

    b = mycc.make_b(eris)
    x0 = mycc.amplitudes_to_vector(t1, t2)
    A = spla.LinearOperator((b.shape[-1], b.shape[-1]), matvec=aop)
    if precond:
        M = spla.LinearOperator((b.shape[-1], b.shape[-1]), matvec=mop)
    else:
        M = None

    niter = [0]
    x_old = [x0]
    def callback(xk):
        locs = inspect.currentframe().f_back.f_locals
        x = locs['x']
        diff_x = max_abs(x - x_old[0])
        x_old[0] = x.copy()
        if method.lower() == 'gcrotmk':
            res = locs.get('r')
        else:
            res = locs.get('r_outer')
        log.info("LCCSD cycle = %s , res = %15.5g , normt = %15.5g, diff t = %10.5g",
                 niter[0], max_abs(res), max_abs(x), diff_x)
        niter[0] += 1

    args = {}
    if method.lower() == 'lgmres':
        drv = spla.lgmres
        args["inner_m"] = mycc.diis_space
    elif method.lower() == 'gcrotmk':
        drv = spla.gcrotmk
    else:
        raise ValueError("Unknown linear solver %s" % method)

    tvec_new, info = drv(A, b, x0=x0, tol=tolnormt, atol=tolnormt,
                         callback=callback, M=M, maxiter=max_cycle,
                         **args)
    conv = (info == 0)

    t1, t2 = mycc.vector_to_amplitudes(tvec_new)
    A = x0 = b = tvec_new = None
    eccsd = mycc.energy(t1, t2, eris)

    log.info('E_corr(LCCSD) = %.15g, conv = %s', eccsd, conv)
    log.timer('LCCSD', *cput0)
    return conv, eccsd, t1, t2

def update_amps(cc, t1, t2, eris, hermi=True):
    assert isinstance(eris, gccsd._PhysicistsERIs)
    nocc, nvir = t1.shape
    fock = eris.fock

    # intermidiates
    tau = t2
    Foo = fock[:nocc, :nocc]
    Fov = fock[:nocc, nocc:]
    Fvv = fock[nocc:, nocc:]
    Woooo = np.asarray(eris.oooo)
    Wvvvv = np.asarray(eris.vvvv)
    Wovvo = np.asarray(eris.ovvo)

    # T1 equation
    t1new  = einsum('ie,ae->ia', t1, Fvv)
    t1new -= einsum('ma,mi->ia', t1, Foo)
    if not hermi:
        t1new += einsum('imae,me->ia', t2, Fov)
    t1new -= einsum('nf,naif->ia', t1, eris.ovov)
    t1new -= 0.5*einsum('imef,maef->ia', t2, eris.ovvv)
    t1new -= 0.5*einsum('mnae,mnie->ia', t2, eris.ooov)

    # T2 equation
    Ftmp = Fvv
    tmp = einsum('ijae,be->ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0,1,3,2)
    Ftmp = Foo
    tmp = einsum('imab,mj->ijab', t2, Ftmp)
    t2new -= (tmp - tmp.transpose(1,0,2,3))

    t2new += einsum('mnab,mnij->ijab', tau, Woooo * 0.5)
    t2new += 0.5 * einsum('ijef,abef->ijab', tau, Wvvvv)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp

    tmp = einsum('ie,jeba->ijab', t1, np.asarray(eris.ovvv))
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma,ijmb->ijab', t1, np.asarray(eris.ooov))
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    return t1new, t2new

def hop(cc, vec, eris, hermi=True):
    vec = vec.ravel()
    t1, t2 = cc.vector_to_amplitudes(vec)
    t1, t2 = cc.update_amps(t1, t2, eris, hermi=hermi)
    return cc.amplitudes_to_vector(t1, t2)

def precond_finv(cc, vec, eris, tol=1e-8):
    """
    Fock inversion as preconditioner.
    """
    vec = vec.ravel()
    t1, t2 = cc.vector_to_amplitudes(vec)
    nocc, nvir = t1.shape
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    eia = mo_e_o[:, None] - mo_e_v
    eia[eia > -tol] = -tol
    t1 /= eia

    for i in range(nocc):
        t2[i] /= lib.direct_sum('a, jb -> jab', eia[i], eia)
    return cc.amplitudes_to_vector(t1, t2)

def precond_diag(cc, vec, eris):
    """
    Diagonal elements as preconditioner, works not well.
    """
    vec = vec.ravel()
    t1, t2 = cc.vector_to_amplitudes(vec)
    nocc, nvir = t1.shape

    fock = eris.fock

    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift
    eia = mo_e_o[:, None] - mo_e_v
    eia -= np.einsum('iaai -> ia', eris.ovvo)
    t1 /= eia

    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    tmp = 0.5 * np.einsum('abab -> ab', eris.vvvv)
    for i in range(nocc):
        eijab[i, i] -= tmp
    tmp = 0.5 * np.einsum('ijij -> ij', eris.oooo)
    for i in range(nvir):
        eijab[:, :, i, i] -= tmp
    t2 /= eijab
    return cc.amplitudes_to_vector(t1, t2)

def make_b(cc, eris):
    nocc = cc.nocc
    t1 = -eris.fock[:nocc, nocc:]
    t2 = -np.asarray(eris.oovv)
    return cc.amplitudes_to_vector(t1, t2)

def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    e = einsum('ia,ia', fock[:nocc,nocc:], t1)
    eris_oovv = np.asarray(eris.oovv)
    e += 0.25*np.einsum('ijab,ijab', t2, eris_oovv, optimize=True)
    e += 0.5 *np.einsum('ia,jb,ijab', t1, t1, eris_oovv, optimize=True)
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in GCCSD energy %s', e)
    return e.real

class LGCCSD(gccsd.GCCSD):

    hermi = True
    conv_tol = getattr(__config__, 'cc_gccsd_GCCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_gccsd_GCCSD_conv_tol_normt', 1e-6)

    energy = energy

    update_amps = update_amps

    hop = hop

    make_b = make_b

    precond_finv = precond_finv

    precond_diag = precond_diag

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None, hermi=True):
        raise NotImplementedError
        from pyscf.cc import gccsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                gccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                    max_cycle=self.max_cycle,
                                    tol=self.conv_tol_normt,
                                    verbose=self.verbose)
        return self.l1, self.l2

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''Un-relaxed 1-particle density matrix in MO space'''
        if not self.hermi:
            raise NotImplementedError

        from pyscf.cc import gccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        doo = -einsum('ia, ja -> ij', t1, t1) - \
               0.5 * einsum('kiab, kjab -> ij', t2, t2)
        dvv =  einsum('ia, ib -> ab', t1, t1) + \
               0.5 * einsum('ijac, ijbc -> ab', t2, t2)
        dov = np.array(t1)
        dvo = dov.conj().T
        d1 = [doo, dov, dvo, dvv]
        return gccsd_rdm._make_rdm1(self, d1, with_frozen=True, ao_repr=ao_repr)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        raise NotImplementedError
        from pyscf.cc import gccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return gccsd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr)

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False, hermi=None,
             method=LINEAR_SOLVER, precond='finv'):
        '''Ground-state linearized GCCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        if hermi is None:
            hermi = self.hermi
        if mbpt2:
            from pyscf.mp import gmp2
            pt = gmp2.GMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocc, nvir = self.t2.shape[1:3]
            self.t1 = np.zeros((nocc,nvir))
            return self.e_corr, self.t1, self.t2

        # Initialize orbspin so that we can attach it to t1, t2
        if getattr(self.mo_coeff, 'orbspin', None) is None:
            orbspin = scf.ghf.guess_orbspin(self.mo_coeff)
            if not np.any(orbspin == -1):
                self.mo_coeff = lib.tag_array(self.mo_coeff, orbspin=orbspin)

        # main kernel
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        self.e_hf = getattr(eris, 'e_hf', None)
        if self.e_hf is None:
            self.e_hf = self._scf.e_tot

        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose, hermi=hermi, method=method,
                       precond=precond)
        self._finalize()

        if getattr(eris, 'orbspin', None) is not None:
            self.t1 = lib.tag_array(self.t1, orbspin=eris.orbspin)
            self.t2 = lib.tag_array(self.t2, orbspin=eris.orbspin)
        return self.e_corr, self.t1, self.t2

    kernel = ccsd

CCSD = LGCCSD

if __name__ == '__main__':
    from pyscf import gto
    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = gto.Mole()
    mol.verbose = 5
    #mol.atom = [['H', (0.,   0., 0.)],
    #            ['H', (0.7430, 0., 0.)]]
    #mol.basis = 'aug-ccpvtz'
    mol.atom = [['H', (0.,   0., 0.)],
                ['F', (1.1, 0., 0.)]]
    mol.basis = 'ccpvdz'
    mol.cart = True
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    mf = scf.addons.convert_to_ghf(mf)
    #gcc = gccsd.GCCSD(mf)

    gcc = LGCCSD(mf)
    gcc.conv_tol = 1e-5
    gcc.conv_tol_normt = 1e-4
    #method = 'gcrotmk'
    method = 'lgmres'
    ecc, t1, t2 = gcc.kernel(method=method, precond='finv')
    print (gcc.e_tot)

    rdm1 = gcc.make_rdm1()
    print (rdm1)
    print (np.trace(rdm1))
