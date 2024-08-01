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
GCCSD using imagninary time evolution.
"""

from functools import partial
import numpy as np
from scipy import linalg as la
from scipy import optimize as opt
from scipy.sparse import linalg as spla

from pyscf import scf
from pyscf import lib
from pyscf.cc import gccsd
from pyscf.cc import gintermediates as imd
from pyscf.cc import ccsd_lambda
from pyscf.cc.gccsd_lambda import make_intermediates
from pyscf.lib import logger

from libdmet.solver.lgccsd import precond_finv, precond_diag
from libdmet.utils import max_abs
from libdmet.utils import logger as log

einsum = lib.einsum

def update_amps_ite(mycc, t1, t2, eris):
    assert isinstance(eris, gccsd._PhysicistsERIs)
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc, nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    # ZHC NOTE ITE typically does not need level shift
    if mycc.ignore_level_shift:
        mo_e_v = eris.mo_energy[nocc:]
    else:
        mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift

    tau = imd.make_tau(t2, t1, t1)

    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)

    # Move energy terms to the other side
    Fvv[np.diag_indices(nvir)] -= mo_e_v
    Foo[np.diag_indices(nocc)] -= mo_e_o

    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new -=  einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new -=  einsum('nf,naif->ia', t1, eris.ovov)
    t1new += (-0.5) * einsum('imef,maef->ia', t2, eris.ovvv)
    t1new += (-0.5) * einsum('mnae,mnie->ia', t2, eris.ooov)
    t1new += fov.conj()

    # T2 equation
    Ftmp = Fvv - 0.5 * einsum('mb, me -> be', t1, Fov)
    tmp = einsum('ijae, be -> ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0,1,3,2)
    Ftmp = Foo + 0.5 * einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab, mj -> ijab', t2, Ftmp)
    t2new -= tmp - tmp.transpose(1,0,2,3)
    t2new += np.asarray(eris.oovv).conj()

    Woooo = imd.cc_Woooo(t1, t2, eris)
    t2new += einsum('mnab, mnij -> ijab', tau, Woooo * 0.5)
    Woooo = None

    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
    t2new += 0.5 * einsum('ijef,abef->ijab', tau, Wvvvv)
    Wvvvv = None
    tau = None

    Wovvo = imd.cc_Wovvo(t1, t2, eris)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    Wovvo = None

    tmp += einsum('ie, ma, mbje -> ijab', t1, t1, eris.ovov)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp
    tmp = einsum('ie, jeba -> ijab', t1, np.asarray(eris.ovvv).conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma, ijmb -> ijab', t1, np.asarray(eris.ooov).conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    # ZHC NOTE imaginary time evolution
    dt = mycc.dt
    eia = mo_e_o[:,None] - mo_e_v
    t1new *= (-dt)
    t1new += t1 * (1.0 + dt * eia)

    eia *= dt
    t2new *= (-dt)
    for i in range(nocc):
        ejab = lib.direct_sum('a, jb -> jab', eia[i] + 1.0, eia)
        t2new[i] += t2[i] * ejab

    return t1new, t2new

def kernel_lambda_ite(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
                      max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if eris is None: eris = mycc.ao2mo()
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda_ite)

def update_lambda_ite(mycc, t1, t2, l1, l2, eris, imds):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    if mycc.ignore_level_shift:
        mo_e_v = eris.mo_energy[nocc:]
    else:
        mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    v1 = imds.v1 - np.diag(mo_e_v)
    v2 = imds.v2 - np.diag(mo_e_o)

    l1new = np.array(fov, copy=True)

    mba = einsum('klca, klcb -> ba', l2, t2) * .5
    mij = einsum('kicd, kjcd -> ij', l2, t2) * .5

    oovv = np.asarray(eris.oovv)
    tau  = np.einsum('ia, jb -> ijab', t1, t1 * 2.0, optimize=True)
    tau += t2
    tmp = einsum('ijcd, klcd -> ijkl', l2, tau)
    tau = None
    m3 = einsum('klab,ijkl->ijab', l2, np.asarray(imds.woooo))
    m3 += einsum('klab,ijkl->ijab', oovv, tmp) * .25
    tmp = None

    tmp = einsum('ijcd,kd -> ijck', l2, t1)
    m3 -= einsum('kcba,ijck->ijab', eris.ovvv, tmp)
    tmp = None
    m3 += einsum('ijcd,cdab->ijab', l2, eris.vvvv) * .5

    l2new = np.array(oovv, copy=True)
    l2new += m3
    l1new += einsum('ijab,jb->ia', m3, t1)
    m3 = None

    fov1 = fov + einsum('kjcb,kc->jb', oovv, t1)
    tmp  = np.einsum('ia, jb -> ijab', l1, fov1)
    tmp += einsum('kica,jcbk->ijab', l2, np.asarray(imds.wovvo))
    tmp = tmp - tmp.transpose(1,0,2,3)
    l2new += tmp - tmp.transpose(0,1,3,2)
    tmp = None

    tmp  = einsum('ka, ijkb -> ijab', l1, eris.ooov)
    tmp += einsum('ijca, cb -> ijab', l2, v1)
    tmp1vv = mba + einsum('ka, kb -> ba', l1, t1)
    tmp += einsum('ca, ijcb -> ijab', tmp1vv, oovv)
    l2new -= tmp - tmp.transpose(0,1,3,2)
    tmp = None

    tmp  = einsum('ic, jcba -> jiba', l1, eris.ovvv)
    tmp += einsum('kiab, jk -> ijab', l2, v2)
    tmp1oo = mij + einsum('ic,kc->ik', l1, t1)
    tmp -= einsum('ik,kjab->ijab', tmp1oo, oovv)
    l2new += tmp - tmp.transpose(1,0,2,3)
    tmp = None

    l1new += einsum('jb, ibaj -> ia', l1, eris.ovvo)
    l1new += einsum('ib, ba -> ia', l1, v1)
    l1new -= einsum('ja, ij -> ia', l1, v2)
    l1new -= einsum('kjca, icjk -> ia', l2, imds.wovoo)
    l1new -= einsum('ikbc, bcak -> ia', l2, imds.wvvvo)
    l1new += einsum('jiba, bj -> ia', l2, imds.w3)
    tmp =(t1 + einsum('kc,kjcb->jb', l1, t2)
          - einsum('bd,jd->jb', tmp1vv, t1)
          - einsum('lj,lb->jb', mij, t1))
    l1new += np.einsum('jiba, jb -> ia', oovv, tmp, optimize=True)
    l1new += np.einsum('icab, bc -> ia', eris.ovvv, tmp1vv, optimize=True)
    l1new -= np.einsum('jika, kj -> ia', eris.ooov, tmp1oo, optimize=True)
    tmp = fov - einsum('kjba, jb -> ka', oovv, t1)
    oovv = None
    l1new -= np.einsum('ik, ka -> ia', mij, tmp, optimize=True)
    l1new -= np.einsum('ca, ic -> ia', mba, tmp, optimize=True)

    #eia = lib.direct_sum('i-j->ij', mo_e_o, mo_e_v)
    #l1new /= eia
    #for i in range(nocc):
    #    l2new[i] /= lib.direct_sum('a, jb -> jab', eia[i], eia)

    # ZHC NOTE imaginary time evolution
    dt = mycc.dt
    eia = lib.direct_sum('i-j->ij', mo_e_o, mo_e_v)
    l1new *= (-dt)
    l1new += l1 * (1.0 + dt * eia)

    eijab = lib.direct_sum('ia, jb -> ijab', eia, eia)
    l2new *= (-dt)
    eijab *= dt
    eijab += 1.0
    l2new += l2 * eijab

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new

def init_amps_ghf(mycc, eris=None):
    """
    GCCSD initial amplitudes from level-shift MP2.
    """
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)

    nocc = mycc.nocc
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    eia = mo_e_o[:, None] - mo_e_v
    t1 = eris.fock[:nocc, nocc:] / eia
    eris_oovv = np.asarray(eris.oovv)
    t2 = np.array(eris_oovv, copy=True).conj()
    emp2 = 0.0
    for i in range(nocc):
        t2[i] /= lib.direct_sum('a, jb -> jab', eia[i], eia)
        emp2 += 0.25 * einsum('jab, jab', t2[i], eris_oovv[i])
    mycc.emp2 = emp2
    log.info('Init t2, MP2 energy = %.15g', mycc.emp2)
    return mycc.emp2, t1, t2

class GCCSDITE(gccsd.GCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, dt=0.01,
                 ignore_level_shift=True):
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.dt = dt
        self.ignore_level_shift = ignore_level_shift
        self._keys = self._keys.union(["dt", "ignore_level_shift"])

    update_amps = update_amps_ite

    init_amps = init_amps_ghf

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                kernel_lambda_ite(self, eris, t1, t2, l1, l2,
                                  max_cycle=self.max_cycle,
                                  tol=self.conv_tol_normt,
                                  verbose=self.verbose)
        return self.l1, self.l2

"""
Runge-Kutta version of ITE.
"""

def update_amps_ite_rk(mycc, t1, t2, eris):
    assert isinstance(eris, gccsd._PhysicistsERIs)
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc, nocc:]
    tau = imd.make_tau(t2, t1, t1)

    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)

    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new -=  einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new -=  einsum('nf,naif->ia', t1, eris.ovov)
    t1new += (-0.5) * einsum('imef,maef->ia', t2, eris.ovvv)
    t1new += (-0.5) * einsum('mnae,mnie->ia', t2, eris.ooov)
    t1new += fov.conj()

    # T2 equation
    Ftmp = Fvv - 0.5 * einsum('mb, me -> be', t1, Fov)
    tmp = einsum('ijae, be -> ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0,1,3,2)
    Ftmp = Foo + 0.5 * einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab, mj -> ijab', t2, Ftmp)
    t2new -= tmp - tmp.transpose(1,0,2,3)
    t2new += np.asarray(eris.oovv).conj()

    Woooo = imd.cc_Woooo(t1, t2, eris)
    t2new += einsum('mnab, mnij -> ijab', tau, Woooo * 0.5)
    Woooo = None

    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
    t2new += 0.5 * einsum('ijef,abef->ijab', tau, Wvvvv)
    Wvvvv = None
    tau = None

    Wovvo = imd.cc_Wovvo(t1, t2, eris)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    Wovvo = None

    tmp += einsum('ie, ma, mbje -> ijab', t1, t1, eris.ovov)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp
    tmp = einsum('ie, jeba -> ijab', t1, np.asarray(eris.ovvv).conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma, ijmb -> ijab', t1, np.asarray(eris.ooov).conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    if getattr(mycc, "frozen_abab", False):
        mycc.remove_t2_abab(t2new)

    return t1new, t2new

def rk4(mycc, t1, t2, eris, h, order=4, fupdate=None):
    if fupdate is None:
        fupdate = mycc.update_amps_ite_rk
    dt11, dt21 = fupdate(t1, t2, eris)
    if order == 1:
        return dt11, dt21
    else:
        t1_, t2_ = t1 + dt11 * (h*0.5), t2 + dt21 * (h*0.5)
        dt1new = dt11
        dt2new = dt21

        dt12, dt22 = fupdate(t1_, t2_, eris)
        t1_, t2_ = t1 + dt12 * (h*0.5), t2 + dt22 * (h*0.5)
        dt1new += (2.0 * dt12)
        dt2new += (2.0 * dt22)
        dt12 = dt22 = None

        dt13, dt23 = fupdate(t1_, t2_, eris)
        t1_, t2_ = t1 + dt13 * h, t2 + dt23 * h
        dt1new += (2.0 * dt13)
        dt2new += (2.0 * dt23)
        dt13 = dt23 = None

        dt14, dt24 = fupdate(t1_, t2_, eris)
        t1_ = t2_ = None
        dt1new += dt14
        dt2new += dt24
        dt14 = dt24 = None

        dt1new *= (1.0 / 6.0)
        dt2new *= (1.0 / 6.0)
        return dt1new, dt2new

def kernel_rk(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
              tolnormt=1e-6, verbose=None):
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

    conv = False
    for istep in range(max_cycle):
        dt = mycc.dt
        t1new, t2new = mycc.rk4(t1, t2, eris, dt, order=4)
        t1new *= (-dt)
        t2new *= (-dt)
        t1new += t1
        t2new += t2

        tmpvec = mycc.amplitudes_to_vector(t1new, t2new)
        tmpvec -= mycc.amplitudes_to_vector(t1, t2)
        normt = np.linalg.norm(tmpvec)
        tmpvec = None
        t1, t2 = t1new, t2new
        t1new = t2new = None
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info('cycle = %d  E_corr(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2

def update_lambda_ite_rk(mycc, l1, l2, eris, t1, t2, imds):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    v1 = imds.v1
    v2 = imds.v2

    l1new = np.array(fov, copy=True)

    mba = einsum('klca, klcb -> ba', l2, t2) * .5
    mij = einsum('kicd, kjcd -> ij', l2, t2) * .5

    oovv = np.asarray(eris.oovv)
    tau  = np.einsum('ia, jb -> ijab', t1, t1 * 2.0, optimize=True)
    tau += t2
    tmp = einsum('ijcd, klcd -> ijkl', l2, tau)
    tau = None
    m3 = einsum('klab,ijkl->ijab', l2, np.asarray(imds.woooo))
    m3 += einsum('klab,ijkl->ijab', oovv, tmp) * .25
    tmp = None

    tmp = einsum('ijcd,kd -> ijck', l2, t1)
    m3 -= einsum('kcba,ijck->ijab', eris.ovvv, tmp)
    tmp = None
    m3 += einsum('ijcd,cdab->ijab', l2, eris.vvvv) * .5

    l2new = np.array(oovv, copy=True)
    l2new += m3
    l1new += einsum('ijab,jb->ia', m3, t1)
    m3 = None

    fov1 = fov + einsum('kjcb,kc->jb', oovv, t1)
    tmp  = np.einsum('ia, jb -> ijab', l1, fov1)
    tmp += einsum('kica,jcbk->ijab', l2, np.asarray(imds.wovvo))
    tmp = tmp - tmp.transpose(1,0,2,3)
    l2new += tmp - tmp.transpose(0,1,3,2)
    tmp = None

    tmp  = einsum('ka, ijkb -> ijab', l1, eris.ooov)
    tmp += einsum('ijca, cb -> ijab', l2, v1)
    tmp1vv = mba + einsum('ka, kb -> ba', l1, t1)
    tmp += einsum('ca, ijcb -> ijab', tmp1vv, oovv)
    l2new -= tmp - tmp.transpose(0,1,3,2)
    tmp = None

    tmp  = einsum('ic, jcba -> jiba', l1, eris.ovvv)
    tmp += einsum('kiab, jk -> ijab', l2, v2)
    tmp1oo = mij + einsum('ic,kc->ik', l1, t1)
    tmp -= einsum('ik,kjab->ijab', tmp1oo, oovv)
    l2new += tmp - tmp.transpose(1,0,2,3)
    tmp = None

    l1new += einsum('jb, ibaj -> ia', l1, eris.ovvo)
    l1new += einsum('ib, ba -> ia', l1, v1)
    l1new -= einsum('ja, ij -> ia', l1, v2)
    l1new -= einsum('kjca, icjk -> ia', l2, imds.wovoo)
    l1new -= einsum('ikbc, bcak -> ia', l2, imds.wvvvo)
    l1new += einsum('jiba, bj -> ia', l2, imds.w3)
    tmp =(t1 + einsum('kc,kjcb->jb', l1, t2)
          - einsum('bd,jd->jb', tmp1vv, t1)
          - einsum('lj,lb->jb', mij, t1))
    l1new += np.einsum('jiba, jb -> ia', oovv, tmp, optimize=True)
    l1new += np.einsum('icab, bc -> ia', eris.ovvv, tmp1vv, optimize=True)
    l1new -= np.einsum('jika, kj -> ia', eris.ooov, tmp1oo, optimize=True)
    tmp = fov - einsum('kjba, jb -> ka', oovv, t1)
    oovv = None
    l1new -= np.einsum('ik, ka -> ia', mij, tmp, optimize=True)
    l1new -= np.einsum('ca, ic -> ia', mba, tmp, optimize=True)

    if getattr(mycc, "frozen_abab", False):
        mycc.remove_t2_abab(l2new)

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new

def rk4_lambda(mycc, t1, t2, l1, l2, eris, imds, h, order=4):
    fupdate = partial(mycc.update_lambda_ite_rk, t1=t1, t2=t2, imds=imds)
    return mycc.rk4(l1, l2, eris, h, order=order, fupdate=fupdate)

def __kernel_lambda_rk(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
                       max_cycle=50, tol=1e-8, verbose=logger.INFO,
                       fintermediates=None, fupdate=None):
    if eris is None: eris = mycc.ao2mo()
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mycc, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = t1
    if l2 is None: l2 = t2
    if fintermediates is None:
        fintermediates = make_intermediates
    if fupdate is None:
        fupdate = update_lambda

    imds = fintermediates(mycc, t1, t2, eris)
    cput0 = log.timer('CCSD lambda initialization', *cput0)

    conv = False

    rk4_lambda = partial(mycc.rk4, t1=t1, t2=t2, imds=imds)

    for istep in range(max_cycle):
        dt = mycc.dt
        l1new, l2new = mycc.rk4_lambda(t1, t2, l1, l2, eris, imds, dt, order=4)
        l1new *= (-dt)
        l2new *= (-dt)
        l1new += l1
        l2new += l2

        normt = np.linalg.norm(mycc.amplitudes_to_vector(l1new, l2new) -
                               mycc.amplitudes_to_vector(l1, l2))
        l1, l2 = l1new, l2new
        l1new = l2new = None
        log.info('cycle = %d  norm(lambda1,lambda2) = %.6g', istep+1, normt)
        cput0 = log.timer('CCSD iter', *cput0)
        if normt < tol:
            conv = True
            break
    return conv, l1, l2

def kernel_lambda_rk(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
                     max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if eris is None: eris = mycc.ao2mo()
    return __kernel_lambda_rk(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda_ite_rk)

class GCCSDITE_RK(gccsd.GCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, dt=0.1,
                 ignore_level_shift=True):
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.dt = dt
        self.ignore_level_shift = ignore_level_shift
        self._keys = self._keys.union(["dt", "ignore_level_shift"])

    def ccsd_(self, t1=None, t2=None, eris=None):
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
                kernel_rk(self, eris, t1, t2, max_cycle=self.max_cycle,
                          tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                          verbose=self.verbose)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state unrestricted (U)CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
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

        e_corr, self.t1, self.t2 = self.ccsd_(t1, t2, eris)
        if getattr(eris, 'orbspin', None) is not None:
            self.t1 = lib.tag_array(self.t1, orbspin=eris.orbspin)
            self.t2 = lib.tag_array(self.t2, orbspin=eris.orbspin)
        return e_corr, self.t1, self.t2

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                kernel_lambda_rk(self, eris, t1, t2, l1, l2,
                                 max_cycle=self.max_cycle,
                                 tol=self.conv_tol_normt,
                                 verbose=self.verbose)
        return self.l1, self.l2

    update_amps_ite_rk = update_amps_ite_rk
    rk4 = rk4

    update_lambda_ite_rk = update_lambda_ite_rk
    rk4_lambda = rk4_lambda

    init_amps = init_amps_ghf

"""
Direct minimize CC residual.
"""

# ZHC NOTE define max_abs to reduce cost and allow termination of 1st iteration
def safe_max_abs(x):
    if np.isfinite(x).all():
        return max(np.max(x), abs(np.min(x)))
    else:
        return 1e+12

def remove_t2_abab(mycc, t2):
    nocc_a, nvir_a = mycc.nocc_a, mycc.nvir_a
    assert nocc_a is not None
    assert nvir_a is not None
    t2[:nocc_a, nocc_a:, :nvir_a, nvir_a:] = 0.0
    t2[:nocc_a, nocc_a:, nvir_a:, :nvir_a] = 0.0
    t2[nocc_a:, :nocc_a, :nvir_a, nvir_a:] = 0.0
    t2[nocc_a:, :nocc_a, nvir_a:, :nvir_a] = 0.0

def kernel_minres(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
                  tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]
    if getattr(mycc, "frozen_abab", False):
        mycc.remove_t2_abab(t2)

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(CCSD) = %.15g', eccsd)

    cycle = [0]
    x0 = mycc.amplitudes_to_vector(t1, t2)

    def f_res(x):
        t1, t2 = mycc.vector_to_amplitudes(x)
        eccsd = mycc.energy(t1, t2, eris)
        t1, t2 = mycc.update_amps_ite_rk(t1, t2, eris)
        res = mycc.amplitudes_to_vector(t1, t2)
        norm = max_abs(res)
        log.info("      cycle = %5d , E = %15.8g , norm(res) = %15.5g", cycle[0],
                 eccsd, norm)
        cycle[0] += 1
        return res

    if mycc.precond == 'finv':
        def mop(x):
            return mycc.precond_finv(x, eris)
        M = spla.LinearOperator((x0.shape[-1], x0.shape[-1]), matvec=mop)
    elif mycc.precond == 'diag':
        def mop(x):
            return mycc.precond_diag(x, eris)
        M = spla.LinearOperator((x0.shape[-1], x0.shape[-1]), matvec=mop)
    else:
        M = None

    if mycc.method == 'krylov':
        inner_m = mycc.inner_m
        outer_k = mycc.outer_k
        res = opt.root(f_res, x0, method='krylov',
                       options={'fatol': tolnormt, 'tol_norm': safe_max_abs,
                                'disp': True, 'maxiter': max_cycle // inner_m,
                                'line_search': 'wolfe',
                                'jac_options': {'rdiff': 1e-6, 'inner_maxiter': 100,
                                                'inner_inner_m': inner_m, 'inner_tol': tolnormt * 0.5,
                                                'outer_k': outer_k, 'inner_M': M}
                               })
    elif mycc.method == 'df-sane':
        res = opt.root(f_res, x0, method='df-sane',
                       options={'fatol': tolnormt, 'disp': True, 'maxfev': max_cycle,
                                'fnorm': max_abs})
    else:
        raise ValueError

    conv = res.success
    t1, t2 = mycc.vector_to_amplitudes(res.x)
    eccsd = mycc.energy(t1, t2, eris)

    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2

def __kernel_lambda_minres(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
                           max_cycle=50, tol=1e-8, verbose=logger.INFO,
                           fintermediates=None, fupdate=None):
    if eris is None: eris = mycc.ao2mo()
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mycc, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = t1
    if l2 is None: l2 = t2
    if fintermediates is None:
        fintermediates = make_intermediates
    if fupdate is None:
        fupdate = update_lambda
    if getattr(mycc, "frozen_abab", False):
        mycc.remove_t2_abab(t2)
        mycc.remove_t2_abab(l2)

    imds = fintermediates(mycc, t1, t2, eris)
    cput0 = log.timer('CCSD lambda initialization', *cput0)

    cycle = [0]
    def f_res(x):
        l1, l2 = mycc.vector_to_amplitudes(x)
        l1, l2 = fupdate(mycc, l1, l2, eris, t1, t2, imds)
        res = mycc.amplitudes_to_vector(l1, l2)
        norm = max_abs(res)
        log.info("      cycle = %5d , norm(res) = %15.5g", cycle[0], norm)
        cycle[0] += 1
        return res

    x0 = mycc.amplitudes_to_vector(l1, l2)
    tolnormt = mycc.conv_tol_normt

    if mycc.precond == 'finv':
        def mop(x):
            return mycc.precond_finv(x, eris)
        M = spla.LinearOperator((x0.shape[-1], x0.shape[-1]), matvec=mop)
    elif mycc.precond == 'diag':
        def mop(x):
            return mycc.precond_diag(x, eris)
        M = spla.LinearOperator((x0.shape[-1], x0.shape[-1]), matvec=mop)
    else:
        M = None

    if mycc.method == 'krylov':
        inner_m = mycc.inner_m
        outer_k = mycc.outer_k
        res = opt.root(f_res, x0, method='krylov',
                       options={'fatol': tolnormt, 'tol_norm': safe_max_abs,
                                'disp': True, 'maxiter': max_cycle // inner_m,
                                'line_search': 'wolfe',
                                'jac_options': {'rdiff': 1e-6, 'inner_maxiter': 100,
                                                'inner_inner_m': inner_m, 'inner_tol': tolnormt * 0.5,
                                                'outer_k': outer_k, 'inner_M': M}
                               })
    elif mycc.method == 'df-sane':
        res = opt.root(f_res, x0, method='df-sane',
                       options={'fatol': tolnormt, 'disp': True, 'maxfev': max_cycle,
                                'fnorm': max_abs})
    else:
        raise ValueError

    conv = res.success
    l1, l2 = mycc.vector_to_amplitudes(res.x)
    return conv, l1, l2

def kernel_lambda_minres(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
                         max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if eris is None: eris = mycc.ao2mo()
    return __kernel_lambda_minres(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                                  verbose, make_intermediates, update_lambda_ite_rk)

class GCCSD_KRYLOV(gccsd.GCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 method='krylov', precond='finv', inner_m=10, outer_k=6,
                 frozen_abab=False, nocc_a=None, nvir_a=None):
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.method = method
        self.precond = precond
        self.inner_m = inner_m
        self.outer_k = outer_k
        self.frozen_abab = frozen_abab
        self.nocc_a = nocc_a
        self.nvir_a = nvir_a
        self._keys = self._keys.union(["method", "precond", "inner_m", "outer_k",
                                       "frozen_abab", "nocc_a", "nvir_a"])

    def dump_flags(self, verbose=None):
        gccsd.GCCSD.dump_flags(self, verbose=verbose)
        logger.info(self, "method  = %s", self.method)
        logger.info(self, "precond = %s", self.precond)
        logger.info(self, "inner_m = %d", self.inner_m)
        logger.info(self, "outer_k = %d", self.outer_k)
        logger.info(self, "frozen_abab  = %s", self.frozen_abab)
        logger.info(self, "nocc_a  = %s", self.nocc_a)
        logger.info(self, "nvir_a  = %s", self.nvir_a)
        return self

    def ccsd_(self, t1=None, t2=None, eris=None):
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
                kernel_minres(self, eris, t1, t2, max_cycle=self.max_cycle,
                              tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                              verbose=self.verbose)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state unrestricted (U)CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
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

        e_corr, self.t1, self.t2 = self.ccsd_(t1, t2, eris)
        if getattr(eris, 'orbspin', None) is not None:
            self.t1 = lib.tag_array(self.t1, orbspin=eris.orbspin)
            self.t2 = lib.tag_array(self.t2, orbspin=eris.orbspin)
        return e_corr, self.t1, self.t2

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                kernel_lambda_minres(self, eris, t1, t2, l1, l2,
                                     max_cycle=self.max_cycle,
                                     tol=self.conv_tol_normt,
                                     verbose=self.verbose)
        return self.l1, self.l2

    update_amps_ite_rk = update_amps_ite_rk
    update_lambda_ite_rk = update_lambda_ite_rk

    precond_finv = precond_finv
    precond_diag = precond_diag

    remove_t2_abab = remove_t2_abab

    init_amps = init_amps_ghf

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.build(verbose=4)
    mf = scf.UHF(mol).run()
    mf = scf.addons.convert_to_ghf(mf)

    # Freeze 1s electrons
    frozen = [0,1,2,3]
    mycc = gccsd.GCCSD(mf, frozen=frozen)
    mycc.kernel()
    rdm1_ref = mycc.make_rdm1(ao_repr=True)

    #gcc = GCCSDITE_RK(mf, frozen=frozen)
    gcc = GCCSD_KRYLOV(mf, frozen=frozen)
    gcc.max_cycle = 500
    gcc.dt = 0.1
    gcc.conv_tol = 1e-6
    gcc.conv_tol_normt = 1e-5

    #ecc, t1, t2 = gcc.kernel()
    gcc.method = 'krylov'
    gcc.precond = 'diag'
    ecc, t1, t2 = gcc.kernel()
    print(ecc - -0.3486987472235819)
    gcc.solve_lambda()
    rdm1 = gcc.make_rdm1(ao_repr=True)

    print (np.linalg.norm(rdm1 - rdm1_ref))

    rdm1 = gcc.make_rdm1()

    print (rdm1)

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.build(verbose=4)
    mf = scf.UHF(mol).run()
    mf = scf.addons.convert_to_ghf(mf)

    mycc = GCCSDITE(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.2133432712431435)

