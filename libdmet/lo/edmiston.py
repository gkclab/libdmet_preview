#!/usr/bin/env python

"""
Edmiston-Ruedenberg localization

Jacobi rotations following the algorithm by
Raffenetti et al. Theor Chim Acta 86, 149 (1992)

Author:
    Bo-Xiao Zheng
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import time
import itertools as it
import numpy as np
import scipy.linalg as la
from math import cos, sin, atan, pi

from pyscf import gto, lib
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.lo import edmiston
from pyscf.soscf import ciah
from pyscf.tools import mo_mapping

from libdmet.lo import scdm
from libdmet.utils import logger as log
from libdmet.utils.misc import mdot, max_abs

def kernel(localizer, mo_coeff=None, callback=None, verbose=None):
    if mo_coeff is not None:
        localizer.mo_coeff = np.asarray(mo_coeff, order='C')
    if localizer.mo_coeff.shape[1] <= 1:
        return localizer.mo_coeff

    if localizer.verbose >= logger.WARN:
        localizer.check_sanity()
    localizer.dump_flags()

    cput0 = (time.process_time(), time.perf_counter())
    log = logger.new_logger(localizer, verbose=verbose)

    if localizer.conv_tol_grad is None:
        conv_tol_grad = np.sqrt(localizer.conv_tol*.1)
        log.info('Set conv_tol_grad to %g', conv_tol_grad)
    else:
        conv_tol_grad = localizer.conv_tol_grad

    if mo_coeff is None:
        if getattr(localizer, 'mol', None) and localizer.mol.natm == 0:
            # For customized Hamiltonian
            u0 = localizer.get_init_guess('random')
        else:
            u0 = localizer.get_init_guess(localizer.init_guess)
    else:
        u0 = localizer.get_init_guess(None)

    rotaiter = ciah.rotate_orb_cc(localizer, u0, conv_tol_grad, verbose=log)
    u, g_orb, stat = next(rotaiter)
    cput1 = log.timer('initializing CIAH', *cput0)

    tot_kf = stat.tot_kf
    tot_hop = stat.tot_hop
    conv = False
    e_last = 0
    for imacro in range(localizer.max_cycle):
        norm_gorb = np.linalg.norm(g_orb)
        u0 = lib.dot(u0, u)
        e = localizer.cost_function(u0)
        e_last, de = e, e-e_last

        log.info('macro= %d  f(x)= %.14g  delta_f= %g  |g|= %g  %d KF %d Hx',
                 imacro+1, e, de, norm_gorb, stat.tot_kf+1, stat.tot_hop)
        cput1 = log.timer('cycle= %d'%(imacro+1), *cput1)

        if (norm_gorb < conv_tol_grad and abs(de) < localizer.conv_tol):
            conv = True

        if callable(callback):
            callback(locals())

        if conv:
            break

        u, g_orb, stat = rotaiter.send(u0)
        tot_kf += stat.tot_kf
        tot_hop += stat.tot_hop

    rotaiter.close()
    log.info('macro X = %d  f(x)= %.14g  |g|= %g  %d intor %d KF %d Hx',
             imacro+1, e, norm_gorb,
             (imacro+1)*2, tot_kf+imacro+1, tot_hop)
# Sort the localized orbitals, to make each localized orbitals as close as
# possible to the corresponding input orbitals
    sorted_idx = mo_mapping.mo_1to1map(u0)
    localizer.mo_coeff = lib.dot(localizer.mo_coeff, u0[:,sorted_idx])
    return localizer.mo_coeff

class EdmistonRuedenberg(edmiston.EdmistonRuedenberg):
    """
    ER localization with CIAH newton minimizer.
    """
    def __init__(self, mol=None, mo_coeff=None, eri=None, jk_func=None,
                 ovlp=None):
        """
        Support customized eri and jk_func.
        jk_func should take two arguments dm and eri.
        """
        try:
            from pyscf.soscf.ciah import CIAHOptimizerMixin
        except ImportError or AttributeError:
            from pyscf.soscf.ciah import CIAHOptimizer as CIAHOptimizerMixin
        CIAHOptimizerMixin.__init__(self)

        self.mo_coeff = np.array(mo_coeff, copy=True)
        nao, nmo = self.mo_coeff.shape
        if eri is None: # real mol
            self.mol = mol
            self.ovlp = mol.intor_symmetric('int1e_ovlp')
            self.jk_func = hf.get_jk
        else: # customized Ham
            assert eri.ndim == 4 or eri.ndim == 2 or eri.ndim == 1
            assert mol is None
            assert jk_func is not None

            mol = gto.Mole()
            mol.incore_anyway = True
            if log.Level[log.verbose] >= log.Level["RESULT"]:
                mol.build(verbose=4, dump_input=False)
            else:
                mol.build(verbose=2, dump_input=False)
            self.mol = None

            if ovlp is None:
                ovlp = np.eye(nao)
            self.ovlp = ovlp
            self.jk_func = jk_func

        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.eri = eri

        keys = set(('conv_tol', 'conv_tol_grad', 'max_cycle', 'max_iters',
                    'max_stepsize', 'ah_trust_region', 'ah_start_tol',
                    'ah_max_cycle', 'init_guess'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def get_jk(self, u):
        mo_coeff = np.dot(self.mo_coeff, u)
        nmo = mo_coeff.shape[-1]
        dms = [np.outer(mo_coeff[:, i], mo_coeff[:, i].conj()) for i in range(nmo)]
        if self.eri is None:
            vj, vk = self.jk_func(self.mol, dms, hermi=1)
        else:
            vj, vk = self.jk_func(dms, self.eri)
        vj = np.asarray([mdot(mo_coeff.T, v, mo_coeff) for v in vj])
        vk = np.asarray([mdot(mo_coeff.T, v, mo_coeff) for v in vk])
        return vj, vk

    def get_init_guess(self, key='atomic', noise=0.0):
        '''Generate initial guess for localization.

        Kwargs:
            key : str or bool or np.ndarray
                If key is
                'atomic': initial guess is based on the projected
                          atomic orbitals.
                'scdm': initial guess is based on the projected
                          atomic orbitals.
                None: use np.eye(nmo) as guess
                ndarray: customized C_mo_lo.
            noise: the amplitude of noise.
        '''
        # ZHC FIXME a better initial guess?
        nmo = self.mo_coeff.shape[1]
        if isinstance(key, np.ndarray): # customized guess
            log.info("Using customized orbitals as ER initial guess.")
            u0 = np.asarray(key)
        elif self.eri is not None: # model Hamiltonian
            log.info("Using SCDM orbitals (model) as ER initial guess.")
            u0 = scdm.scdm_model(self.mo_coeff, return_C_mo_lo=True)[1][0]
        elif isinstance(key, str) and key.lower() == 'atomic':
            log.info("Using projected AOs as ER initial guess.")
            u0 = boys.atomic_init_guess(self.mol, self.mo_coeff)
        elif isinstance(key, str) and key.lower() == 'scdm':
            log.info("Using SCDM orbitals (mol) as ER initial guess.")
            u0 = scdm.scdm_mol(self.mol, self.mo_coeff,
                               return_C_mo_lo=True)[1][0]
        else:
            log.warn("Using identity as ER initial guess.")
            u0 = np.eye(nmo)
        if noise > 1e-10:
            # Add noise to kick initial guess out of saddle point
            dr = (np.random.random((nmo-1) * nmo // 2) - 0.5) * noise
            u_noise = u0.dot(self.extract_rotation(dr))
            u0 = u0.dot(u_noise)
        if nmo > 2:
            log.info("Init cost func: %.13f, |g|: %.5e", self.cost_function(u0),
                     la.norm(self.get_grad(u0)))
        else:
            log.info("Init cost func: %.13f, |g|: 0.0", self.cost_function(u0))

        return u0

    kernel = kernel

ER = Edmiston = EdmistonRuedenberg

def ER_model(mo_coeff, eri, jk_func=None, num_rand=5, noise=1.0, guess=None,
             conv_tol=1e-10):
    """
    ER localization wrapper for model Hamiltonian.
    """
    assert mo_coeff.ndim == 2
    verbose = log.verbose
    if jk_func is None:
        from libdmet.solver import scf as scf_hp
        jk_func = scf_hp._get_jk

    localizer = ER(mo_coeff=mo_coeff, eri=eri, jk_func=jk_func)
    if guess is None:
        localizer.init_guess = 'scdm'
        log.verbose = 'RESULT'
        localizer.verbose = 3
    else:
        localizer.init_guess = guess
        C_ao_lo_old = np.dot(mo_coeff, guess)
        localizer.verbose = 3
    localizer.conv_tol = conv_tol
    C_ao_lo = localizer.kernel()
    C_mo_lo = np.dot(mo_coeff.conj().T, C_ao_lo)
    f_max = localizer.cost_function(np.eye(mo_coeff.shape[-1]))
    log.result("Default guess cost function: %s", f_max)

    if guess is None:
        for i in range(num_rand):
            localizer = ER(mo_coeff=mo_coeff, eri=eri, jk_func=jk_func)
            localizer.init_guess = localizer.get_init_guess(noise=noise)
            localizer.conv_tol = conv_tol
            localizer.verbose = 3
            C_ao_lo_tmp = localizer.kernel()
            C_mo_lo_tmp = np.dot(mo_coeff.conj().T, C_ao_lo_tmp)
            f_tmp = localizer.cost_function(np.eye(mo_coeff.shape[-1]))
            if f_tmp > f_max:
                f_max = f_tmp
                C_ao_lo = C_ao_lo_tmp
                C_mo_lo = C_mo_lo_tmp
            log.result("Random guess %s cost function: %s", i, f_max)
    else:
            log.result("orbital change: %.3e", max_abs(C_ao_lo - C_ao_lo_old))
    log.verbose = verbose
    return C_ao_lo, C_mo_lo

class Localizer(object):
    def __init__(self, Int2e, C_mo_lo=None, copy=True):
        """
        ER localization using Jacobi rotation.

        Args:
            Int2e[i, j, k, l] = (ij | kl), with 1-fold or 4-fold symmetry, in LO basis.
            C_mo_lo: initial guess.
            copy: whether to copy Int2e.
        """
        # ZHC TODO support ERI with symmetry
        self.Int2e = np.array(Int2e, copy=copy)
        if self.Int2e.ndim == 4:
            self.norbs = self.Int2e.shape[-1]
            self.eri_format = 's1'
        elif self.Int2e.ndim == 2:
            self.norbs = int(np.sqrt(Int2e.shape[-1] * 2))
            self.eri_format = 's4'
        else:
            self.norbs = int(np.sqrt(int(np.sqrt(Int2e.shape[-1] * 2)) * 2))
            self.eri_format = 's8'
        if C_mo_lo is None:
            self.coefs = np.eye(self.norbs) # C_lo_mo
        else:
            self.coefs = np.array(C_mo_lo).T

    def transformInt(self, i, j, theta):
        r"""
        Transform 2e integrals wrt Jacobi rotation
        J_ii = J_jj = cos\theta, J_ij = sin\theta, J_ji = -sin\theta
        restrict to i < j
        The scaling of this transformation is O(n^3)
        """
        # this works even for general cases where Int2e does not have symmetry
        delta = np.asarray([[cos(theta)-1, sin(theta)],[-sin(theta), cos(theta)-1]])
        # four index part O(1)
        g4 = self.Int2e[np.ix_([i, j],[i, j],[i, j],[i, j])]
        g4 = np.einsum("pi, qj, rk, sl, ijkl -> pqrs",
                       delta, delta, delta, delta, g4, optimize=True)
        # three index part O(n)
        g3_1 = self.Int2e[np.ix_(range(self.norbs), [i, j], [i, j], [i, j])]
        g3_1 = np.einsum("qj, rk, sl, pjkl -> pqrs", delta, delta, delta, g3_1, optimize=True)
        g3_2 = self.Int2e[np.ix_([i, j], range(self.norbs), [i, j], [i, j])]
        g3_2 = np.einsum("pi, rk, sl, iqkl -> pqrs", delta, delta, delta, g3_2, optimize=True)
        g3_3 = self.Int2e[np.ix_([i, j], [i, j], range(self.norbs), [i, j])]
        g3_3 = np.einsum("pi, qj, sl, ijrl -> pqrs", delta, delta, delta, g3_3, optimize=True)
        g3_4 = self.Int2e[np.ix_([i, j], [i, j], [i, j], range(self.norbs))]
        g3_4 = np.einsum("pi, qj, rk, ijks -> pqrs", delta, delta, delta, g3_4, optimize=True)
        # two index part O(n^2)
        g2_12 = self.Int2e[np.ix_(range(self.norbs), range(self.norbs), [i, j], [i, j])]
        g2_12 = np.einsum("rk, sl, pqkl -> pqrs", delta, delta, g2_12, optimize=True)
        g2_13 = self.Int2e[np.ix_(range(self.norbs), [i, j], range(self.norbs), [i, j])]
        g2_13 = np.einsum("qj, sl, pjrl -> pqrs", delta, delta, g2_13, optimize=True)
        g2_14 = self.Int2e[np.ix_(range(self.norbs), [i, j], [i, j], range(self.norbs))]
        g2_14 = np.einsum("qj, rk, pjks -> pqrs", delta, delta, g2_14, optimize=True)
        g2_23 = self.Int2e[np.ix_([i, j], range(self.norbs), range(self.norbs), [i, j])]
        g2_23 = np.einsum("pi, sl, iqrl -> pqrs", delta, delta, g2_23, optimize=True)
        g2_24 = self.Int2e[np.ix_([i, j], range(self.norbs), [i, j], range(self.norbs))]
        g2_24 = np.einsum("pi, rk, iqks -> pqrs", delta, delta, g2_24, optimize=True)
        g2_34 = self.Int2e[np.ix_([i, j], [i, j], range(self.norbs), range(self.norbs))]
        g2_34 = np.einsum("pi, qj, ijrs -> pqrs", delta, delta, g2_34, optimize=True)
        # one index part O(n^3)
        g1_1 = self.Int2e[[i, j], :, :, :]
        g1_1 = np.einsum("pi, iqrs -> pqrs", delta, g1_1, optimize=True)
        g1_2 = self.Int2e[:, [i, j], :, :]
        g1_2 = np.einsum("qj, pjrs -> pqrs", delta, g1_2, optimize=True)
        g1_3 = self.Int2e[:, :, [i, j], :]
        g1_3 = np.einsum("rk, pqks -> pqrs", delta, g1_3, optimize=True)
        g1_4 = self.Int2e[:, :, :, [i, j]]
        g1_4 = np.einsum("sl, pqrl -> pqrs", delta, g1_4, optimize=True)
        # sum over all rotations
        self.Int2e[np.ix_([i, j],[i, j],[i, j],[i, j])] += g4
        self.Int2e[np.ix_(range(self.norbs), [i, j], [i, j], [i, j])] += g3_1
        self.Int2e[np.ix_([i, j], range(self.norbs), [i, j], [i, j])] += g3_2
        self.Int2e[np.ix_([i, j], [i, j], range(self.norbs), [i, j])] += g3_3
        self.Int2e[np.ix_([i, j], [i, j], [i, j], range(self.norbs))] += g3_4
        self.Int2e[np.ix_(range(self.norbs), range(self.norbs), [i, j], [i, j])] += g2_12
        self.Int2e[np.ix_(range(self.norbs), [i, j], range(self.norbs), [i, j])] += g2_13
        self.Int2e[np.ix_(range(self.norbs), [i, j], [i, j], range(self.norbs))] += g2_14
        self.Int2e[np.ix_([i, j], range(self.norbs), range(self.norbs), [i, j])] += g2_23
        self.Int2e[np.ix_([i, j], range(self.norbs), [i, j], range(self.norbs))] += g2_24
        self.Int2e[np.ix_([i, j], [i, j], range(self.norbs), range(self.norbs))] += g2_34
        self.Int2e[[i, j], :, :, :] += g1_1
        self.Int2e[:, [i, j], :, :] += g1_2
        self.Int2e[:, :, [i, j], :] += g1_3
        self.Int2e[:, :, :, [i, j]] += g1_4

    def transformCoef(self, i, j, theta):
        U = np.eye(self.norbs)
        U[i, i] = U[j, j] = cos(theta)
        U[i, j] = sin(theta)
        U[j, i] = -sin(theta)
        self.coefs = np.dot(U, self.coefs)

    def predictor(self, i, j):
        r"""
        For the rotation between orbitals i, j
        we restrict theta in -pi/4, +pi/4
        compute (i'i'||i'i')+(j'j'||j'j')-(ii||ii)-(jj||jj)
        i'=i*cos\theta+j*sin\theta
        j'=j*cos\theta-i*sin\theta
        (i'i'||i'i')+(j'j'||j'j') = [(ii||ii)+(jj||jj)][3/4+1/4*cos4\theta]
        + [(ii||ij)+...-(ij||jj)-...]*1/4*sin4\theta
        + [(ii||jj)+...][1/4-1/4*cos4\theta]
        """
        A = self.Int2e[i, i, i, i] + self.Int2e[j, j, j, j]
        B = self.Int2e[i, i, i, j] + self.Int2e[i, i, j, i] + self.Int2e[i, j, i, i] + self.Int2e[j, i, i, i] \
                - self.Int2e[i, j, j, j] - self.Int2e[j, i, j, j] - self.Int2e[j, j, i, j] - self.Int2e[j, j, j, i]
        C = self.Int2e[i, i, j, j] + self.Int2e[i, j, i, j] + self.Int2e[i, j, j, i] + self.Int2e[j, i, i, j] \
                + self.Int2e[j, i, j, i] + self.Int2e[j, j, i, i]

        def get_dL(theta):
            return 0.25 * ((cos(4 * theta) - 1) * (A - C) + sin(4 * theta) * B)

        def get_theta():
            # solve dL/dtheta = 0, take theta that corresponds to maximum
            if abs(A - C) > 1e-8:
                alpha = atan(B / (A - C))
            else:
                alpha = pi * 0.5
            if alpha > 0:
                theta = [alpha * 0.25, (alpha - pi) * 0.25]
            else:
                theta = [alpha * 0.25, (alpha + pi) * 0.25]
            vals = [get_dL(x) for x in theta]
            if vals[0] > vals[1]:
                return theta[0], vals[0]
            else:
                return theta[1], vals[1]

        return get_theta()

    def getL(self):
        return np.einsum('iiii ->', self.Int2e, optimize=True)

    def optimize(self, thr=1e-7, MaxIter=50000):
        r"""
        Edmiston-Ruedenberg: maximizing self-energy
        L = \sum_p (pp||pp)
        each Jacobian step \theta between -pi/4 to pi/4
        """
        if self.norbs < 2:
            log.info("Norb = %d, too few to localize", self.norbs)
            return
        Iter = 0
        log.info("Edmiston-Ruedenberg localization")
        initL = self.getL()
        log.debug(1, "Iter        L            dL     (i , j)   theta/pi")
        sweep = []
        for i, j in it.combinations(range(self.norbs), 2):
            sweep.append((i, j) + self.predictor(i, j))
        sweep.sort(key = lambda x: x[3])
        i, j, theta, dL = sweep[-1]
        log.debug(1, "%4d %12.6g %12.6g %3d %3d  %10.6g",
                  Iter, self.getL(), dL, i, j, theta/pi)
        while dL > thr and Iter < MaxIter:
            self.transformInt(i, j, theta)
            self.transformCoef(i, j, theta)
            Iter += 1
            sweep = []
            for i, j in it.combinations(range(self.norbs), 2):
                sweep.append((i, j) + self.predictor(i, j))
            sweep.sort(key = lambda x: x[3])
            i, j, theta, dL = sweep[-1]
            log.debug(1, "%4d %12.6g %12.6g %3d %3d  %10.6g",
                      Iter, self.getL(), dL, i, j, theta/pi)

        # mapping to original orbitals
#        sorted_idx = mo_mapping.mo_1to1map(self.coefs.T)
#        self.coefs = self.coefs[sorted_idx]

        log.info("Localization converged after %4d iterations", Iter)
        log.info("Cost function: init %12.6g   final %12.6g", initL, self.getL())

if __name__ == '__main__':
    from pyscf import gto, scf
    from libdmet.solver import scf as scf_hp

    mol = gto.Mole()
    mol.atom = '''
         He   0.    0.     0.2
         H    0.   -0.5   -0.4
         H    0.    0.5   -0.4
      '''
    mol.basis = 'sto-3g'
    mol.build(verbose=4)
    mf = scf.RHF(mol).run()

    eri = mf._eri
    jk_func = scf_hp._get_jk

    localizer = ER(mol=None, eri=eri, jk_func=jk_func, mo_coeff=mf.mo_coeff[:,:2])
    localizer.init_guess = 'scdm'
    mo = localizer.kernel()
    #mo = ER(mol).kernel(mf.mo_coeff[:,:2], verbose=4)
