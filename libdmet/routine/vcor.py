#!/usr/bin/env python

# define type/symmetry of correlation potential
# potential fitting algorithms using the symmetry
# initial guess

import types
import itertools as it
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from pyscf.lib.numpy_helper import pack_tril
from pyscf.pbc.lib.kpts_helper import KPT_DIFF_TOL
from libdmet.system.lattice import round_to_FBZ
from libdmet.utils import logger as log
from libdmet.utils import max_abs

class Vcor(object):
    def __init__(self):
        self.param = None
        self.value = None
        self.local = True
        self.is_vcor_kpts = False

    def update(self, param):
        self.param = param
        self.value = self.evaluate()

    def islocal(self):
        return self.local

    def is_local(self):
        return self.local

    def get(self, i=0, kspace=True):
        """
        i is k-point index.
        """
        log.eassert(self.value is not None, "Vcor not initialized yet")
        if self.value.ndim == 4: # (nkpts, spin, nlo, nlo)
            return self.value[i]
        else:
            if kspace or i == 0:
                return self.value
            else:
                return np.zeros_like(self.value)

    def evaluate(self):
        log.error("function evaulate() is not implemented")

    def gradient(self):
        log.error("function gradient() is not implemented")

    def length(self):
        log.error("function len() is not implemented")

    def assign(self, v0):
        if self.is_local():
            log.eassert(v0.shape == self.gradient().shape[1:],
                        "The correlation potential should have shape %s, rather than %s",
                        self.gradient().shape[1:], v0.shape)

            v0prime = v0
            param = np.empty(self.length())
            g = self.gradient()
            for i in range(self.length()):
                gnorm = np.einsum('spq, spq -> ', g[i], g[i], optimize=True)
                param[i] = np.einsum('spq, spq ->', g[i], v0prime, optimize=True) / gnorm
            self.update(param)
            diff = max_abs(v0-self.get())
            if diff > 1e-7:
                log.warn("symmetrization imposed on initial guess, diff = %.5g", diff)
        else:
            v0prime = v0
            param = np.empty((self.length()))
            g = self.gradient()
            steps = self.steps
            kpts_map = self.kpts_map
            for i, step in enumerate(steps):
                grad = g[i]
                kpts_group = kpts_map[i]
                ndeg = len(kpts_group)
                if ndeg == 1:
                    k1 = kpts_group[0]
                    proj = np.einsum('xsqp, spq -> x', grad.conj(), v0prime[k1], optimize=True)
                    norm = np.einsum('xsqp, xspq -> x', grad.conj(), grad, optimize=True)
                    param[step] = proj.real / norm.real
                else:
                    k1, k2 = kpts_group
                    norm = np.einsum('xsqp, xspq -> x', grad.conj(), grad, optimize=True)
                    proj  = np.einsum('xsqp, spq -> x', grad.conj(), v0prime[k1], optimize=True)
                    proj += np.einsum('xsqp, spq -> x', grad.conj(), v0prime[k2].conj(), optimize=True)
                    param[step] = proj.real / norm.real

            self.update(param)
            for k in range(self.nkpts):
                if max_abs(v0[k] - self.get(k)) > 1e-7:
                    log.warn("symmetrization imposed on initial guess")
                    break

    def __str__(self):
        return self.evaluate().__str__()

def VcorNonLocal(restricted, bogoliubov, Lat, idx_range=None, \
        bogo_res=False):
    """
    Non-Local correlation potential.

    Args:
        restricted: restricted on v.
        bogoliubov: whether include delta.
        Lat: lattice obj.
        idx_range: index list to add correlation potential.
        bogo_res: if True, delta is restricted so that D = D.T

    Returns:
        vcor: vcor object.
    """
    nscsites = Lat.nscsites
    ncells = Lat.nkpts

    if idx_range is None:
        idx_range = list(range(0, nscsites))
    nidx = len(idx_range)

    # compute the related cell indices by inversion.
    hermi_list = -np.ones((ncells,), dtype=int)
    weight_list = np.ones((ncells,), dtype=int)
    param_range = [0]
    for R in range(ncells):
        if weight_list[R] == 1:
            idx = Lat.cell_pos2idx(-Lat.cell_idx2pos(R))
            hermi_list[R] = idx
            if idx != R:
                weight_list[R] = 2
                weight_list[idx] = 0

    # security checks:
    # elements in weight_list must be 0 or 1 or 2.
    assert np.max(weight_list) <= 2
    # for 1 elements, hermi[R] must be R.
    assert (hermi_list[weight_list==1] == \
            np.arange(ncells)[weight_list==1]).all()
    # the total weight must be ncells
    assert weight_list.sum() == ncells

    nV_1_per_spin = nidx * (nidx + 1) // 2
    nV_2_per_spin = nidx * nidx
    if restricted:
        nV_1 = nV_1_per_spin
        nV_2 = nV_2_per_spin
    else:
        nV_1 = nV_1_per_spin * 2
        nV_2 = nV_2_per_spin * 2
    if bogoliubov:
        if restricted or bogo_res:
            nD_1 = nV_1_per_spin
            nD_2 = nV_2_per_spin
        else:
            nD_1 = nidx * nidx
            nD_2 = nidx * nidx * 2
    else:
        nD_1 = nD_2 = 0

    for R in range(ncells):
        if weight_list[R] == 1:
            param_range.append(param_range[-1] + nV_1 + nD_1)
        elif weight_list[R] == 2:
            param_range.append(param_range[-1] + nV_2 + nD_2)
        else:
            param_range.append(param_range[-1])
    nparam = param_range[-1]

    v = Vcor()
    v.restricted = restricted
    v.bogoliubov = bogoliubov
    v.bogo_res = bogo_res
    v.grad = None
    v.grad_k = None

    if restricted and not bogoliubov:
        def evaluate(self):
            V = np.zeros((1, ncells, nscsites, nscsites))
            for R in range(ncells):
                if weight_list[R] == 1:
                    for idx, (i, j) in enumerate(\
                            it.combinations_with_replacement(idx_range, 2), \
                            start=param_range[R]):
                        V[0, R, i, j] = \
                        V[0, R, j, i] = \
                        self.param[idx]
                elif weight_list[R] == 2:
                    for idx, (i, j) in enumerate(\
                            it.product(idx_range, repeat=2), \
                            start=param_range[R]):
                        V[0, R, i, j] = \
                        V[0, hermi_list[R], j, i] = \
                        self.param[idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nparam, 1, ncells, nscsites, nscsites))
                for R in range(ncells):
                    if weight_list[R] == 1:
                        for idx, (i, j) in enumerate(\
                                it.combinations_with_replacement(idx_range, 2), \
                                start=param_range[R]):
                            g[idx, 0, R, i, j] = \
                            g[idx, 0, R, j, i] = 1
                    elif weight_list[R] == 2:
                        for idx, (i, j) in enumerate(\
                                it.product(idx_range, repeat=2), \
                                start=param_range[R]):
                            g[idx, 0, R, i, j] = \
                            g[idx, 0, hermi_list[R], j, i] = 1
                self.grad = g
                self.grad_k = np.asarray([Lat.R2k(g[i]) for i in \
                        range(g.shape[0])])
            return self.grad

    elif not restricted and not bogoliubov:
        def evaluate(self):
            V = np.zeros((2, ncells, nscsites, nscsites))
            for R in range(ncells):
                if weight_list[R] == 1:
                    for idx, (i, j) in enumerate(\
                            it.combinations_with_replacement(idx_range, 2), \
                            start=param_range[R]):
                        V[0, R, i, j] = \
                        V[0, R, j, i] = \
                        self.param[idx]

                        V[1, R, i, j] = \
                        V[1, R, j, i] = \
                        self.param[idx + nV_1_per_spin]
                elif weight_list[R] == 2:
                    for idx, (i, j) in enumerate(\
                            it.product(idx_range, repeat=2), \
                            start=param_range[R]):
                        V[0, R, i, j] = \
                        V[0, hermi_list[R], j, i] = \
                        self.param[idx]

                        V[1, R, i, j] = \
                        V[1, hermi_list[R], j, i] = \
                        self.param[idx + nV_2_per_spin]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nparam, 2, ncells, nscsites, nscsites))
                for R in range(ncells):
                    if weight_list[R] == 1:
                        for idx, (i, j) in enumerate(\
                                it.combinations_with_replacement(idx_range, 2), \
                                start=param_range[R]):
                            g[idx, 0, R, i, j] = \
                            g[idx, 0, R, j, i] = 1

                            g[idx + nV_1_per_spin, 1, R, i, j] = \
                            g[idx + nV_1_per_spin, 1, R, j, i] = 1
                    elif weight_list[R] == 2:
                        for idx, (i, j) in enumerate(\
                                it.product(idx_range, repeat=2), \
                                start=param_range[R]):
                            g[idx, 0, R, i, j] = \
                            g[idx, 0, hermi_list[R], j, i] = 1

                            g[idx + nV_2_per_spin, 1, R, i, j] = \
                            g[idx + nV_2_per_spin, 1, hermi_list[R], j, i] = 1

                self.grad = g
                self.grad_k = np.asarray([Lat.R2k(g[i]) for i in \
                        range(g.shape[0])])
            return self.grad

    elif restricted and bogoliubov:
        def evaluate(self):
            V = np.zeros((3, ncells, nscsites, nscsites))
            for R in range(ncells):
                if weight_list[R] == 1:
                    for idx, (i, j) in enumerate(\
                            it.combinations_with_replacement(idx_range, 2), \
                            start=param_range[R]):
                        V[0, R, i, j] = \
                        V[0, R, j, i] = \
                        self.param[idx]

                        V[2, R, i, j] = \
                        V[2, R, j, i] = \
                        self.param[idx + nV_1]

                elif weight_list[R] == 2:
                    for idx, (i, j) in enumerate(\
                            it.product(idx_range, repeat=2), \
                            start=param_range[R]):
                        V[0, R, i, j] = \
                        V[0, hermi_list[R], j, i] = \
                        self.param[idx]

                        V[2, R, i, j] = \
                        V[2, hermi_list[R], j, i] = \
                        self.param[idx + nV_2]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nparam, 3, ncells, nscsites, nscsites))
                for R in range(ncells):
                    if weight_list[R] == 1:
                        for idx, (i, j) in enumerate(\
                                it.combinations_with_replacement(idx_range, 2), \
                                start=param_range[R]):
                            g[idx, 0, R, i, j] = \
                            g[idx, 0, R, j, i] = 1

                            g[idx + nV_1, 2, R, i, j] = \
                            g[idx + nV_1, 2, R, j, i] = 1

                    elif weight_list[R] == 2:
                        for idx, (i, j) in enumerate(\
                                it.product(idx_range, repeat=2), \
                                start=param_range[R]):
                            g[idx, 0, R, i, j] = \
                            g[idx, 0, hermi_list[R], j, i] = 1

                            g[idx + nV_2, 2, R, i, j] = \
                            g[idx + nV_2, 2, hermi_list[R], j, i] = 1
                self.grad = g
                self.grad_k = np.asarray([Lat.R2k(g[i]) for i in \
                        range(g.shape[0])])
            return self.grad

    else: # not restricted and bogoliubov
        if bogo_res:
            def evaluate(self):
                V = np.zeros((3, ncells, nscsites, nscsites))
                for R in range(ncells):
                    if weight_list[R] == 1:
                        for idx, (i, j) in enumerate(\
                                it.combinations_with_replacement(idx_range, 2), \
                                start=param_range[R]):
                            V[0, R, i, j] = \
                            V[0, R, j, i] = \
                            self.param[idx]

                            V[1, R, i, j] = \
                            V[1, R, j, i] = \
                            self.param[idx + nV_1_per_spin]

                            V[2, R, i, j] = \
                            V[2, R, j, i] = \
                            self.param[idx + nV_1]

                    elif weight_list[R] == 2:
                        for idx, (i, j) in enumerate(\
                                it.product(idx_range, repeat=2), \
                                start=param_range[R]):
                            V[0, R, i, j] = \
                            V[0, hermi_list[R], j, i] = \
                            self.param[idx]

                            V[1, R, i, j] = \
                            V[1, hermi_list[R], j, i] = \
                            self.param[idx + nV_2_per_spin]

                            V[2, R, i, j] = \
                            V[2, hermi_list[R], j, i] = \
                            self.param[idx + nV_2]
                return V

            def gradient(self):
                if self.grad is None:
                    g = np.zeros((nparam, 3, ncells, nscsites, nscsites))
                    for R in range(ncells):
                        if weight_list[R] == 1:
                            for idx, (i, j) in enumerate(\
                                    it.combinations_with_replacement(idx_range, 2), \
                                    start=param_range[R]):
                                g[idx, 0, R, i, j] = \
                                g[idx, 0, R, j, i] = 1

                                g[idx + nV_1_per_spin, 1, R, i, j] = \
                                g[idx + nV_1_per_spin, 1, R, j, i] = 1

                                g[idx + nV_1, 2, R, i, j] = \
                                g[idx + nV_1, 2, R, j, i] = 1

                        elif weight_list[R] == 2:
                            for idx, (i, j) in enumerate(\
                                    it.product(idx_range, repeat=2), \
                                    start=param_range[R]):
                                g[idx, 0, R, i, j] = \
                                g[idx, 0, hermi_list[R], j, i] = 1

                                g[idx + nV_2_per_spin, 1, R, i, j] = \
                                g[idx + nV_2_per_spin, 1, hermi_list[R], j, i] = 1

                                g[idx + nV_2, 2, R, i, j] = \
                                g[idx + nV_2, 2, hermi_list[R], j, i] = 1
                    self.grad = g
                    self.grad_k = np.asarray([Lat.R2k(g[i]) for i in \
                            range(g.shape[0])])
                return self.grad

        else:
            def evaluate(self):
                V = np.zeros((3, ncells, nscsites, nscsites))
                for R in range(ncells):
                    if weight_list[R] == 1:
                        for idx, (i, j) in enumerate(\
                                it.combinations_with_replacement(idx_range, 2), \
                                start=param_range[R]):
                            V[0, R, i, j] = \
                            V[0, R, j, i] = \
                            self.param[idx]

                            V[1, R, i, j] = \
                            V[1, R, j, i] = \
                            self.param[idx + nV_1_per_spin]

                        for idx, (i, j) in enumerate(\
                                it.product(idx_range, repeat=2),
                                start=param_range[R]+nV_1):
                            V[2, R, i, j] = self.param[idx]

                    elif weight_list[R] == 2:
                        for idx, (i, j) in enumerate(\
                                it.product(idx_range, repeat=2), \
                                start=param_range[R]):
                            V[0, R, i, j] = \
                            V[0, hermi_list[R], j, i] = \
                            self.param[idx]

                            V[1, R, i, j] = \
                            V[1, hermi_list[R], j, i] = \
                            self.param[idx + nV_2_per_spin]

                            V[2, R, i, j] = \
                            self.param[idx + nV_2]

                            V[2, hermi_list[R], i, j] = \
                            self.param[idx + nV_2 + nidx * nidx]
                return V

            def gradient(self):
                if self.grad is None:
                    g = np.zeros((nparam, 3, ncells, nscsites, nscsites))
                    for R in range(ncells):
                        if weight_list[R] == 1:
                            for idx, (i, j) in enumerate(\
                                    it.combinations_with_replacement(idx_range, 2), \
                                    start=param_range[R]):
                                g[idx, 0, R, i, j] = \
                                g[idx, 0, R, j, i] = 1

                                g[idx + nV_1_per_spin, 1, R, i, j] = \
                                g[idx + nV_1_per_spin, 1, R, j, i] = 1

                            for idx, (i, j) in enumerate(\
                                    it.product(idx_range, repeat=2),
                                    start=param_range[R]+nV_1):
                                g[idx, 2, R, i, j] = 1

                        elif weight_list[R] == 2:
                            for idx, (i, j) in enumerate(\
                                    it.product(idx_range, repeat=2), \
                                    start=param_range[R]):
                                g[idx, 0, R, i, j] = \
                                g[idx, 0, hermi_list[R], j, i] = 1

                                g[idx + nV_2_per_spin, 1, R, i, j] = \
                                g[idx + nV_2_per_spin, 1, hermi_list[R], j, i] = 1

                                g[idx + nV_2, 2, R, i, j] = 1
                                g[idx + nV_2 + nidx * nidx, 2, hermi_list[R], i, j] = 1

                    self.grad = g
                    self.grad_k = np.asarray([Lat.R2k(g[i]) for i in \
                            range(g.shape[0])])
                return self.grad

    def update(self, param):
        assert len(param) == self.length()
        self.param = param
        self.value = self.evaluate()
        self.value_k = Lat.R2k(self.value)

    def get(self, i=0, kspace=True, return_all=False):
        log.eassert(self.value is not None, "Vcor not initialized yet")
        if kspace:
            if return_all:
                return self.value_k
            else:
                return self.value_k[:, i]
        else:
            if return_all:
                return self.value
            else:
                return self.value[:, i]

    def assign(self, v0):
        log.eassert(v0.shape == self.gradient().shape[1:], \
            "The correlation potential should have shape %s, rather than %s",
            self.gradient().shape[1:], v0.shape)
        v0prime = v0
        param = np.empty(self.length())
        g = self.gradient()
        for i in range(self.length()):
            param[i] = np.sum(g[i] * v0prime) / np.sum(g[i] * g[i])
        self.update(param)
        log.check(la.norm(v0-self.get(kspace=False, return_all=True)) < 1e-7, \
                "symmetrization imposed on initial guess")

    v.local = False
    v.evaluate = types.MethodType(evaluate, v)
    v.gradient = types.MethodType(gradient, v)
    v.length = types.MethodType(lambda self: nparam, v)
    v.update = types.MethodType(update, v)
    v.get = types.MethodType(get, v)
    v.assign = types.MethodType(assign, v)
    return v

def get_kpts_map(kpts_scaled, tol=KPT_DIFF_TOL):
    nkpts = len(kpts_scaled)
    kpts_round = round_to_FBZ(kpts_scaled, tol=tol)
    weights = np.ones(nkpts, dtype=int)
    kpts_map = []
    for i, ki in enumerate(kpts_round):
        if weights[i] == 1:
            for j in range(i+1, nkpts):
                sum_ij = ki + kpts_round[j]
                sum_ij -= np.round(sum_ij)
                if max_abs(sum_ij) < tol:
                    weights[i] = 2
                    weights[j] = 0
                    kpts_map.append([i, j])
                    break
            else:
                kpts_map.append([i])
    assert np.sum(weights) == nkpts
    return kpts_map

def VcorKpoints(restricted, bogoliubov, lattice, idx_range=None, bogo_res=False,
                v_idx=None, d_idx=None, ghf=False):
    """
    k-points adapted correlation potential.

    Args:
        restricted: restricted on v.
        bogoliubov: whether include delta.
        nscsites: number of orbitals.
        idx_range: index list to add correlation potential.
        bogo_res: if True, delta is restricted so that D = D.T

    Returns:
        vcor: vcor object.
    """
    # nparam_per_k = nV + nD

    # param (nkpts_irep, ndeg, nparam_per_k) a list
    # ndeg can be 1 (real) or 2 (real + imag)

    # value (nkpts, spin, nlo, nlo) complex
    # grad is (nparam_per_k, spin, nlo, nlo) complex
    # (same for all kpts)
    kpts = lattice.kpts
    nscsites = lattice.nscsites
    nkpts = len(kpts)
    kpts_map = get_kpts_map(lattice.kpts_scaled)
    #dtypes_irep = [float if len(k) == 1 else complex for k in kpts_map]
    nkpts_irep = len(kpts_map)
    ndegs = [len(k) for k in kpts_map]

    if idx_range is None:
        idx_range = list(range(0, nscsites))
    nidx = len(idx_range)

    if v_idx is None:
        if restricted:
            nv      = nidx * nidx
            nv_real = nidx * (nidx + 1) // 2
            nv_imag = nidx * (nidx - 1) // 2
        else:
            nv      = nidx * nidx * 2
            nv_real = nidx * (nidx + 1)
            nv_imag = nidx * (nidx - 1)
    else:
        raise NotImplementedError

    if d_idx is None:
        if bogoliubov and restricted:
            nd  = nidx * (nidx + 1) // 2
        elif bogoliubov and not restricted:
            if bogo_res:
                nd = nidx * (nidx + 1) // 2
            else:
                nd = nidx * nidx
        else:
            nd = 0
    else:
        raise NotImplementedError

    nparam_kpts = [nv_real + nd if len(kpts) == 1 else nv_real + nv_imag + nd for kpts in kpts_map]

    param_idx_start = 0
    param_idx_end   = 0
    param_k_slices  = []

    for nparam_k in nparam_kpts:
        if restricted:
            param_idx_end   = param_idx_start + nparam_k
            param_k_slices.append(
                slice(param_idx_start, param_idx_end)
                )
            param_idx_start = param_idx_end
        else:
            param_idx_end    = param_idx_start + nparam_k
            param_k_slices.append((
                slice(param_idx_start, param_idx_end),
                slice(param_idx_start, param_idx_start + nparam_k // 2),
                slice(param_idx_start + nparam_k // 2, param_idx_end)
            ))
            param_idx_start = param_idx_end

    nparam_tot = param_idx_end

    v = Vcor()
    v.restricted = restricted
    v.bogoliubov = bogoliubov
    v.bogo_res   = bogo_res
    v.grad       = None
    v.diag_idx   = None
    v.nkpts      = nkpts
    v.kpts_map   = kpts_map
    v.nparam_kpts       = nparam_kpts
    v.param_k_slices    = param_k_slices

    v.is_vcor_kpts = True

    if restricted and not bogoliubov:
        if v_idx is not None:
            raise NotImplementedError

        def evaluate(self):
            V = np.zeros((nkpts, 2, nscsites, nscsites), dtype=complex)

            for k, k_group in enumerate(kpts_map):
                if len(k_group) == 1:
                    idx1, idy1 = np.tril_indices(nscsites)

                    k1       = k_group[0]
                    k_slice  = param_k_slices[k]
                    param_k  = self.param[k_slice]

                    V[k1, 0, idx1, idy1] = param_k
                    V[k1, 0, idy1, idx1] = param_k
                    V[k1, 1, idx1, idy1] = param_k
                    V[k1, 1, idy1, idx1] = param_k

                else:
                    idx1, idy1 = np.tril_indices(nscsites)
                    idx2, idy2 = np.tril_indices(nscsites, -1)

                    k1, k2   = k_group
                    k_slice  = param_k_slices[k]
                    param_k  = self.param[k_slice]

                    param_k_real = param_k[:nv_real]
                    param_k_imag = param_k[nv_real:]

                    V[k1, 0, idx1, idy1]  = param_k_real
                    V[k1, 0, idy1, idx1]  = param_k_real
                    V[k1, 1, idx1, idy1]  = param_k_real
                    V[k1, 1, idy1, idx1]  = param_k_real

                    V[k1, 0, idx2, idy2] += param_k_imag * 1j
                    V[k1, 0, idy2, idx2] -= param_k_imag * 1j
                    V[k1, 1, idx2, idy2] += param_k_imag * 1j
                    V[k1, 1, idy2, idx2] -= param_k_imag * 1j

                    V[k2, 0, idx1, idy1]  = param_k_real
                    V[k2, 0, idy1, idx1]  = param_k_real
                    V[k2, 1, idx1, idy1]  = param_k_real
                    V[k2, 1, idy1, idx1]  = param_k_real

                    V[k2, 0, idx2, idy2] -= param_k_imag * 1j
                    V[k2, 0, idy2, idx2] += param_k_imag * 1j
                    V[k2, 1, idx2, idy2] -= param_k_imag * 1j
                    V[k2, 1, idy2, idx2] += param_k_imag * 1j

            return V

        def gradient(self):
            raise NotImplementedError

        def diag_indices(self):
            if self.diag_idx is None:
                self.diag_idx = [utils.triu_diag_indices(len(idx_range))]
            return self.diag_idx

    elif not restricted and not bogoliubov:
        if v_idx is not None:
            raise NotImplementedError

        def evaluate(self):
            V = np.zeros((nkpts, 2, nscsites, nscsites), dtype=complex)

            for k, k_group in enumerate(kpts_map):
                if len(k_group) == 1:
                    idx1, idy1 = np.tril_indices(nscsites)

                    k1 = k_group[0]

                    k_slice, k_slice_alpha, k_slice_beta = param_k_slices[k]

                    param_k_alpha = self.param[k_slice_alpha]
                    param_k_beta  = self.param[k_slice_beta]

                    V[k1, 0, idx1, idy1] = param_k_alpha
                    V[k1, 0, idy1, idx1] = param_k_alpha
                    V[k1, 1, idx1, idy1] = param_k_beta
                    V[k1, 1, idy1, idx1] = param_k_beta

                else:
                    idx1, idy1 = np.tril_indices(nscsites)
                    idx2, idy2 = np.tril_indices(nscsites, -1)

                    k1, k2   = k_group
                    k_slice, k_slice_alpha, k_slice_beta = param_k_slices[k]

                    param_k_alpha = self.param[k_slice_alpha]
                    param_k_beta  = self.param[k_slice_beta]

                    param_k_alpha_real = param_k_alpha[:nv_real//2]
                    param_k_alpha_imag = param_k_alpha[nv_real//2:]

                    param_k_beta_real  = param_k_beta[:nv_real//2]
                    param_k_beta_imag  = param_k_beta[nv_real//2:]

                    V[k1, 0, idx1, idy1]  = param_k_alpha_real
                    V[k1, 0, idy1, idx1]  = param_k_alpha_real
                    V[k1, 1, idx1, idy1]  = param_k_beta_real
                    V[k1, 1, idy1, idx1]  = param_k_beta_real

                    V[k1, 0, idx2, idy2] += param_k_alpha_imag * 1j
                    V[k1, 0, idy2, idx2] -= param_k_alpha_imag * 1j
                    V[k1, 1, idx2, idy2] += param_k_beta_imag * 1j
                    V[k1, 1, idy2, idx2] -= param_k_beta_imag * 1j

                    V[k2, 0, idx1, idy1]  = param_k_alpha_real
                    V[k2, 0, idy1, idx1]  = param_k_alpha_real
                    V[k2, 1, idx1, idy1]  = param_k_beta_real
                    V[k2, 1, idy1, idx1]  = param_k_beta_real

                    V[k2, 0, idx2, idy2] -= param_k_alpha_imag * 1j
                    V[k2, 0, idy2, idx2] += param_k_alpha_imag * 1j
                    V[k2, 1, idx2, idy2] -= param_k_beta_imag * 1j
                    V[k2, 1, idy2, idx2] += param_k_beta_imag * 1j

            return V

        def gradient(self):
            raise NotImplementedError

        def diag_indices(self):
            if self.diag_idx is None:
                idx = utils.triu_diag_indices(len(idx_range))
                self.diag_idx = [idx, np.asarray(idx) + nv // 2]
            return self.diag_idx

    elif restricted and bogoliubov:
        if ghf:
            raise NotImplementedError
        else:
            raise NotImplementedError

    elif not restricted and bogoliubov:
        if ghf:
            raise NotImplementedError
        else:
            raise NotImplementedError

    else: # not restricted and bogoliubov
        raise NotImplementedError

    def show(self):
        vcor_mat = self.get()
        string = "vcor\n"
        string += "nao %d \n" % vcor_mat.shape[-1]
        string += "idx range %s, length %s\n" % (self.idx_range, len(self.idx_range))
        string += "res: %s, bogo: %s, bogo res: %s\n" %(self.restricted, self.bogoliubov, self.bogo_res)
        string += str(vcor_mat[np.ix_(np.arange(vcor_mat.shape[0]), idx_range,
                                      idx_range)])
        return string

    v.evaluate = types.MethodType(evaluate, v)
    v.gradient = types.MethodType(gradient, v)
    v.diag_indices = types.MethodType(diag_indices, v)
    v.length = types.MethodType(lambda self: nparam_tot, v)
    v.steps = param_k_slices
    v.ndegs = ndegs
    v.kpts_map = kpts_map
    v.show = types.MethodType(show, v)
    v.idx_range = idx_range

    param0 = np.zeros(nparam_tot)
    v.update(param0)
    v.local = False
    return v
