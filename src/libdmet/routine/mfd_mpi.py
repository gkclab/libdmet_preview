#! /usr/bin/env python

"""
Mean-field routines with MPI.

Author:
    Zhi-hao Cui <zhcui0408@gmail.com>
"""

from mpi4pyscf.tools import mpi
import numpy as np
import scipy.linalg as la

from pyscf.pbc.lib.kpts_helper import KPT_DIFF_TOL
from libdmet.system.lattice import round_to_FBZ

from libdmet.routine import ftsystem
from libdmet.utils.misc import max_abs
from libdmet.settings import IMAG_DISCARD_TOL
from libdmet.utils import logger as log

comm = mpi.comm
rank = mpi.rank

def _task_location(n, task=rank):
    neach, extras = divmod(n, mpi.pool.size)
    section_sizes = ([0] + extras * [neach+1] + (mpi.pool.size-extras) * [neach])
    div_points = np.cumsum(section_sizes)
    loc0 = div_points[task]
    loc1 = div_points[task + 1]
    return loc0, loc1

def get_kpairs_kidx(cell, kpts, tol=KPT_DIFF_TOL):
    nkpts = len(kpts)
    kpts_scaled = cell.get_scaled_kpts(kpts)
    kpts_round = round_to_FBZ(kpts_scaled, tol=tol)
    weights = np.ones(nkpts, dtype=int)

    kpairs = []
    for i, ki in enumerate(kpts_round):
        if weights[i] == 1:
            for j in range(i+1, nkpts):
                sum_ij = ki + kpts_round[j]
                sum_ij -= np.round(sum_ij)
                if max_abs(sum_ij) < tol:
                    weights[i] = 2
                    weights[j] = 0
                    kpairs.append((i, j))
                    break
            else:
                kpairs.append((i,))
    assert np.sum(weights) == nkpts
    kidx = np.where(weights > 0)[0]
    return kpairs, kidx

@mpi.parallel_call(skip_args=[1], skip_kwargs=['GFock'])
def DiagGHF_symm(cell, GFock, vcor_mat, mu, kpairs, kidx):
    ntasks = mpi.pool.size
    if rank == 0:
        nkpts, nso, _ = GFock.shape
        nao = nso // 2
        GFock_ibz = np.asarray(GFock[kidx], order='C')
        nkpts_ibz = len(GFock_ibz)
        klocs = [_task_location(nkpts_ibz, task_id) for task_id in range(ntasks)]
        GFock_ibz_seg = [GFock_ibz[i:j] for i, j in klocs]
        nso, nao = comm.bcast((nso, nao))
    else:
        nkpts_ibz = None
        GFock_ibz = None
        GFock_ibz_seg = None
        nso, nao = comm.bcast(None)

    # distribute irreduceble fock
    GFock_own = mpi.scatter_new(GFock_ibz_seg, data=GFock_ibz)
    nkpts_own = len(GFock_own)

    ew = np.empty((nkpts_own, nso))
    ev = np.empty((nkpts_own, nso, nso), dtype=np.complex128)

    GFock_own[:, :nao, :nao] += vcor_mat[0]
    GFock_own[:, nao:, nao:] += vcor_mat[1]
    GFock_own[:, nao:, :nao] += vcor_mat[2].conj().T

    if mu is not None:
        GFock_own[:, range(nao), range(nao)] -= mu
        GFock_own[:, range(nao, nso), range(nao, nso)] += mu

    for i in range(nkpts_own):
        ew[i], ev[i] = la.eigh(GFock_own[i], overwrite_a=True,
                               check_finite=False, lower=True)

    # gather ired fock
    ew = mpi.gather_new(ew)
    ev = mpi.gather_new(ev)

    # reconstruct full fock
    if rank == 0:
        ew_full = np.empty((nkpts, nso))
        ev_full = np.empty((nkpts, nso, nso), dtype=np.complex128)
        for k in range(nkpts_ibz):
            kp = kpairs[k]
            if len(kp) == 1:
                ew_full[kp[0]] = ew[k]
                ev_full[kp[0]] = ev[k]
            else:
                ew_full[kp[0]] = ew[k]
                ev_full[kp[0]] = ev[k]
                ew_full[kp[1]] = ew[k]
                ev_full[kp[1]] = ev[k].conj()
    else:
        ew_full = ew
        ev_full = ev

    return ew_full, ev_full

@mpi.parallel_call(skip_args=[1, 2, 3, 4], skip_kwargs=['ew', 'ev', 'drho', 'dV_dparam'])
def get_dw_dparam(cell, ew, ev, drho, dV_dparam, val, mu_quasi, beta, fix_mu, fit_idx, kpairs, kidx):
    ntasks = mpi.pool.size
    if rank == 0:
        nkpts, nso = ew.shape
        nao = nso // 2

        ew_ibz = np.asarray(ew[kidx], order='C')
        ev_ibz = np.asarray(ev[kidx], order='C')

        nkpts_ibz = len(ew_ibz)
        klocs = [_task_location(nkpts_ibz, task_id) for task_id in range(ntasks)]
        ew_ibz_seg = [ew_ibz[i:j] for i, j in klocs]
        ev_ibz_seg = [ev_ibz[i:j] for i, j in klocs]
        nkpts, nso, nao, klocs = comm.bcast((nkpts, nso, nao, klocs))
        drho = mpi.bcast(drho)
        dV_dparam = mpi.bcast(dV_dparam)
    else:
        nkpts_ibz = None
        ew_ibz = None
        ew_ibz_seg = None
        ev_ibz = None
        ev_ibz_seg = None
        nkpts, nso, nao, klocs = comm.bcast(None)
        drho = mpi.bcast(None)
        dV_dparam = mpi.bcast(None)

    # distribute irreduceble ew, ev
    ew_own = mpi.scatter_new(ew_ibz_seg, data=ew_ibz)
    ev_own = mpi.scatter_new(ev_ibz_seg, data=ev_ibz)
    nkpts_own = len(ew_own)

    dw_dparam = np.zeros((dV_dparam.shape[0],), dtype=dV_dparam.dtype)
    start, end = klocs[rank]
    for k in range(nkpts_own):
        dw_dv = ftsystem.get_dw_dv(ew_own[k], ev_own[k], drho, mu_quasi, beta, fix_mu=fix_mu,
                                   fit_idx=fit_idx,
                                   compact=(dV_dparam.ndim == 2))
        kp = kpairs[k+start]
        if len(kp) == 1:
            dw_dparam += dV_dparam.reshape(dV_dparam.shape[0], -1).dot(dw_dv.real.ravel())
        else:
            dw_dparam += dV_dparam.reshape(dV_dparam.shape[0], -1).dot(dw_dv.real.ravel()) * 2.0

    dw_dparam /= (2.0 * val * np.sqrt(2.0) * nkpts)
    dw_dparam = mpi.reduce_inplace(dw_dparam)
    return dw_dparam
