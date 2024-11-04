#!/usr/bin/env python

"""
MPI paralleled eri transform.

Author:
    Zhi-Hao Cui
"""

import numpy as np

from pyscf import lib
from pyscf.pbc import df

from libdmet.basis_transform.make_basis import multiply_basis
from libdmet.system.lattice import (get_phase_R2k, kpt_member)
from libdmet.utils.misc import max_abs, add_spin_dim
from libdmet.utils import logger as log
from libdmet.basis_transform.eri_transform import (get_basis_k, _pack_tril, get_weights_t_reversal,
                                                   get_naoaux, sr_loop, transform_ao_to_emb,
                                                   _Lij_s4_to_eri, eri_restore, _Lij_s4_to_eri_gso,
                                                   ERI_IMAG_TOL, ERI_SLICE, KPT_DIFF_TOL)
from mpi4pyscf.tools import mpi
comm = mpi.comm
rank = mpi.rank

def _task_location(n, task=rank):
    neach, extras = divmod(n, mpi.pool.size)
    section_sizes = ([0] + extras * [neach+1] + (mpi.pool.size-extras) * [neach])
    div_points = np.cumsum(section_sizes)
    loc0 = div_points[task]
    loc1 = div_points[task + 1]
    return loc0, loc1

def assign_workload(weights, n):
    idx_1 = np.where(weights == 1)[0]
    idx_2 = np.where(weights == 2)[0]
    n_1 = len(idx_1)
    n_2 = len(idx_2)
    nibz = n_1 + n_2

    klocs = [_task_location(nibz, task_id) for task_id in range(n)]
    ns = [j - i for i, j in klocs]

    kids = [[] for i in range(n)]
    # first assign 1
    for i, idx in enumerate(idx_1):
        kids[i%n].append(idx)

    start = 0
    for i, kid in enumerate(kids):
        end = start + (ns[i] - len(kid))
        kid.extend(idx_2[start:end])
        start = end
    return kids

@mpi.parallel_call
def get_emb_eri_fast_gdf(cell, cderi, kpts, C_ao_lo=None, basis=None, feri=None,
                         kscaled_center=None, symmetry=4, max_memory=None,
                         kconserv_tol=KPT_DIFF_TOL, unit_eri=False, swap_idx=None,
                         t_reversal_symm=True, incore=True, fout="H2.h5"):
    """
    Fast routine to compute embedding space ERI on the fly, with MPI.

    Args:
        C_ao_lo: (spin, nkpts, nao, nlo), AO to LO basis in k
        basis: (spin, ncells, nlo, nemb), embedding basis in R
        symmetry: embedding ERI symmetry
        max_memory: maximum memory
        t_reversal_symm: whether to use time reversal symmetry

    Returns:
        eri: embedding ERI.
    """
    log.verbose = comm.bcast(log.verbose)
    if rank == 0:
        log.info("Get ERI from Lpq in AO using MPI, nproc = %5s", mpi.pool.size)

    # gdf variables
    nao = cell.nao_nr()
    nkpts = len(kpts)
    mydf = df.GDF(cell, kpts)
    mydf._cderi = cderi
    # treat the possible drop of aux-basis at some kpts.
    naux = get_naoaux(mydf)

    # If C_ao_lo and basis not given, this routine is k2gamma AO transformation
    if C_ao_lo is None:
        C_ao_lo = np.zeros((nkpts, nao, nao), dtype=np.complex128)
        C_ao_lo[:, range(nao), range(nao)] = 1.0 # identity matrix for each k

    # add spin dimension for restricted C_ao_lo
    if C_ao_lo.ndim == 3:
        C_ao_lo = C_ao_lo[np.newaxis]

    # possible kpts shift
    kscaled = cell.get_scaled_kpts(kpts)
    if kscaled_center is not None:
        kscaled -= kscaled_center

    # basis related
    if basis is None:
        basis = np.eye(nkpts * nao).reshape(1, nkpts, nao, nkpts * nao)
    if basis.shape[0] < C_ao_lo.shape[0]:
        basis = add_spin_dim(basis, C_ao_lo.shape[0])
    if C_ao_lo.shape[0] < basis.shape[0]:
        C_ao_lo = add_spin_dim(C_ao_lo, basis.shape[0])

    if unit_eri: # unit ERI for DMFT
        C_ao_emb = C_ao_lo / (nkpts**0.75)
    else:
        phase = get_phase_R2k(cell, kpts)
        C_ao_emb = multiply_basis(C_ao_lo, get_basis_k(basis, phase)) / (nkpts**(0.75))
    spin, _, _, nemb = C_ao_emb.shape
    nemb_pair = nemb * (nemb+1) // 2
    res_shape = (spin * (spin+1) // 2, nemb_pair, nemb_pair)

    # ERI construction
    # ZHC NOTE outcore ERI is aa, bb, ab order.
    if t_reversal_symm:
        weights = get_weights_t_reversal(cell, kpts)
        if rank == 0:
            log.debug(2, "time reversal symm used, weights of kpts:\n%s", weights)
        if incore:
            eri = np.zeros(res_shape)
        else:
            eri = lib.H5TmpFile(filename=fout, mode='a')
            if not "ccdd" in eri.keys():
                eri.create_dataset("ccdd", res_shape, 'f8')
            elif eri["ccdd"].shape != res_shape:
                del eri["ccdd"]
                eri.create_dataset("ccdd", res_shape, 'f8')
            eri["ccdd"][:] = 0.0
    else:
        weights = np.ones((nkpts,), dtype=int)
        if rank == 0:
            log.debug(2, "time reversal symm not used.")
        if incore:
            eri = np.zeros(res_shape, dtype=np.complex128)
        else:
            raise NotImplementedError
    nibz = np.sum(weights != 0)

    if max_memory is None:
        max_memory = max(2000, (mydf.max_memory-lib.current_memory()[0]) * 0.9)
    blksize = max_memory * 1e6 / 16 / (nao**2 * 2)
    blksize = max(16, min(int(blksize), mydf.blockdim))
    Lij_s4 = np.empty((spin, naux, nemb_pair), dtype=np.complex128)
    buf = np.empty((spin * blksize * nemb_pair,), dtype=np.complex128)

    # parallel over kL
    ntasks = mpi.pool.size
    klocs = [_task_location(nibz, task_id) for task_id in range(ntasks)]
    kL_ids = assign_workload(weights, ntasks)
    kL_ids_own = kL_ids[rank]

    for nL_acc, kL in enumerate(kL_ids_own):
        assert weights[kL] > 0
        Lij_s4[:] = 0.0
        i_visited = np.zeros((nkpts,), dtype=bool)
        for i, kpti in enumerate(kpts):
            if i_visited[i]:
                continue
            i_visited[i] = True
            for j, kptj in enumerate(kpts):
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                if max_abs(np.round(kconserv) - kconserv) > kconserv_tol:
                    continue

                if t_reversal_symm:
                    jm = kpt_member(-kscaled[j], kscaled)
                    assert len(jm) == 1
                    jm = jm[0]

                step0, step1 = 0, 0
                for Lpq in sr_loop(mydf, [kpti, kptj], max_memory=max_memory,
                                   compact=False, blksize=blksize):
                    lchunk = Lpq.shape[0]
                    step0, step1 = step1, step1 + lchunk
                    Lpq_beta = None
                    Lij_loc = transform_ao_to_emb(Lpq, C_ao_emb, i, j, \
                              Lpq_beta=Lpq_beta).reshape(-1, nemb, nemb)

                    if t_reversal_symm and (not i_visited[jm]):
                        lib.hermi_sum(Lij_loc, axes=(0, 2, 1),
                                      hermi=lib.SYMMETRIC, inplace=True)

                    lib.pack_tril(Lij_loc, out=buf)
                    Lij_s4[:, step0:step1] += buf[:(spin * lchunk * \
                                              nemb_pair)].reshape(spin, \
                                              lchunk, nemb_pair)

                Lpq = Lpq_beta = Lij_loc = None
                if t_reversal_symm:
                    i_visited[jm] = True

        log.debug(1, "ERI contraction for kL: %5s / %-5s @ rank %5s",
                  nL_acc + klocs[rank][0] + 1, nibz, rank)
        _Lij_s4_to_eri(Lij_s4, eri, weight=weights[kL],
                       t_reversal_symm=t_reversal_symm)
    Lij_s4 = buf = None

    try:
        if rank == 0:
            log.debug(1, "ERI reduce in place")
        eri = mpi.reduce_inplace(eri)
    except AttributeError:
        if rank == 0:
            log.debug(1, "ERI reduce not in place")
        eri = mpi.reduce(eri)

    if rank == 0:
        if isinstance(eri, np.ndarray):
            if not t_reversal_symm:
                eri_imag_norm = max_abs(eri.imag)
                log.info('ERI imaginary = %s', eri_imag_norm)
                if eri_imag_norm > ERI_IMAG_TOL:
                    log.warn("ERI has imaginary part > %s (%s)", ERI_IMAG_TOL,
                             eri_imag_norm)
                eri = eri.real
            log.debug(1, "ERI restore")
            eri = eri_restore(eri, symmetry, nemb)
    return eri

@mpi.parallel_call
def get_emb_eri_gso(cell, cderi, kpts, C_ao_lo=None, basis=None, feri=None,
                    kscaled_center=None, symmetry=4, max_memory=None,
                    kconserv_tol=KPT_DIFF_TOL, unit_eri=False, swap_idx=None,
                    t_reversal_symm=True, basis_k=None, incore=True, fout="H2.h5"):
    """
    Fast routine to compute embedding space ERI with partial p-h transform, with MPI.

    Args:
        C_ao_lo: ((spin,) nkpts, nao, nlo), AO to LO basis in k
        basis: (ncells, nlo * 2, nemb), embedding basis in R
        symmetry: embedding ERI symmetry
        max_memory: maximum memory
        t_reversal_symm: whether to use time reversal symmetry

    Returns:
        eri: embedding ERI.
    """
    log.verbose = comm.bcast(log.verbose)
    if rank == 0:
        log.info("Get ERI from Lpq in AO using MPI, nproc = %5s", mpi.pool.size)

    # gdf variables
    nao = cell.nao_nr()
    nkpts = len(kpts)
    mydf = df.GDF(cell, kpts)
    mydf._cderi = cderi
    # treat the possible drop of aux-basis at some kpts.
    naux = get_naoaux(mydf)

    # add spin dimension for restricted C_ao_lo
    # here should always have 2 spin flavors
    C_ao_lo = add_spin_dim(C_ao_lo, 2)

    # possible kpts shift
    kscaled = cell.get_scaled_kpts(kpts)
    if kscaled_center is not None:
        kscaled -= kscaled_center

    # basis related
    if basis_k is None:
        assert basis is not None and basis.ndim == 3
        phase = get_phase_R2k(cell, kpts)
        basis_k = get_basis_k(basis[None], phase)[0]
    if basis_k.ndim == 3:
        from libdmet.routine.spinless import separate_basis
        basis_k = np.asarray(tuple(separate_basis(basis_k)))

    if unit_eri: # unit ERI for DMFT
        C_ao_emb = C_ao_lo / (nkpts**0.75)
    else:
        # AO to embedding basis
        C_ao_emb = multiply_basis(C_ao_lo, basis_k) / (nkpts**(0.75))
    spin, _, _, nemb = C_ao_emb.shape
    nemb_pair = nemb * (nemb+1) // 2
    res_shape = (1, nemb_pair, nemb_pair)

    if t_reversal_symm:
        weights = get_weights_t_reversal(cell, kpts)
        if rank == 0:
            log.debug(2, "time reversal symm used, weights of kpts:\n%s", weights)
        if incore:
            eri = np.zeros(res_shape)
        else:
            eri = lib.H5TmpFile(filename=fout, mode='a')
            if not "ccdd" in eri.keys():
                eri.create_dataset("ccdd", res_shape, 'f8')
            elif eri["ccdd"].shape != res_shape:
                del eri["ccdd"]
                eri.create_dataset("ccdd", res_shape, 'f8')
            eri["ccdd"][:] = 0.0
    else:
        weights = np.ones((nkpts,), dtype=int)
        if rank == 0:
            log.debug(2, "time reversal symm not used.")
        if incore:
            eri = np.zeros(res_shape, dtype=np.complex128) # 4-fold symmetry
        else:
            raise NotImplementedError
    nibz = np.sum(weights != 0)

    if max_memory is None:
        max_memory = max(2000, (mydf.max_memory-lib.current_memory()[0]) * 0.9)
    blksize = max_memory * 1e6 / 16 / (nao**2 * 2)
    blksize = max(16, min(int(blksize), mydf.blockdim))
    Lij_s4 = np.empty((spin, naux, nemb_pair), dtype=np.complex128)
    buf = np.empty((spin * blksize * nemb_pair,), dtype=np.complex128)

    # parallel over kL
    ntasks = mpi.pool.size
    klocs = [_task_location(nibz, task_id) for task_id in range(ntasks)]
    kL_ids = assign_workload(weights, ntasks)
    kL_ids_own = kL_ids[rank]

    for nL_acc, kL in enumerate(kL_ids_own):
        assert weights[kL] > 0
        Lij_s4[:] = 0.0
        i_visited = np.zeros((nkpts,), dtype=bool)
        for i, kpti in enumerate(kpts):
            if i_visited[i]:
                continue
            i_visited[i] = True
            for j, kptj in enumerate(kpts):
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                if max_abs(np.round(kconserv) - kconserv) > kconserv_tol:
                    continue

                if t_reversal_symm:
                    jm = kpt_member(-kscaled[j], kscaled)
                    assert len(jm) == 1
                    jm = jm[0]

                step0, step1 = 0, 0
                for Lpq in sr_loop(mydf, [kpti, kptj], max_memory=max_memory,
                                   compact=False, blksize=blksize):
                    lchunk = Lpq.shape[0]
                    step0, step1 = step1, step1 + lchunk
                    Lpq_beta = None
                    Lij_loc = transform_ao_to_emb(Lpq, C_ao_emb, i, j,
                              Lpq_beta=Lpq_beta).reshape(-1, nemb, nemb)

                    if t_reversal_symm and (not i_visited[jm]):
                        lib.hermi_sum(Lij_loc, axes=(0, 2, 1),
                                      hermi=lib.SYMMETRIC, inplace=True)

                    lib.pack_tril(Lij_loc, out=buf)
                    Lij_s4[:, step0:step1] += buf[:(spin * lchunk * \
                                              nemb_pair)].reshape(spin, \
                                              lchunk, nemb_pair)

                Lpq = Lpq_beta = Lij_loc = None
                if t_reversal_symm:
                    i_visited[jm] = True

        log.debug(1, "ERI contraction for kL: %5s / %-5s @ rank %5s",
                  nL_acc + klocs[rank][0] + 1, nibz, rank)
        _Lij_s4_to_eri_gso(Lij_s4, eri, weight=weights[kL],
                           t_reversal_symm=t_reversal_symm)
    Lij_s4 = buf = None

    try:
        if rank == 0:
            log.debug(1, "ERI reduce in place")
        eri = mpi.reduce_inplace(eri)
    except AttributeError:
        if rank == 0:
            log.debug(1, "ERI reduce not in place")
        eri = mpi.reduce(eri)

    if rank == 0:
        if isinstance(eri, np.ndarray):
            if not t_reversal_symm:
                eri_imag_norm = max_abs(eri.imag)
                log.info('ERI imaginary = %s', eri_imag_norm)
                if eri_imag_norm > ERI_IMAG_TOL:
                    log.warn("ERI has imaginary part > %s (%s)", ERI_IMAG_TOL, eri_imag_norm)
                eri = eri.real

            log.debug(1, "ERI restore")
            eri = eri_restore(eri, symmetry, nemb)
    return eri

if __name__ == '__main__':
    pass
