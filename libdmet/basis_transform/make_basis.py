#!/usr/bin/env python

"""
Making local basis for preprocess of DMET and DMFT.

Author:
    Zhi-Hao Cui
    Tianyu Zhu
"""

import os
import sys
import numpy as np
import scipy.linalg as la
from pyscf import lib
from libdmet.utils.misc import (mdot, kdot, get_spin_dim, add_spin_dim,
                                max_abs, Iterable)
from libdmet.utils import logger as log
from libdmet.lo.proj_wannier import get_proj_string

# *****************************************************************************
# Wannier related functions:
# Main wrapper: get_C_ao_lo_wannier
# *****************************************************************************

def get_C_ao_lo_wannier(lattice, kmf, proj_val, proj_virt, proj_core=None,
                        num_iter=2000, dis_num_iter=0, extra_keywords='',
                        A_core=None, A_val=None, A_virt=None, full_return=False):
    """
    Main wrapper to get Wannier C_ao_lo.
    
    Args:
        lattice: lattice object.
        kmf: kmf object.
        proj_val: projection string for valence
        proj_virt: projection string for virtual
        proj_core: projection string for core
        num_iter: number of wannier iteration
        extra_keywords: extra keywords for wannier90
        A_core: customize A matrix for core
        A_val: customize A matrix for valence
        A_virt: customize A matrix for virtual
        full_return: if true, return u_val, u_virt, u_core as well
    
    Returns:
        C_ao_lo (and U matrices if full_return == True)
    """
    # ZHC TODO unrestricted case
    from libdmet.lo import pywannier90
    C_ao_mo = np.asarray(kmf.mo_coeff)
    cell = lattice.mol
    kmesh = lattice.kmesh
    ncore = lattice.ncore
    nval = lattice.nval
    nvirt = lattice.nvirt
    
    # check nmo = ntot
    ntot = ncore + nval + nvirt
    nmo = C_ao_mo.shape[-1]
    log.eassert(ntot == nmo, "ncore(%s) + nval(%s) + nvirt(%s) != nmo(%s)",
                ncore, nval, nvirt, nmo)

    string_val, string_virt, string_core = \
            _get_exclude_bands_strings(nval=nval, nvirt=nvirt, ncore=ncore)

    # possible core
    if ncore > 0:
        log.info("Wannier localization on core")
        assert(proj_core is not None)
        num_wann = ncore
        keywords = \
        '''
        num_iter = %s
        dis_num_iter = %s
        begin projections
        %s
        end projections
        %s
        %s
        '''%(num_iter, dis_num_iter, proj_core, string_core, extra_keywords)
        
        w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords=keywords)
        w90.kernel(A_matrix=A_core)
        u_core = np.array(w90.U_matrix.transpose(2, 0, 1), order='C')
    else:
        u_core = None
    
    # valence
    log.info("Wannier localization on valence")
    num_wann = nval
    keywords = \
    '''
    num_iter = %s
    dis_num_iter = %s
    begin projections
    %s
    end projections
    %s
    %s
    '''%(num_iter, dis_num_iter, proj_val, string_val, extra_keywords)
    
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords=keywords)
    w90.kernel(A_matrix=A_val)
    u_val = np.array(w90.U_matrix.transpose(2, 0, 1), order='C')
    
    # possible virt
    if nvirt > 0:
        log.info("Wannier localization on virtual")
        num_wann = nvirt
        keywords = \
        '''
        num_iter = %s
        dis_num_iter = %s
        begin projections
        %s
        end projections
        %s
        %s
        '''%(num_iter, dis_num_iter, proj_virt, string_virt, extra_keywords)
        
        w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords=keywords)
        w90.kernel(A_matrix=A_virt)
        u_virt = np.array(w90.U_matrix.transpose(2, 0, 1), order='C')
    else:
        u_virt = None

    C_mo_lo = tile_u_matrix(u_val, u_virt=u_virt, u_core=u_core)
    C_ao_lo = multiply_basis(C_ao_mo, C_mo_lo)
    if full_return:
        return C_ao_lo, u_val, u_virt, u_core
    else:
        return C_ao_lo

def tile_u_matrix(u_val, u_virt=None, u_core=None):
    r"""
    Tile the u matrix from different subspaces.
    u has shape (nkpts, nmo, nlo)
    return C_mo_lo.
    
    Args:
        u_val: valence
        u_virt: virtual
        u_core: core

    Returns:
        u_tiled: C_mo_lo.
    """
    nkpts = u_val.shape[-3]
    if u_virt is None:
        u_virt = np.zeros((nkpts, 0, 0), dtype=u_val.dtype)
    if u_core is None:
        u_core = np.zeros((nkpts, 0, 0), dtype=u_val.dtype)
    nval  = u_val.shape[-1] # num of LO
    nvirt = u_virt.shape[-1]
    ncore = u_core.shape[-1]
    nlo = nmo = nval + nvirt + ncore
    if u_val.ndim == 3:
        u_tiled  = np.zeros((nkpts, nmo, nlo), dtype=u_val.dtype)
        for k in range(nkpts):
            u_tiled[k] = la.block_diag(u_core[k], u_val[k], u_virt[k])
    else:
        spin = u_val.shape[0]
        u_core = add_spin_dim(u_core, spin)
        u_virt = add_spin_dim(u_virt, spin)
        u_tiled  = np.zeros((spin, nkpts, nmo, nlo), dtype=u_val.dtype)
        for s in range(spin):
            for k in range(nkpts):
                u_tiled[s, k] = la.block_diag(u_core[s, k], u_val[s, k],
                                              u_virt[s, k])
    return u_tiled

def _get_exclude_bands_strings(nval=0, nvirt=0, ncore=0):
    """
    Generate exclude_bands strings for val, virt and core states.
    
    Args:
        nval:  number of valence orb
        nvirt: number of virtual orb
        ncore: number of core orb

    Returns:
        a string for exlcude bands.
    """
    norb = nval + nvirt + ncore 
    assert nval >= 0 and nvirt >= 0 and ncore >= 0 and norb > 0
    # exclude val and virt for core
    string_core = 'exclude_bands : %s-%s \n'%(ncore+1, norb) 
    # exclude core and val for virt
    string_virt = 'exclude_bands : %s-%s \n'%(1, ncore+nval) 

    if ncore == 0:
        if nvirt == 0:
            string_val = ' \n'
        else:
            # exclude virt for val
            string_val = 'exclude_bands : %s-%s \n'%(ncore+nval+1, norb) 
    else:
        if nvirt == 0:
            # exclude core for val
            string_val = 'exclude_bands : %s-%s \n'%(1, ncore) 
        else:
            # exclude core and virt for val
            string_val = 'exclude_bands : %s-%s, %s-%s \n'%(1, ncore,
                                                            ncore+nval+1, norb) 
    return string_val, string_virt, string_core

# *****************************************************************************
# IAO related functions:
# Main wrapper: get_C_ao_lo_iao
# *****************************************************************************

def get_C_ao_lo_iao(lattice, kmf, minao='minao', orth_virt=True,
                    full_virt=False, full_return=False, pmol_val=None,
                    pmol_core=None, max_ovlp=False, tol=1e-10,
                    return_labels=False, allow_smearing=False, nocc=None):
    """
    Main wrapper to get IAO C_ao_lo.

    Args:
        lattice: lattice object
        kmf: kmf object
        minao: IAO reference
        orth_virt: othogonalize PAO virtuals
        full_virt: use full PAOs as virtuals
        full_return: return C_ao_lo, C_ao_val, C_ao_virt, (C_ao_core)
        pmol_val: customize IAO valence reference
        pmol_core: customize IAO core reference
        max_ovlp: use max overlap to define PAOs.
        tol: tolerance for IAO orthogonal check
        allow_smearing: whether allow mo_occ to be fractional occupied mo_occ.
        nocc: specify the number of orbitals with occupied pattern. 
              Default is to use mo_occ > 0 as occupied (for allow_smearing=False),
              mo_occ > 0.5 (for allow_smearing=True).
              can be a float or array with shape (spin, nkpts). should not include core.
    Returns:
        C_ao_lo if full_return == False, else
        C_ao_lo, C_ao_val, C_ao_virt, (C_ao_core).
    """
    from libdmet.lo import iao
    kpts = lattice.kpts
    cell = lattice.mol
    ovlp = np.asarray(kmf.get_ovlp())
    C_ao_mo = np.asarray(kmf.mo_coeff)
    mo_occ  = np.asarray(kmf.mo_occ)
    
    # store the old shape of C_ao_mo
    old_shape = C_ao_mo.shape
    if len(old_shape) == 3:
        C_ao_mo = C_ao_mo[np.newaxis]
        mo_occ = mo_occ[np.newaxis]
    spin, nkpts, nao, nmo = C_ao_mo.shape
    if spin == 1:
        # RHF case, mo_occ convert to [0, 1]
        mo_occ = mo_occ * 0.5
    
    if (any((mo_occ.ravel() != 0.0) & (mo_occ.ravel() != 1.0))) and (nocc is None):
        if allow_smearing:
            log.debug(0, "IAO construction with fractional occupation.")
        else:
            log.debug(0, "IAO construction without fractional occupation.")
            log.warn("IAO construction has fractional occupation, "
                     "please set allow_smearing=True .")

    ncore = lattice.ncore
    if ncore > 0:
        # First treat core
        log.debug(1, "IAO core")
        assert pmol_core is not None
        assert pmol_val  is not None
        C_ao_core = C_ao_mo[:, :, :, :ncore]
        C_ao_lo_core = np.zeros_like(C_ao_core)
        for s in range(spin):
            # remove warning in cell.build
            with lib.temporary_env(sys, stderr=open(os.devnull, "w")):
                C_ao_lo_core[s] = iao.iao(cell, C_ao_core[s], minao=None,
                                          kpts=kpts, pmol=pmol_core,
                                          mo_coeff_B1=C_ao_core[s])
            C_ao_lo_core[s] = iao.vec_lowdin_k(C_ao_lo_core[s], ovlp)
        core_labels = pmol_core.ao_labels()
        log.debug(1, "-" * 79)
        log.debug(1, "core labels")
        for lab in core_labels:
            log.debug(1, "%s", lab)
        log.debug(1, "-" * 79)

        # Then valence
        log.debug(1, "IAO valence")
        nval = pmol_val.nao_nr()
        if full_virt: # all PAO
            nvirt = nao - ncore
        else: # part of PAO
            nvirt = nao - ncore - nval
        log.eassert(nvirt >= 0, "IAO nvirt (%s) should be non-negative! "
                    "Please check your IAO reference.", nvirt)
        
        val_labels = pmol_val.ao_labels()
        log.debug(1, "-" * 79)
        log.debug(1, "val labels")
        for lab in val_labels:
            log.debug(1, "%s", lab)
        log.debug(1, "-" * 79)
        
        nlo = ncore + nval + nvirt
        nxcore = nval + nvirt
        C_ao_mo = C_ao_mo[:, :, :, ncore:]
        mo_occ = mo_occ[:, :, ncore:]

        # store the final results
        C_ao_lo = np.zeros((spin, nkpts, nao, nlo), dtype=np.complex128)
        C_ao_lo_val = np.zeros((spin, nkpts, nao, nval), dtype=np.complex128)
        C_ao_lo_virt = np.zeros((spin, nkpts, nao, nvirt), dtype=np.complex128)
        for s in range(spin):
            # IAO valence
            with lib.temporary_env(sys, stderr=open(os.devnull, "w")): 
                if allow_smearing:
                    if isinstance(nocc, Iterable):
                        nocc_ = nocc[s]
                    else:
                        nocc_ = nocc
                    C_val = iao.iao(cell, C_ao_mo[s], minao=None, kpts=kpts,
                                    pmol=pmol_val, mo_coeff_B1=C_ao_mo[s],
                                    mo_occ=mo_occ[s], nocc=nocc_)
                else:
                    if nocc is None:
                        mo_coeff_occ = [C_ao_mo[s, k][:, mo_occ[s, k] > 0]
                                        for k in range(nkpts)]
                    else:
                        if isinstance(nocc, Iterable):
                            mo_coeff_occ = [C_ao_mo[s, k][:, :nocc[s][k]]
                                            for k in range(nkpts)]
                        else:
                            mo_coeff_occ = [C_ao_mo[s, k][:, :nocc]
                                            for k in range(nkpts)]

                    C_val = iao.iao(cell, mo_coeff_occ, minao=None, kpts=kpts,
                                    pmol=pmol_val, mo_coeff_B1=C_ao_mo[s])
            C_val = iao.vec_lowdin_k(C_val, ovlp)
            
            # IAO virtual
            # tile valence and core
            C_core_val = tile_C_ao_iao(C_ao_lo_core[s], C_val)
            if C_val.shape[-1] == nao - ncore: # no virtual
                C_virt = None
            else:
                with lib.temporary_env(sys, stderr=open(os.devnull, "w")): 
                    C_virt = iao.get_iao_virt(cell, C_core_val, ovlp, minao=minao,
                                              full_virt=full_virt,
                                              max_ovlp=max_ovlp, verbose=(s==0))
                if orth_virt: # orthogonalize virtual
                    assert full_virt == False
                    C_virt = iao.vec_lowdin_k(C_virt, ovlp)
            
            C_ao_lo[s] = tile_C_ao_iao(C_val, C_virt, C_core=C_ao_lo_core[s])
            C_ao_lo_val[s] = C_val
            C_ao_lo_virt[s] = C_virt
    else:
        C_ao_lo_core = [None for s in range(spin)]
        if pmol_val is None:
            with lib.temporary_env(sys, stderr=open(os.devnull, "w")): 
                pmol_val = iao.reference_mol(cell, minao)
        
        val_labels = pmol_val.ao_labels()
        log.debug(1, "-" * 79)
        log.debug(1, "val labels")
        for lab in val_labels:
            log.debug(1, "%s", lab)
        log.debug(1, "-" * 79)
        
        nval = pmol_val.nao_nr()
        if full_virt: # all PAO
            nvirt = nao - ncore
        else: # part of PAO
            nvirt = nao - ncore - nval
        log.eassert(nvirt >= 0, "IAO nvirt (%s) should be non-negative! "
                    "Please check your IAO reference.", nvirt)
        nlo = ncore + nval + nvirt

        C_ao_lo = np.zeros((spin, nkpts, nao, nlo), dtype=np.complex128)
        C_ao_lo_val = np.zeros((spin, nkpts, nao, nval), dtype=np.complex128)
        C_ao_lo_virt = np.zeros((spin, nkpts, nao, nvirt), dtype=np.complex128)
        for s in range(spin):
            # IAO valence
            with lib.temporary_env(sys, stderr=open(os.devnull, "w")): 
                if allow_smearing:
                    if isinstance(nocc, Iterable):
                        nocc_ = nocc[s]
                    else:
                        nocc_ = nocc
                    C_val = iao.iao(cell, C_ao_mo[s], minao=None, kpts=kpts,
                                    pmol=pmol_val, mo_occ=mo_occ[s], nocc=nocc_)
                else:
                    if nocc is None:
                        mo_coeff_occ = [C_ao_mo[s, k][:, mo_occ[s, k]>0]
                                        for k in range(nkpts)]
                    else:
                        if isinstance(nocc, Iterable):
                            mo_coeff_occ = [C_ao_mo[s, k][:, :nocc[s][k]]
                                            for k in range(nkpts)]
                        else:
                            mo_coeff_occ = [C_ao_mo[s, k][:, :nocc]
                                            for k in range(nkpts)]
                    C_val = iao.iao(cell, mo_coeff_occ, minao=None, kpts=kpts,
                                    pmol=pmol_val)
            C_val = iao.vec_lowdin_k(C_val, ovlp)
            # IAO virtual
            if C_val.shape[-1] == nao - ncore: # no virtual
                C_virt = None
            else:
                with lib.temporary_env(sys, stderr=open(os.devnull, "w")): 
                    C_virt = iao.get_iao_virt(cell, C_val, ovlp, minao=None,
                                              full_virt=full_virt, pmol=pmol_val,
                                              max_ovlp=max_ovlp, verbose=(s==0))
                if orth_virt: # orthogonalize virtual
                    assert full_virt == False
                    C_virt = iao.vec_lowdin_k(C_virt, ovlp)
            C_ao_lo[s] = tile_C_ao_iao(C_val, C_virt, C_core=C_ao_lo_core[s])
            C_ao_lo_val[s] = C_val
            C_ao_lo_virt[s] = C_virt
    
    for s in range(spin):
        is_orth = iao.check_orthonormal(C_ao_lo[s], ovlp, tol=tol)
        if not is_orth:
            log.warn("IAO set is not orthogonal!")

    if len(old_shape) == 3:
        C_ao_lo = C_ao_lo[0]
        C_ao_lo_val = C_ao_lo_val[0]
        C_ao_lo_virt = C_ao_lo_virt[0]
        C_ao_lo_core = C_ao_lo_core[0]

    if full_return:
        if ncore > 0:
            res = [C_ao_lo, C_ao_lo_val, C_ao_lo_virt, C_ao_lo_core]
        else:
            res = [C_ao_lo, C_ao_lo_val, C_ao_lo_virt]
        if return_labels:
            if pmol_val is not None:
                val_labels = pmol_val.ao_labels()
            else:
                val_labels = None
            if pmol_core is not None:
                core_labels = pmol_core.ao_labels()
            else:
                core_labels = None

            with lib.temporary_env(sys, stderr=open(os.devnull, "w")): 
                labels = iao.get_labels(cell, minao=minao, full_virt=full_virt,
                                        B2_labels=val_labels,
                                        core_labels=core_labels)[0]
            res.append(labels)
    else:
        res = C_ao_lo
    return res

def tile_C_ao_iao(C_val, C_virt=None, C_core=None):
    r"""
    Tile the C matrix (IAO) from different subspaces.
    C_{(s), (k), AO, LO}
    
    Args:
        C_val: coefficent of valence orb
        C_virt: coefficent of virtual orb
        C_core: coefficent of core orb

    Returns:
        C_tiled: tiled coeffcient.
    """
    C_val = np.asarray(C_val)
    nao = C_val.shape[-2]
    if C_val.ndim == 2:
        spin = 0
        nkpts = 0
        if C_core is None:
            C_core = np.zeros((nao, 0), dtype=C_val.dtype)
        if C_virt is None:
            C_virt = np.zeros((nao, 0), dtype=C_val.dtype)
    elif C_val.ndim == 3:
        spin = 0
        nkpts  = C_val.shape[-3]
        if C_core is None:
            C_core = np.zeros((nkpts, nao, 0), dtype=C_val.dtype)
        if C_virt is None:
            C_virt = np.zeros((nkpts, nao, 0), dtype=C_val.dtype)
    else:
        spin = C_val.shape[-4]
        nkpts  = C_val.shape[-3]
        if C_core is None:
            C_core = np.zeros((spin, nkpts, nao, 0), dtype=C_val.dtype)
        if C_virt is None:
            C_virt = np.zeros((spin, nkpts, nao, 0), dtype=C_val.dtype)
    
    nval  = C_val.shape[-1]
    nvirt = C_virt.shape[-1]
    ncore = C_core.shape[-1]
    nlo = nval + nvirt + ncore
    if C_val.ndim == 2:
        C_tiled = np.hstack((C_core, C_val, C_virt))
    elif C_val.ndim == 3:
        C_tiled  = np.zeros((nkpts, nao, nlo), dtype=C_val.dtype)
        for k in range(nkpts):
            C_tiled[k] = np.hstack((C_core[k], C_val[k], C_virt[k]))
    else:
        spin = C_val.shape[0]
        C_tiled  = np.zeros((spin, nkpts, nao, nlo), dtype=C_val.dtype)
        for s in range(spin):
            for k in range(nkpts):
                C_tiled[s, k] = np.hstack((C_core[s, k], C_val[s, k], C_virt[s, k]))
    return C_tiled

# *****************************************************************************
# Lowdin related functions:
# Main wrapper: get_C_ao_lo_lowdin
# *****************************************************************************

def get_C_ao_lo_lowdin(mf_or_lattice, method='meta_lowdin', s=None):
    from libdmet.lo.lowdin import lowdin
    return lowdin(mf_or_lattice, method=method, s=s)

# *****************************************************************************
# Transform functions AO -> LO and LO -> AO
# for h1 and rdm1
# *****************************************************************************

def transform_h1_to_lo(h_ao_ao, C_ao_lo):
    r"""
    Transform h1 to lo basis, with kpts.
    h^{LO} = C^{\dagger} h^{AO} C
    """
    h_ao_ao = np.asarray(h_ao_ao)
    C_ao_lo = np.asarray(C_ao_lo)
    nkpts  = C_ao_lo.shape[-3]
    nlo = C_ao_lo.shape[-1]
    res_type = np.result_type(h_ao_ao.dtype, C_ao_lo.dtype)

    # treat the special case where h is 0 or [0, 0]
    if h_ao_ao.ndim == 0: # scalar
        return np.ones((nkpts, nlo, nlo), dtype=res_type) * h_ao_ao
    elif h_ao_ao.ndim == 1: # [0, 0]
        spin = len(h_ao_ao)
        h_lo_lo = np.ones((spin, nkpts, nlo, nlo), dtype=res_type)
        for s in range(spin):
            h_lo_lo[s] *= h_ao_ao[s]
        return h_lo_lo
    
    if C_ao_lo.ndim == 3 and h_ao_ao.ndim == 3:
        h_lo_lo  = np.zeros((nkpts, nlo, nlo), dtype=res_type)
        for k in range(nkpts):
            h_lo_lo[k] = mdot(C_ao_lo[k].conj().T, h_ao_ao[k], C_ao_lo[k])
    else:
        spin = get_spin_dim((h_ao_ao, C_ao_lo))
        h_ao_ao = add_spin_dim(h_ao_ao, spin)
        C_ao_lo = add_spin_dim(C_ao_lo, spin)
        assert h_ao_ao.ndim == C_ao_lo.ndim
        h_lo_lo  = np.zeros((spin, nkpts, nlo, nlo), dtype=res_type)
        for s in range(spin):
            for k in range(nkpts):
                h_lo_lo[s, k] = mdot(C_ao_lo[s, k].conj().T, h_ao_ao[s, k], C_ao_lo[s, k])
    return h_lo_lo

def transform_h1_to_ao(h_lo_lo, C_ao_lo, S_ao_ao):
    r"""
    Transform h1 to ao basis, with kpts.
    h^{LO} = C^{-1 \dagger} h^{AO} C^{-1}
    C^{-1} = C^{\dagger} S
    """
    h_lo_lo = np.asarray(h_lo_lo)
    C_ao_lo = np.asarray(C_ao_lo)
    nkpts  = C_ao_lo.shape[-3]
    nao = C_ao_lo.shape[-2]
    res_type = np.result_type(h_lo_lo.dtype, C_ao_lo.dtype, S_ao_ao.dtype)

    # S_ao_ao is assumed to be spin unrelated
    if C_ao_lo.ndim == 3 and h_lo_lo.ndim == 3:
        h_ao_ao  = np.zeros((nkpts, nao, nao), dtype=res_type)
        for k in range(nkpts):
            C_inv = C_ao_lo[k].conj().T.dot(S_ao_ao[k])
            h_ao_ao[k] = mdot(C_inv.conj().T, h_lo_lo[k], C_inv)
    else:
        spin = get_spin_dim((h_lo_lo, C_ao_lo))
        h_lo_lo = add_spin_dim(h_lo_lo, spin)
        C_ao_lo = add_spin_dim(C_ao_lo, spin)
        assert h_lo_lo.ndim == C_ao_lo.ndim
        h_ao_ao  = np.zeros((spin, nkpts, nao, nao), dtype=res_type)
        for s in range(spin):
            for k in range(nkpts):
                C_inv = C_ao_lo[s, k].conj().T.dot(S_ao_ao[k])
                h_ao_ao[s, k] = mdot(C_inv.conj().T, h_lo_lo[s, k], C_inv)
    return h_ao_ao

def transform_rdm1_to_lo(dm_ao_ao, C_ao_lo, S_ao_ao):
    r"""
    Transform rdm1 to lo basis, with kpts.
    \gamma^{LO} = C^{-1} \gamma^{AO} (C^{-1})^{\dagger}
    C^{-1} = C^{\dagger} S
    """
    dm_ao_ao = np.asarray(dm_ao_ao)
    C_ao_lo = np.asarray(C_ao_lo)
    nkpts  = C_ao_lo.shape[-3]
    nlo = C_ao_lo.shape[-1]
    res_type = np.result_type(dm_ao_ao.dtype, C_ao_lo.dtype, S_ao_ao.dtype)

    # S_ao_ao is assumed to be spin unrelated
    if C_ao_lo.ndim == 3 and dm_ao_ao.ndim == 3:
        dm_lo_lo  = np.zeros((nkpts, nlo, nlo), dtype=res_type)
        for k in range(nkpts):
            C_inv = C_ao_lo[k].conj().T.dot(S_ao_ao[k])
            dm_lo_lo[k] = mdot(C_inv, dm_ao_ao[k], C_inv.conj().T)
    else:
        spin = get_spin_dim((dm_ao_ao, C_ao_lo))
        dm_ao_ao = add_spin_dim(dm_ao_ao, spin)
        C_ao_lo = add_spin_dim(C_ao_lo, spin)
        assert dm_ao_ao.ndim == C_ao_lo.ndim
        dm_lo_lo  = np.zeros((spin, nkpts, nlo, nlo), dtype=res_type)
        for s in range(spin):
            for k in range(nkpts):
                C_inv = C_ao_lo[s, k].conj().T.dot(S_ao_ao[k])
                dm_lo_lo[s, k] = mdot(C_inv, dm_ao_ao[s, k], C_inv.conj().T)
    return dm_lo_lo

def transform_rdm1_to_ao(dm_lo_lo, C_ao_lo):
    r"""
    Transform rdm1 to ao basis, with kpts.
    \gamma^{AO} = C \gamma^{LO} C^{\dagger}
    """
    dm_lo_lo = np.asarray(dm_lo_lo)
    C_ao_lo = np.asarray(C_ao_lo)
    nkpts  = C_ao_lo.shape[-3]
    nao = C_ao_lo.shape[-2]
    res_type = np.result_type(dm_lo_lo, C_ao_lo)

    if C_ao_lo.ndim == 3 and dm_lo_lo.ndim == 3:
        dm_ao_ao  = np.zeros((nkpts, nao, nao), dtype=res_type)
        for k in range(nkpts):
            dm_ao_ao[k] = mdot(C_ao_lo[k], dm_lo_lo[k], C_ao_lo[k].conj().T)
    else:
        spin = get_spin_dim((dm_lo_lo, C_ao_lo))
        dm_lo_lo = add_spin_dim(dm_lo_lo, spin)
        C_ao_lo = add_spin_dim(C_ao_lo, spin)
        assert dm_lo_lo.ndim == C_ao_lo.ndim
        dm_ao_ao  = np.zeros((spin, nkpts, nao, nao), dtype=res_type)
        for s in range(spin):
            for k in range(nkpts):
                dm_ao_ao[s, k] = mdot(C_ao_lo[s, k], dm_lo_lo[s, k], C_ao_lo[s, k].conj().T)
    return dm_ao_ao

# *****************************************************************************
### functions for molecular calculations
# *****************************************************************************

def get_C_ao_lo_iao_mol(mf, minao='minao', orth_virt=True, full_virt=False,
                        full_return=False, pmol_val=None, pmol_core=None,
                        tol=1e-10):
    """
    Main wrapper to get IAO C_ao_lo. This function is for molecule.

    Args:
        mf: mf object
        minao: IAO reference
        orth_virt: othogonalize PAO virtuals
        full_virt: use full PAOs as virtuals
        full_return: return C_ao_lo, C_ao_val, C_ao_virt, (C_ao_core)
        pmol_val: customize IAO valence reference
        pmol_core: customize IAO core reference
        tol: tolerance for IAO orthogonal check

    Returns:
        C_ao_lo if full_return == False, else
        C_ao_lo, C_ao_val, C_ao_virt, (C_ao_core).
    """
    from libdmet.lo import iao
    mol = mf.mol
    ovlp = np.asarray(mf.get_ovlp())
    C_ao_mo = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)
    
    # store the old shape of C_ao_mo
    old_shape = C_ao_mo.shape
    if len(old_shape) == 2:
        C_ao_mo = C_ao_mo[np.newaxis]
        mo_occ = mo_occ[np.newaxis]
    spin, nao, nmo = C_ao_mo.shape
    
    if pmol_core is not None: # has core
        # First treat core
        ncore = pmol_core.nao_nr()
        log.debug(1, "IAO core")
        assert pmol_val is not None
        C_ao_core = C_ao_mo[:, :, :ncore]
        C_ao_lo_core = np.zeros_like(C_ao_core)
        for s in range(spin):
            C_ao_lo_core[s] = iao.iao(mol, C_ao_core[s], minao=None,
                                      pmol=pmol_core, mo_coeff_B1=C_ao_core[s])
            C_ao_lo_core[s] = iao.vec_lowdin_k(C_ao_lo_core[s], ovlp)
        
        # Then valence
        log.debug(1, "IAO valence")
        nval = pmol_val.nao_nr()
        if full_virt: # all PAO
            nvirt = nao - ncore
        else: # part of PAO
            nvirt = nao - ncore - nval
        log.eassert(nvirt >= 0, "IAO nvirt (%s) should be non-negative! "
                    "Please check your IAO reference.", nvirt)
        nlo = ncore + nval + nvirt
        nxcore = nval + nvirt
        C_ao_mo = C_ao_mo[:, :, ncore:]
        mo_occ = mo_occ[:, ncore:]

        # store the final results
        C_ao_lo = np.zeros((spin, nao, nlo), dtype=C_ao_mo.dtype)
        C_ao_lo_val = np.zeros((spin, nao, nval), dtype=C_ao_mo.dtype)
        C_ao_lo_virt = np.zeros((spin, nao, nvirt), dtype=C_ao_mo.dtype)
        for s in range(spin):
            mo_coeff_occ = C_ao_mo[s][:, mo_occ[s] > 0]
            # IAO valence
            C_val = iao.iao(mol, mo_coeff_occ, minao=None, pmol=pmol_val,
                            mo_coeff_B1=C_ao_mo[s])
            C_val = iao.vec_lowdin_k(C_val, ovlp)
            
            # IAO virtual
            # tile valence and core
            C_core_val = tile_C_ao_iao(C_ao_lo_core[s], C_val)
            if C_val.shape[-1] == nao - ncore: # no virtual
                C_virt = None
            else:
                C_virt = iao.get_iao_virt(mol, C_core_val, ovlp, minao=minao,
                                          full_virt=full_virt)
                if orth_virt: # orthogonalize virtual
                    assert full_virt == False
                    C_virt = iao.vec_lowdin_k(C_virt, ovlp)
            
            C_ao_lo[s] = tile_C_ao_iao(C_val, C_virt, C_core=C_ao_lo_core[s])
            C_ao_lo_val[s] = C_val
            C_ao_lo_virt[s] = C_virt
    else:
        ncore = 0
        C_ao_lo_core = [None for s in range(spin)]
        if pmol_val is None:
            with lib.temporary_env(sys, stderr=open(os.devnull, "w")): 
                pmol_val = iao.reference_mol(mol, minao)
        
        nval = pmol_val.nao_nr()
        if full_virt: # all PAO
            nvirt = nao - ncore
        else: # part of PAO
            nvirt = nao - ncore - nval
        log.eassert(nvirt >= 0, "IAO nvirt (%s) should be non-negative! "
                "Please check your IAO reference.", nvirt)
        nlo = ncore + nval + nvirt

        C_ao_lo = np.zeros((spin, nao, nlo), dtype=C_ao_mo.dtype)
        C_ao_lo_val = np.zeros((spin, nao, nval), dtype=C_ao_mo.dtype)
        C_ao_lo_virt = np.zeros((spin, nao, nvirt), dtype=C_ao_mo.dtype)
        for s in range(spin):
            mo_coeff_occ = C_ao_mo[s][:, mo_occ[s] > 0]
            # IAO valence
            C_val = iao.iao(mol, mo_coeff_occ, minao=None, pmol=pmol_val)
            C_val = iao.vec_lowdin_k(C_val, ovlp)
            # IAO virtual
            if C_val.shape[-1] == nao - ncore: # no virtual
                C_virt = None
            else:
                C_virt = iao.get_iao_virt(mol, C_val, ovlp, minao=None,
                                          full_virt=full_virt, pmol=pmol_val)
                if orth_virt: # orthogonalize virtual
                    assert full_virt == False
                    C_virt = iao.vec_lowdin_k(C_virt, ovlp)
            C_ao_lo[s] = tile_C_ao_iao(C_val, C_virt, C_core=C_ao_lo_core[s])
            C_ao_lo_val[s] = C_val
            C_ao_lo_virt[s] = C_virt
    
    for s in range(spin):
        is_orth = iao.check_orthonormal(C_ao_lo[s], ovlp, tol=tol)
        log.eassert(is_orth, "IAO set is not orthogonal!")

    if len(old_shape) == 2:
        C_ao_lo = C_ao_lo[0]
        C_ao_lo_val = C_ao_lo_val[0]
        C_ao_lo_virt = C_ao_lo_virt[0]
        C_ao_lo_core = C_ao_lo_core[0]

    if full_return:
        if ncore > 0:
            return C_ao_lo, C_ao_lo_val, C_ao_lo_virt, C_ao_lo_core
        else:
            return C_ao_lo, C_ao_lo_val, C_ao_lo_virt
    else:
        return C_ao_lo

def transform_h1_to_ao_mol(h_mo_mo, C_ao_mo, S_ao_ao):
    h_mo_mo = np.asarray(h_mo_mo)
    C_ao_mo = np.asarray(C_ao_mo)
    
    nao = C_ao_mo.shape[-2]
    # spin should be encoded in C_ao_mo,
    # h_mo_mo may be spin unrelated
    if C_ao_mo.ndim < h_mo_mo.ndim:
        C_ao_mo = add_spin_dim(C_ao_mo, h_mo_mo.shape[0], non_spin_dim=2)
    if C_ao_mo.ndim == 2:
        C_inv = C_ao_mo.conj().T.dot(S_ao_ao)
        h_ao_ao = mdot(C_inv.conj().T, h_mo_mo, C_inv)
    else:
        spin = C_ao_mo.shape[0]
        h_mo_mo = add_spin_dim(h_mo_mo, spin, non_spin_dim=2)
        assert h_mo_mo.ndim == C_ao_mo.ndim
        h_ao_ao  = np.zeros((spin, nao, nao), dtype=C_ao_mo.dtype)
        for s in range(spin):
            C_inv = C_ao_mo[s].conj().T.dot(S_ao_ao)
            h_ao_ao[s] = mdot(C_inv.conj().T, h_mo_mo[s], C_inv)
    return h_ao_ao

def transform_rdm1_to_ao_mol(dm_mo_mo, C_ao_mo):
    r"""
    Transform rdm1 to ao basis. [For molecular calculations, no kpts]
    \gamma^{AO} = C \gamma^{MO} C^{\dagger}
    """
    dm_mo_mo = np.asarray(dm_mo_mo)
    C_ao_mo = np.asarray(C_ao_mo)
    
    nao = C_ao_mo.shape[-2]
    # spin should be encoded in C_ao_mo,
    # dm_mo_mo may be spin unrelated
    if C_ao_mo.ndim < dm_mo_mo.ndim:
        C_ao_mo = add_spin_dim(C_ao_mo, dm_mo_mo.shape[0], non_spin_dim=2)
    if C_ao_mo.ndim == 2:
        dm_ao_ao = mdot(C_ao_mo, dm_mo_mo, C_ao_mo.conj().T)
    else:
        spin = C_ao_mo.shape[0]
        dm_mo_mo = add_spin_dim(dm_mo_mo, spin, non_spin_dim=2)
        assert dm_mo_mo.ndim == C_ao_mo.ndim
        dm_ao_ao  = np.zeros((spin, nao, nao), dtype=C_ao_mo.dtype)
        for s in range(spin):
            dm_ao_ao[s] = mdot(C_ao_mo[s], dm_mo_mo[s], C_ao_mo[s].conj().T)
    return dm_ao_ao

def transform_h1_to_mo_mol(h_ao_ao, C_ao_mo):
    h_ao_ao = np.asarray(h_ao_ao)
    C_ao_mo = np.asarray(C_ao_mo)
    return transform_rdm1_to_ao_mol(h_ao_ao, np.swapaxes(C_ao_mo.conj(), -1, -2))

transform_h1_to_lo_mol = transform_h1_to_mo_mol

def transform_rdm1_to_mo_mol(rdm1_ao_ao, C_ao_mo, S_ao_ao):
    rdm1_ao_ao = np.asarray(rdm1_ao_ao)
    C_ao_mo = np.asarray(C_ao_mo)
    nao = C_ao_mo.shape[-2]
    # spin should be encoded in C_ao_mo,
    # rdm1_ao_ao may be spin unrelated
    if C_ao_mo.ndim < rdm1_ao_ao.ndim:
        C_ao_mo = add_spin_dim(C_ao_mo, rdm1_ao_ao.shape[0], non_spin_dim=2)
    if C_ao_mo.ndim == 2:
        C_inv = C_ao_mo.conj().T.dot(S_ao_ao)
        rdm1_mo_mo = mdot(C_inv, rdm1_ao_ao, C_inv.conj().T)
    else:
        spin = C_ao_mo.shape[0]
        rdm1_ao_ao = add_spin_dim(rdm1_ao_ao, spin, non_spin_dim=2)
        assert(rdm1_ao_ao.ndim == C_ao_mo.ndim)
        rdm1_mo_mo  = np.zeros((spin, nao, nao), dtype=C_ao_mo.dtype)
        for s in range(spin):
            C_inv = C_ao_mo[s].conj().T.dot(S_ao_ao)
            rdm1_mo_mo[s] = mdot(C_inv, rdm1_ao_ao[s], C_inv.conj().T)
    return rdm1_mo_mo

transform_rdm1_to_lo_mol = transform_rdm1_to_mo_mol

def transform_rdm2_to_ao_mol(rdm2_mo, C_ao_mo):
    r"""
    Transform rdm2 to ao basis. [For molecular calculations, no kpts]
    \gamma^{AO} = C C \rdm2^{MO} C^{\dagger} C^{\dagger}
    assume aaaa, aabb, bbbb order
    """
    rdm2_mo = np.asarray(rdm2_mo)
    C_ao_mo = np.asarray(C_ao_mo)
    
    # spin should be encoded in C_ao_mo,
    # rdm2_mo may be spin unrelated
    if C_ao_mo.ndim == 2 and rdm2_mo.ndim == 5:
        C_ao_mo = add_spin_dim(C_ao_mo, 2, non_spin_dim=2)

    if C_ao_mo.ndim == 2:
        rdm2_ao = _transform_rdm2_to_ao_mol(rdm2_mo, C_ao_mo)
    else:
        spin = rdm2_mo.shape[0]
        nao = C_ao_mo.shape[-2]
        rdm2_ao = np.zeros((spin, nao, nao, nao, nao), dtype=rdm2_mo.dtype)
        # ZHC NOTE assume aaaa, aabb, bbbb order
        if spin == 1:
            rdm2_ao[0] = _transform_rdm2_to_ao_mol(rdm2_mo[0], C_ao_mo[0])
        elif spin == 3:
            # aaaa
            rdm2_ao[0] = _transform_rdm2_to_ao_mol(rdm2_mo[0], C_ao_mo[0])
            # aabb
            rdm2_ao[1] = _transform_rdm2_to_ao_mol(rdm2_mo[1], C_ao_mo[0], C_ao_mo[1])
            # bbbb
            rdm2_ao[2] = _transform_rdm2_to_ao_mol(rdm2_mo[2], C_ao_mo[1])
        else:
            raise ValueError
    return rdm2_ao

def _transform_rdm2_to_ao_mol(rdm2_mo, C_a, C_b=None):
    if C_b is None:
        C_b = C_a
    assert(C_a.shape == C_b.shape)
    nao, nmo = C_a.shape[-2:]
    # (M1M2|M3M4) -> (A1M2|M3M4)
    rdm2_ao = np.dot(C_a, rdm2_mo.reshape(nmo,-1))
    # (A1M2|M3M4) -> (A1M2|M3B4)
    rdm2_ao = np.dot(rdm2_ao.reshape(-1,nmo), C_b.conj().T)
    # (A1M2|M3B4) -> (M3B4|A1M2)
    rdm2_ao = rdm2_ao.reshape((nao,nmo,nmo,nao)).transpose(2,3,0,1)
    # (M3B4|A1M2) -> (B3B4|A1M2)
    rdm2_ao = np.dot(C_b, rdm2_ao.reshape(nmo,-1))
    # (B3B4|A1M2) -> (B3B4|A1A2)
    rdm2_ao = np.dot(rdm2_ao.reshape(-1,nmo), C_a.conj().T)
    # (B3B4|A1A2) -> (A1A2|B3B4)
    rdm2_ao = rdm2_ao.reshape([nao]*4).transpose((2,3,0,1))
    return rdm2_ao

# *****************************************************************************
# basis rotation related
# *****************************************************************************

def multiply_basis(C_ao_lo, C_lo_eo):
    """
    Get a new basis for C_ao_eo = C_ao_lo * C_lo_eo.
    Final shape would be (spin, nkpts, nao, neo) if either has spin
    (nkpts, nao, neo) otherwise.
    
    Args:
        C_ao_lo: ((spin,), nkpts, nao, nlo)
        C_lo_eo: ((spin,), nkpts, nlo, neo)
    
    Returns:
        C_ao_eo: ((spin,), nkpts, nao, neo)
    """
    C_ao_lo = np.asarray(C_ao_lo)
    C_lo_eo = np.asarray(C_lo_eo)
    nkpts, nlo, neo = C_lo_eo.shape[-3:]
    nao = C_ao_lo.shape[-2]
    
    if C_ao_lo.ndim == 3 and C_lo_eo.ndim == 3:
        C_ao_eo = kdot(C_ao_lo, C_lo_eo)
    else:
        if C_ao_lo.ndim == 3 and C_lo_eo.ndim == 4:
            spin = C_lo_eo.shape[0]
            C_ao_lo = add_spin_dim(C_ao_lo, spin)
        elif C_ao_lo.ndim == 4 and C_lo_eo.ndim == 3:
            spin = C_ao_lo.shape[0]
            C_lo_eo = add_spin_dim(C_lo_eo, spin)
        elif C_ao_lo.ndim == 4 and C_lo_eo.ndim == 4:
            spin = max(C_ao_lo.shape[0], C_lo_eo.shape[0])
            C_ao_lo = add_spin_dim(C_ao_lo, spin)
            C_lo_eo = add_spin_dim(C_lo_eo, spin)
        else:
            raise ValueError("invalid shape for multiply_basis: "
                             "C_ao_lo shape %s, C_lo_eo shape: %s"
                             %(C_ao_lo.shape, C_lo_eo.shape))
        C_ao_eo = np.zeros((spin, nkpts, nao, neo), 
                           dtype=np.result_type(C_ao_lo.dtype, C_lo_eo.dtype))
        for s in range(spin):
            C_ao_eo[s] = kdot(C_ao_lo[s], C_lo_eo[s])
    return C_ao_eo

def rotate_emb_basis(basis_R, mo_coeff):
    """
    Rotate the last axis of embedding basis.
    Used for the solver with a mean-field reference calcualtion or CAS.
    """
    if basis_rot.ndim == 3:
        # Rpm, mn -> Rpn
        basis_rot = np.tensordot(basis_R, mo_coeff, axes=(-1, -2))
    else:
        basis_rot = np.zeros_like(basis_R)
        # sRpm, smn -> sR
        for s in range(basis_rot.shape[0]):
            basis_rot[s] = np.tensordot(basis_R[s], mo_coeff[s], axes=(-1, -2))
    return basis_rot

def trans_mo(mo_coeff, u):
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim == 2:
        res = np.dot(mo_coeff, u)
    else:
        spin, nao, nmo = mo_coeff.shape
        res = np.zeros((spin, nao, nmo), dtype=mo_coeff.dtype)
        for s in range(spin):
            res[s] = np.dot(mo_coeff[s], u[s])
    return res

def get_mo_ovlp(mo1, mo2, ovlp):
    """
    Get MO overlap, C_1.conj().T ovlp C_2.

    Args:
        mo1: (nao, nmo1), can with spin and kpts dimension.
        mo2: (nao, nmo2), can with spin and kpts dimension.
        ovlp: can be (nao, nao) or (nkpts, nao, nao).

    Returns:
        res: (nmo1, nmo2), can with spin and kpts dimension.
    """
    ovlp = np.asarray(ovlp)
    mo1 = np.asarray(mo1)
    mo2 = np.asarray(mo2)
    if ovlp.ndim == 3: # with kpts
        nkpts, nao, _ = ovlp.shape
        nmo1, nmo2 = mo1.shape[-1], mo2.shape[-1]
        if mo1.ndim == 3:
            res = np.zeros((nkpts, nmo1, nmo2), dtype=np.result_type(mo1, mo2))
            for k in range(nkpts):
                res[k] = mdot(mo1[k].conj().T, ovlp[k], mo2[k])
        else:
            assert mo1.shape[0] == mo2.shape[0]
            spin = mo1.shape[0]
            res = np.zeros((spin, nkpts, nmo1, nmo2), 
                           dtype=np.result_type(mo1, mo2))
            for s in range(spin):
                for k in range(nkpts):
                    res[s, k] = mdot(mo1[s, k].conj().T, ovlp[k], mo2[s, k])
    else: # without kpts
        if mo1.ndim == 2:
            res = mdot(mo1.conj().T, ovlp, mo2)
        else:
            assert mo1.shape[0] == mo2.shape[0]
            spin, nao, nmo1 = mo1.shape
            nmo2 = mo2.shape[-1]
            res = np.zeros((spin, nmo1, nmo2), dtype=np.result_type(mo1, mo2))
            for s in range(spin):
                res[s] = mdot(mo1[s].conj().T, ovlp, mo2[s])
    return res

get_mo_ovlp_k = get_mo_ovlp

def find_closest_mo(mo_coeff, mo_coeff_ref, ovlp=None, return_rotmat=False):
    """
    Given mo_coeff and a reference mo_coeff_ref,
    find the U matrix so that |mo_coeff.dot(U) - mo_coeff_ref|_F is minimal.
    i.e. so-called orthogonal Procrustes problem
    
    Args:
        mo_coeff: MOs need to be rotated
        mo_coeff_ref: target reference MOs
        ovlp: overlap matrix for AOs
        return_rotmat: return rotation matrix

    Returns:
        closest MO (and rotation matrix if return_rotmat == True).
    """
    mo_coeff = np.asarray(mo_coeff)
    mo_coeff_ref = np.asarray(mo_coeff_ref)
    mo_shape = mo_coeff.shape
    if mo_coeff.ndim == 2:
        mo_coeff = mo_coeff[None]
    if mo_coeff_ref.ndim == 2:
        mo_coeff_ref = mo_coeff_ref[None]
    spin, nao, nmo = mo_coeff.shape
    if ovlp is None:
        ovlp = np.eye(nao)
    
    rotmat = np.zeros((spin, nmo, nmo), dtype=np.result_type(mo_coeff, ovlp))
    mo_coeff_closest = np.zeros_like(mo_coeff)
    for s in range(spin):
        ovlp_mo = mdot(mo_coeff[s].conj().T, ovlp, mo_coeff_ref[s])
        u, sigma, vt = la.svd(ovlp_mo)
        rotmat[s] = np.dot(u, vt)
        mo_coeff_closest[s] = np.dot(mo_coeff[s], rotmat[s])
    
    mo_coeff_closest = mo_coeff_closest.reshape(mo_shape)
    if len(mo_shape) == 2:
        rotmat = rotmat[0]
    if return_rotmat:
        return mo_coeff_closest, rotmat
    else:
        return mo_coeff_closest

# *************************************************************************************************************
# Functions for removing imaginary part of integrals.
# *************************************************************************************************************

def symmetrize_kmf(kmf, lattice, tol=1e-10):
    return lattice.symmetrize_kmf(kmf, tol=1e-10)

def parity(orb):
    orb_strip = orb.strip('1234567890')
    if (orb_strip[0] == 's'): 
        return   1
    if (orb_strip[0] == 'p'): 
        return  -1
    if (orb_strip[0] == 'd'): 
        return   1
    if (orb_strip[0] == 'f'): 
        return  -1
    if (orb_strip[0] == 'g'): 
        return   1

def detect_inv_sym(cell):
    """
    Detect inversion symmetry for a cell.
    Return orbital inversion index array and parity.
    Modified from Mario Motta's code.
    """
    at = cell._atom
    natom = len(at)
    norbs = cell.nao_nr()

    # atom inversion check
    inv_atm = np.zeros(natom, dtype=int)
    for ia, a in enumerate(at):
        Za, Ra = a[0], np.asarray(a[1])
        for ib in range(ia+1, natom):
            b = at[ib]
            Zb, Rb = b[0], np.asarray(b[1])
            if ((Za == Zb) and (np.allclose(Ra, -Rb))):
                log.info("   >>> atom %s %s %s", ia, Za, Ra)
                log.info("   >>> matches atom %s %s %s", ib, Zb, Rb)
                inv_atm[ia] = ib
                inv_atm[ib] = ia
    log.info("   >>> atom permutation, %s", inv_atm)
    
    # orbital inversion check
    inv = np.zeros(norbs, dtype=int)
    offset_info = np.asarray(cell.offset_nr_by_atom())
    for ia in range(natom):
        ib = inv_atm[ia]
        (sh0_a, sh1_a, ao0_a, ao1_a) = offset_info[ia]
        (sh0_b, sh1_b, ao0_b, ao1_b) = offset_info[ib]
        idx = np.arange(ao0_a, ao1_a)
        inv[idx] = ao0_b + (idx-ao0_a)
    log.info("   >>> orb permutation, %s", inv)
    
    # orbital parity
    sgn = np.zeros(norbs)
    ao_labels = cell.ao_labels()
    for ix in range(norbs):
        iatm, Z, orb = ao_labels[ix].split()
        log.info("   >>> orbital %s %s %s, parity: %s", iatm, Z, orb, parity(orb))
        sgn[ix] = parity(orb)

    return inv, sgn

def build_Martin_basis(orb_inv, parity_sgn, ovlp, hcore=None, imag_tol=1e-8):
    """
    Build Martin basis to remove the imaginary part for the system 
    with inversion symmetry.
    """
    norbs = len(parity_sgn)
    C = np.eye(norbs, dtype=np.complex128)
    C_ao_rao = np.zeros_like(C)
    imag_ovlp = max_abs(ovlp.imag)
    if hcore is None:
        imag_hcore = 0.0
    else:
        imag_hcore = max_abs(hcore.imag)
    
    if imag_ovlp > imag_tol or imag_hcore > imag_tol:
        for mu in range(norbs):
            if (orb_inv[mu] > mu):
                C_ao_rao[:, mu] = C[:, mu] + parity_sgn * C[orb_inv, mu].conj()
            else:
                C_ao_rao[:, mu] = (C[:, mu] - parity_sgn * C[orb_inv, mu].conj())*1j
            
            psi = C_ao_rao[:, mu]
            norm = mdot(psi.T.conj(), ovlp, psi)
            if norm.real < 0.0:
                norm = 0.0
            else:
                norm = norm.real
            norm = np.sqrt(norm)
            
            log.info("orbital %s norm %s", mu, norm)
            if (not np.allclose(norm, 0.0)):
                C_ao_rao[:, mu] = psi / norm
            else:
                log.warn("orbital norm close to 0.0. norm:\n%s", norm)
                raise ValueError
    else:
        log.info("Already in the real form. Martin basis is identity.")
        C_ao_rao = C
    return C_ao_rao

if __name__ == '__main__':
    np.set_printoptions(3, linewidth=1000, suppress=False)
