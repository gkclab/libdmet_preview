#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Paul J. Robinson <pjrobinson@ucla.edu>
#
# Modified for k-points sampling, IAO virtuals, cores by 
#   Zhi-Hao Cui <zhcui0408@gmail.com>

"""
Intrinsic Atomic Orbitals (IAOs) and projected atomic orbitals (PAOs).
Ref. Knizia, J. Chem. Theory Comput. 2013, 9, 11, 4834.
     Cui et al. J. Chem. Theory Comput. 2020, 16, 1, 119.
"""

import sys
import collections
import numpy as np
import scipy.linalg as la

from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf import __config__
from pyscf import lo
from pyscf.lo.iao import reference_mol, fast_iao_mullikan_pop

from libdmet.utils import logger as log
from libdmet.utils.misc import mdot, kdot, format_idx, Iterable
from libdmet.lo.lowdin import orth_cano, vec_lowdin, vec_lowdin_k, \
        check_orthonormal, check_orthogonal_two_sets, get_ao_labels

# Alternately, use ANO for minao
# orthogonalize iao by orth.lowdin(c.T*mol.intor(ovlp)*c)
MINAO = getattr(__config__, 'lo_iao_minao', 'minao')

def iao(mol, orbocc, minao=MINAO, kpts=None, pmol=None, mo_coeff_B1=None, \
        mo_occ=None, tol=1e-12):
    """
    Intrinsic Atomic Orbitals. [Ref. JCTC, 9, 4834]

    Args:
        mol : the molecule or cell object
        orbocc : (nao, nmo) or a list of (nao, nmo). occupied orbitals.
        minao: B2 basis.
        kpts: absolute kpoints.
        pmol: if given, will use as reference mol and negelect minao.
        mo_coeff_B1: use to linear combine B1 basis.
        mo_occ: given to support the smearing, should be [1.0, 0.99, ..., 0.1, 0.0]
        tol: tolerance for discard singular values in inversion.

    Returns:
        non-orthogonal IAO orbitals.  Orthogonalize them as C (C^T S C)^{-1/2},
        eg using :func:`orth.lowdin`

        >>> orbocc = mf.mo_coeff[:,mf.mo_occ>0]
        >>> c = iao(mol, orbocc)
        >>> np.dot(c, orth.lowdin(reduce(np.dot, (c.T,s,c))))
    """
    if mol.has_ecp():
        logger.warn(mol, 'ECP/PP is used. MINAO is not a good reference AO basis in IAO.')
    
    if pmol is None:
        pmol = reference_mol(mol, minao)

    # For PBC, we must use the pbc code for evaluating the integrals lest the
    # pbc conditions be ignored.
    # DO NOT import pbcgto early and check whether mol is a cell object.
    # "from pyscf.pbc import gto as pbcgto and isinstance(mol, pbcgto.Cell)"
    # The code should work even pbc module is not availabe.
    if getattr(mol, 'pbc_intor', None):  # cell object has pbc_intor method
        from pyscf.pbc import gto as pbcgto
        s1  = np.asarray(mol.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
        s2  = np.asarray(pmol.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
        s12 = np.asarray(pbcgto.cell.intor_cross('int1e_ovlp', mol, pmol, kpts=kpts))
    else:
        s1  = mol.intor_symmetric('int1e_ovlp')
        s2  = pmol.intor_symmetric('int1e_ovlp')
        s12 = gto.mole.intor_cross('int1e_ovlp', mol, pmol)
    
    # allow to customize the B1 basis, 
    # i.e. B1 can be a linear combination of large bases.
    if mo_coeff_B1 is not None:
        mo_coeff_B1 = np.asarray(mo_coeff_B1)
        if kpts is None:
            orbocc = mdot(mo_coeff_B1.conj().T, s1, orbocc)
            s1     = mdot(mo_coeff_B1.conj().T, s1, mo_coeff_B1)
            s12    = np.dot(mo_coeff_B1.conj().T, s12)
        else:
            nkpts, nao, nmo = mo_coeff_B1.shape
            orbocc_new = [] # may have different dimension at different k
            s1_new  = np.zeros((nkpts, nmo, nmo), dtype=s1.dtype)
            s12_new = np.zeros((nkpts, nmo, s12.shape[-1]), dtype=s12.dtype)
            for k in range(nkpts):
                orbocc_new.append(mdot(mo_coeff_B1[k].conj().T, s1[k], orbocc[k]))
                s1_new[k] = mdot(mo_coeff_B1[k].conj().T, s1[k], mo_coeff_B1[k])
                s12_new[k] = np.dot(mo_coeff_B1[k].conj().T, s12[k])
            orbocc = orbocc_new
            s1 = s1_new
            s12 = s12_new
    
    if mo_occ is not None:
        assert np.max(mo_occ) < 1.0 + tol
        assert np.min(mo_occ) > -tol

    if len(s1.shape) == 2:
        s21 = s12.T
        s1cd = la.cho_factor(s1)
        s2cd = la.cho_factor(s2)
        p12 = la.cho_solve(s1cd, s12)

        if np.asarray(orbocc).size == 0:
            a = p12
        else:
            C = orbocc
            ctild = la.cho_solve(s2cd, np.dot(s21, C))
            ctild = la.cho_solve(s1cd, np.dot(s12, ctild))
            # orthogonalize ctild using canonical orthogonalize
            if mo_occ is None:
                ctild = orth_cano(ctild, s1, tol=tol)
                ccs1 = mdot(C, C.conj().T, s1)
            else:
                ctild = orth_cano(ctild, s1, tol=tol, f=np.sqrt(mo_occ))
                ccs1 = mdot(C * mo_occ, C.conj().T, s1)

            ccs2 = mdot(ctild, ctild.conj().T, s1)
            a = (p12 + mdot(ccs1, ccs2, p12) * 2 \
                    - np.dot(ccs1, p12) - np.dot(ccs2, p12))
    else: # k point sampling
        s21 = np.swapaxes(s12, -1, -2).conj()
        nkpts = len(kpts)
        a = np.zeros((nkpts, s1.shape[-1], s2.shape[-1]), dtype=np.complex128)
        for k in range(nkpts):
            s1cd_k = la.cho_factor(s1[k])
            s2cd_k = la.cho_factor(s2[k])
            p12_k = la.cho_solve(s1cd_k, s12[k])
            
            if np.asarray(orbocc[k]).size == 0:
                # at some kpts, there is possibly no occupied MO.
                a[k] = p12_k
            else:
                C_k = orbocc[k]

                ctild_k = la.cho_solve(s2cd_k, np.dot(s21[k], C_k))
                ctild_k = la.cho_solve(s1cd_k, np.dot(s12[k], ctild_k))
                # orthogonalize ctild using canonical orthogonalize
                if mo_occ is None:
                    ctild_k = orth_cano(ctild_k, s1[k], tol=tol)
                    ccs1_k = mdot(C_k, C_k.conj().T, s1[k])
                else:
                    ctild_k = orth_cano(ctild_k, s1[k], tol=tol, f=np.sqrt(mo_occ[k]))
                    ccs1_k = mdot(C_k * mo_occ[k], C_k.conj().T, s1[k])
                ccs2_k = mdot(ctild_k, ctild_k.conj().T, s1[k])
                
                #a is the set of IAOs in the original basis
                a[k] = (p12_k + mdot(ccs1_k, ccs2_k, p12_k) * 2 \
                        - np.dot(ccs1_k, p12_k) - np.dot(ccs2_k, p12_k))
                
    if mo_coeff_B1 is not None:
        if kpts is None:
            a = np.dot(mo_coeff_B1, a)
        else:
            a = kdot(mo_coeff_B1, a)
    return a

def get_iao_virt(cell, C_ao_iao, S, minao=MINAO, full_virt=False, \
        pmol=None, max_ovlp=False): 
    """
    Get virtual orbitals from orthogonal IAO orbitals, C_ao_iao.
    Math: (1 - |IAO><IAO|) |B1> where B1 only choose the remaining virtual AO basis.
    """
    if max_ovlp:
        return get_iao_virt_max_ovlp(cell, C_ao_iao, S, minao=minao, \
                full_virt=full_virt, pmol=pmol)

    C_ao_iao = np.asarray(C_ao_iao)
    S = np.asarray(S)
    if pmol is None:
        pmol = reference_mol(cell, minao)

    B1_labels = cell.ao_labels()
    if full_virt:
        B2_labels = []
    else:
        B2_labels = pmol.ao_labels()
    virt_idx = [idx for idx, label in enumerate(B1_labels) \
            if (not label in B2_labels)]
    log.debug(1, "IAO: virt_idx %s", format_idx(virt_idx))
    log.debug(1, "IAO: virt_labels:\n%s", np.asarray(B1_labels)[virt_idx])
    nB1 = len(B1_labels)
    nB2 = len(B2_labels)
    nvirt = len(virt_idx)
    assert nB2 + nvirt == nB1
    
    if S.ndim == 3: # with kpts:
        nkpts = C_ao_iao.shape[-3]
        if C_ao_iao.ndim == 3:
            C_virt = np.zeros((nkpts, nB1, nvirt), dtype=np.complex128) 
            for k in range(nkpts):
                CCdS_k = mdot(C_ao_iao[k], C_ao_iao[k].conj().T, S[k])
                C_virt[k] = (np.eye(nB1) - CCdS_k)[:, virt_idx]
        else:
            spin = C_ao_iao.shape[0]
            C_virt = np.zeros((spin, nkpts, nB1, nvirt), dtype=np.complex128) 
            for s in range(spin):
                for k in range(nkpts):
                    CCdS_sk = mdot(C_ao_iao[s, k], C_ao_iao[s, k].conj().T, S[k])
                    C_virt[s, k] = (np.eye(nB1) - CCdS_sk)[:, virt_idx]
    else:
        if C_ao_iao.ndim == 2:
            CCdS = mdot(C_ao_iao, C_ao_iao.conj().T, S)
            C_virt = (np.eye(nB1) - CCdS)[:, virt_idx]
        else:
            spin = C_ao_iao.shape[0]
            C_virt = np.zeros((spin, nB1, nvirt), dtype=C_ao_iao.dtype)
            for s in range(spin):
                CCdS_s = mdot(C_ao_iao[s], C_ao_iao[s].conj().T, S)
                C_virt[s] = (np.eye(nB1) - CCdS_s)[:, virt_idx]
    return C_virt

def get_iao_virt_max_ovlp(cell, C_ao_iao, S, minao=MINAO, full_virt=False, pmol=None): 
    """
    Get virtual orbitals from orthogonal IAO orbitals.
    Find the maximum overlap with high AOs.
    """
    from libdmet.basis_transform import find_closest_mo
    C_ao_iao = np.asarray(C_ao_iao)
    S = np.asarray(S)
    if pmol is None:
        pmol = reference_mol(cell, minao)

    B1_labels = cell.ao_labels()
    if full_virt:
        B2_labels = []
    else:
        B2_labels = pmol.ao_labels()
    virt_idx = [idx for idx, label in enumerate(B1_labels) \
            if (not label in B2_labels)]
    nB1 = len(B1_labels)
    nB2 = len(B2_labels)
    nvirt = len(virt_idx)
    assert nB2 + nvirt == nB1
    
    if S.ndim == 3: # with kpts:
        nkpts = C_ao_iao.shape[-3]
        if C_ao_iao.ndim == 3:
            C_virt = np.zeros((nkpts, nB1, nvirt), dtype=np.complex128)
            for k in range(nkpts):
                CCdS_k = mdot(C_ao_iao[k], C_ao_iao[k].conj().T, S[k])
                C_virt_full = np.eye(nB1) - CCdS_k
                # use AOs as reference
                C_virt_ref = np.eye(nB1)[:, virt_idx]
                # or use PAOs as reference
                #C_virt_ref = C_virt_full[:, virt_idx]
                ew, ev = la.eigh(mdot(C_virt_full.conj().T, S[k], C_virt_full))
                U = ev[:, -nvirt:]
                C_virt_new = np.dot(C_virt_full, U)
                C_virt[k] = find_closest_mo(C_virt_new, C_virt_ref, ovlp=S[k])
        else:
            spin = C_ao_iao.shape[0]
            C_virt = np.zeros((spin, nkpts, nB1, nvirt), dtype=np.complex128) 
            for s in range(spin):
                for k in range(nkpts):
                    CCdS_sk = mdot(C_ao_iao[s, k], C_ao_iao[s, k].conj().T, S[k])
                    C_virt_full = np.eye(nB1) - CCdS_sk
                    # use AOs as reference
                    C_virt_ref = np.eye(nB1)[:, virt_idx]
                    # or use PAOs as reference
                    #C_virt_ref = C_virt_full[:, virt_idx]
                    ew, ev = la.eigh(mdot(C_virt_full.conj().T, S[k], C_virt_full))
                    U = ev[:, -nvirt:]
                    C_virt_new = np.dot(C_virt_full, U)
                    C_virt[s, k] = find_closest_mo(C_virt_new, C_virt_ref, ovlp=S[k])
    else:
        if C_ao_iao.ndim == 2:
            CCdS = mdot(C_ao_iao, C_ao_iao.conj().T, S)
            C_virt_full = np.eye(nB1) - CCdS
            C_virt_ref = np.eye(nB1)[:, virt_idx]
            ew, ev = la.eigh(mdot(C_virt_full.conj().T, S, C_virt_full))
            U = ev[:, -nvirt:]
            C_virt_new = np.dot(C_virt_full, U)
            C_virt = find_closest_mo(C_virt_new, C_virt_ref, ovlp=S)
        else:
            spin = C_ao_iao.shape[0]
            C_virt = np.zeros((spin, nB1, nvirt), dtype=C_ao_iao.dtype)
            for s in range(spin):
                CCdS_s = mdot(C_ao_iao[s], C_ao_iao[s].conj().T, S)
                C_virt_full = np.eye(nB1) - CCdS_s
                C_virt_ref = np.eye(nB1)[:, virt_idx]
                ew, ev = la.eigh(mdot(C_virt_full.conj().T, S, C_virt_full))
                U = ev[:, -nvirt:]
                C_virt_new = np.dot(C_virt_full, U)
                C_virt[s] = find_closest_mo(C_virt_new, C_virt_ref, ovlp=S)
    return C_virt

def get_labels(cell, minao=MINAO, full_virt=False, B2_labels=None, core_labels=None):
    """
    Get labels of all, val and virt.
    """
    mol = cell
    if core_labels is None:
        core_labels = []

    B1_labels = mol.ao_labels()
    if B2_labels is None:
        if full_virt:
            B2_labels = []
        else:
            pmol = reference_mol(mol, minao)
            B2_labels = pmol.ao_labels()
    
    virt_labels = [label for idx, label in enumerate(B1_labels) \
            if ((not label in B2_labels) and (not label in core_labels))]
    nB1 = len(B1_labels)
    nB2 = len(B2_labels)
    nvirt = len(virt_labels)
    ncore = len(core_labels)
    log.eassert(nB2 + nvirt + ncore == nB1, \
            "nB2 (%s) + nvirt (%s) + ncore (%s) != nB1 (%s)", \
            nB2, nvirt, ncore, nB1)
    
    labels = B2_labels + virt_labels
    return labels, B2_labels, virt_labels

def get_idx_each(cell=None, minao=MINAO, full_virt=False, labels=None, 
                 B2_labels=None, core_labels=None, kind='atom', symbols=None):
    """
    Get orbital index for all atom / orbital in the cell.
    
    Args:
        cell: cell
        minao: IAO projection reference
        full_virt: whether to exclude the orbitals with the same names
        B2_labels  : if give, will use as IAO reference labels
        core_labels: if give, will use as core labels
        kind: since the label is formatted as 'id + 'atom' + 'nl' + 'lz',
              support:
              'id atom': id + atom name
              'atom': atom name
              'atom nl': atom name + nl
              'atom nl lz' or 'atom nlm': atom name + nl + lz
              'id atom nl': id + atom name + nl
              'atom l': atom name + l
              'id atom l': id + atom name + l
              'all': keys includes all information
    
    Returns:
        dic: {atom name: index}. OrderedDict, not include core indices.
    """
    kind = kind.lower()
    if labels is None:
        labels = get_labels(cell, minao=minao, full_virt=full_virt, 
                            B2_labels=B2_labels, core_labels=core_labels)[0]
    if symbols is None:
        if kind == 'id atom':
            symbols = [" ".join(label.split()[:2]) for label in labels]
        elif kind == 'atom':
            symbols = [label.split()[1] for label in labels]
        elif kind == 'atom nl':
            symbols = []
            for label in labels:
                ele = label.split()
                for i, x in enumerate(ele[2], start=1):
                    if not x.isdigit():
                        symbols.append(ele[1] + " " + ele[2][:i])
                        break
        elif kind == 'atom nl lz' or kind == 'atom nlm':
            symbols = [" ".join(label.split()[1:]) for label in labels]
        elif kind == 'id atom nl':
            symbols = []
            for label in labels:
                ele = label.split()
                for i, x in enumerate(ele[2], start=1):
                    if not x.isdigit():
                        symbols.append(ele[0] + " " + ele[1] + " " + ele[2][:i])
                        break
        elif kind == 'atom l':
            symbols = []
            for label in labels:
                ele = label.split()
                for x in ele[2]:
                    if not x.isdigit():
                        symbols.append(ele[1] + " " + x)
                        break
        elif kind == 'atom lm':
            symbols = []
            for label in labels:
                ele = label.split()
                for i, x in enumerate(ele[2]):
                    if not x.isdigit():
                        symbols.append(ele[1] + " " + ele[2][i:])
                        break
        elif kind == 'id atom l':
            symbols = []
            for label in labels:
                ele = label.split()
                for x in ele[2]:
                    if not x.isdigit():
                        symbols.append(ele[0] + " " + ele[1] + " " + x)
                        break
        elif kind == 'all':
            symbols = labels
        else:
            raise ValueError

    dic = collections.OrderedDict()
    for i, lab in enumerate(symbols):
        if lab in dic:
            dic[lab].append(i)
        else:
            dic[lab] = [i]
    return dic

def get_idx_each_atom(cell=None, minao=MINAO, full_virt=False, B2_labels=None, 
                      core_labels=None, kind='atom'):
    return get_idx_each(cell=cell, minao=minao, full_virt=full_virt, 
                        B2_labels=B2_labels, core_labels=core_labels, kind=kind)

def get_idx_each_orbital(cell=None, minao=MINAO, full_virt=False, B2_labels=None, 
                         core_labels=None, kind='atom nl'):
    return get_idx_each(cell=cell, minao=minao, full_virt=full_virt, 
                        B2_labels=B2_labels, core_labels=core_labels, kind=kind)

def get_idx_to_ao_labels(cell, minao=MINAO, labels=None):
    if labels is None:
        labels = get_labels(cell, minao)[0]
    #atom_ids = [int(lab.split()[0]) for lab in labels]
    #idx = np.argsort(atom_ids, kind='mergesort')
    ao_labels = cell.ao_labels()
    vals = [ao_labels.index(lab) for lab in labels]
    idx  = np.argsort(vals, kind='mergesort')
    return idx

def get_idx(labels, atom_num, offset=0):
    """
    Get orbital index for a given atom_num.
    atom_num: a list of atoms, e.g. atom_num=[1, 2] (the 2nd and 3rd atom).
    """
    if not isinstance(atom_num, Iterable):
        atom_num = [atom_num]
    atom_num = [str(atom) for atom in atom_num]
    idx = [i+offset for i, label in enumerate(labels) if label.split()[0] in atom_num]
    if log.__verbose() >= log.Level["DEBUG1"]:
        log.debug(1, "Get idx for atom_num = %s", atom_num)
        log.debug(1, "idx   label")
        for i in idx:
            log.debug(1, " %s    %s", i, labels[i-offset]) 
    return idx

def proj_ref_ao(mol, minao=MINAO, kpts=None, pmol=None, return_labels=False):
    """
    Get a set of reference AO spanned by the calculation basis.
    Not orthogonalized.
    
    Args:
        return_labels: if True, return the labels as well.
    """
    nao = mol.nao_nr()
    if pmol is None:
        pmol = reference_mol(mol, minao)
    if kpts is None:
        orbocc = np.empty((nao, 0))
    else:
        orbocc = np.empty((len(kpts), nao, 0), dtype=np.complex128)
    
    C_ao_lo = iao(mol, orbocc, minao=minao, kpts=kpts, pmol=pmol)
    if return_labels:
        labels = pmol.ao_labels()
        return C_ao_lo, labels
    else:
        return C_ao_lo

del(MINAO)
