#!/usr/bin/env python

"""
Restricted DFT+U with kpoint sampling.
Based on KRHF routine.

Refs: PRB, 1998, 57, 1505.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import copy
import itertools as it
import numpy as np
import scipy.linalg as la

from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.pbc.dft import krks
from pyscf.data.nist import HARTREE2EV

from libdmet.basis_transform import make_basis
from libdmet.lo import iao, lowdin_k
from libdmet.utils import format_idx, mdot 

DIAG = False

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1, 
             kpts=None, kpts_band=None):
    """
    Coulomb + XC functional + Hubbard U terms.

    .. note::
        This is a replica of pyscf.dft.rks.get_veff with kpts added.
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        Veff : (nkpts, nao, nao) or (*, nkpts, nao, nao) ndarray
        Veff = J + Vxc + V_U.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts

    # J + V_xc
    vxc = super(ks.__class__, ks).get_veff(cell=cell, dm=dm, dm_last=dm_last,
                                           vhf_last=vhf_last, hermi=hermi, kpts=kpts,
                                           kpts_band=kpts_band)
    
    # V_U
    C_ao_lo = ks.C_ao_lo
    ovlp = ks.get_ovlp()
    nkpts = len(kpts)
    nlo = C_ao_lo.shape[-1]

    rdm1_lo  = np.zeros((nkpts, nlo, nlo), dtype=np.complex128)
    for k in range(nkpts):
        C_inv = np.dot(C_ao_lo[k].conj().T, ovlp[k])
        rdm1_lo[k] = mdot(C_inv, dm[k], C_inv.conj().T)
    
    is_ibz = hasattr(kpts, "kpts_ibz")
    if is_ibz:
        rdm1_lo_0 = kpts.dm_at_ref_cell(rdm1_lo)

    E_U = 0.0
    weight = getattr(kpts, "weights_ibz", np.array([1.0/nkpts,]*nkpts))
    logger.info(ks, "-" * 79)
    with np.printoptions(precision=5, suppress=True, linewidth=1000):
        for idx, val, lab in zip(ks.U_idx, ks.U_val, ks.U_lab):
            lab_string = " "
            for l in lab:
                lab_string += "%9s" %(l.split()[-1])
            lab_sp = lab[0].split()
            logger.info(ks, "local rdm1 of atom %s: ",
                        " ".join(lab_sp[:2]) + " " + lab_sp[2][:2])
            U_mesh = np.ix_(idx, idx)
            P_loc = 0.0
            for k in range(nkpts):
                S_k = ovlp[k]
                C_k = C_ao_lo[k][:, idx]
                P_k = rdm1_lo[k][U_mesh]
                
                SC = np.dot(S_k, C_k)
                vxc[k] += mdot(SC, (np.eye(P_k.shape[-1]) - P_k)
                               * (val * 0.5), SC.conj().T)
                E_U += weight[k] * (val * 0.5) * (P_k.trace() - np.dot(P_k, P_k).trace() * 0.5)
                #vxc[k] += np.dot(SC * ((1.0 - P_k.diagonal()) * (val * 0.5)),
                #                 SC.conj().T)
                #E_U += weight[k] * (val * 0.5) * (P_k.trace() - np.sum(np.diag(P_k) * np.diag(P_k)) * 0.5)
                if not is_ibz:
                    P_loc += P_k
            if is_ibz:
                P_loc = rdm1_lo_0[U_mesh].real
            else:
                P_loc = P_loc.real / nkpts
            logger.info(ks, "%s\n%s", lab_string, P_loc)
            logger.info(ks, "-" * 79)

    if E_U.real < 0.0 and all(np.asarray(ks.U_val) > 0):
        logger.warn(ks, "E_U (%s) is negative...", E_U.real)
    vxc = lib.tag_array(vxc, E_U=E_U)
    return vxc

def get_veff_diag(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1, 
             kpts=None, kpts_band=None):
    """
    Coulomb + XC functional + Hubbard U terms.

    .. note::
        This is a replica of pyscf.dft.rks.get_veff with kpts added.
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        Veff : (nkpts, nao, nao) or (*, nkpts, nao, nao) ndarray
        Veff = J + Vxc + V_U.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts

    # J + V_xc
    vxc = super(ks.__class__, ks).get_veff(cell=cell, dm=dm, dm_last=dm_last,
                                           vhf_last=vhf_last, hermi=hermi, kpts=kpts,
                                           kpts_band=kpts_band)
    
    # V_U
    C_ao_lo = ks.C_ao_lo
    ovlp = ks.get_ovlp()
    nkpts = len(kpts)
    nlo = C_ao_lo.shape[-1]

    rdm1_lo  = np.zeros((nkpts, nlo, nlo), dtype=np.complex128)
    for k in range(nkpts):
        C_inv = np.dot(C_ao_lo[k].conj().T, ovlp[k])
        rdm1_lo[k] = mdot(C_inv, dm[k], C_inv.conj().T)
    
    is_ibz = hasattr(kpts, "kpts_ibz")
    if is_ibz:
        rdm1_lo_0 = kpts.dm_at_ref_cell(rdm1_lo)

    E_U = 0.0
    weight = getattr(kpts, "weights_ibz", np.array([1.0/nkpts,]*nkpts))
    logger.info(ks, "-" * 79)
    with np.printoptions(precision=5, suppress=True, linewidth=1000):
        for idx, val, lab in zip(ks.U_idx, ks.U_val, ks.U_lab):
            lab_string = " "
            for l in lab:
                lab_string += "%9s" %(l.split()[-1])
            lab_sp = lab[0].split()
            logger.info(ks, "local rdm1 of atom %s: ",
                        " ".join(lab_sp[:2]) + " " + lab_sp[2][:2])
            U_mesh = np.ix_(idx, idx)
            P_loc = 0.0
            for k in range(nkpts):
                S_k = ovlp[k]
                C_k = C_ao_lo[k][:, idx]
                P_k = rdm1_lo[k][U_mesh]
                
                SC = np.dot(S_k, C_k)
                #vxc[k] += mdot(SC, (np.eye(P_k.shape[-1]) - P_k)
                #               * (val * 0.5), SC.conj().T)
                #E_U += weight[k] * (val * 0.5) * (P_k.trace() - np.dot(P_k, P_k).trace() * 0.5)
                vxc[k] += np.dot(SC * ((1.0 - P_k.diagonal()) * (val * 0.5)),
                                 SC.conj().T)
                E_U += weight[k] * (val * 0.5) * (P_k.trace() - np.sum(np.diag(P_k) * np.diag(P_k)) * 0.5)
                if not is_ibz:
                    P_loc += P_k
            if is_ibz:
                P_loc = rdm1_lo_0[U_mesh].real
            else:
                P_loc = P_loc.real / nkpts
            logger.info(ks, "%s\n%s", lab_string, P_loc)
            logger.info(ks, "-" * 79)

    if E_U.real < 0.0 and all(np.asarray(ks.U_val) > 0):
        logger.warn(ks, "E_U (%s) is negative...", E_U.real)
    vxc = lib.tag_array(vxc, E_U=E_U)
    return vxc

def energy_elec(ks, dm_kpts=None, h1e_kpts=None, vhf=None):
    """
    Electronic energy for KRKSpU.
    """
    if h1e_kpts is None: h1e_kpts = ks.get_hcore(ks.cell, ks.kpts)
    if dm_kpts is None: dm_kpts = ks.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.cell, dm_kpts)

    weight = getattr(ks.kpts, "weights_ibz",
                     np.array([1.0/len(h1e_kpts),]*len(h1e_kpts)))
    e1 = np.einsum('k,kij,kji', weight, h1e_kpts, dm_kpts)
    tot_e = e1 + vhf.ecoul + vhf.exc + vhf.E_U
    ks.scf_summary['e1'] = e1.real
    ks.scf_summary['coul'] = vhf.ecoul.real
    ks.scf_summary['exc'] = vhf.exc.real
    ks.scf_summary['E_U'] = vhf.E_U.real
    logger.debug(ks, 'E1 = %s  Ecoul = %s  Exc = %s  EU = %s', e1, vhf.ecoul,
                 vhf.exc, vhf.E_U)
    return tot_e.real, vhf.ecoul + vhf.exc + vhf.E_U

def set_U(ks, U_idx, U_val):
    """
    Regularize the U_idx and U_val to each atom,
    and set ks.U_idx, ks.U_val, ks.U_lab.
    """
    assert len(U_idx) == len(U_val)
    ks.U_val = []
    ks.U_idx = []
    ks.U_lab = []

    lo_labels = np.asarray(ks.cell.ao_labels())
    for i, idx in enumerate(U_idx):
        if isinstance(idx, str):
            lab_idx = ks.cell.search_ao_label(idx)
            labs = lo_labels[lab_idx]
            labs = zip(lab_idx, labs)
            for j, idxj in it.groupby(labs, key=lambda x: x[1].split()[0]):
                ks.U_idx.append(list(list(zip(*idxj))[0]))
                ks.U_val.append(U_val[i])
        else:
            ks.U_idx.append(copy.deepcopy(idx))
            ks.U_val.append(U_val[i])
    ks.U_val = np.asarray(ks.U_val) / HARTREE2EV
    logger.info(ks, "-" * 79)
    logger.debug(ks, 'U indices and values: ')
    for idx, val in zip(ks.U_idx, ks.U_val):
        ks.U_lab.append(lo_labels[idx])
        logger.debug(ks, '%6s [%.6g eV] ==> %-100s', format_idx(idx),
                     val * HARTREE2EV, "".join(lo_labels[idx]))
    logger.info(ks, "-" * 79)

def make_minao_lo(ks, minao_ref, pmol=None):
    """
    Construct minao local orbitals.
    """
    cell = ks.cell
    nao = cell.nao_nr()
    kpts = getattr(ks.kpts, "kpts_ibz", ks.kpts)
    nkpts = len(kpts)
    ovlp = ks.get_ovlp()
    C_ao_minao, labels = iao.proj_ref_ao(cell, minao=minao_ref, 
                                         kpts=kpts, pmol=pmol, 
                                         return_labels=True)
    C_ao_minao = iao.vec_lowdin(C_ao_minao, ovlp)
    labels = np.asarray(labels)
    
    C_ao_lo = np.zeros((nkpts, nao, nao), dtype=np.complex128)
    for idx, lab in zip(ks.U_idx, ks.U_lab):                    
        idx_minao = [i for i, l in enumerate(labels) if l in lab]
        assert len(idx_minao) == len(idx)
        C_ao_sub = C_ao_minao[:, :, idx_minao]
        C_ao_lo[:, :, idx] = C_ao_sub
    return C_ao_lo

def make_lowdin(ks):
    """
    Construct minao local orbitals.
    """
    def lowdin(s):
        e, v = la.eigh(s)
        idx = e > 1e-15
        if not all(idx):
            logger.warn(ks, "Lowdin has zero eigenvalues:\n%s", e)
        return np.dot(v[:,idx]/np.sqrt(e[idx]), v[:,idx].conj().T)

    cell = ks.cell
    nao = cell.nao_nr()
    kpts = getattr(ks.kpts, "kpts_ibz", ks.kpts)
    nkpts = len(kpts)
    ovlp = ks.get_ovlp()
    C_ao_lo = np.zeros((nkpts, nao, nao), dtype=np.complex128)
    #C_ao_lo[:, range(nao), range(nao)] = 1.0
    #for k in range(nkpts):
    #    ld = lowdin(ovlp[k])
    #    C_ao_lo[k] = ld
    for idx, lab in zip(ks.U_idx, ks.U_lab):
        for k in range(nkpts):
            C_ao_lo[k][idx, idx] = 1.0
            ld = lowdin(ovlp[k][np.ix_(idx, idx)])
            C_ao_lo[k][:, idx] = np.dot(C_ao_lo[k][:, idx], ld)
    return C_ao_lo

class KRKSpU(krks.KRKS):
    """
    RKSpU class adapted for PBCs with k-point sampling.
    """
    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'), 
                 U_idx=[], U_val=[], C_ao_lo='minao', **kwargs):
        """
        DFT+U args:
            U_idx: can be 
                   list of list: each sublist is a set of LO indices to add U.
                   list of string: each string is one kind of LO orbitals, 
                                   e.g. ['Ni 3d', '1 O 2pz'], in this case,
                                   LO should be aranged as ao_labels order.
                   or a combination of these two.
            U_val: a list of effective U [in eV], i.e. U-J in Dudarev's DFT+U.
                   each U corresponds to one kind of LO orbitals, should have
                   the same length as U_idx.
            C_ao_lo: LO coefficients, can be 
                     np.array, shape ((spin,), nkpts, nao, nlo),
                     string, in 'minao', 'meta-lowdin', 'lowdin'.
                     default is 'minao'.
        
        Kwargs:
            minao_ref: reference for minao orbitals, default is 'MINAO'. 
            pmol: reference pmol for minao orbitals. default is None.
            pre_orth_ao: can be 
                         ANO: [default] using ANO as reference basis for constructing 
                               (meta)-Lowdin C_ao_lo
                         otherwise use identity (AO) as reference.
        """
        try:
            super(self.__class__, self).__init__(cell, kpts, xc=xc, exxdiv=exxdiv)
        except TypeError:
            # backward compatibility
            super(self.__class__, self).__init__(cell, kpts)
            self.xc = xc
            self.exxdiv = exxdiv
        
        set_U(self, U_idx, U_val)
        
        lo_info = {"type": None, "minao_ref": None, "pmol": None, "pre_orth_ao": None}
        if isinstance(C_ao_lo, str): 
            if C_ao_lo == 'minao':
                minao_ref = kwargs.get("minao_ref", "MINAO")
                pmol = kwargs.get("pmol", None)
                self.C_ao_lo = make_minao_lo(self, minao_ref, pmol)
                lo_info["type"] = C_ao_lo
                lo_info["minao_ref"] = minao_ref
                lo_info["pmol"] = pmol
            else: # (meta)-lowdin, w/ or w/o ref AO.
                pre_orth_ao = kwargs.get("pre_orth_ao", 'ANO')
                self.C_ao_lo = lowdin_k(self, method=C_ao_lo, pre_orth_ao=pre_orth_ao)
                lo_info["type"] = C_ao_lo
                lo_info["pre_orth_ao"] = pre_orth_ao
        else:
            self.C_ao_lo = np.asarray(C_ao_lo)
            lo_info["type"] = "customize"
        logger.info(self, "-" * 79)
        logger.info(self, "DFT + U orbitals")
        logger.info(self, "%s", lo_info)
        logger.info(self, "-" * 79)

        if self.C_ao_lo.ndim == 4:
            self.C_ao_lo = self.C_ao_lo[0]
        self._keys = self._keys.union(["U_idx", "U_val", "C_ao_lo", "U_lab"])

    if DIAG:
        get_veff = get_veff_diag
    else:
        get_veff = get_veff
    energy_elec = energy_elec

    def nuc_grad_method(self):
        raise NotImplementedError

if __name__ == '__main__':
    from pyscf.pbc import gto
    from libdmet.system import lattice
    np.set_printoptions(3, linewidth=1000, suppress=True)
    cell = gto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 7
    cell.build()
    kmesh = [2, 2, 2]
    kpts = cell.make_kpts(kmesh, wrap_around=True) 
    Lat = lattice.Lattice(cell, kmesh)
    #U_idx = ["2p", "2s"]
    #U_val = [5.0, 2.0]
    U_idx = ["1 C 2p"]
    U_val = [5.0]
    
    mf = KRKSpU(cell, kpts, U_idx=U_idx, U_val=U_val, C_ao_lo='minao',
            minao_ref='gth-szv')
    #mf.C_ao_lo = make_lowdin(mf)
    
    mf.conv_tol = 1e-10
    print (mf.U_idx)
    print (mf.U_val)
    print (mf.C_ao_lo.shape)
    print (mf.kernel())
    Lat.analyze(mf)

