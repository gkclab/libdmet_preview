#!/usr/bin/env python

"""
DFT in LO basis.
Author: Zhi-Hao Cui
"""

import time
import copy
import numpy as np

import pyscf
from pyscf import scf, ao2mo, dft, lib
from pyscf.lib import logger
from pyscf.dft.rks import prune_small_rho_grids_

from libdmet.basis_transform import make_basis
from libdmet.lo import lowdin, iao
from libdmet.utils.misc import max_abs

np.set_printoptions(3, linewidth=1000, suppress=True)

def get_veff_lo(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional

    .. note::
        This function will modify the input ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        dm_last : ndarray or a list of ndarrays or 0
            The density matrix baseline.  If not 0, this function computes the
            increment of HF potential w.r.t. the reference HF potential matrix.
        vhf_last : ndarray or a list of ndarrays or 0
            The reference Vxc potential matrix.
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

    Returns:
        matrix Veff = J + Vxc.  Veff can be a list matrices, if the input
        dm is a list of density matrices.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    ground_state = (isinstance(dm, np.ndarray) and dm.ndim == 2)
    
    # ZHC NOTE
    dm_ao = make_basis.transform_rdm1_to_ao_mol(dm, ks.C_ao_lo)

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            # Filter grids the first time setup grids
            # ZHC NOTE
            #ks.grids = prune_small_rho_grids_(ks, mol, dm, ks.grids)
            ks.grids = prune_small_rho_grids_(ks, mol, dm_ao, ks.grids)
        t0 = logger.timer(ks, 'setting up grids', *t0)
    if ks.nlc != '':
        if ks.nlcgrids.coords is None:
            ks.nlcgrids.build(with_non0tab=True)
            if ks.small_rho_cutoff > 1e-20 and ground_state:
                # Filter grids the first time setup grids
                # ZHC NOTE
                #ks.nlcgrids = prune_small_rho_grids_(ks, mol, dm, ks.nlcgrids)
                ks.nlcgrids = prune_small_rho_grids_(ks, mol, dm_ao, ks.nlcgrids)
            t0 = logger.timer(ks, 'setting up nlc grids', *t0)

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        # ZHC NOTE
        #n, exc, vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
        n, exc, vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm_ao, max_memory=max_memory)
        if ks.nlc != '':
            assert('VV10' in ks.nlc.upper())
            # ZHC NOTE
            #_, enlc, vnlc = ni.nr_rks(mol, ks.nlcgrids, ks.xc+'__'+ks.nlc, dm,
            #                          max_memory=max_memory)
            _, enlc, vnlc = ni.nr_rks(mol, ks.nlcgrids, ks.xc+'__'+ks.nlc,
                    dm_ao, max_memory=max_memory)
            exc += enlc
            vxc += vnlc
        # ZHC NOTE
        vxc = make_basis.transform_h1_to_mo_mol(vxc, ks.C_ao_lo)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)

    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        vk = None
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vj', None) is not None):
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj = ks.get_j(mol, ddm, hermi)
            vj += vhf_last.vj
        else:
            vj = ks.get_j(mol, dm, hermi)
        vxc += vj
    else:
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vk', None) is not None):
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                vklr = ks.get_k(mol, ddm, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vj += vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
            vk *= hyb
            if abs(omega) > 1e-10:
                vklr = ks.get_k(mol, dm, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
        vxc += vj - vk * .5

        if ground_state:
            exc -= np.einsum('ij,ji', dm, vk).real * .5 * .5

    if ground_state:
        ecoul = np.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc

class RKS_LO(dft.rks.RKS):
    get_veff = get_veff_lo

def test_dft_lo_mol(xc='hf'):
    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = '631g',
        verbose = 4
    )

    # ****************************************************
    # normal DFT in AO
    # ****************************************************
    mf = mol.KS()
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.kernel()
    E_mf = mf.e_tot

    hcore = mf.get_hcore()
    ovlp  = mf.get_ovlp()
    rdm1  = mf.make_rdm1()
    fock  = mf.get_fock()
    C_ao_mo = mf.mo_coeff
    mo_occ = mf.mo_occ

    # ****************************************************
    # construct LO
    # ****************************************************
    minao = 'minao'
    pmol = iao.reference_mol(mol, minao=minao)
    basis = pmol._basis
    basis_val = {}
    basis_core = {}

    # F core is 1s and 2s
    # H has no core
    basis_core["F"] = copy.deepcopy(basis["F"][:1])
    basis_val["F"] = copy.deepcopy(basis["F"][1:])
    basis_val["H"] = copy.deepcopy(basis["H"])

    pmol_core = pmol.copy()
    pmol_core.basis = basis_core
    pmol_core.build()

    pmol_val = pmol.copy()
    pmol_val.basis = basis_val
    pmol_val.build()

    ncore = len(pmol_core.ao_labels())
    nval = pmol_val.nao_nr()
    nvirt = mol.nao_nr() - ncore - nval
    nlo = ncore + nval + nvirt
    nxcore = nval + nvirt

    print ("ncore:", ncore)
    print (pmol_core.ao_labels())
    print ("nval:", nval)
    print (pmol_val.ao_labels())
    print ("nvirt:", nvirt)
            
    C_ao_lo, C_ao_lo_val, C_ao_lo_virt, C_ao_lo_core = \
            make_basis.get_C_ao_lo_iao_mol(mf, pmol_val=pmol_val, \
            pmol_core=pmol_core, full_return=True)
    C_ao_lo_xcore = np.hstack((C_ao_lo_val, C_ao_lo_virt))

    print ("Check IAOs")
    print ("core: ", lowdin.check_orthonormal(C_ao_lo_core, ovlp))
    print ("val: ", lowdin.check_orthonormal(C_ao_lo_val, ovlp))
    print ("virt: ", lowdin.check_orthonormal(C_ao_lo_virt, ovlp))
    print ("all: ", lowdin.check_orthonormal(C_ao_lo, ovlp))

    print ("core val orth:", lowdin.check_orthogonal(C_ao_lo_core, C_ao_lo_val, ovlp))
    print ("core virt orth:", lowdin.check_orthogonal(C_ao_lo_core, C_ao_lo_virt, ovlp))
    print ("val virt orth:", lowdin.check_orthogonal(C_ao_lo_virt, C_ao_lo_val, ovlp))

    print ("core span the same space:", lowdin.check_span_same_space(C_ao_lo_core,
        C_ao_mo[:, :ncore], ovlp))
    print ("xcore span the same space:", lowdin.check_span_same_space(C_ao_lo_xcore,
        C_ao_mo[:, ncore:], ovlp))

    assert lowdin.check_orthonormal(C_ao_lo_core, ovlp)
    assert lowdin.check_orthonormal(C_ao_lo_val, ovlp)
    assert lowdin.check_orthonormal(C_ao_lo_virt, ovlp)
    assert lowdin.check_orthonormal(C_ao_lo, ovlp)

    assert lowdin.check_orthogonal(C_ao_lo_core, C_ao_lo_val, ovlp)
    assert lowdin.check_orthogonal(C_ao_lo_core, C_ao_lo_virt, ovlp)
    assert lowdin.check_orthogonal(C_ao_lo_virt, C_ao_lo_val, ovlp)

    assert lowdin.check_span_same_space(C_ao_lo_core, C_ao_mo[:, :ncore], ovlp)
    assert lowdin.check_span_same_space(C_ao_lo_xcore, C_ao_mo[:, ncore:], ovlp)

    # ****************************************************
    # Active space Hamiltonian
    # ****************************************************
    rdm1_core = mf.make_rdm1(C_ao_mo[:, :ncore], mo_occ[:ncore])
    veff_core = mol.HF().get_veff(mol, dm=rdm1_core)
    E_core = np.dot((hcore + veff_core * 0.5), rdm1_core).trace()

    hcore_lo = make_basis.transform_h1_to_mo_mol(hcore + veff_core, C_ao_lo_xcore)
    ovlp_lo = make_basis.transform_h1_to_mo_mol(ovlp, C_ao_lo_xcore)
    rdm1_lo = make_basis.transform_rdm1_to_mo_mol(rdm1 - rdm1_core, C_ao_lo_xcore, ovlp)
    eri_lo = ao2mo.kernel(mf._eri, C_ao_lo_xcore)

    nao = mol.nao_nr()
    mol.nelectron = mol.nelectron - ncore * 2
    mol.nao_nr = lambda *args: nao - ncore

    mf_lo = RKS_LO(mol)
    mf_lo.C_ao_lo = C_ao_lo_xcore
    mf_lo.get_hcore = lambda *args: hcore_lo 
    mf_lo.get_ovlp = lambda *args: ovlp_lo
    mf_lo._eri = eri_lo
    mf_lo.energy_nuc = lambda *args: mol.energy_nuc() + E_core
    mf_lo.xc = mf.xc
    mf_lo.kernel(dm0=rdm1_lo)
    E_mf2 = mf_lo.e_tot
    rdm1_lo2 = mf_lo.make_rdm1()

    diff_E = abs(E_mf2 - E_mf)
    diff_rdm1 = max_abs(rdm1_lo2 - rdm1_lo)
    print ("diff E: ", diff_E)
    print ("diff rdm1: ", diff_rdm1)
    if xc == 'hf':
        assert diff_E < 1e-10
        assert diff_rdm1 < 1e-8

if __name__ == "__main__":
    test_dft_lo_mol(xc='hf')
