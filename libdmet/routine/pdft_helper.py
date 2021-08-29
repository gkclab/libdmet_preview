#! /usr/bin/env python

"""
Helper functions for pyscf PBC dft module.
Including KRKS_LO, get_veff_lo_rks, KUKS_LO, get_veff_lo_uks, DFTpU etc.

Author:
    Zhihao Cui <zhcui0408@gmail.com>
"""

import time
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import khf
from pyscf.pbc.scf import kuhf
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import multigrid
from pyscf.pbc.dft import krks
from pyscf.pbc.dft import kuks
from pyscf import __config__

from libdmet.basis_transform import make_basis
from libdmet.routine import krkspu
from libdmet.routine import kukspu
try:
    from libdmet.routine import krkspu_ksymm
    from libdmet.routine import kukspu_ksymm
except ImportError:
    pass

NELEC_ERROR_TOL = getattr(__config__, 'pbc_dft_rks_prune_error_tol', 0.02)

# *********************************************************************
# DFT+U
# *********************************************************************

def KRKSpU(cell, *args, **kwargs):
    for arg in args:
        if hasattr(arg, "kpts_ibz"):
            return krkspu_ksymm.KRKSpU(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if hasattr(kwargs['kpts'], "kpts_ibz"):
            return krkspu_ksymm.KRKSpU(cell, *args, **kwargs)
    return krkspu.KRKSpU(cell, *args, **kwargs)

def KUKSpU(cell, *args, **kwargs):
    for arg in args:
        if hasattr(arg, "kpts_ibz"):
            return kukspu_ksymm.KUKSpU(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if hasattr(kwargs['kpts'], "kpts_ibz"):
            return kukspu_ksymm.KUKSpU(cell, *args, **kwargs)
    return kukspu.KUKSpU(cell, *args, **kwargs)

# *********************************************************************
# KRKS_LO
# *********************************************************************

def get_hybrid_param(ks):
    """
    Get omega, alpha = LR_HFX, hyb = SR_HFX for a KS object.
    """
    mol = getattr(ks, "cell", ks.mol)
    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
    return omega, alpha, hyb

def get_veff(kmf, vj, vk, vxc=None):
    vj = np.asarray(vj)
    vk = np.asarray(vk)
    if vxc is not None:
        vxc = np.asarray(vxc)
        if isinstance(kmf, kuks.KUKS):
            omega, alpha, hyb = get_hybrid_param(kmf)
            if omega > 0.0:
                raise NotImplementedError
            if hyb == 0.0:
                veff = vj[0] + vj[1] + vxc
            else:
                veff = vj[0] + vj[1] - (vk * hyb) + vxc
        elif isinstance(kmf, krks.KRKS):
            omega, alpha, hyb = get_hybrid_param(kmf)
            if omega > 0.0:
                raise NotImplementedError
            if hyb == 0.0:
                veff = vj + vxc
            else:
                veff = vj - (vk * (hyb * 0.5)) + vxc
        else:
            raise ValueError
    else:
        if isinstance(kmf, kuhf.KUHF):
            veff = vj[0] + vj[1] - vk
        elif isinstance(kmf, khf.KRHF):
            veff = vj - vk * 0.5
        else:
            raise ValueError
    return veff

def get_vxc(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    """
    Get vxc, no J or K.
    """
    if isinstance(ks, kuks.KUKS):
        return get_vxc_kuks(ks, cell=cell, dm=dm, dm_last=dm_last, 
                            vhf_last=vhf_last, hermi=hermi, kpts=kpts, 
                            kpts_band=kpts_band)
    elif isinstance(ks, krks.KRKS):
        return get_vxc_krks(ks, cell=cell, dm=dm, dm_last=dm_last, 
                            vhf_last=vhf_last, hermi=hermi, kpts=kpts, 
                            kpts_band=kpts_band)
    else:
        raise ValueError

def prune_small_rho_grids_(ks, cell, dm, grids, kpts):
    rho = ks.get_rho(dm, grids, kpts)
    n = np.dot(rho, grids.weights)
    nelec_core = getattr(ks, "nelec_core", 0)
    if abs(n - cell.nelectron - nelec_core) < NELEC_ERROR_TOL * n:
        rho *= grids.weights
        idx = abs(rho) > ks.small_rho_cutoff / grids.weights.size
        logger.debug(ks, 'Drop grids %d',
                     grids.weights.size - np.count_nonzero(idx))
        grids.coords  = np.asarray(grids.coords [idx], order='C')
        grids.weights = np.asarray(grids.weights[idx], order='C')
        grids.non0tab = grids.make_mask(cell, grids.coords)
    return grids

def get_veff_lo_krks(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                     kpts=None, kpts_band=None):
    """
    Coulomb + XC functional, in LO basis.

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
        Veff = J + Vxc.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10 or abs(alpha) > 1e-10
    
    # ZHC NOTE
    if getattr(ks, "C_ao_lo", None) is not None:
        dm_ao = make_basis.transform_rdm1_to_ao(dm, ks.C_ao_lo)
    else:
        dm_ao = dm
    if getattr(ks, "dm_core_ao", None) is not None:
        dm_ao += ks.dm_core_ao

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        # ZHC NOTE
        #n, exc, vxc = multigrid.nr_rks(ks.with_df, ks.xc, dm, hermi,
        #                               kpts, kpts_band,
        #                               with_j=True, return_j=False)
        n, exc, vxc = multigrid.nr_rks(ks.with_df, ks.xc, dm_ao, hermi,
                                       kpts, kpts_band,
                                       with_j=True, return_j=False)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        
        # ZHC NOTE
        if getattr(ks, "C_ao_lo", None) is not None:
            vxc = make_basis.transform_h1_to_lo(vxc, ks.C_ao_lo)
        return vxc

    # ndim = 3 : dm.shape = (nkpts, nao, nao)
    ground_state = (isinstance(dm, np.ndarray) and dm.ndim == 3 and
                    kpts_band is None)

# For UniformGrids, grids.coords does not indicate whehter grids are initialized
    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            # ZHC NOTE
            #ks.grids = rks.prune_small_rho_grids_(ks, cell, dm, ks.grids, kpts)
            ks.grids = prune_small_rho_grids_(ks, cell, dm_ao, ks.grids, kpts)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        # ZHC NOTE
        #n, exc, vxc = ks._numint.nr_rks(cell, ks.grids, ks.xc, dm, 0,
        #                                kpts, kpts_band)
        n, exc, vxc = ks._numint.nr_rks(cell, ks.grids, ks.xc, dm_ao, 0,
                                        kpts, kpts_band)
        if getattr(ks, "C_ao_lo", None) is not None:
            vxc = make_basis.transform_h1_to_lo(vxc, ks.C_ao_lo)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    weight = 1./len(kpts)
    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
        vxc += vj
    else:
        if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
            ks.with_df._j_only = False
        vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
        vk *= hyb
        if abs(omega) > 1e-10:
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        try:
            vxc += vj - vk * .5
        except: # ZHC NOTE for the case where vxc is a ndarray number.
            vxc = vxc + (vj - vk * .5)

        if ground_state:
            exc -= np.einsum('Kij,Kji', dm, vk).real * .5 * .5 * weight

    if ground_state:
        ecoul = np.einsum('Kij,Kji', dm, vj).real * .5 * weight
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

def get_vxc_krks(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
    """
    XC functional without JK.

    .. note::
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        Vxc : (nkpts, nao, nao) or (*, nkpts, nao, nao) ndarray, no JK.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())
    
    # ZHC NOTE
    if getattr(ks, "C_ao_lo", None) is not None:
        dm = make_basis.transform_rdm1_to_ao(dm, ks.C_ao_lo)
    if getattr(ks, "dm_core_ao", None) is not None:
        dm += ks.dm_core_ao

    hybrid = False
    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        raise NotImplementedError

    # ndim = 3 : dm.shape = (nkpts, nao, nao)
    ground_state = (isinstance(dm, np.ndarray) and dm.ndim == 3 and
                    kpts_band is None)

    # For UniformGrids, grids.coords does not indicate whehter grids are initialized
    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = prune_small_rho_grids_(ks, cell, dm, ks.grids, kpts)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        n, exc, vxc = ks._numint.nr_rks(cell, ks.grids, ks.xc, dm, 0,
                                        kpts, kpts_band)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
    
    # ZHC NOTE
    if getattr(ks, "C_ao_lo", None) is not None:
        vxc = make_basis.transform_h1_to_lo(vxc, ks.C_ao_lo)
    return np.asarray(vxc)

class KRKS_LO(krks.KRKS):
    """
    RKS class adapted for PBCs with k-point sampling, in LO basis and allow frozen core.
    """
    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN', 
                 C_ao_lo=None, dm_core_ao=None):
        krks.KRKS.__init__(self, cell, kpts=kpts)
        self.xc = xc
        if C_ao_lo is not None:
            C_ao_lo = np.asarray(C_ao_lo)
        self.C_ao_lo = C_ao_lo
        
        if dm_core_ao is not None:
            self.dm_core_ao = np.asarray(dm_core_ao)
            ovlp = np.asarray(self.get_ovlp())
            self.nelec_core = np.einsum('kpq, kqp ->', self.dm_core_ao, ovlp).real / len(kpts)
        else:
            self.dm_core_ao = None
            self.nelec_core = 0
        logger.info(self, 'KRKS_LO: nelec_core %s', self.nelec_core)
        self._keys = self._keys.union(["dm_core_ao", "nelec_core", "C_ao_lo"])
    
    get_veff = get_veff_lo_krks
    
    get_vxc = get_vxc_krks

# *********************************************************************
# KUKS_LO
# *********************************************************************

def get_veff_lo_kuks(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                     kpts=None, kpts_band=None):
    '''Coulomb + XC functional for UKS.  See pyscf/pbc/dft/uks.py
    :func:`get_veff` fore more details.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10

    if getattr(ks, "C_ao_lo", None) is not None:
        dm_ao = make_basis.transform_rdm1_to_ao(dm, ks.C_ao_lo)
    else:
        dm_ao = dm
    if getattr(ks, "dm_core_ao", None) is not None:
        dm_ao += ks.dm_core_ao

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        n, exc, vxc = multigrid.nr_uks(ks.with_df, ks.xc, dm_ao, hermi,
                                       kpts, kpts_band,
                                       with_j=True, return_j=False)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        if getattr(ks, "C_ao_lo", None) is not None:
            vxc = make_basis.transform_h1_to_lo(vxc, ks.C_ao_lo)
        return vxc

    # ndim = 4 : dm.shape = ([alpha,beta], nkpts, nao, nao)
    ground_state = (dm.ndim == 4 and dm.shape[0] == 2 and kpts_band is None)

    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = prune_small_rho_grids_(ks, cell, dm_ao, ks.grids, kpts)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        n, exc, vxc = ks._numint.nr_uks(cell, ks.grids, ks.xc, dm_ao, 0,
                                        kpts, kpts_band)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        if getattr(ks, "C_ao_lo", None) is not None:
            vxc = make_basis.transform_h1_to_lo(vxc, ks.C_ao_lo)

    weight = 1./len(kpts)

    if not hybrid:
        vj = ks.get_j(cell, dm[0]+dm[1], hermi, kpts, kpts_band)
        vxc += vj
    else:
        if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
            ks.with_df._j_only = False
        vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
        vj = vj[0] + vj[1]
        vk *= hyb
        if abs(omega) > 1e-10:
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        try:
            vxc += vj - vk
        except:
            vxc = vj - vk

        if ground_state:
            exc -= (np.einsum('Kij,Kji', dm[0], vk[0]) +
                    np.einsum('Kij,Kji', dm[1], vk[1])).real * .5 * weight

    if ground_state:
        ecoul = np.einsum('Kij,Kji', dm[0]+dm[1], vj).real * .5 * weight
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

def get_vxc_kuks(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
    """
    XC functional for UKS, no JK.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    if getattr(ks, "C_ao_lo", None) is not None:
        dm = make_basis.transform_rdm1_to_ao(dm, ks.C_ao_lo)
    if getattr(ks, "dm_core_ao", None) is not None:
        dm += ks.dm_core_ao
    
    hybrid = False
    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        raise NotImplementedError

    # ndim = 4 : dm.shape = ([alpha,beta], nkpts, nao, nao)
    ground_state = (dm.ndim == 4 and dm.shape[0] == 2 and kpts_band is None)

    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = prune_small_rho_grids_(ks, cell, dm, ks.grids, kpts)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        n, exc, vxc = ks._numint.nr_uks(cell, ks.grids, ks.xc, dm, 0,
                                        kpts, kpts_band)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
    if getattr(ks, "C_ao_lo", None) is not None:
        vxc = make_basis.transform_h1_to_lo(vxc, ks.C_ao_lo)
    return np.asarray(vxc)

class KUKS_LO(kuks.KUKS):
    """
    RKS class adapted for PBCs with k-point sampling, in LO basis and allow frozen core.
    """
    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN', 
                 C_ao_lo=None, dm_core_ao=None):
        kuks.KUKS.__init__(self, cell, kpts=kpts)
        self.xc = xc
        if C_ao_lo is not None:
            C_ao_lo = np.asarray(C_ao_lo)
        self.C_ao_lo = C_ao_lo
        
        if dm_core_ao is not None:
            self.dm_core_ao = np.asarray(dm_core_ao)
            ovlp = np.asarray(self.get_ovlp())
            self.nelec_core = np.einsum('skpq, kqp ->', self.dm_core_ao, ovlp).real / len(kpts)
        else:
            self.dm_core_ao = None
            self.nelec_core = 0
        logger.info(self, 'KUKS_LO: nelec_core %s', self.nelec_core)
        self._keys = self._keys.union(["dm_core_ao", "nelec_core", "C_ao_lo"])

    get_veff = get_veff_lo_kuks
    
    get_vxc = get_vxc_kuks
