#!/usr/bin/env python

"""
Lattice and LatticeModel class.

Author:
    Zhi-Hao Cui
    Bo-Xiao Zheng
"""

import numpy as np
import scipy.linalg as la
import itertools as it

from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft

import libdmet.system.hamiltonian as ham
from  libdmet.utils import logger as log
from libdmet.utils.misc import add_spin_dim, max_abs, format_idx, Iterable

from libdmet.system.fourier import *
from libdmet.system.analyze import *
from libdmet.utils.iotools  import *

# ***********************************************************************************
# Lattice class
# *********************************************************************************** 

class Lattice(object):
    def __init__(self, cell, kmesh):
        # cell related:
        self.mol = self.cell = self.supercell = cell
        self.kmesh = kmesh
        self.nscsites = self.nao = self.supercell.nsites = int(self.cell.nao_nr())
        names, coords = zip(*cell._atom)
        self.names, self.coords = np.asarray(names), np.asarray(coords)
         
        self.dim = cell.dimension # dim          
        self.csize = np.asarray(kmesh) # cellsize, kmesh
        self.ncells = np.product(self.csize) # num of cells
        # cells' coordinates
        self.cells = lib.cartesian_prod((np.arange(self.kmesh[0]),
                                         np.arange(self.kmesh[1]),
                                         np.arange(self.kmesh[2]))) 
        # map: cell coord -> cell idx
        self.celldict = dict(zip(map(tuple, self.cells), range(self.ncells))) 
        self.kpts_scaled = self.make_kpts_scaled()
        self.kpts = self.kpts_abs = self.make_kpts_abs()
        self.nkpts = len(self.kpts)
        
        # phase
        self.bigcell, self.phase = get_phase(self.cell, self.kpts, self.kmesh) # +iRk / sqrt(N)
        self.phase_k2R = self.phase.copy().T / np.sqrt(self.nkpts) # +ikR / N
        self.phase_R2k = self.phase_k2R.conj().T * self.nkpts # -iRk
        self.nsites = self.bigcell.nao_nr()

        # basis related:
        self.val_idx  = []
        self.virt_idx = []
        self.core_idx = [] 
        
        # Hamiltonian related:
        self.kmf = None
        self.kmf_lo = None
        self.kmf_sc = None
        self.df = None
        self.Ham = None
        self.C_ao_lo = None

        self.hcore_ao_k = None
        self.fock_ao_k = None
        self.rdm1_ao_k = None
        self.vxc_ao_k = None

        self.hcore_lo_k = None
        self.fock_lo_k = None
        self.rdm1_lo_k = None
        self.vxc_lo_k = None

        self.hcore_lo_R = None
        self.fock_lo_R = None
        self.rdm1_lo_R = None
        self.vxc_lo_R = None

        self.JK_imp = None
        self.JK_emb = None # will be evaluated when construct ImpHam
        self.JK_core = None # will be evaluated when construct ImpHam

        # initialization
        self.has_Ham = False
        self.restricted = None
        self.is_model = False

    """
    oribtal information.
    """
    @property
    def ncore(self):
        return len(self.core_idx)
    
    @property
    def nval(self):
        return len(self.val_idx)
    
    @property
    def nvirt(self):
        return len(self.virt_idx)
    
    @property
    def nimp(self):
        return self.nval + self.nvirt
    
    limp = nimp
    
    @property
    def imp_idx(self):
        return list(self.val_idx) + list(self.virt_idx)
    
    @property
    def has_orb_info(self):
        return (self.nimp != 0)

    def set_val_virt_core(self, val, virt, core):
        """
        Set valence, virtual and core indices.
        
        Args:
            val : valence indices, list or number
            virt: virtual indices, list or number
            core: core    indices, list or number
        """
        if isinstance(core, Iterable):
            self.core_idx = list(core)
        else:
            self.core_idx = list(range(0, core))
        
        if isinstance(val, Iterable):
            self.val_idx = list(val)
        else:
            self.val_idx  = list(range(self.ncore, self.ncore + val))
        
        if isinstance(virt, Iterable):
            self.virt_idx = list(virt)
        else:
            self.virt_idx = list(range(self.ncore + self.nval, \
                    self.ncore + self.nval + virt))

        if self.ncore + self.nval + self.nvirt != self.nao:
            log.warn("ncore (%s) + nval (%s) + nvirt (%s) != nao (%s), \n" 
                     "set_val_virt_core may be incorrect.", 
                     self.ncore, self.nval, self.nvirt, self.nao)
        log.info("ncore: %s, nval: %s, nvirt: %s", 
                 self.ncore, self.nval, self.nvirt)
        log.debug(0, "core_idx : %s", format_idx(self.core_idx))
        log.debug(0, "val_idx  : %s", format_idx(self.val_idx))
        log.debug(0, "virt_idx : %s", format_idx(self.virt_idx))
    
    """
    cell functions:
    """
    def lattice_vectors(self):
        return self.cell.lattice_vectors()

    def real_coords(self):
        return self.coords

    def frac_coords(self):
        return real2frac(self.lattice_vectors(), self.coords)

    def make_kpts_scaled(self):
        return make_kpts_scaled(self.kmesh)

    def make_kpts_abs(self):
        if self.kpts_scaled is None:
            kpts_scaled = self.make_kpts_scaled()
        else:
            kpts_scaled = self.kpts_scaled
        kpts_abs = self.cell.get_abs_kpts(kpts_scaled)
        return kpts_abs

    def cell_idx2pos(self, idx):
        return self.cells[idx % self.ncells]

    def cell_pos2idx(self, pos):
        return self.celldict[tuple(pos % self.csize)]

    def add(self, i, j):
        return self.cell_pos2idx(self.cell_idx2pos(i) + self.cell_idx2pos(j))

    def subtract(self, i, j):
        return self.cell_pos2idx(self.cell_idx2pos(i) - self.cell_idx2pos(j))
    
    """
    Functions on matrices in the system:
    """
    def FFTtoK(self, A):
        return FFTtoK(A, self.kmesh)
    
    def FFTtoT(self, B, tol=IMAG_DISCARD_TOL):
        return FFTtoT(B, self.kmesh, tol=tol)

    def k2R(self, A, tol=IMAG_DISCARD_TOL):
        return k2R(A, self.kmesh, tol=tol)
    
    def R2k(self, B):
        return R2k(B, self.kmesh)
    
    def k2R_H2(self, H2_k, tol=IMAG_DISCARD_TOL):
        return k2R_H2(H2_k, self.phase_k2R, tol=tol)

    def R2k_H2(self, H2_R):
        return R2k_H2(H2_R, self.phase_R2k)

    def expand(self, A, dense=False):
        """
        Expand ncells * nscsites * nscsites translational invariant matrix to full
        nsites * nsites matrix (stripe to full)
        """
        assert A.shape[-3] == self.ncells
        nscsites = A.shape[-1]
        nsites = A.shape[-1] * A.shape[-3] 
        if A.ndim == 3:
            bigA = np.zeros((nsites, nsites), dtype=A.dtype)
            if dense:
                for i, j in it.product(range(self.ncells), repeat=2):
                    idx = self.add(i, j)
                    bigA[idx*nscsites:(idx+1)*nscsites, j*nscsites:(j+1)*nscsites] = A[i]
            else:
                nonzero = filter(lambda j: not np.allclose(A[j], 0.), range(self.ncells))
                for i, j in it.product(nonzero, range(self.ncells)):
                    idx = self.add(i, j)
                    bigA[idx*nscsites:(idx+1)*nscsites, j*nscsites:(j+1)*nscsites] = A[i]
        elif A.ndim == 4:
            spin = A.shape[0]
            bigA = np.zeros((spin, nsites, nsites), dtype=A.dtype)
            if dense:
                for i, j in it.product(range(self.ncells), repeat=2):
                    idx = self.add(i, j)
                    bigA[:, idx*nscsites:(idx+1)*nscsites, j*nscsites:(j+1)*nscsites] = A[:, i]
            else:
                nonzero = filter(lambda j: not np.allclose(A[:, j], 0.), range(self.ncells))
                for i, j in it.product(nonzero, range(self.ncells)):
                    idx = self.add(i, j)
                    bigA[:, idx*nscsites:(idx+1)*nscsites, j*nscsites:(j+1)*nscsites] = A[:, i]
        else:
            raise ValueError("unknown shape of A, %s" % A.shape)
        return bigA
    
    def extract_stripe(self, A):
        """
        Full to stripe, inverse function of expand.
        """
        ncells = self.ncells
        nscsites = A.shape[-1] // ncells
        if A.ndim == 2:
            return A.reshape((ncells, nscsites, ncells, nscsites))[:, :, 0]
        elif A.ndim == 3:
            spin = A.shape[0]
            return A.reshape((spin, ncells, nscsites, ncells, nscsites))[:, :, :, 0]
        else:
            raise ValueError("unknown shape of A, %s"%A.shape)
    
    def expand_orb(self, C):
        """
        Expand ncells * nao * nmo translational invariant wannier function C^{T, 0}_{pm}
        to C^{T, R}_{pm}
        """
        ncells = C.shape[-3]
        assert ncells == self.ncells
        nao = C.shape[-2]
        nmo = C.shape[-1]
        nao_sc = nao * ncells
        nmo_sc = nmo * ncells
        if C.ndim == 3:
            bigC = np.zeros((nao_sc, nmo_sc), dtype=C.dtype)
            for i, j in it.product(range(self.ncells), repeat=2):
                idx = self.add(i, j)
                bigC[idx*nao:(idx+1)*nao, j*nmo:(j+1)*nmo] = C[i]
        elif C.ndim == 4:
            spin = C.shape[0]
            bigC = np.zeros((spin, nao_sc, nmo_sc), dtype=C.dtype)
            for i, j in it.product(range(self.ncells), repeat=2):
                idx = self.add(i, j)
                bigC[:, idx*nao:(idx+1)*nao, j*nmo:(j+1)*nmo] = C[:, i]
        else:
            raise ValueError("unknown shape of C, %s"%C.shape)
        return bigC
    
    def transpose(self, A):
        """
        return the transpose of ncells * nscsites * nscsites translational invariant matrix
        """
        AT = np.zeros_like(A)
        if A.ndim == 3:
            for n in range(self.ncells):
                AT[n] = A[self.cell_pos2idx(-self.cell_idx2pos(n))].T
        elif A.ndim == 4:
            for s in range(A.shape[0]):
                for n in range(self.ncells):
                    AT[s, n] = A[s, self.cell_pos2idx(-self.cell_idx2pos(n))].T
        else:
            raise ValueError("unknown shape of A, %s"%A.shape)
            
        return AT

    """
    basis functions:
    """
    def k2R_basis(self, basis_k):
        """
        Transform k-basis to R-basis
        1/Nk factor for k to R.
        """
        return self.k2R(basis_k)
    
    def R2k_basis(self, basis_R):
        """
        Transform R-basis to k-basis
        No factor for R to k.
        """
        return self.R2k(basis_R)
    
    """
    Hamiltonian functions:
    """
    def set_Ham(self, kmf, df, C_ao_lo, eri_symmetry=1, 
                ovlp=None, hcore=None, rdm1=None, fock=None, veff=None, vhf=None,
                vj=None, vk=None, vxc=None, use_hcore_as_emb_ham=False, H0=0.0):
        """
        Set Hamiltonian: 
        hcore, fock, ovlp, rdm1, (vhf), (vxc) in kAO, kLO, RLO, 
        and eri_symmetry.
        """
        from libdmet.routine import pbc_helper as pbc_hp
        log.info("\n" + "-" * 79)
        log.info("Set DMET mean-field Hamiltonian.")

        self.kmf = kmf
        self.df  = df
        self.C_ao_lo = np.asarray(C_ao_lo)
        
        # get: mo_coeff, ovlp, hcore, rdm1, fock
        log.info("set_Ham: set hcore, ovlp, rdm1")
        if ovlp is None:
            ovlp  = self.kmf.get_ovlp()
        if hcore is None:
            hcore = self.kmf.get_hcore()
        if rdm1 is None:
            rdm1  = self.kmf.make_rdm1()
        ovlp = np.asarray(ovlp)
        hcore = np.asarray(hcore)
        rdm1 = np.asarray(rdm1)
        
        # vj, vk
        log.info("set_Ham: set vj, vk")
        if vj is None or vk is None:
            vj, vk = self.kmf.get_jk(dm_kpts=rdm1)
        vj, vk = np.asarray(vj), np.asarray(vk)
        
        # veff, vhf
        log.info("set_Ham: set veff, vhf")
        if isinstance(self.kmf, pdft.rks.KohnShamDFT): # DFT
            if vxc is None:
                vxc = pbc_hp.get_vxc(self.kmf, dm=rdm1)
            if veff is None:
                veff = pbc_hp.get_veff(kmf, vj, vk, vxc)
            if vhf is None:
                vhf  = pbc_hp.get_veff(kmf, vj, vk, vxc=None)
            self.kmf_sc = self.kmf.__class__(self.bigcell.copy()).density_fit()
            self.kmf_sc.xc = self.kmf.xc
            self.kmf_sc.exxdiv = self.kmf.exxdiv
            # reset nelec for prune small grids
            self.kmf_sc.cell.nelectron = (self.ncore + self.nval) * 2
            self.kmf_sc.cell.rcut = 1e-5
        else: # HF
            if vhf is None:
                vhf = pbc_hp.get_veff(kmf, vj, vk, vxc=None)
            if veff is None:
                veff = vhf
        # fock, fock_hf
        if fock is None:
            fock = hcore + veff
        fock_hf = hcore + vhf

        self.ovlp_ao_k  = np.asarray(ovlp)
        self.hcore_ao_k = np.asarray(hcore)
        self.rdm1_ao_k  = np.asarray(rdm1)
        self.fock_ao_k  = np.asarray(fock)
        self.fock_hf_ao_k = np.asarray(fock_hf)
        
        self.vj_ao_k   = np.asarray(vj)
        self.vk_ao_k   = np.asarray(vk)
        self.veff_ao_k = np.asarray(veff)
        self.vhf_ao_k  = np.asarray(vhf)
        if vxc is not None:
            self.vxc_ao_k = np.asarray(vxc)
        
        # set spin and restricted
        if self.C_ao_lo.ndim == 3:
            self.spin = 1
            self.restricted = True
        else:
            self.spin = C_ao_lo.shape[0]
            self.restricted = (self.spin == 1)
         
        # set eri symmetry
        self.eri_symmetry = eri_symmetry
        assert self.eri_symmetry in [1, 4, 8]
        if not self.restricted: # unrestricted does not support 8-fold symmetry
            assert self.eri_symmetry != 8

        # transform hcore, fock, rdm1 to LO
        self.transform_obj_to_lo()
        self.H0 = H0
        self.has_Ham = True
        self.use_hcore_as_emb_ham = use_hcore_as_emb_ham
        if self.use_hcore_as_emb_ham:
            log.warn("You are using hcore to construct embedding Hamiltonian...")
        log.info("-" * 79 + "\n")
    
    def set_Ham_model(self, Ham, rdm1=None, fock=None, ovlp=None, \
            eri_symmetry=1, vj=None, vk=None, vxc=None, \
            use_hcore_as_emb_ham=True):
        # TODO DFT with model.
        self.Ham = Ham
        self.hcore_lo_R = self.Ham.getH1()
        self.hcore_lo_k = self.R2k(self.hcore_lo_R)

        if ovlp is None:
            self.ovlp_lo_R = np.zeros((self.nkpts, self.nao, self.nao))
            self.ovlp_lo_R[0] = np.eye(self.nao)
        else:
            self.ovlp_lo_R = ovlp
        self.ovlp_lo_k = self.R2k(self.ovlp_lo_R)
        
        if fock is None:
            self.fock_lo_R = self.Ham.getFock()
        else:
            self.fock_lo_R = fock
        self.fock_lo_k = self.R2k(self.fock_lo_R)
        
        self.rdm1_lo_R = rdm1
        if self.rdm1_lo_R is not None:
            self.rdm1_lo_k = self.R2k(self.rdm1_lo_R)

        self.vxc_lo_R = vxc
        if self.vxc_lo_R is not None:
            self.vxc_lo_k = self.R2k(self.vxc_lo_R)

        self.check_imag()
        self.eri_symmetry = eri_symmetry
        self.use_hcore_as_emb_ham = use_hcore_as_emb_ham
        if self.use_hcore_as_emb_ham:
            log.warn("You are using hcore to construct embedding Hamiltonian...")
        self.has_Ham = True

        # check the format of H2
        self.H2_format = self.Ham.H2_format
        log.info("Lattice H2 format: %s, H2 shape: %s", \
                self.H2_format, self.Ham.H2.shape)
        self.H0 = self.Ham.getH0()
    
    setHam = set_Ham
    
    setHam_model = set_Ham_model

    def update_Ham(self, rdm1_lo_R, veff=None, vhf=None, **kwargs):
        """
        Update fock matrix based on the new rdm from DMET,
        essentially follow Knizia JCTC 2013
        """
        from libdmet.basis_transform import make_basis 
        log.info("Update DMET mean-field Hamiltonian.")
        self.rdm1_lo_R = rdm1_lo_R
        self.rdm1_lo_k = self.R2k(self.rdm1_lo_R)
        self.rdm1_ao_k = make_basis.transform_rdm1_to_ao(self.rdm1_lo_k, self.C_ao_lo)
        if veff is None and vhf is None:
            vj = vk = None
        else:
            # veff can be passed from out of the function
            # no need to update vj and vk
            vj = self.vj_ao_k
            vk = self.vk_ao_k
        
        # vj, vk, vxc, fock will be re-evaluated by rdm1_ao_k.
        self.set_Ham(self.kmf, self.df, self.C_ao_lo, self.eri_symmetry, \
            ovlp=self.ovlp_ao_k, hcore=self.hcore_ao_k, rdm1=self.rdm1_ao_k, \
            fock=None, veff=veff, vhf=vhf, vj=vj, vk=vk, vxc=None, \
            use_hcore_as_emb_ham=self.use_hcore_as_emb_ham, H0=self.H0)
    
    def transform_obj_to_lo(self):
        """
        Transform objects (hcore, fock, rdm1) to klo and RLO basis.
        """
        from libdmet.basis_transform import make_basis 
        # transform to LO basis
        self.hcore_lo_k = \
                make_basis.transform_h1_to_lo(self.hcore_ao_k, self.C_ao_lo)
        self.ovlp_lo_k = \
                make_basis.transform_h1_to_lo(self.ovlp_ao_k, self.C_ao_lo)
        self.fock_lo_k = \
                make_basis.transform_h1_to_lo(self.fock_ao_k, self.C_ao_lo)
        self.veff_lo_k = \
                make_basis.transform_h1_to_lo(self.veff_ao_k, self.C_ao_lo)
        self.vhf_lo_k = \
                make_basis.transform_h1_to_lo(self.vhf_ao_k, self.C_ao_lo)
        self.rdm1_lo_k = \
                make_basis.transform_rdm1_to_lo(self.rdm1_ao_k, self.C_ao_lo, self.ovlp_ao_k)

        # Add extra dimension for restricted case
        self.hcore_lo_k = add_spin_dim(self.hcore_lo_k, self.spin)
        self.fock_lo_k = add_spin_dim(self.fock_lo_k, self.spin)
        self.veff_lo_k = add_spin_dim(self.veff_lo_k, self.spin)
        self.vhf_lo_k = add_spin_dim(self.vhf_lo_k, self.spin)
        self.rdm1_lo_k = add_spin_dim(self.rdm1_lo_k, self.spin)
        
        # FFT to real basis (stripe)
        self.hcore_lo_R = self.k2R(self.hcore_lo_k)
        self.ovlp_lo_R = self.k2R(self.ovlp_lo_k)
        self.fock_lo_R = self.k2R(self.fock_lo_k)
        self.veff_lo_R = self.k2R(self.veff_lo_k)
        self.vhf_lo_R = self.k2R(self.vhf_lo_k)
        self.rdm1_lo_R = self.k2R(self.rdm1_lo_k)
        
        # DFT vxc
        if self.vxc_ao_k is not None:
            self.vxc_lo_k = make_basis.transform_h1_to_lo(self.vxc_ao_k, self.C_ao_lo)
            self.vxc_lo_k = add_spin_dim(self.vxc_lo_k, self.spin)
            self.vxc_lo_R = self.k2R(self.vxc_lo_k)
        self.check_imag()

    def check_imag(self):
        if self.hcore_lo_R is not None:
            imag_hcore = max_abs(self.hcore_lo_R.imag)
            if imag_hcore < IMAG_DISCARD_TOL:
                self.hcore_lo_R = self.hcore_lo_R.real
        else:
            imag_hcore = 0.0
        
        if self.fock_lo_R is not None:
            imag_fock = max_abs(self.fock_lo_R.imag)
            if imag_fock < IMAG_DISCARD_TOL:
                self.fock_lo_R = self.fock_lo_R.real
        else:
            imag_fock = 0.0

        if self.rdm1_lo_R is not None:
            imag_rdm1 = max_abs(self.rdm1_lo_R.imag)
            if imag_rdm1 < IMAG_DISCARD_TOL:
                self.rdm1_lo_R = self.rdm1_lo_R.real
        else:
            imag_rdm1 = 0.0
        
        if self.vxc_lo_R is not None:
            imag_vxc = max_abs(self.vxc_lo_R.imag)
            if imag_vxc < IMAG_DISCARD_TOL:
                self.vxc_lo_R = self.vxc_lo_R.real
        else:
            imag_vxc = 0.0

        log.info("Imag of LO hcore: %s, fock: %s, rdm1: %s, vxc: %s", \
                imag_hcore, imag_fock, imag_rdm1, imag_vxc)

    def update_lo(self, C_ao_lo):
        """
        Update lo.
        """
        log.info("Update lo.")
        
        self.C_ao_lo = np.asarray(C_ao_lo)
        self.transform_obj_to_lo()
    
    def getH1(self, kspace=True):
        if kspace:
            return self.hcore_lo_k
        else:
            return self.hcore_lo_R

    def getFock(self, kspace=True):
        if kspace:
            return self.fock_lo_k
        else:
            return self.fock_lo_R
    
    def get_ovlp(self, kspace=True):
        if kspace:
            return self.ovlp_lo_k
        else:
            return self.ovlp_lo_R

    def get_JK_emb(self):
        return self.JK_emb
    
    def get_JK_core(self):
        return self.JK_core

    def getH2(self, kpts=None, compact=False, kspace=True, use_Ham=False):
        if use_Ham or self.is_model: # use H2 from Ham object.
            if kspace:
                raise NotImplementedError
            else:
                return self.Ham.getH2()
        else:
            from libdmet.routine import pbc_helper as pbc_hp
            H2_k = pbc_hp.get_eri_7d(self.cell, self.df, kpts=kpts, compact=compact)
            if kspace:
                return H2_k
            else:
                return self.R2k_H2(H2_k)
    
    def getImpJK(self):
        if self.JK_imp is not None:
            return self.JK_imp
        elif self.Ham is not None:
            return self.Ham.getImpJK()
        else:
            return None
     
    def getH0(self):
        return self.H0
    
    check_lo = check_lo

    symmetrize_lo = symmetrize_lo
    
    symmetrize_kmf = symmetrize_kmf

    analyze = analyze
    
    @staticmethod
    def analyze_kmo(kmf, **kwargs):
        return analyze_kmo(kmf, **kwargs)
    
    mulliken_lo = mulliken_lo

    mulliken_lo_R0 = mulliken_lo_R0

    get_JK_imp = getImpJK

# ***********************************************************************************
# Lattice Models
# *********************************************************************************** 

class LatticeModel(Lattice):
    def __init__(self, sc, size):
        self.supercell = sc
        self.dim = sc.dim
        self.csize = self.kmesh = np.array(size)
        self.size = np.dot(np.diag(self.csize), sc.size)
        self.ncells = np.product(self.csize)
        
        # a fake cell for pyscf functions 
        if log.Level[log.verbose] > log.Level["DEBUG2"]:
            verbose = 5
        elif log.Level[log.verbose] > log.Level["RESULT"]:
            verbose = 4
        else:
            verbose = 3
        self.mol = self.cell = pgto.M(
             unit='A',
             a=np.eye(3),
             verbose=verbose, 
             dump_input=False)

        kpts_scaled = self.make_kpts_scaled()
        self.nkpts, shape_tmp = kpts_scaled.shape
        # if low dimensional system, pad kpts to (nkpts, 3)
        self.kpts_scaled = np.zeros((self.nkpts, 3))
        self.kpts_scaled[:, :shape_tmp] = kpts_scaled
        self.kpts = self.kpts_abs = self.make_kpts_abs()
        self.nsites = sc.nsites * self.ncells

        self.cells, self.sites = translateSites(sc.sites, sc.size, size)
        self.names = sc.names * self.ncells

        self.celldict = dict(zip(map(tuple, self.cells), range(self.ncells)))
        self.sitedict = dict(zip(map(tuple, self.sites), range(self.nsites)))
        self.neighborDist = []
        
        self.nao = self.nscsites = self.supercell.nsites
        self.val_idx  = list(range(self.nao))
        self.virt_idx = []
        self.core_idx = []
        
        self.kmf = None
        self.kmf_lo = None
        self.eri_symmetry = None
        
        # model set_Ham only accept Ham object
        self.set_Ham = self.setHam = self.set_Ham_model
        self.JK_imp = None
        self.is_model = True
        self.has_Ham = False

    def __str__(self):
        r = self.supercell.__str__()
        r += "Lattice Shape\n%s\n" % self.csize
        r += "Number of SuperCells: %4d\n" % self.ncells
        r += "Number of Sites:      %4d\n" % self.nsites
        return r

    """
    functions on translations
    """
    def kpoints(self):
        return [np.fft.fftfreq(self.csize[d], \
                1.0/(2.0*np.pi)) for d in range(self.dim)]
    
    def site_idx2pos(self, idx):
        #return self.sites[idx % self.ncells]
        return self.sites[idx]

    def site_pos2idx(self, pos):
        return self.sitedict[tuple(pos % np.diag(self.size))]

    """
    get neighbor sites
    """
    def neighbor(self, dis=1.0, max_range=1, sitesA=None, sitesB=None, \
            search_range=1):
        # siteA, siteB are indices, not positions
        if sitesA is None:
            sitesA = range(self.nsites)
        if sitesB is None:
            sitesB = range(self.nsites)

        nscsites = self.nscsites
        cellshifts = [self.cell_pos2idx(np.asarray(s)) for s in
            it.product(range(-max_range, max_range+1), repeat=self.dim)]
        
        shifts = [np.asarray(s) for s in it.product(range(-search_range, \
                search_range + 1), repeat=self.dim)]

        neighbors = []
        for siteA in sitesA:
            cellA = siteA // nscsites
            cellB = [self.add(cellA, x) for x in cellshifts]
            psitesB = list(set(sitesB) & \
                set(it.chain.from_iterable(map(lambda c:range(c*nscsites, \
                (c+1)*nscsites), cellB))))

            for siteB in psitesB:
                for shift in shifts:
                    if abs(la.norm(self.sites[siteA] - self.sites[siteB] \
                            - np.dot(shift, self.size)) - dis) < 1e-5:
                        neighbors.append((siteA, siteB))
                        break
        return neighbors
    
    def update_Ham(self, rdm1_lo_R, fock_lo_k=None, ghf=False, **kwargs):
        """
        Update fock matrix based on the new rdm from DMET,
        essentially follow Knizia JCTC 2013
        """
        from libdmet.routine import pbc_helper as pbc_hp
        log.info("Update DMET mean-field Hamiltonian.")
        assert self.has_Ham
        
        # update rdm1
        self.rdm1_lo_R = rdm1_lo_R
        if self.rdm1_lo_R.ndim == 3:
            self.rdm1_lo_R = self.rdm1_lo_R[np.newaxis]
        self.rdm1_lo_k = self.R2k(self.rdm1_lo_R)
        
        # update fock
        if fock_lo_k is None:
            # based on the format of H2
            if self.H2_format == "local":
                eri = self.getH2(compact=False, kspace=False)
                vj, vk = pbc_hp.get_jk_from_eri_local(eri, self.rdm1_lo_k)
            elif self.H2_format == "nearest":
                eri = self.getH2(compact=False, kspace=False)
                vj, vk = pbc_hp.get_jk_from_eri_nearest(eri, \
                        self.rdm1_lo_k, self)
            elif self.H2_format == "full":
                eri = self.getH2(compact=False, kspace=True)
                vj, vk = pbc_hp.get_jk_from_eri_7d(eri, self.rdm1_lo_k)
            else:
                raise ValueError
            
            spin = self.rdm1_lo_R.shape[0]
            if spin == 1:
                JK = vj - vk * 0.5 
            else:
                JK = vj[0] + vj[1] - vk
            self.fock_lo_k = self.hcore_lo_k + JK
        else:
            self.fock_lo_k = fock_lo_k
        self.fock_lo_R = self.k2R(self.fock_lo_k)
        self.check_imag()
    
    def getH2(self, compact=False, kspace=False, use_Ham=True):
        assert use_Ham
        if kspace:
            raise NotImplementedError
        else:
            return self.Ham.getH2()

class UnitCell(object):
    def __init__(self, size, sites):
        # unit cell shape
        self.size = np.array(size)
        log.eassert(self.size.shape[0] == self.size.shape[1], \
                "Invalid unitcell constants")
        self.dim = self.size.shape[0]
        self.sites = []
        self.names = []
        for s in sites:
            log.eassert(s[0].shape == (self.dim,), \
                    "Invalid position for the site")
            self.sites.append(np.asarray(s[0]))
            self.names.append(s[1])
        self.nsites = len(self.sites)
        self.sitedict = dict(zip(map(tuple, self.sites), range(self.nsites)))

    def __str__(self):
        r = "UnitCell Shape\n%s\nSites:\n" % self.size
        for i in range(len(self.sites)):
          r += "%-10s%-10s\t" % (self.names[i], self.sites[i])
          if (i+1)%6 == 0:
            r+= "\n"
        r += "\n\n"
        return r

class SuperCell(object):
    def __init__(self, uc, size): # uc is unitcell
        self.unitcell = uc
        self.dim = uc.dim
        self.csize = np.array(size)
        self.size = np.dot(np.diag(self.csize), uc.size)
        self.ncells = np.product(self.csize)
        self.nsites = uc.nsites * self.ncells

        self.cells, self.sites = translateSites(uc.sites, uc.size, size)
        self.names = uc.names * self.ncells

        self.celldict = dict(zip(map(tuple, self.cells), range(self.ncells)))
        self.sitedict = dict(zip(map(tuple, self.sites), range(self.nsites)))

    def __str__(self):
        r = self.unitcell.__str__()
        r += "SuperCell Shape\n"
        r += self.size.__str__()
        r += "\nNumber of Sites:%d\n" % self.nsites
        r += "\n"
        return r

def translateSites(baseSites, usize, csize):
    # csize = [3,3], then cells = [0,0], [0,1], [0,2], [0,3], [1,0], ..., [3,3]
    cells = [np.asarray(x) for x in it.product(*tuple(map(range, csize)))]
    sites = list(it.chain.from_iterable(map(lambda c: \
            map(lambda s: np.dot(c, usize) + s, baseSites), cells)))
    return cells, sites

def BipartiteSquare(impsize):
    subA = []
    subB = []
    for idx, pos in enumerate(it.product(*map(range, impsize))):
        if np.sum(pos) % 2 == 0:
            subA.append(idx)
        else:
            subB.append(idx)
    log.eassert(len(subA) == len(subB), \
        "The impurity cannot be divided into two sublattices")
    return subA, subB

# ***********************************************************************************
# Some systems.
# *********************************************************************************** 

def ChainLattice(length, scsites):
    """
    1D 1-band model.
    """
    log.eassert(length % scsites == 0, \
            "incompatible lattice and supercell sizes")
    uc = UnitCell(np.eye(1), [(np.array([0]), "X")])
    sc = SuperCell(uc, np.asarray([scsites]))
    lat = LatticeModel(sc, np.asarray([length // scsites]))
    lat.neighborDist = [1.0, 2.0, 3.0]
    return lat

def SquareLattice(lx, ly, scx, scy):
    """
    2D 1-band model.
    """
    log.eassert(lx % scx == 0 and ly % scy == 0, \
            "incompatible lattice and supercell sizes")
    uc = UnitCell(np.eye(2), [(np.array([0, 0]), "X")])
    sc = SuperCell(uc, np.asarray([scx, scy]))
    lat = LatticeModel(sc, np.asarray([lx // scx, ly // scy]))
    lat.neighborDist = [1.0, np.sqrt(2.0), 2.0]
    return lat

def SquareAFM(lx, ly, scx, scy):
    """
    2D 1-band model, AFM cell.
        A - - - A
        |       |
        -   B   -
        |       |
        A - - - A

    """
    log.eassert(lx % scx == 0 and ly % scy == 0, \
            "incompatible latticeand supercell sizes")
    uc = UnitCell(np.eye(2) * np.sqrt(2.0), 
        [(np.zeros(2),                        "A"),
         (np.ones (2) * (np.sqrt(2.0) * 0.5), "B")])
    sc = SuperCell(uc, np.asarray([scx, scy]))
    lat = LatticeModel(sc, np.asarray([lx // scx, ly // scy]))
    lat.neighborDist = [1.0, np.sqrt(2.0), 2.0]
    return lat

def Square3Band(lx, ly, scx, scy):
    """
    2D 3-band model, 1 CuO2 per unit cell.
        Cu - O - Cu
        |         |
       2O         O
        |         |
       0Cu -1O - Cu
        0    1    2
    """
    log.eassert(lx % scx == 0 and ly % scy == 0, \
            "incompatible latticeand supercell sizes")
    uc = UnitCell(np.eye(2) * 2.0, 
        [(np.array([0.0, 0.0]), "Cu"),
         (np.array([1.0, 0.0]),  "O"),
         (np.array([0.0, 1.0]),  "O")])
    sc = SuperCell(uc, np.asarray([scx, scy]))
    lat = LatticeModel(sc, np.asarray([lx // scx, ly // scy]))
    lat.neighborDist = [1.0, np.sqrt(2.0), 2.0]
    return lat

def Square3BandAFM(lx, ly, scx, scy, symm=True):
    """
    2D 3-band model, AFM cell with 2 CuO2 per unit cell.
    if symm == True:
                        vec2
                    -O+

               +          +
              4O         5O
               -          -     
               -          -
         -O+ +0Cu+ -3O+ +1Cu+ -O+ 
               -          -
               +          +   
               O          O
               -          -

                   -2O+
                        vec1
    if symm == False:
                        vec2
                    -O+

               +          +
              4O          O
               -          -     
               -          -
         -2O+ +0Cu+ -3O+ +1Cu+ -O+ 
               -          -
               +          +   
              5O          O
               -          -

                   -O+
                        vec1

    """
    log.eassert(lx % scx == 0 and ly % scy == 0, \
            "incompatible latticeand supercell sizes")
    if symm:
        uc = UnitCell(np.array([[2.0, -2.0], \
                                [2.0,  2.0]]), 
            [(np.array([1.0,  0.0]), "Cu"),
             (np.array([3.0,  0.0]), "Cu"),
             (np.array([2.0, -2.0]), "O"),
             (np.array([2.0,  0.0]), "O"),
             (np.array([1.0,  1.0]), "O"),
             (np.array([3.0,  1.0]), "O")])
    else: 
        uc = UnitCell(np.array([[2.0, -2.0], \
                                [2.0,  2.0]]), 
            [(np.array([1.0,  0.0]), "Cu"),
             (np.array([3.0,  0.0]), "Cu"),
             (np.array([0.0,  0.0]), "O"),
             (np.array([2.0,  0.0]), "O"),
             (np.array([1.0,  1.0]), "O"),
             (np.array([1.0, -1.0]), "O")])

    sc = SuperCell(uc, np.asarray([scx, scy]))
    lat = LatticeModel(sc, np.asarray([lx // scx, ly // scy]))
    lat.neighborDist = [1.0, np.sqrt(2.0), 2.0]
    return lat

def Square3BandSymm(lx, ly, scx=1, scy=1):
    """
    2D 3-band model, 2x2 symmetric supercells.
             |
            4O
             |          |
          - 3Cu - 5O - 6Cu - 7O -
             |          |
            2O         8O
             |          |
     - 1O - 0Cu - 11O - 9Cu -
             |          |
                      10O
                        |
    """
    uc = UnitCell(np.eye(2) * 4.0, [
      (np.array([1.0, 1.0]), "Cu"),
      (np.array([0.0, 1.0]), "O"),
      (np.array([1.0, 2.0]), "O"),
      (np.array([1.0, 3.0]), "Cu"),
      (np.array([1.0, 4.0]), "O"),
      (np.array([2.0, 3.0]), "O"),
      (np.array([3.0, 3.0]), "Cu"),
      (np.array([4.0, 3.0]), "O"),
      (np.array([3.0, 2.0]), "O"),
      (np.array([3.0, 1.0]), "Cu"),
      (np.array([3.0, 0.0]), "O"),
      (np.array([2.0, 1.0]), "O"),
    ])
    sc = SuperCell(uc, np.asarray([scx, scy]))
    lat = LatticeModel(sc, np.asarray([lx, ly]))
    lat.neighborDist = [1.0, np.sqrt(2.0), 2.0]
    return lat

def HoneycombLattice(lx, ly, scx, scy):
    log.error("honeycomb lattice not implemented yet")

def CubicLattice(lx, ly, lz, scx, scy, scz):
    """
    3D 1-band model.
    """
    log.eassert(lx % scx == 0 and ly % scy == 0 and lz % scz == 0, \
            "incompatible lattice and supercell size")
    uc = UnitCell(np.eye(3), [(np.array([0.0, 0.0, 0.0]), "X")])
    sc = SuperCell(uc, np.asarray([scx, scy, scz]))
    lat = LatticeModel(sc, np.asarray([lx // scx, ly // scy, lz // scz]))
    lat.neighborDist = [1.0, np.sqrt(2.0), np.sqrt(3.0)]
    return lat

def HChain(nH=2, R=1.5, vac=10.0, shift=np.zeros(3)):
    """
    Creat a cell with hydrogen chain.
    
    Args:
        nH: number of H.
        R: bond length.
        vac: vacuum on x and y direction.
        shift: shift of origin.

    Returns:
        cell: H-chain
    """
    length = nH * R
    cell = pgto.Cell()
    cell.a = np.diag([vac, vac, length])
    cell.atom = [['H', np.array([shift[0], shift[1], i*R + shift[2]])] \
            for i in range(nH)]
    cell.unit = 'A'
    return cell

if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    import pyscf.pbc.scf as pscf
    np.set_printoptions(3, linewidth=1000)

    # test ab initio systems
    cell = pgto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''

    cell.atom = ''' H 0.      0.      0.
                    H 0.      0.      1.5 '''
    #cell.basis = 'sto3g'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 4
    cell.precision = 1e-15
    cell.build(unit='Angstrom')
   
    # Latice class
    kmesh = [1, 1, 3]  
    Lat = Lattice(cell, kmesh)
    nscsites = Lat.nscsites
    kpts_abs = Lat.kpts_abs

    mf = pscf.KRHF(cell, kpts_abs)
    mf = mf.density_fit()
    mf.conv_tol = 1e-15
    mf.kernel()

    dm_k = mf.make_rdm1()
    dm_R = Lat.k2R(dm_k)
    dm_R_full = Lat.expand(dm_R)
    
    # supercell calculation
    scell = Lat.bigcell
    mf = pscf.KRHF(scell)
    mf = mf.density_fit()
    mf.conv_tol = 1e-15
    mf.kernel()
    dm_R_ref = mf.make_rdm1()
    print (np.max(np.abs(dm_R_ref - dm_R_full)))

    

