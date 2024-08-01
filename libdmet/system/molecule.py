#!/usr/bin/env python

"""
Molecule class.

Author:
    Tianyu Zhu
    Zhi-Hao Cui
"""

import numpy as np
import itertools as it

from pyscf import lib, scf, dft

from  libdmet.utils import logger as log
from libdmet.utils.misc import add_spin_dim, format_idx, Iterable

# ***********************************************************************************
# Molecule class
# *********************************************************************************** 

class Molecule(object):
    def __init__(self, mol):
        # mol related:
        self.mol = self.cell = self.supercell = mol
        self.kmesh = [1,1,1]
        self.nscsites = self.nao = self.supercell.nsites = self.mol.nao_nr()
        names, coords = zip(*mol._atom)
        self.names, self.coords = np.asarray(names), np.asarray(coords)
         
        self.dim = 0 # dim          
        self.csize = np.asarray(self.kmesh) # cellsize, kmesh
        self.ncells = np.prod(self.csize) # num of cells
        # cells' coordinates
        self.cells = lib.cartesian_prod((np.arange(self.kmesh[0]),
                                         np.arange(self.kmesh[1]),
                                         np.arange(self.kmesh[2]))) 
        # map: cell coord -> cell idx
        self.celldict = dict(zip(map(tuple, self.cells), range(self.ncells))) 
        self.kpts_scaled = [[0,0,0]]
        self.kpts = self.kpts_abs = [[0,0,0]]
        self.nkpts = len(self.kpts)
        
        # phase
        self.bigcell = mol
        self.phase = [[1.]]
        self.phase_k2R = self.phase
        self.phase_R2k = self.phase
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
    mol functions:
    """
    def real_coords(self):
        return self.coords

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
        return A
    
    def FFTtoT(self, B):
        return B

    def k2R(self, A):
        return A
    
    def R2k(self, B):
        return B
    
    def k2R_H2(self, H2_k):
        return H2_k

    def R2k_H2(self, H2_R):
        return H2_R

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
    def set_Ham(self, mf, C_ao_lo, eri_symmetry=4, 
                ovlp=None, hcore=None, rdm1=None, fock=None, vj=None, vk=None, 
                vxc=None, use_hcore_as_emb_ham=False, H0=0.0):
        """
        Set Hamiltonian:
        hcore, fock, ovlp, rdm1, (vhf), (vxc) in kAO, kLO, RLO,
        and eri_symmetry.
        """
        log.info("Set DMET mean-field Hamiltonian.")
        self.mf = mf
        self.kmf = mf
        self.C_ao_lo = np.asarray(C_ao_lo)

        # get: mo_coeff, ovlp, hcore, rdm1, fock
        log.info("set_Ham: set hcore, ovlp, rdm1")
        if ovlp is None:
            ovlp = self.mf.get_ovlp()
        if hcore is None:
            hcore = self.mf.get_hcore()
        if rdm1 is None:
            rdm1 = self.mf.make_rdm1()
        ovlp = np.asarray(ovlp)
        hcore = np.asarray(hcore)
        rdm1 = np.asarray(rdm1)

        # vj, vk
        log.info("set_Ham: set vj, vk")
        if vj is None or vk is None:
            vj, vk = self.mf.get_jk(dm=rdm1)
        vj, vk = np.asarray(vj), np.asarray(vk)

        # veff, vhf
        log.info("set_Ham: set veff, vhf")
        if isinstance(self.mf, dft.rks.KohnShamDFT): # DFT
            if vxc is None:
                vxc = self.mf.get_vxc(dm=rdm1)
            # TODO: do not re-calculate get_veff
            veff = self.mf.get_veff(dm=rdm1)
            if isinstance(self.mf, dft.uks.UKS):
                vhf = vj[0] + vj[1] - vk
            elif isinstance(self.mf, dft.rks.RKS):
                vhf = vj - vk * 0.5
            else:
                raise ValueError
        else: # HF
            if isinstance(self.mf, scf.uhf.UHF):
                veff = vhf = vj[0] + vj[1] - vk
            elif isinstance(self.mf, scf.hf.RHF):
                veff = vhf = vj - vk * 0.5
            else:
                raise ValueError
        # fock, fock_hf
        if fock is None:
            fock = hcore + veff
        fock_hf = hcore + vhf

        # set spin and restricted
        if self.C_ao_lo.ndim == 2:
            self.spin = 1
            self.restricted = True
        else:
            self.spin = C_ao_lo.shape[0]
            self.restricted = (self.spin == 1)

        # add extra dimension for fake Gamma point
        ovlp = ovlp[None]
        hcore = hcore[None]
        if self.restricted:
            self.C_ao_lo = self.C_ao_lo[None]
            rdm1 = rdm1[None]
            fock = fock[None]
            fock_hf = fock_hf[None]
            vj = vj[None]
            vk = vk[None]
            veff = veff[None]
            vhf = vhf[None]
        else:
            self.C_ao_lo = self.C_ao_lo[:, None]
            rdm1 = rdm1[:, None]
            fock = fock[:, None]
            fock_hf = fock_hf[:, None]
            vj = vj[:, None]
            vk = vk[:, None]
            veff = veff[:, None]
            vhf = vhf[:, None]

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
            if self.restricted:
                vxc = vxc[None]
            else:
                vxc = vxc[:, None]
            self.vxc_ao_k = np.asarray(vxc)

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

    setHam = set_Ham

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

    def getH2(self, aosym='s4', use_Ham=False):
        '''
        AO/LO basis H2 (ERI) for whole molecule
        '''
        if use_Ham: # use H2 from Ham object.
            H2 = self.Ham.getH2()
        else:
            H2 = self.mol.intor('int2e', aosym=aosym)
        return H2

    def getImpJK(self):
        if self.JK_imp is not None:
            return self.JK_imp
        elif self.Ham is not None:
            return self.Ham.getImpJK()
        else:
            return None
     
    def getH0(self):
        return self.H0
 
    get_JK_imp = getImpJK

if __name__ == '__main__':
    from pyscf import gto, lo
    mol = gto.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'sto3g',
    )
    Mole = Molecule(mol)
    
    mf = scf.HF(mol)
    mf.kernel()
    C_ao_lo = lo.orth_ao(mf)
    Mole.set_Ham(mf, C_ao_lo)

    print (Mole.getH2().shape)
    

