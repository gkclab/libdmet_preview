#!/usr/bin/env python

"""
wannier90 interface with pyscf.

Original code:
pyWannier90: Wannier90 for PySCF, https://github.com/hungpham2017/pyWannier90
Author:
    Hung Q. Pham <pqh3.14@gmail.com>

BSD 3-Clause License

Copyright (c) 2017, Hung Pham
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This version:
only need libwannier.so (use 'make lib' in wannier90 directory to generate).
Modified by
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import numpy as np
import scipy.linalg as la
import ctypes
from ctypes import *
from pyscf import lib
from pyscf.pbc import df
from pyscf.pbc import dft as pdft
import pyscf.lib.parameters as param
from pyscf.data.nist import HARTREE2EV
from libdmet.lo.lowdin import lowdin_k, get_ao_labels
from libdmet.lo.scdm import get_grid_uniform_cell, get_grid_becke_cell
from libdmet.utils.misc import mdot, max_abs
from libdmet.utils import logger as log

# The path for the wannier90 library
libwann_path = "/home/zhcui/program/wannier90/wannier90_dev/libwannier.so"
#libwann_path = "/home/zcui/program/wannier90/wannier90_dev/libwannier.so"
try:
    libwann = lib.load_library(libwann_path)
except OSError:
    log.debug(2, "wannier90 library path not set or incorrectly.")
    libwann = None

NUM_NNMAX = 12
STRING_LENGTH = 20

class W90:
    def __init__(self, kmf, kmesh, num_wann, gamma_only=False, spinors=False, \
            restricted=True, spin_up=None, other_keywords=None):
        """
        Main class to hold the input and output parameters for Wannier90.
        """
        self.kmf = kmf
        self.cell = kmf.cell
        self.num_wann = num_wann
        self.keywords = other_keywords
        self.num_nnmax = NUM_NNMAX

        # input for setup function
        self.seed_name = "%-*s"%(STRING_LENGTH, "wannier90")
        self.mp_grid   = np.array(kmesh, dtype=np.int32)
        self.num_kpts  = len(self.kmf.kpts)
        assert self.num_kpts == np.prod(self.mp_grid)
        self.real_lattice  = np.array(self.cell.lattice_vectors() * \
                param.BOHR, order='F')
        self.recip_lattice = np.array(self.cell.reciprocal_vectors() /
                param.BOHR, order='F')
        self.kpt_latt = np.array(self.cell.get_scaled_kpts(self.kmf.kpts).T,
                order='F') # frac coord
        #self.num_bands_tot = self.cell.nao_nr()
        self.num_bands_tot = np.asarray(kmf.mo_coeff).shape[-1]
        self.num_atoms     = self.cell.natm
        self.atom_symbols  = np.array([("%-*s"%(STRING_LENGTH, \
                atom[0])).encode('utf-8') for atom in self.cell._atom])
        self.atoms_cart    = np.array(np.array([atom[1] for atom in
            self.cell._atom]).T, order='F') * param.BOHR
        self.gamma_only    = int(gamma_only)
        self.spinors       = int(spinors)

        # output for setup function
        self.nntot = None
        self.nnlist = None
        self.nncell = None
        self.num_bands = None
        self.proj_site = None
        self.proj_l = None
        self.proj_m = None
        self.proj_radial = None
        self.proj_z = None
        self.proj_x = None
        self.proj_zona = None
        self.exclude_bands = None
        self.proj_s = None
        self.proj_s_qaxis = None

        # input for run function
        self.M_matrix = None
        self.A_matrix = None
        self.eigenvalues = None

        # output for run function
        self.U_matrix = None
        self.U_matrix_opt = None
        self.lwindow = None
        self.wann_centres = None
        self.wann_spreads = None
        self.spread = None

        self.band_included_list = None
        self.use_bloch_phases = False
        self.use_scdm = False
        self.use_atomic = False
        self.guiding_centres = True
        self.check_complex = False
        self.restricted = restricted
        self.spin_up = spin_up
        if self.restricted:
            self.mo_energy = self.kmf.mo_energy
            self.mo_coeff = self.kmf.mo_coeff
        else:
            if spin_up:
                self.mo_energy = self.kmf.mo_energy[0]
                self.mo_coeff = self.kmf.mo_coeff[0]
            else:
                self.mo_energy = self.kmf.mo_energy[1]
                self.mo_coeff = self.kmf.mo_coeff[1]

    def kernel(self, A_matrix=None, M_matrix=None, **kwargs):
        """
        Main kernel for wannier90.
        """
        self.make_win()
        self.setup()
        if M_matrix is None:
            self.M_matrix = self.get_M_mat()
        else:
            log.info("Wannier90: use given M matrix.")
            self.M_matrix = np.asarray(M_matrix, dtype=np.complex128, \
                    order='F')
        assert self.M_matrix.shape == (self.num_bands, self.num_bands, \
            self.nntot, self.num_kpts)
        if A_matrix is None:
            self.A_matrix = self.get_A_mat(**kwargs)
        else:
            log.info("Wannier90: use customized guess.")
            self.A_matrix = np.asarray(A_matrix, dtype=np.complex128, \
                    order='F')
        assert self.A_matrix.shape == (self.num_bands, self.num_wann, \
                self.num_kpts)
        self.eigenvalues = self.get_epsilon_mat()
        self.run()

    def make_win(self, fname=None):
        """
        Make a basic *.win file for wannier90.
        """
        if fname is None:
            fname = self.seed_name.strip() + ".win"
        win_file = open(fname, "w")
        win_file.write('! Basic input\n\n')
        win_file.write('num_bands = %d\n' % (self.num_bands_tot))
        win_file.write('num_wann  = %d\n\n' % (self.num_wann))

        # lattice
        win_file.write('Begin Unit_Cell_Cart\n')
        for vec in self.real_lattice:
            win_file.write('%25.20f  %25.20f  %25.20f\n' \
                    %(vec[0], vec[1], vec[2]))
        win_file.write('End Unit_Cell_Cart\n\n')

        # atoms
        win_file.write('Begin atoms_cart\n')
        for i, symbol in enumerate(self.atom_symbols):
            win_file.write('%s  %25.20f  %25.20f  %25.20f\n' \
                    %(symbol.decode(), self.atoms_cart[0, i], self.atoms_cart[1, i], \
                    self.atoms_cart[2, i]))
        win_file.write('End atoms_cart\n\n')

        # kpts related
        win_file.write('mp_grid = %d %d %d\n' %(self.mp_grid[0], \
                self.mp_grid[1], self.mp_grid[2]))
        if self.gamma_only == 1:
            win_file.write('gamma_only : true\n')
        win_file.write('begin kpoints\n')
        for k in range(self.num_kpts):
            win_file.write('%25.20f  %25.20f  %25.20f\n' \
                    %(self.kpt_latt[0, k], self.kpt_latt[1, k], \
                      self.kpt_latt[2, k]))
        win_file.write('End Kpoints\n\n')

        # additional
        if self.keywords != None:
            win_file.write('! Additional keywords\n')
            win_file.write(self.keywords)
            win_file.write("\n")
        if self.use_bloch_phases:
            win_file.write('use_bloch_phases = T\n')
        if self.guiding_centres:
            win_file.write('guiding_centres = T\n')
        win_file.close()

    def setup(self):
        """
        Execute the wannier90_setup functon.
        input args:
            seed_name     : 20 char string
            mp_grid       : kmesh 1D-F-int32
            num_kpts      : number of kpts, int32
            real_lattice  : real lattice vec in A, 2D-F-real, (3, 3)
            recip_lattice : recip lattice vec in A^-1, 2D-F-real, (3, 3)
            kpt_latt      : scaled kpts, 2D-F-real, (3, nkpts)
            num_bands_tot : nbands (total), int32
            num_atoms     : natm, int32
            atom_symbols  : atom names, 1D-F-string
            atoms_cart    : atom pos, 2D-F-real, (3, natoms)
            gamma_only    : gamma point kpt or not, bool, c_int
            spinors       : spinor wfn or not, bool, c_int
        """
        log.info("Wannier90: setup start.")
        seed_name = self.seed_name
        mp_grid = np.asarray(self.mp_grid, dtype=np.int32, order='F')
        num_kpts = self.num_kpts
        real_lattice = self.real_lattice.ravel(order='F')
        recip_lattice = self.recip_lattice.ravel(order='F')
        kpt_latt = self.kpt_latt.ravel(order='F')
        num_bands_tot = self.num_bands_tot
        num_atoms = self.num_atoms
        atom_symbols = self.atom_symbols
        atoms_cart = self.atoms_cart.ravel(order='F')
        gamma_only = self.gamma_only
        spinors = self.spinors

        """
        output args:
            nntot         : number of nearest neighbors, c_int
            nnlist        : nearest kpts starting id, k+b in FBZ,
                            2D-F-int32, (nkpts, num_nnmax)
            nncell        : 3 integers specifies G, that moves k+b in FBZ,
                            3D-F-int32, (3, nkpts, num_nnmax)
            num_bands     : nbands (included), c_int
            num_wann      : nwann, c_int
            proj_site     : center, 2D-F-real, (3, num_bands_tot)
            proj_l        : l, 1D-F-int32, (num_bands_tot)
            proj_m        : m_r, 1D-F-int32, (num_bands_tot)
            proj_radial   : 1D-F-int32, (num_bands_tot)
            proj_z        : specify the zaxis of theta, 2D-F-real,
                            (3, num_bands_tot)
            proj_x        : specify the zaxis of phi, 2D-F-real,
                            (3, num_bands_tot)
            proj_zona     : z/a value, 1D-F-real, (num_bands_tot)
            exclude_bands : exclude_bands, 1D-F-int32, (num_bands_tot)
            proj_s        : 1 or -1, project to up or down spin states,
                            1D-F-int32, (num_bands_tot)
            proj_s_qaxis  : spin quantisation axis in cart coords,
                            2D-F-real, (3, num_bands_tot)
        """
        num_nnmax = self.num_nnmax
        nntot = c_int(0)
        nnlist = np.zeros((num_kpts, num_nnmax), dtype=np.int32, order='F')
        nncell = np.zeros((3, num_kpts, num_nnmax), dtype=np.int32, order='F')
        num_bands = c_int(0)
        num_wann = c_int(0)
        proj_site = np.zeros((3, num_bands_tot), dtype=np.double, order='F')
        proj_l = np.zeros((num_bands_tot), dtype=np.int32, order='F')
        proj_m = np.zeros((num_bands_tot), dtype=np.int32, order='F')
        proj_radial = np.zeros((num_bands_tot), dtype=np.int32, order='F')
        proj_z = np.zeros((3, num_bands_tot), dtype=np.double, order='F')
        proj_x = np.zeros((3, num_bands_tot), dtype=np.double, order='F')
        proj_zona = np.zeros((num_bands_tot), dtype=np.double, order='F')
        exclude_bands = np.zeros((num_bands_tot), dtype=np.int32, order='F')
        proj_s  = np.zeros((num_bands_tot), dtype=np.int32, order='F')
        proj_s_qaxis = np.zeros((num_bands_tot, 3), dtype=np.double, order='F')

        """
        call setup
        """
        wannier_setup = libwann.wannier_setup_
        wannier_setup(c_char_p(seed_name.encode('utf-8')), \
            mp_grid.ctypes.data_as(c_void_p), byref(c_int(num_kpts)), \
            real_lattice.ctypes.data_as(c_void_p), \
            recip_lattice.ctypes.data_as(c_void_p), \
            kpt_latt.ctypes.data_as(c_void_p), byref(c_int(num_bands_tot)), \
            byref(c_int(num_atoms)), atom_symbols.ctypes.data_as(c_void_p), \
            atoms_cart.ctypes.data_as(c_void_p), byref(c_int(gamma_only)), \
            byref(c_int(spinors)), \
            byref(nntot), nnlist.ctypes.data_as(c_void_p), \
            nncell.ctypes.data_as(c_void_p), \
            byref(num_bands), byref(num_wann), \
            proj_site.ctypes.data_as(c_void_p), \
            proj_l.ctypes.data_as(c_void_p), \
            proj_m.ctypes.data_as(c_void_p), \
            proj_radial.ctypes.data_as(c_void_p), \
            proj_z.ctypes.data_as(c_void_p), \
            proj_x.ctypes.data_as(c_void_p),\
            proj_zona.ctypes.data_as(c_void_p), \
            exclude_bands.ctypes.data_as(c_void_p), \
            proj_s.ctypes.data_as(c_void_p), \
            proj_s_qaxis.ctypes.data_as(c_void_p), \
            c_long(STRING_LENGTH), c_long(STRING_LENGTH))
            # ZHC NOTE: the last two args are length of the string.

        """
        post procecessing of output
        """
        nntot = nntot.value
        num_bands = num_bands.value
        num_wann = num_wann.value
        assert num_wann == self.num_wann
        # save to self
        self.nntot = nntot
        self.nnlist = nnlist
        self.nncell = nncell
        self.num_bands = num_bands
        self.proj_site = proj_site
        self.proj_l = proj_l
        self.proj_m = proj_m
        self.proj_radial = proj_radial
        self.proj_z = proj_z
        self.proj_x = proj_x
        self.proj_zona = proj_zona
        self.exclude_bands = exclude_bands
        self.band_included_list = [i for i in range(self.num_bands_tot) \
                if (i + 1) not in self.exclude_bands] # not standard
        self.proj_s = proj_s
        self.proj_s_qaxis = proj_s_qaxis
        log.info("Wannier90: setup complete.")

    def get_M_mat(self):
        r"""
        Construct the ovelap matrix: M_{m,n}^{(\mathbf{k,b})}
        Eq. 25 in PRB, 56, 12847.
        M_{mnbk} = <u_m k | u_n k+b>, 4D-F-complex
            (num_bands, num_bands, nntot, num_kpts)
        k = kpt_latt[:, k_idx]
        k+b = kpt_latt[:, nnlist[k_idx, b_idx]-1] + nncell[:, k_idx, b_idx]
        ft_ao_pair function:
            \sum_T exp(i k_j * T) \int dr exp(-i(G+q)r) i(r) j(r-T)
        ZHC NOTE: pyscf's documentation has a minus sign in the exp,
                  which should be a typo.
        """
        log.debug(1, "Wannier90: M matrix start.")
        M_matrix = np.empty((self.num_bands, self.num_bands, \
            self.nntot, self.num_kpts), dtype=np.complex128, order='F')
        for k_idx in range(self.num_kpts):
            for b_idx in range(self.nntot):
                # k
                k1 = self.cell.get_abs_kpts(self.kpt_latt[:, k_idx])
                # k + b
                k2_idx = self.nnlist[k_idx, b_idx] - 1
                k2_scaled = self.kpt_latt[:, k2_idx] + \
                        self.nncell[:, k_idx, b_idx]
                k2 = self.cell.get_abs_kpts(k2_scaled)
                # G = b, kj = (k + b), q = 0
                ovlp_ao = df.ft_ao.ft_aopair(self.cell, Gv=k2-k1, \
                        kpti_kptj=[k1, k2], q=np.zeros(3))[0]
                Cm = self.mo_coeff[k_idx][:, self.band_included_list]
                Cn = self.mo_coeff[k2_idx][:, self.band_included_list]
                M_matrix[:, :, b_idx, k_idx] = mdot(Cm.conj().T, ovlp_ao, Cn)
        log.debug(1, "Wannier90: M matrix complete.")
        return M_matrix

    def get_A_mat(self, **kwargs):
        r"""
        Construct the projection matrix: A_{m,n}^{\mathbf{k}}
        Eq. 62 in PRB, 56, 12847 or Eq. 22 in PRB, 65, 035109.
        Amn_k = <psi_{mk}| g_n>, 3D-F-(m, n, k)
        """
        log.debug(1, "Wannier90: A matrix start.")
        A_matrix = np.empty((self.num_bands, self.num_wann, self.num_kpts), \
                dtype=np.complex128, order='F')
        if self.use_bloch_phases:
            log.info("Wannier90: use bloch guess.")
            Amn = np.zeros((self.num_bands, self.num_wann), order='F')
            np.fill_diagonal(Amn, 1.0)
            for k in range(A_matrix.shape[-1]):
                A_matrix[:, :, k] = Amn
        elif self.use_scdm:
            # ZHC TODO FIXME smearing?
            # different number of bands at different kpts?
            log.info("Wannier90: use scdm guess.")
            from libdmet.lo.scdm import scdm_k
            cell = self.cell
            mo_coeff = np.asarray(self.mo_coeff)[:, :, self.band_included_list]
            kpts = self.kmf.kpts
            nlo = self.num_wann
            use_gamma_perm = kwargs.get("use_gamma_perm", True)
            grid = kwargs.get("grid", "becke")
            mesh = kwargs.get("mesh", None)
            level = kwargs.get("level", 5)
            A_matrix[:] = scdm_k(cell, mo_coeff, kpts, grid=grid, \
                    return_C_mo_lo=True, use_gamma_perm=use_gamma_perm, \
                    nlo=nlo, smear_func=None, mesh=mesh, level=level)[1][0]\
                    .transpose(1, 2, 0)
        elif self.use_atomic:
            log.info("Wannier90: use atomic guess.")
            rand = kwargs.get("rand", False)
            mo_coeff = np.asarray(self.mo_coeff)[:, :, self.band_included_list]
            A_matrix[:] = atomic_init_guess(self.kmf, mo_coeff, rand=rand)\
                    .transpose(1, 2, 0)
        else:
            log.info("Wannier90: use projection guess.")
            from libdmet.lo.proj_wannier import g_r
            cell = self.cell
            kpts = self.kmf.kpts
            grid = kwargs.get("grid", "becke")
            mesh = kwargs.get("mesh", None)
            level = kwargs.get("level", 5)
            if grid.strip().startswith(('b', 'B')):
                coords, weights = get_grid_becke_cell(cell, level=level)
            else:
                coords, weights = get_grid_uniform_cell(cell, mesh=mesh)
            for k in range(self.num_kpts):
                ao_g = pdft.numint.eval_ao(cell, coords, kpt=kpts[k], deriv=0)
                #from pyscf import dft
                #ao_g = dft.numint.eval_ao(cell, coords)
                for i in range(self.num_wann):
                    frac_site = self.proj_site[:, i]
                    # ZHC NOTE the unit here are all in Bohr
                    abs_site = frac_site.dot(self.real_lattice) / param.BOHR
                    l = self.proj_l[i]
                    mr = self.proj_m[i]
                    r = self.proj_radial[i]
                    z_axis = self.proj_z[:, i]
                    x_axis = self.proj_x[:, i]
                    zona = self.proj_zona[i]
                    gr = g_r(coords, abs_site, l, mr, r, zona, x_axis, z_axis,\
                            unit='B')
                    mo_included = self.mo_coeff[k][:, self.band_included_list]
                    A_matrix[:, i, k] = np.dot(mo_included.conj().T, \
                            np.dot(ao_g.conj().T, gr * weights))
        log.debug(1, "Wannier90: A matrix complete.")
        return A_matrix

    def get_epsilon_mat(self):
        r"""
        Construct the eigenvalues matrix: \epsilon_{n}^(\mathbf{k})
        2D-F-(n, k), in eV.
        """
        return np.asarray((np.asarray(self.mo_energy)[:, self.band_included_list]\
                * HARTREE2EV).T, order='F')

    def run(self):
        """
        Execute the wannier90 run.
        """
        log.info("Wannier90: run start.")
        assert self.num_wann is not None
        assert isinstance(self.M_matrix, np.ndarray)
        assert isinstance(self.A_matrix, np.ndarray)
        assert isinstance(self.eigenvalues, np.ndarray)
        """
        input args:
            seed_name     : 20 char string
            mp_grid       : kmesh 1D-F-int32
            num_kpts      : number of kpts, int32
            real_lattice  : real lattice vec in A, 2D-F-real, (3, 3)
            recip_lattice : recip lattice vec in A^-1, 2D-F-real, (3, 3)
            kpt_latt      : scaled kpts, 2D-F-real, (3, nkpts)
            num_bands     : nbands (included), int32
            num_wann      : nwann, int32
            nntot         : number of nearest neighbors, int32
            num_atoms     : natm, int32
            atom_symbols  : atom names, 1D-F-string
            atoms_cart    : atom pos, 2D-F-real, (3, natoms)
            gamma_only    : gamma point kpt or not, bool, c_int
            M_matrix      : M_{mnbk}, 4D-F-complex, (nbands, nbands, nb, nk)
            A_matrix      : A_{mnk},  3D-F-complex, (nbands, nwann, nk)
            eigenvalues   : e_{nk},   2D-F-real, (nbands, nk)
        """
        seed_name = self.seed_name
        mp_grid = np.asarray(self.mp_grid, dtype=np.int32, order='F')
        num_kpts = self.num_kpts
        real_lattice = self.real_lattice.ravel(order='F')
        recip_lattice = self.recip_lattice.ravel(order='F')
        kpt_latt = self.kpt_latt.ravel(order='F')
        num_bands = self.num_bands
        num_wann = self.num_wann
        nntot = self.nntot
        num_atoms = self.num_atoms
        atom_symbols = self.atom_symbols
        atoms_cart = self.atoms_cart.ravel(order='F')
        gamma_only = self.gamma_only
        M_matrix = self.M_matrix.ravel(order='F')
        A_matrix = self.A_matrix.ravel(order='F')
        eigenvalues = self.eigenvalues.ravel(order='F')

        """
        output args:
            U_matrix     : U_k  3D-F-complex, (nwann, nwann, nkpts)
            U_matrix_opt : U_k to subspace 3D-F-complex, (nbands, nwann, nkpts)
            lwindow      : whether each band is in outer energy window,
                           2D-F-int32, (nbands, nkpts)
            wann_centres : w center for each wann, 2D-F-real, (3, nwann)
            wann_spreads : w spreads for each wann, 1D-F-real, (nwann)
            spread       : Omega, Omega_I, Omega_tilde, 1D-F-real, (3)
        """
        U_matrix = np.zeros((num_wann, num_wann, num_kpts), \
                dtype=np.complex128, order='F')
        U_matrix_opt = np.zeros((num_bands, num_wann, num_kpts), \
                dtype=np.complex128, order='F')
        lwindow = np.zeros((num_bands, num_kpts), dtype=np.int32, order='F')
        wann_centres = np.zeros((3, num_wann), dtype=np.double, order='F')
        wann_spreads = np.zeros((num_wann), dtype=np.double, order='F')
        spread = np.zeros((3), dtype=np.double, order='F')

        """
        call run
        """
        wannier_run = libwann.wannier_run_
        wannier_run(c_char_p(seed_name.encode('utf-8')), mp_grid.ctypes.data_as(c_void_p), \
                byref(c_int(num_kpts)), real_lattice.ctypes.data_as(c_void_p),\
                recip_lattice.ctypes.data_as(c_void_p), \
                kpt_latt.ctypes.data_as(c_void_p), byref(c_int(num_bands)), \
                byref(c_int(num_wann)), byref(c_int(nntot)), \
                byref(c_int(num_atoms)), \
                atom_symbols.ctypes.data_as(c_void_p), \
                atoms_cart.ctypes.data_as(c_void_p), \
                byref(c_int(gamma_only)), \
                M_matrix.ctypes.data_as(c_void_p), \
                A_matrix.ctypes.data_as(c_void_p), \
                eigenvalues.ctypes.data_as(c_void_p), \
                U_matrix.ctypes.data_as(c_void_p), \
                U_matrix_opt.ctypes.data_as(c_void_p), \
                lwindow.ctypes.data_as(c_void_p), \
                wann_centres.ctypes.data_as(c_void_p), \
                wann_spreads.ctypes.data_as(c_void_p), \
                spread.ctypes.data_as(c_void_p),\
                c_long(STRING_LENGTH), c_long(STRING_LENGTH))
                # ZHC NOTE: the last two args are length of the string.

        """
        post processing
        """
        self.U_matrix = U_matrix
        self.U_matrix_opt = U_matrix_opt
        self.lwindow = (lwindow != 0) # non-zero for True
        self.wann_centres = wann_centres
        self.wann_spreads = wann_spreads
        self.spread = spread
        log.info("Wannier90: run complete.")

    def export_AME(self):
        r"""
        Export A_{m,n}^{\mathbf{k}} and M_{m,n}^{(\mathbf{k,b})}
        and \epsilon_{n}^(\mathbf{k})
        """
        if self.M_matrix is None or self.A_matrix is None:
            self.make_win()
            self.setup()
            self.M_matrix = self.get_M_mat()
            self.A_matrix = self.get_A_mat()
            self.eigenvalues = self.get_epsilon_mat()

        with open('%s.mmn'%(self.seed_name.strip()), 'w') as f:
            f.write('Generated by the pyWannier90\n')
            f.write('    %d    %d    %d\n' % (self.num_bands, \
                    self.num_kpts, self.nntot))
            for k_id in range(self.num_kpts):
                for nn in range(self.nntot):
                    k_id1 = k_id + 1
                    k_id2 = self.nnlist[k_id, nn]
                    nnn, nnm, nnl = self.nncell[:, k_id, nn]
                    f.write('    %d  %d    %d  %d  %d\n' \
                            % (k_id1, k_id2, nnn, nnm, nnl))
                    for m in range(self.num_bands):
                        for n in range(self.num_bands):
                            f.write('    %22.18f  %22.18f\n' \
                                    % (self.M_matrix[m, n, nn, k_id].real, \
                                       self.M_matrix[m, n, nn, k_id].imag))

        with open('%s.amn'%(self.seed_name.strip()), 'w') as f:
            f.write('    %d\n' % (self.num_bands*self.num_kpts*self.num_wann))
            f.write('    %d    %d    %d\n' \
                    % (self.num_bands, self.num_kpts, self.num_wann))
            for k_id in range(self.num_kpts):
                for ith_wann in range(self.num_wann):
                    for band in range(self.num_bands):
                        f.write('    %d    %d    %d    %22.18f    %22.18f\n' \
                                % (band+1, ith_wann+1, k_id+1, \
                                self.A_matrix[band, ith_wann, k_id].real, \
                                self.A_matrix[band, ith_wann, k_id].imag))

        with open('%s.eig'%(self.seed_name.strip()), 'w') as f:
            for k_id in range(self.num_kpts):
                for band in range(self.num_bands):
                        f.write('    %d    %d    %22.18f\n' % (band+1, k_id+1,\
                                self.eigenvalues[band, k_id]))

    def export_unk(self, mesh=[50, 50, 50]):
        """
        Export the periodic part of BF in a real space grid,
        for plotting with wannier90.
        """
        from scipy.io import FortranFile
        coords, weights = get_grid_uniform_cell(self.cell, mesh=mesh, order='F')
        for k_id in range(self.num_kpts):
            if (self.spin_up is not None) and (not self.spin_up):
                spin = '.2'
            else:
                spin = '.1'
            kpt = self.cell.get_abs_kpts(self.kpt_latt[:, k_id])
            ao_g = pdft.numint.eval_ao(self.cell, coords, kpt=kpt)
            u_ao = np.exp(-1.0j * np.dot(coords, kpt))[:, None] * ao_g
            unk_file = FortranFile('UNK' + "%05d" % (k_id + 1) + spin, 'w')
            unk_file.write_record(np.asarray((mesh[0], mesh[1], mesh[2], k_id +
                1, self.num_bands), dtype=np.int32))
            mo_included = self.mo_coeff[k_id][:, self.band_included_list]
            u_mo = np.dot(u_ao, mo_included)
            for band in range(len(self.band_included_list)):
                unk_file.write_record(np.asarray(u_mo[:, band], \
                        dtype=np.complex128))
            unk_file.close()

def get_A_mat_from_lo(C_ao_mo, S_ao_ao, C_ao_lo):
    if C_ao_lo is None:
        return None
    nkpts, nao, nlo = C_ao_lo.shape[-3:]
    nmo = C_ao_mo.shape[-1]
    if C_ao_lo.ndim == 3:
        A_mat = np.zeros((nkpts, nlo, nmo), dtype=np.complex128)
        for k in range(nkpts):
            A_mat[k] = mdot(C_ao_lo[k].T, S_ao_ao[k].T, C_ao_mo[k].conj())
    else:
        spin = C_ao_lo.shape[0]
        A_mat = np.zeros((spin, nkpts, nlo, nmo), dtype=np.complex128)
        for s in range(spin):
            for k in range(nkpts):
                A_mat[s, k] = mdot(C_ao_lo[s, k].T, S_ao_ao[k].T, \
                        C_ao_mo[s, k].conj())
    return A_mat

def unpack_uniq_var(v):
    nmo = int(np.sqrt(v.size*2)) + 1
    idx = np.tril_indices(nmo, -1)
    mat = np.zeros((nmo,nmo))
    mat[idx] = v
    return mat - mat.conj().T

def extract_rotation(dr, u0=1):
    dr = unpack_uniq_var(dr)
    return np.dot(u0, la.expm(dr))

def add_noise(u_mat, noise_amp=1e-3):
    u_mat = np.asarray(u_mat)
    nkpts, nmo, nlo = u_mat.shape
    dr = np.cos(np.arange((nmo-1)*nmo//2)) * noise_amp
    for k in range(nkpts):
        u_mat[k] = extract_rotation(dr, u0=u_mat[k])
    return u_mat

def atomic_init_guess(kmf, mo_coeff, rand=False, noise_amp=1e-3):
    """
    Use closest lowdin orbital as initial guess.
    """
    ovlp = kmf.get_ovlp()
    c_lowdin = lowdin_k(kmf)
    if c_lowdin.ndim == 4:
        c_lowdin = c_lowdin[0]
    nkpts = len(mo_coeff)
    umat_col = []
    for k in range(nkpts):
        # indices are ip
        ovlp_mo = mdot(mo_coeff[k].conj().T, ovlp[k], c_lowdin[k])
        # only keep the AOs which have largest overlap to MOs
        idx = np.argsort(np.einsum('ip, ip -> p', ovlp_mo.conj(), \
                ovlp_mo).real, kind='mergesort')
        nmo = ovlp_mo.shape[0]
        idx = sorted(idx[-nmo:])
        # Rotate mo_coeff, make it as close as possible to AOs
        u, sigma, vt = la.svd(ovlp_mo[:, idx])
        umat_col.append(np.dot(u, vt))
    umat_col = np.asarray(umat_col)
    if rand:
        u_mat_col = add_noise(u_mat_col, noise_amp=noise_amp)
    return umat_col

if __name__ == '__main__':
    import numpy as np
    from pyscf import scf, gto
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    log.verbose = "DEBUG2"
    #import pywannier90
    cell = pgto.Cell()
    cell.atom = '''
     C                  3.17500000    3.17500000    3.17500000
     H                  2.54626556    2.54626556    2.54626556
     H                  3.80373444    3.80373444    2.54626556
     H                  2.54626556    3.80373444    3.80373444
     H                  3.80373444    2.54626556    3.80373444
    '''
    cell.basis = 'sto3g'
    cell.a = np.eye(3) * 6.35
    #cell.a = np.eye(3) * 10.0
    cell.precision = 1e-10
    cell.verbose = 5
    cell.build()

    kmesh = nk = [1, 1, 1]
    abs_kpts = cell.make_kpts(nk)

    kmf = pscf.KRHF(cell, abs_kpts)
    kmf = kmf.density_fit()
    kmf.conv_tol = 1e-10
    ekpt = kmf.run()

    #num_wann = 9
    #keywords = \
    #"""
    #num_iter = 200
    #dis_num_iter = 200
    #begin projections
    #C:l=0;sp3
    #H:l=0
    #end projections
    #"""
    num_wann = 4
    keywords = \
    """
    begin projections
    C:sp3
    end projections
    exclude_bands : 1,6-%s
    """%(cell.nao_nr())

    # wannier run
    w90 = W90(kmf, kmesh, num_wann, other_keywords=keywords)
    #w90.use_atomic = True
    #w90.use_scdm = True
    #w90.use_bloch_phases = True
    w90.kernel()
    w90.export_AME()
    ## Plotting using Wannier90
    #keywords_plot = keywords + 'wannier_plot = True\n wannier_plot_supercell = 1\n'
    #w90 = W90(kmf, kmesh, num_wann, other_keywords = keywords_plot)
    #w90.kernel()
