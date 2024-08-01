#! /usr/bin/env python

"""
Vistualization code.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import h5py
import numpy as np
import scipy.linalg as la
from scipy.interpolate import UnivariateSpline

from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.data.nist import HARTREE2EV

from libdmet.utils.lattice_plot import LatticePlot, plot_3band_order
from libdmet.utils import cubegen
from libdmet.utils.misc import max_abs
from libdmet.utils import logger as log

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import Normalize
from matplotlib.collections import LineCollection
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# set font to 42 for Type2 fonts:
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype'] = 42

# ****************************************************************************
# plot curve
# ****************************************************************************

def plot_smooth(x, y, x_plot=None, label=None, color='black', marker='o',
                linestyle='-', smooth=1e-4, remove_neg=False, n0left=None,
                n0right=None, do_plot=True, **kwargs):
    """
    Plot a y-x curve with spline.

    Args:
        x: x
        y: y
        x_plot: a list of fine mesh of x, if not provide, will be a linspace of
                x with 100 points.
        label:  [None]
        color:  ['black']
        marker: ['o']
        linestyle: ['-']
        smooth: s for spline.
        remove_neg: remove the negative values to 0.
        n0left: left zero points.
        n0right: right zero points.
        do_plot: plot.

    Returns:
        x_plot: dense x points.
        y_plot: spline curve.
    """
    spl = UnivariateSpline(x, y, s=smooth)
    if x_plot is None:
        x_plot = np.linspace(np.min(x), np.max(x), 100)
    y_plot = spl(x_plot)
    if remove_neg:
        y_plot[y_plot < 0.0] = 0.0
    if n0left is not None:
        y_plot[:n0left] = 0.0
    if n0right is not None:
        y_plot[-n0right:] = 0.0

    if do_plot:
        plt.plot(x_plot, y_plot, color=color, marker='', linestyle=linestyle, **kwargs)
        plt.plot(x, y, color=color, marker=marker, linestyle='', label=label, **kwargs)
    return x_plot, y_plot

def get_even_spacing_xy(x, y, figsize=(1, 1), spacing=0.03, x_only=False, y_only=False):
    """
    Get x, y, idx for even spacing, useful for scattering plots.
    """
    x_scaled = np.asarray(x).ravel()
    y_scaled = np.asarray(y).ravel()
    x_scaled = x_scaled / max_abs(x_scaled) * figsize[0]
    y_scaled = y_scaled / max_abs(y_scaled) * figsize[1]
    if x_only:
        y_scaled[:] = 0.0
    if y_only:
        x_scaled[:] = 0.0

    dis_cum = 0.0
    dis = np.sqrt(np.diff(x_scaled)**2 + np.diff(y_scaled)**2)
    idx = [0]
    for i in range(len(dis)):
        if dis_cum > spacing:
            idx.append(i + 1)
            dis_cum = 0.0
        else:
            dis_cum += dis[i]

    return x[idx], y[idx], idx

# ****************************************************************************
# plot periodic orbitals
# ****************************************************************************

class Grids(object):
    def __init__(self, cube):
        self.coords = cube.get_coords()
        self.ngrids = cube.get_ngrids()
        self.weights = np.empty(self.ngrids)
        self.weights[:] = abs(np.linalg.det(cube.box)) / float(self.ngrids)
        self.non0tab = None
        self.mol = cube.mol
        self.cutoff = None

    def build(self, with_non0tab=False):
        pass

def plot_orb_k(cell, outfile, coeff, kpts_abs, nx=80, ny=80, nz=80, resolution=None,
               margin=5.0, latt_vec=None, boxorig=None, box=None):
    """
    Calculate orbital value on real space grid and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        coeff : 2D array
            coeff coefficient (nkpts, nao).

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
    """
    cc = cubegen.Cube(cell, nx, ny, nz, resolution, margin=margin,
                      latt_vec=latt_vec, boxorig=boxorig, box=box)

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = np.empty(ngrids)
    coeff = coeff / np.sqrt(len(kpts_abs))
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = np.asarray(cell.pbc_eval_gto('GTOval', coords[ip0:ip1], kpts=kpts_abs))
        if ao.ndim == 2: # 1 kpt
            ao = ao[np.newaxis]
        #: np.einsum('kgp, kp -> g')
        orb_on_grid[ip0:ip1] = np.tensordot(ao, coeff, axes=((0, 2), (0, 1))).real

    orb_on_grid = orb_on_grid.reshape(cc.nx, cc.ny, cc.nz)
    # Write out orbital to the .cube file
    cc.write(orb_on_grid, outfile, comment='Orbital value in real space (1/Bohr^3)')

def plot_orb_k_all(cell, outfile, coeffs, kpts_abs, nx=80, ny=80, nz=80, resolution=None,
                   margin=5.0, latt_vec=None, boxorig=None, box=None):
    """
    Plot all k-dependent orbitals in the reference cell.
    """
    cc = cubegen.Cube(cell, nx, ny, nz, resolution, margin=margin,
                      latt_vec=latt_vec, boxorig=boxorig, box=box)
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)

    coeffs = np.asarray(coeffs)
    coeffs = coeffs / np.sqrt(len(kpts_abs))
    nmo = coeffs.shape[-1]

    if len(coeffs.shape) == 3:
        for i in range(nmo):
            cc.write(None, outfile+"_mo%s.cube"%i,
                     comment='Orbital value in real space (1/Bohr^3)',
                     header_only=True)

        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            ao = np.asarray(cell.pbc_eval_gto('GTOval', coords[ip0:ip1], kpts=kpts_abs))
            if ao.ndim == 2: # 1 kpt
                ao = ao[np.newaxis]
            for i in range(nmo):
                #: np.einsum('kgp, kp -> g')
                orb_on_grid = np.tensordot(ao, coeffs[:, :, i], axes=((0, 2), (0, 1))).real
                cc.write_field(orb_on_grid, outfile+"_mo%s.cube"%i)
    else:
        spin = coeffs.shape[0]
        for s in range(spin):
            for i in range(nmo):
                cc.write(None, outfile+"_s%s_mo%s.cube"%(s, i),
                         comment='Orbital value in real space (1/Bohr^3)',
                         header_only=True)

        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            ao = np.asarray(cell.pbc_eval_gto('GTOval', coords[ip0:ip1], kpts=kpts_abs))
            if ao.ndim == 2: # 1 kpt
                ao = ao[np.newaxis]
            for s in range(spin):
                for i in range(nmo):
                    #: np.einsum('kgp, kp -> g')
                    orb_on_grid = np.tensordot(ao, coeffs[s, :, :, i], axes=((0, 2), (0, 1))).real
                    cc.write_field(orb_on_grid, outfile+"_s%s_mo%s.cube"%(s, i))

def plot_density_k(cell, outfile, dm, kpts_abs, nx=80, ny=80, nz=80, resolution=None,
                   margin=5.0, latt_vec=None, boxorig=None, box=None, skip_calc=False):
    """
    Calculates electron density and write out in cube format.

    Args:
        cell : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        dm : ndarray
            Density matrix of molecule.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
    """

    cc = cubegen.Cube(cell, nx, ny, nz, resolution, margin=margin,
                      latt_vec=latt_vec, boxorig=boxorig, box=box)

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    rho = np.empty(ngrids)
    if skip_calc:
        rho[:] = 0.0
    else:
        kni = numint.KNumInt(kpts_abs)
        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            ao = np.asarray(cell.pbc_eval_gto('GTOval', coords[ip0:ip1], kpts=kpts_abs))
            if ao.ndim == 2: # 1 kpt
                ao = ao[np.newaxis]
            #rho[ip0:ip1] = numint.eval_rho(cell, ao, dm)
            rho[ip0:ip1] = kni.eval_rho(cell, ao, dm, non0tab=None, xctype='LDA', hermi=1)

    rho = rho.reshape(cc.nx,cc.ny,cc.nz)

    # Write out density to the .cube file
    cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')

def plot_density_matrix_k(cell, outfile, dm, kpts_abs, R0, nx=80, ny=80, nz=80, resolution=None,
                          margin=5.0, latt_vec=None, boxorig=None, box=None,
                          skip_calc=False, coord0_idx=None, fname=None):
    r"""
    Calculates density matrix \gamma(R0, r) and write out in cube format.
    where R0 is the reference point.

    Args:
        cell : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        dm : ndarray
            Density matrix of molecule.
        R0: reference point, [x0, y0, z0] in bohr. Will find the nearest point.
        coord0_idx: if given, will use the grid (coords[coord0_idx]) as the reference point.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
    """
    cc = cubegen.Cube(cell, nx, ny, nz, resolution, margin=margin,
                      latt_vec=latt_vec, boxorig=boxorig, box=box)

    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    rho = np.empty(ngrids)

    if coord0_idx is not None:
        coord0 = coords[coord0_idx]
    else:
        coord0 = R0
    log.info("R0 %s (nearest grid %s)", R0, coord0)

    if skip_calc:
        rho[:] = 0.0
    else:
        ao_0 = np.asarray(cell.pbc_eval_gto('GTOval', coord0[None], kpts=kpts_abs))
        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            ao = np.asarray(cell.pbc_eval_gto('GTOval', coords[ip0:ip1], kpts=kpts_abs))
            if ao.ndim == 2: # 1 kpt
                ao = ao[np.newaxis]
            rho[ip0:ip1] = np.einsum("kP, kPQ, kgQ -> g", ao_0[:, 0], dm,
                                     ao.conj(), optimize=True).real
    rho = rho.reshape(cc.nx, cc.ny, cc.nz)
    if fname is not None:
        f = h5py.File("%s"%fname, 'w')
        f["n"] = np.asarray((cc.nx, cc.ny, cc.nz))
        f["coords"] = coords
        f["R0"] = coord0
        f["box"] = cc.box
        f["rho"] = rho
        f.close()
    cc.write(rho, outfile, comment='Electron density matrix in real space (e/Bohr^3)')

def plot_elf(mol, cube, dms, fname="ELF", elf_reg=1e-5, small_rho_tol=1e-8,
             max_memory=None, spin_average=False, kpts=None, save_rho=True):
    if hasattr(mol, 'pbc_intor'):
        return plot_elf_pbc(mol, cube, dms, fname=fname, elf_reg=elf_reg,
                            small_rho_tol=small_rho_tol, max_memory=max_memory,
                            spin_average=spin_average, kpts=kpts,
                            save_rho=save_rho)
    else:
        return plot_elf_mol(mol, cube, dms, fname=fname, elf_reg=elf_reg,
                            small_rho_tol=small_rho_tol, max_memory=max_memory,
                            spin_average=spin_average, save_rho=save_rho)

def plot_elf_mol(mol, cube, dms, fname="ELF", elf_reg=1e-5, small_rho_tol=1e-8,
                 max_memory=None, spin_average=False, save_rho=True):
    """
    Electron localization function. molecular version.

    Args:
        mol: Mole object.
        cube: Cube object.
        dms: density matrix in AO basis, (nao, nao) or (2, nao, nao).
        fname: cube file name for visualization.
        elf_reg: convergence factor for ELF far from molecule.
        small_rho_tol: tolerance for discard small density values.

    """
    assert not hasattr(mol, 'pbc_intor')
    from pyscf.dft import numint
    dms = np.asarray(dms)
    assert dms.ndim == 2 or (dms.ndim == 3 and dms.shape[0] == 2)
    nao = dms.shape[-1]
    if max_memory is None:
        max_memory = mol.max_memory

    ao_deriv = 2
    ni = numint.NumInt()
    grids = Grids(cube)
    ngrids = grids.ngrids

    f = h5py.File("%s.h5"%fname, 'w')
    f["n"] = np.asarray((cube.nx, cube.ny, cube.nz))
    f["coords"] = np.asarray(grids.coords)
    ip, iq = 0, 0

    if dms.ndim == 2:
        f.create_dataset("elf", (ngrids,), 'f8')
        if save_rho:
            f.create_dataset("rho", (ngrids,), 'f8')

        cube.write(None, fname + ".cube", comment='ELF value in real space (1/Bohr^3)', header_only=True)
        for ao, mask, weight, coords in \
                ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            rho = ni.eval_rho(mol, ao, dms, mask, xctype='MGGA', hermi=1)
            rho, dx_rho, dy_rho, dz_rho, nab2_rho, tau = rho
            rho[rho <= 0.0] = 0.0
            n0tab = rho > small_rho_tol

            # D(r)
            D = tau
            D[n0tab] -= 0.125 * ((dx_rho[n0tab]**2 + dy_rho[n0tab]**2 + dz_rho[n0tab]**2) / rho[n0tab])

            # D0(r)
            D0 = (0.3 * (3.0 * np.pi**2)**(2.0 / 3.0)) * (rho ** (5.0 / 3.0))

            # ELF(r)
            ELF = np.zeros_like(D)
            ELF[n0tab] = 1.0 / (1.0 + ((D[n0tab] + elf_reg)/ D0[n0tab])**2)

            cube.write_field(ELF, fname + ".cube")

            iq = ip + ELF.shape[-1]
            f["elf"][ip:iq] = np.asarray(ELF)
            if save_rho:
                f["rho"][ip:iq] = np.asarray(rho)

            ip = iq
    else:
        if spin_average:
            f.create_dataset("elf", (ngrids,), 'f8')
            if save_rho:
                f.create_dataset("rho", (ngrids,), 'f8')

            cube.write(None, fname + ".cube", comment='ELF value in real space (1/Bohr^3)', header_only=True)
            for ao, mask, weight, coords in \
                    ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
                D_sum = 0.0
                D0_sum = 0.0
                n0tab_sum = np.ones(coords.shape[0], dtype=bool)
                rho_sum = 0.0
                for dm in dms:
                    rho = ni.eval_rho(mol, ao, dm, mask, xctype='MGGA', hermi=1)
                    rho, dx_rho, dy_rho, dz_rho, nab2_rho, tau = rho
                    rho[rho <= 0.0] = 0.0
                    n0tab = rho > small_rho_tol
                    n0tab_sum &= n0tab

                    # D(r)
                    D = tau
                    D[n0tab] -= 0.125 * ((dx_rho[n0tab]**2 + dy_rho[n0tab]**2 + dz_rho[n0tab]**2) / rho[n0tab])
                    D_sum += D

                    # D0(r)
                    D0 = (0.3 * (6.0 * np.pi**2)**(2.0 / 3.0)) * (rho ** (5.0 / 3.0))
                    D0_sum += D0

                    if save_rho:
                        rho_sum += rho

                # ELF(r)
                ELF = np.zeros_like(D_sum)
                ELF[n0tab_sum] = 1.0 / (1.0 + ((D_sum[n0tab_sum] + elf_reg) / D0_sum[n0tab_sum])**2)

                cube.write_field(ELF, fname + ".cube")
                iq = ip + ELF.shape[-1]
                f["elf"][ip:iq] = np.asarray(ELF)
                if save_rho:
                    f["rho"][ip:iq] = np.asarray(rho_sum)
                ip = iq
        else:
            f.create_dataset("elf_0", (ngrids,), 'f8')
            f.create_dataset("elf_1", (ngrids,), 'f8')
            if save_rho:
                f.create_dataset("rho_0", (ngrids,), 'f8')
                f.create_dataset("rho_1", (ngrids,), 'f8')

            cube.write(None, fname + "_0.cube", comment='ELF value in real space (1/Bohr^3)', header_only=True)
            cube.write(None, fname + "_1.cube", comment='ELF value in real space (1/Bohr^3)', header_only=True)
            for ao, mask, weight, coords in \
                    ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
                for i, dm in enumerate(dms):
                    rho = ni.eval_rho(mol, ao, dm, mask, 'MGGA', hermi=1)
                    rho, dx_rho, dy_rho, dz_rho, nab2_rho, tau = rho
                    rho[rho <= 0.0] = 0.0
                    n0tab = rho > small_rho_tol

                    # D(r)
                    D = tau
                    D[n0tab] -= 0.125 * ((dx_rho[n0tab]**2 + dy_rho[n0tab]**2 + dz_rho[n0tab]**2) / rho[n0tab])

                    # D0(r)
                    D0 = (0.3 * (6.0 * np.pi**2)**(2.0 / 3.0)) * (rho ** (5.0 / 3.0))

                    # ELF(r)
                    ELF = np.zeros_like(D)
                    ELF[n0tab] = 1.0 / (1.0 + ((D[n0tab] + elf_reg)/ D0[n0tab])**2)

                    cube.write_field(ELF, fname + "_%s.cube"%i)

                    iq = ip + ELF.shape[-1]
                    f["elf_%s"%i][ip:iq] = np.asarray(ELF)
                    if save_rho:
                        f["rho_%s"%i][ip:iq] = np.asarray(rho)

                ip = iq
    f.close()

def plot_elf_pbc(mol, cube, dms, fname="ELF", elf_reg=1e-5, small_rho_tol=1e-8,
                 max_memory=None, spin_average=False, kpts=None, save_rho=True):
    """
    Electron localization function. Gamma point version.

    Args:
        mol: Mole object.
        cube: Cube object.
        dms: density matrix in AO basis, (nao, nao) or (2, nao, nao).
        fname: cube file name for visualization.
        elf_reg: convergence factor for ELF far from molecule.
        small_rho_tol: tolerance for discard small density values.

    """
    assert hasattr(mol, 'pbc_intor')
    from pyscf.pbc.dft import numint

    dms = np.asarray(dms)
    nao = dms.shape[-1]
    if kpts is None:
        kpts = np.zeros((1, 3))
        if dms.ndim == 2:
            dms = dms.reshape(1, nao, nao)
        elif dms.ndim == 3:
            dms = dms.reshape(2, 1, nao, nao)
        else:
            raise ValueError
    assert dms.ndim == 3 or (dms.ndim == 4 and dms.shape[0] == 2)
    nkpts = len(kpts)

    if max_memory is None:
        max_memory = mol.max_memory

    ao_deriv = 2
    ni = numint.KNumInt()
    grids = Grids(cube)
    ngrids = grids.ngrids

    f = h5py.File("%s.h5"%fname, 'w')
    f["n"] = np.asarray((cube.nx, cube.ny, cube.nz))
    f["coords"] = np.asarray(grids.coords)
    ip, iq = 0, 0

    if dms.ndim == 3:
        f.create_dataset("elf", (ngrids,), 'f8')
        if save_rho:
            f.create_dataset("rho", (ngrids,), 'f8')

        cube.write(None, fname + ".cube", comment='ELF value in real space (1/Bohr^3)', header_only=True)
        for _, ao, mask, weight, coords in \
                ni.block_loop(mol, grids, nao=nao, deriv=ao_deriv, kpts=kpts,
                              max_memory=max_memory):
            rho = ni.eval_rho(mol, ao, dms, mask, xctype='MGGA', hermi=1)
            rho, dx_rho, dy_rho, dz_rho, nab2_rho, tau = rho
            rho[rho <= 0.0] = 0.0
            n0tab = rho > small_rho_tol

            # D(r)
            D = tau
            D[n0tab] -= 0.125 * ((dx_rho[n0tab]**2 + dy_rho[n0tab]**2 + dz_rho[n0tab]**2) / rho[n0tab])

            # D0(r)
            D0 = (0.3 * (3.0 * np.pi**2)**(2.0 / 3.0)) * (rho ** (5.0 / 3.0))

            # ELF(r)
            ELF = np.zeros_like(D)
            ELF[n0tab] = 1.0 / (1.0 + ((D[n0tab] + elf_reg) / D0[n0tab])**2)

            cube.write_field(ELF, fname + ".cube")
            iq = ip + ELF.shape[-1]
            f["elf"][ip:iq] = np.asarray(ELF)
            if save_rho:
                f["rho"][ip:iq] = np.asarray(rho)
            ip = iq
    else:
        if spin_average:
            f.create_dataset("elf", (ngrids,), 'f8')
            if save_rho:
                f.create_dataset("rho", (ngrids,), 'f8')
            cube.write(None, fname + ".cube", comment='ELF value in real space (1/Bohr^3)', header_only=True)
            for _, ao, mask, weight, coords in \
                    ni.block_loop(mol, grids, nao=nao, deriv=ao_deriv, kpts=kpts,
                                  max_memory=max_memory):
                D_sum = 0.0
                D0_sum = 0.0
                rho_sum = 0.0
                n0tab_sum = np.ones(coords.shape[0], dtype=bool)
                for dm in dms:
                    rho = ni.eval_rho(mol, ao, dm, mask, xctype='MGGA', hermi=1)
                    rho, dx_rho, dy_rho, dz_rho, nab2_rho, tau = rho
                    rho[rho <= 0.0] = 0.0
                    n0tab = rho > small_rho_tol
                    n0tab_sum &= n0tab

                    # D(r)
                    D = tau
                    D[n0tab] -= 0.125 * ((dx_rho[n0tab]**2 + dy_rho[n0tab]**2 + dz_rho[n0tab]**2) / rho[n0tab])
                    D_sum += D

                    # D0(r)
                    D0 = (0.3 * (6.0 * np.pi**2)**(2.0 / 3.0)) * (rho ** (5.0 / 3.0))
                    D0_sum += D0

                    rho_sum += rho

                # ELF(r)
                ELF = np.zeros_like(D_sum)
                ELF[n0tab_sum] = 1.0 / (1.0 + ((D_sum[n0tab_sum] + elf_reg) / D0_sum[n0tab_sum])**2)

                cube.write_field(ELF, fname + ".cube")
                iq = ip + ELF.shape[-1]
                f["elf"][ip:iq] = np.asarray(ELF)
                if save_rho:
                    f["rho"][ip:iq] = np.asarray(rho_sum)

                ip = iq
        else:
            f.create_dataset("elf_0", (ngrids,), 'f8')
            f.create_dataset("elf_1", (ngrids,), 'f8')
            if save_rho:
                f.create_dataset("rho_0", (ngrids,), 'f8')
                f.create_dataset("rho_1", (ngrids,), 'f8')

            cube.write(None, fname + "_0.cube", comment='ELF value in real space (1/Bohr^3)', header_only=True)
            cube.write(None, fname + "_1.cube", comment='ELF value in real space (1/Bohr^3)', header_only=True)
            for _, ao, mask, weight, coords in \
                    ni.block_loop(mol, grids, nao=nao, deriv=ao_deriv, kpts=kpts,
                                  max_memory=max_memory):
                for i, dm in enumerate(dms):
                    rho = ni.eval_rho(mol, ao, dm, mask, xctype='MGGA', hermi=1)
                    rho, dx_rho, dy_rho, dz_rho, nab2_rho, tau = rho
                    rho[rho <= 0.0] = 0.0
                    n0tab = rho > small_rho_tol

                    # D(r)
                    D = tau
                    D[n0tab] -= 0.125 * ((dx_rho[n0tab]**2 + dy_rho[n0tab]**2 + dz_rho[n0tab]**2) / rho[n0tab])

                    # D0(r)
                    D0 = (0.3 * (6.0 * np.pi**2)**(2.0 / 3.0)) * (rho ** (5.0 / 3.0))

                    # ELF(r)
                    ELF = np.zeros_like(D)
                    ELF[n0tab] = 1.0 / (1.0 + ((D[n0tab] + elf_reg) / D0[n0tab])**2)

                    cube.write_field(ELF, fname + "_%s.cube"%i)
                    iq = ip + ELF.shape[-1]
                    f["elf_%s"%i][ip:iq] = np.asarray(ELF)
                    if save_rho:
                        f["rho_%s"%i][ip:iq] = np.asarray(rho)

                ip = iq
    f.close()

def get_ao_g_mol(mol, nx=40, ny=40, nz=40, resolution=None, coords=None):
    cc = cubegen.Cube(mol, nx, ny, nz, resolution)
    if coords is None:
        coords = cc.get_coords()
        ngrids = cc.get_ngrids()
    else:
        ngrids = coords.shape[0]
    blksize = min(8000, ngrids)
    nao = mol.nao_nr()
    orb_on_grid = np.empty((ngrids, nao))
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1])
        orb_on_grid[ip0:ip1] = ao
    return orb_on_grid

def get_ao_g_k(cell, kpts_abs, nx=80, ny=80, nz=80, resolution=None,
               margin=5.0, latt_vec=None, boxorig=None, box=None, coords=None):
    cc = cubegen.Cube(cell, nx, ny, nz, resolution, margin=margin,
                      latt_vec=latt_vec, boxorig=boxorig, box=box)
    if coords is None:
        coords = cc.get_coords()
        ngrids = cc.get_ngrids()
    else:
        ngrids = coords.shape[0]

    blksize = min(8000, ngrids)
    nao = cell.nao_nr()
    nkpts = len(kpts_abs)
    orb_on_grid = np.empty((nkpts, ngrids, nao))
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = np.asarray(cell.pbc_eval_gto('GTOval', coords[ip0:ip1], kpts=kpts_abs))
        if ao.ndim == 2: # 1 kpt
            ao = ao[np.newaxis]
        orb_on_grid[:, ip0:ip1] = ao
    return orb_on_grid

# ****************************************************************************
# plot spin-spin correlation functions
# ****************************************************************************

def get_rho_pair(ao_g, mo):
    assert (mo.ndim == 3)
    ngrids = ao_g.shape[0]
    spin, nao, nmo = mo.shape
    mo_g = np.zeros((spin, ngrids, nmo))
    for s in range(spin):
        mo_g[s] = ao_g.dot(mo[s])
    rho_sji = np.einsum('sgi, sgj -> sgji', mo_g.conj(), mo_g)
    return rho_sji

def eval_spin_corr_func_R(rho_pair0, rho_pair, rdm1, rdm2):
    """
    Spin-spin correlation function, <S(r0) S(r)>. Origin is r0.
    """
    spin, ng2, nmo, _ = rho_pair.shape
    cf = np.zeros((ng2,), dtype=rho_pair.dtype)

    # one-body terms
    rho_pair0 = rho_pair0.reshape((spin, nmo, nmo))
    rg0 = np.zeros((spin, nmo, nmo))
    for s in range(spin):
        rg0[s] = rho_pair0[s].conj().T.dot(rdm1[s])
        #cf += np.trace(rho_pair[s].dot(rg0[s]), axis1=1, axis2=2)
        cf += lib.einsum('gij, ji -> g', rho_pair[s], rg0[s])

    for s in range(spin):
        #tr2 = np.trace(rho_pair[s].dot(rdm1[s]), axis1=1, axis2=2)
        tr2 = lib.einsum('gij, ji -> g', rho_pair[s], rdm1[s])
        for t in range(spin):
            factor = (-1)**s * (-1)**t
            tr1 = rg0[t].trace()
            cf -= (factor*tr1)*tr2

    # two-body terms:
    rdm2_red = np.zeros((spin*spin, nmo, nmo))
    # t=0, s=0, ss=0
    rdm2_red[0] = lib.einsum('sr, srpq -> pq', rho_pair0[0].conj(), rdm2[0])
    # t=0, s=1, ss=1
    rdm2_red[1] = lib.einsum('sr, srpq -> pq', rho_pair0[0].conj(), rdm2[1])
    # t=1, s=1, ss=2
    rdm2_red[2] = lib.einsum('sr, srpq -> pq', rho_pair0[1].conj(), rdm2[2])
    # t=1, s=0, ss=3
    rdm2_red[3] = lib.einsum('sr, srpq -> pq', rho_pair0[1].conj(), rdm2[1].conj().transpose(2, 3, 0, 1))

    cf += lib.einsum('gqp, pq -> g', rho_pair[0], rdm2_red[0])
    cf += lib.einsum('gqp, pq -> g', rho_pair[1], rdm2_red[2])
    cf -= lib.einsum('gqp, pq -> g', rho_pair[1], rdm2_red[1])
    cf -= lib.einsum('gqp, pq -> g', rho_pair[0], rdm2_red[3])

    return cf

def get_rho_pair_q(ao_g, mo, q, nz, nxy, z_val):
    assert (mo.ndim == 3)
    ngrids = ao_g.shape[0]
    spin, nao, nmo = mo.shape
    assert(nz*nxy == ngrids)
    mo_g = np.zeros((spin, ngrids, nmo), dtype=np.complex128)
    for s in range(spin):
        mo_g[s] = ao_g.dot(mo[s])
    mo_g = mo_g.reshape((spin, nz, nxy, nmo))

    rho_sz = np.einsum('szai, szaj -> szji', mo_g.conj(), mo_g, optimize=True)
    qz = np.einsum('q, z -> qz', q, z_val)
    eiqz = np.exp(1j*qz)
    rho_q = np.einsum('szji, qz -> sqji', rho_sz, eiqz, optimize=True)
    return rho_q

def eval_spin_corr_func_q(rho_q, rdm1, rdm2, factor):
    """
    Stucture factor of 1D system.
    """
    spin, nq, nmo, _ = rho_q.shape
    cf = np.zeros((nq,), dtype=np.complex128)

    # one-body terms
    rdg = np.zeros((spin, nq, nmo, nmo), np.complex128)
    for s in range(spin):
        rdg[s] = (rho_q[s].conj().transpose((0, 2, 1))).dot(rdm1[s])
        # finite size correction, add 1.0 in the end.
        #cf += np.einsum('qij, qji -> q', rdg[s], rho_q[s], optimize=True)

    for s in range(spin):
        tr2 = lib.einsum('qij, ji -> q', rho_q[s], rdm1[s])
        for t in range(spin):
            factor = (-1)**s * (-1)**t
            tr1 = np.sum(rdg[t], axis=(1, 2))
            cf -= (factor) * tr1 * tr2

    # two-body terms:
    rdm2_red = np.zeros((spin*spin, nq, nmo, nmo), np.complex128)
    # t=0, s=0, ss=0
    rdm2_red[0] = lib.einsum('Qsr, srpq -> Qpq', rho_q[0].conj(), rdm2[0])
    # t=0, s=1, ss=1
    rdm2_red[1] = lib.einsum('Qsr, srpq -> Qpq', rho_q[0].conj(), rdm2[1])
    # t=1, s=1, ss=2
    rdm2_red[2] = lib.einsum('Qsr, srpq -> Qpq', rho_q[1].conj(), rdm2[2])
    # t=1, s=0, ss=3
    rdm2_red[3] = lib.einsum('Qsr, srpq -> Qpq', rho_q[1].conj(), rdm2[1].conj().transpose(2, 3, 0, 1))

    cf += np.einsum('Qqp, Qpq -> Q', rho_q[0], rdm2_red[0], optimize=True)
    cf += np.einsum('Qqp, Qpq -> Q', rho_q[1], rdm2_red[2], optimize=True)
    cf -= np.einsum('Qqp, Qpq -> Q', rho_q[1], rdm2_red[1], optimize=True)
    cf -= np.einsum('Qqp, Qpq -> Q', rho_q[0], rdm2_red[3], optimize=True)

    return cf*factor + 1.0

def eval_spin_corr_func_lo(rdm1_lo, rdm2_lo, idx1, idx2, Sz_only=False):
    r"""
    Evaluate the spin correlation function based on LO indices.
    \sum_{i in idx1, j in idx2} <S_i S_j>

    Args:
        rdm1_lo: rdm1 in lo
        rdm2_lo: rdm2 in lo
        idx1: idx for the first atom
        idx2: idx for the second atom

    Returns:
        a float number for correlation function value.
    """
    rdm1_a, rdm1_b = rdm1_lo
    rdm2_aa, rdm2_ab, rdm2_bb = rdm2_lo
    norb = rdm1_a.shape[-1]
    mesh_1 = np.ix_(idx1, idx2)
    mesh = np.ix_(idx1, idx1, idx2, idx2)

    # prod
    delta = np.eye(norb)
    S  = np.einsum('ij, ij ->', rdm1_a[mesh_1], delta[mesh_1], optimize=True)
    S += np.einsum('ij, ij ->', rdm1_b[mesh_1], delta[mesh_1], optimize=True)
    if Sz_only:
        S *= 0.25
    else:
        S *= 0.75

    # Az
    S += 0.25 * np.einsum("iijj ->", rdm2_aa[mesh], optimize=True)
    S += 0.25 * np.einsum("iijj ->", rdm2_bb[mesh], optimize=True)
    S -= 0.25 * np.einsum("iijj ->", rdm2_ab[mesh], optimize=True)
    S -= 0.25 * np.einsum("jjii ->", rdm2_ab[np.ix_(idx2, idx2, idx1, idx1)], optimize=True)

    # Axy
    if not Sz_only:
        S -= 0.5 * np.einsum('ijji ->', rdm2_ab[np.ix_(idx1, idx2, idx2, idx1)], optimize=True)
        S -= 0.5 * np.einsum('jiij ->', rdm2_ab[np.ix_(idx2, idx1, idx1, idx2)], optimize=True)
    return S

# ****************************************************************************
# plot density of states DOS
# ****************************************************************************

def get_dos(mo_energy, ndos=301, e_min=None, e_max=None, e_fermi=None,
            sigma=0.005, mo_coeff=None, ovlp=None, elist=None):
    """
    Compute density of states for a given set of MOs (with kpts).
    If mo_coeff is None, the total (spin-)dos is calculated,
    Otherwise, orbital-based (spin-)pdos is calculated.
    DOS shape: ((spin,) ndos)
    PDOS shape: ((spin,), nlo, ndos)

    Args:
        mo_energy: ((spin,), nkpts, nmo)
        ndos: number of points to plot
        e_min: left boundary of plot range
        e_max: right boundary of plot range
        e_fermi: fermi level, if given shift the zero as fermi level.
        sigma: smearing value
        mo_coeff: C_lo_mo for character analysis (PDOS),
                  shape ((spin,) nkpts, nlo, nmo)
        efermi

    Returns:
        elist: (ndos)
        dos: ((spin,), (nlo,), ndos)
    """
    mo_energy = np.asarray(mo_energy)
    if e_fermi is not None:
        mo_energy = mo_energy - e_fermi
    nkpts, nmo = mo_energy.shape[-2:]
    mo_energy_min = mo_energy.min()
    mo_energy_max = mo_energy.max()
    margin = max(10 * sigma, 0.05 * (mo_energy_max - mo_energy_min)) # margin
    if e_min is None:
        if e_fermi is None:
            e_min = mo_energy_min - margin
        else:
            e_min = max(-15.0 / HARTREE2EV, mo_energy_min) - 1.0
    if e_max is None:
        if e_fermi is None:
            e_max = mo_energy_max + margin
        else:
            e_max = min(15.0 / HARTREE2EV, mo_energy_max) + 1.0

    if elist is None:
        elist = np.linspace(e_min, e_max, ndos)
    ndos = ne = len(elist)
    norm = sigma * np.sqrt(2 * np.pi)
    tsigma = 2.0 * sigma ** 2
    if mo_energy.ndim == 2:
        if mo_coeff is None: # total dos
            dos = np.zeros_like(elist)
            for i, e_curr in enumerate(elist):
                dos[i] = np.sum(np.exp(-((mo_energy-e_curr)**2) / tsigma))
        else: # pdos
            mo_coeff = np.asarray(mo_coeff)
            nao, nmo = mo_coeff.shape[-2:]
            dos = np.zeros((nao, ndos))

            if ovlp is None:
                log.warn("plot PDOS ovlp is not given ... Use identity instead.")
                ovlp = np.zeros((nkpts, nao, nao), dtype=complex)
                ovlp[:, range(nao), range(nao)] = 1.0
            mo_sq = np.einsum('kpm, kpq, kqm -> pkm',
                              mo_coeff.conj(), ovlp, mo_coeff,
                              optimize=True).real
            for i, e_curr in enumerate(elist):
                # pkm, km -> p
                dos[:, i] = np.sum(mo_sq * np.exp(-((mo_energy-e_curr)**2) / tsigma),
                                   axis=(1, 2))
    else:
        spin = mo_energy.shape[0]
        if mo_coeff is None:
            dos = np.zeros((spin,) + elist.shape)
            for s in range(spin):
                for i, e_curr in enumerate(elist):
                    dos[s, i] = np.sum(np.exp(-((mo_energy[s]-e_curr)**2) / tsigma))
        else:
            mo_coeff = np.asarray(mo_coeff)
            nao, nmo = mo_coeff.shape[-2:]
            dos = np.zeros((spin, nao) + elist.shape)
            if ovlp is None:
                log.warn("plot PDOS ovlp is not given ... Use identity instead.")
                ovlp = np.zeros((nkpts, nao, nao), dtype=complex)
                ovlp[:, range(nao), range(nao)] = 1.0
            mo_sq = np.einsum('skpm, kpq, skqm -> spkm',
                              mo_coeff.conj(), ovlp, mo_coeff,
                              optimize=True).real
            for s in range(spin):
                for i, e_curr in enumerate(elist):
                    # pkm, km -> pkm -> p
                    dos[s, :, i] = np.sum(mo_sq[s] * np.exp(-((mo_energy[s] - e_curr)**2) / tsigma),
                                          axis=(1, 2))
    return elist, dos / (nkpts * norm)

def get_dos_k(mo_energy, ndos=301, e_min=None, e_max=None, e_fermi=None,
              sigma=0.005, mo_coeff=None, ovlp=None, elist=None):
    """
    Compute density of states (per k-point) for a given set of MOs (with kpts).
    If mo_coeff is None, the total (spin-)dos is calculated,
    Otherwise, orbital-based (spin-)pdos is calculated.
    DOS shape: ((spin,), nkpts, ndos)
    PDOS shape: ((spin,), nkpts, nlo, ndos)

    Args:
        mo_energy: ((spin,), nkpts, nmo)
        ndos: number of points to plot
        e_min: left boundary of plot range
        e_max: right boundary of plot range
        e_fermi: fermi level, if given shift the zero as fermi level.
        sigma: smearing value
        mo_coeff: C_lo_mo for character analysis (PDOS),
                  shape ((spin,) nkpts, nlo, nmo)
        efermi

    Returns:
        elist: (ndos)
        dos: ((spin,), nkpts, (nlo,), ndos)
    """
    mo_energy = np.asarray(mo_energy)
    if e_fermi is not None:
        mo_energy = mo_energy - e_fermi
    nkpts, nmo = mo_energy.shape[-2:]
    mo_energy_min = mo_energy.min()
    mo_energy_max = mo_energy.max()
    margin = max(10 * sigma, 0.05 * (mo_energy_max - mo_energy_min)) # margin
    if e_min is None:
        if e_fermi is None:
            e_min = mo_energy_min - margin
        else:
            e_min = max(-15.0 / HARTREE2EV, mo_energy_min) - 1.0
    if e_max is None:
        if e_fermi is None:
            e_max = mo_energy_max + margin
        else:
            e_max = min(15.0 / HARTREE2EV, mo_energy_max) + 1.0

    if elist is None:
        elist = np.linspace(e_min, e_max, ndos)
    ndos = ne = len(elist)
    norm = sigma * np.sqrt(2 * np.pi)
    tsigma = 2.0 * sigma ** 2
    if mo_energy.ndim == 2:
        if mo_coeff is None: # total dos
            dos = np.zeros((nkpts, ne))
            for i, e_curr in enumerate(elist):
                dos[:, i] = np.einsum('km -> k', np.exp(-((mo_energy-e_curr)**2) / tsigma))
        else: # pdos
            mo_coeff = np.asarray(mo_coeff)
            nao, nmo = mo_coeff.shape[-2:]
            dos = np.zeros((nkpts, nao, ndos))
            if ovlp is None:
                log.warn("plot PDOS ovlp is not given ... Use identity instead.")
                ovlp = np.zeros((nkpts, nao, nao), dtype=complex)
                ovlp[:, range(nao), range(nao)] = 1.0
            mo_sq = np.einsum('kpm, kpq, kqm -> pkm',
                              mo_coeff.conj(), ovlp, mo_coeff,
                              optimize=True).real
            for i, e_curr in enumerate(elist):
                dos[:, :, i] = np.einsum('pkm, km -> kp', mo_sq, np.exp(-((mo_energy-e_curr)**2) / tsigma))
    else:
        spin = mo_energy.shape[0]
        if mo_coeff is None:
            dos = np.zeros((spin, nkpts,) + elist.shape)
            for s in range(spin):
                for i, e_curr in enumerate(elist):
                    dos[s, :, i] = np.einsum('km -> k', np.exp(-((mo_energy[s]-e_curr)**2) / tsigma))
        else:
            mo_coeff = np.asarray(mo_coeff)
            nao, nmo = mo_coeff.shape[-2:]
            dos = np.zeros((spin, nkpts, nao) + elist.shape)
            if ovlp is None:
                log.warn("plot PDOS ovlp is not given ... Use identity instead.")
                ovlp = np.zeros((nkpts, nao, nao), dtype=complex)
                ovlp[:, range(nao), range(nao)] = 1.0
            mo_sq = np.einsum('skpm, kpq, skqm -> spkm',
                              mo_coeff.conj(), ovlp, mo_coeff,
                              optimize=True).real
            for s in range(spin):
                for i, e_curr in enumerate(elist):
                    # pkm, km -> pkm -> p
                    dos[s, :, :, i] = np.einsum('pkm, km -> kp', mo_sq[s], np.exp(-((mo_energy[s] - e_curr)**2) / tsigma))
    return elist, dos / (norm)

def plot_dos(elist, pdos, idx_dic=None, color_dic=None,
             fig_size=(12, 6), fig_name="pdos.pdf", unit='eV', text=None,
             **kwargs):
    """
    Plot (projected) density of states.

    Args:
        elist: energy range, shape (ndos,)
        pdos: density of states, shape ((nlo,), ndos)
        idx_dic: a dictionary required for pdos plotting,
                 should be {"orbital name", idx}
        color_dic: a dictionary for pdos coloring,
                 should be {"orbital name", "color name"},
                 if provided, only plot the lines that have color.
        fig_size: size of figure, default is (12, 6)
        fig_name: figure name
        unit: unit of E in figure
        text: a label on the left upper corner.

    Returns:
        plt: matplotlib plot object.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    ax = plt.gca()

    if unit == 'eV':
        elist = np.asarray(elist) * HARTREE2EV

    if pdos.ndim == 1: # restricted total DOS
        plt.plot(elist, pdos, label='total', color='grey', linewidth=1)
    elif pdos.ndim == 2 and (idx_dic is not None): # restricted PDOS
        dos = pdos.sum(axis=0)
        plt.plot(elist, dos, label='total', color='grey', linewidth=1)
        if color_dic is None:
            for orb_name, idx in idx_dic.items():
                pdos_i = pdos[idx].sum(axis=0)
                plt.plot(elist, pdos_i, label=orb_name, linewidth=1)
        else:
            for orb_name, idx in idx_dic.items():
                if orb_name in color_dic:
                    pdos_i = pdos[idx].sum(axis=0)
                    plt.plot(elist, pdos_i, label=orb_name,
                             color=color_dic[orb_name], linewidth=1)
    elif pdos.ndim == 2 and (idx_dic is None): # unrestricted total DOS
        assert pdos.shape[0] == 2
        plt.plot(elist, pdos[0], label='total', color='grey', linewidth=1)
        plt.plot(elist, -pdos[1], color='grey', linewidth=1)
    elif pdos.ndim == 3: # unrestricted PDOS
        assert idx_dic is not None
        dos = pdos.sum(axis=1)
        plt.plot(elist, dos[0], label='total', color='grey', linewidth=1)
        plt.plot(elist, -dos[1], color='grey', linewidth=1)
        if color_dic is None:
            for orb_name, idx in idx_dic.items():
                pdos_i = pdos[:, idx].sum(axis=1)
                tmp = plt.plot(elist, pdos_i[0], label=orb_name,
                               linewidth=1)[0]
                plt.plot(elist, -pdos_i[1], color=tmp.get_color(),
                         linewidth=1)
        else:
            for orb_name, idx in idx_dic.items():
                if orb_name in color_dic:
                    pdos_i = pdos[:, idx].sum(axis=1)
                    tmp = plt.plot(elist, pdos_i[0], label=orb_name,
                                   color=color_dic[orb_name], linewidth=1)[0]
                    plt.plot(elist, -pdos_i[1], color=tmp.get_color(),
                             linewidth=1)
    else:
        raise ValueError("Unknown pdos shape %s" %(str(pdos.shape)))

    ax.legend(fancybox=False, framealpha=1.0, edgecolor='black', fontsize=10,
              frameon=False, loc='upper right')

    # plot efermi line
    efermi_x = [0.0, 0.0]
    efermi_y = ax.get_ylim()
    plt.plot(efermi_x, efermi_y, linestyle='--', color='black', linewidth=1)
    ax.set_ylim(efermi_y)

    plt.xlabel("$E$ [%s]"%(unit), fontsize=10)
    plt.ylabel("PDOS", fontsize=10)
    if text is not None:
        plt.text(0.02, 0.96, text, horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes,
                 fontsize=10)
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    return plt

def plot_bands(ax, kdis, ew, weights=None, cmap=None, linewidth=4, alpha=1.0):
    """
    Plot bands for given ax object.

    Args:
        ax:
        kdis: kpoint distance, (nkpts)
        ew: mo energy, (nkpts, nbands)
        weights: weights for color map, should be 0 - 1, (nkpts, nbands)
        cmap: colormap type.
        linewidth: line width.

    Returns:
        line: collection of lines.
    """
    norm = Normalize(0.0, 1.0)
    kdis = np.asarray(kdis)
    ew   = np.asarray(ew)
    nbands = ew.shape[-1]
    if weights is None:
        weights = np.ones_like(ew)
    if cmap is None:
        cmap = 'Greys'

    for n in range(nbands):
        x = kdis
        y = ew[:, n]
        points = np.array((x, y)).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        dydx = weights[:, n]
        lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=alpha)
        lc.set_array(dydx)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)

    return line


def get_kdis(kpts, kpts_sp=None, latt_vec=None, tol=1e-10):
    """
    Get k-distance in the BZ.

    Args:
        kpts: scaled kpts for bands, (nkpts, 3).
        kpts_sp: special kpoints for segmentation, (nkpts_sp, 3),
                 the path go through these points should include kpts.
        tol: tolerance for zero.
        latt_vec: lattice vector for non-cubic cell.

    Returns:
        kdis: distance in the BZ, use for plot bands.
        kdis_sp: distance for special kpts.
    """
    from libdmet.system.fourier import frac2real, real2frac
    def get_pos(kpt, kleft, kright):
        diff0 = kpt - kleft
        diff1 = kpt - kright
        prod = diff0 * diff1

        if max_abs(diff0) < tol:
            pos = 'left'
        elif max_abs(diff1) < tol:
            pos = 'right'
        elif (prod < tol).all():
            for i, ele in enumerate(prod):
                if abs(ele) < tol and abs(kleft[i] - kright[i]) > tol:
                    pos = 'out'
                    break
            else:
                pos = 'in'
        else:
            pos = 'out'
        return pos


    if kpts_sp is None:
        kdis = np.diff(kpts, axis=0)
        if latt_vec is not None:
            kdis_sp = frac2real(latt_vec, kdis_sp)
            kdis = frac2real(latt_vec, kdis)
        kdis = np.hstack((0.0, np.cumsum(la.norm(kdis, axis=1))))
        kdis_sp = None
    else:
        kdis_sp = np.diff(kpts_sp, axis=0)
        if latt_vec is not None:
            kdis_sp = frac2real(latt_vec, kdis_sp)
        kdis_sp = np.hstack((0.0, np.cumsum(la.norm(kdis_sp, axis=1))))
        kdis = []
        idx0 = 0
        for kpt in kpts:
            for i in range(idx0, len(kpts_sp) - 1):
                pos = get_pos(kpt, kpts_sp[idx0], kpts_sp[idx0+1])
                if pos == 'right':
                    idx0 += 1
                elif pos == 'out':
                    idx0 += 1
                    continue
                break
            else:
                raise ValueError("kpt (%s) are not in the special kpoints path."%(kpt))
            if latt_vec is not None:
                kdis.append(kdis_sp[idx0] + la.norm(frac2real(latt_vec, kpt - kpts_sp[idx0])))
            else:
                kdis.append(kdis_sp[idx0] + la.norm(kpt - kpts_sp[idx0]))

        kdis = np.asarray(kdis)
    return kdis, kdis_sp


def get_fermi_surface(mo_coeff, mo_energy, latt, mu=0.0, sigma=0.1, latt_uc=None,
                      fname=None):
    """
    Compute the fermi surface.

    Args:
        mo_coeff: (nkpts, nao, nmo) MO coefficients.
        mo_energy: (nkpts, nmo) MO energies.
        mu: energy level to compute.
        sigma: smearing for gaussian.
        latt: lattice.
        latt_uc: if given, will unfold the fermi_surface to the unit cell.
        fname: if given, will write the fs to a file named fname.

    Returns:
        kpts_scaled: kpts scaled.
        fs: fermi surface.
    """
    mo_energy = np.asarray(mo_energy)
    nmo = mo_energy.shape[-1]
    f_occ = mo_energy - mu
    f_occ **= 2
    f_occ /= (-2 * sigma**2)
    f_occ = np.exp(f_occ)
    f_occ /= (np.sqrt(2 * np.pi * sigma**2) * nmo)

    if latt_uc is not None:
        from libdmet.system.fourier import unfold_mo_coeff, unfold_mo_energy
        mo_coeff = np.asarray(mo_coeff)
        mo_coeff_uc = unfold_mo_coeff(mo_coeff, latt_uc, latt)
        weights = np.einsum('kpm, kpm -> km', mo_coeff_uc.conj(), mo_coeff_uc, optimize=True).real
        f_occ_uc = unfold_mo_energy(f_occ, latt_uc, latt, tol=1e-10)
        f_occ_uc *= (weights * (latt_uc.nkpts / latt.nkpts))
        f_occ_uc = f_occ_uc.sum(axis=-1)
        f_occ = f_occ_uc
        kpts_scaled = latt_uc.kpts_scaled
    else:
        f_occ = f_occ.sum(axis=-1)
        kpts_scaled = latt.kpts_scaled

    if fname is not None:
        with open(fname, 'w') as f:
            for kpt, fo in zip(kpts_scaled, f_occ):
                f.write("%15.8f %15.8f %15.8f %15.8f \n"%(*kpt, fo))

    return kpts_scaled, f_occ

get_fs = get_fermi_surface

def plot_fermi_surface(fname, fname_save="fs.pdf", wrap_around=True, vmin=None, vmax=None,
                       cmap='viridis', interpolation=None):
    xs, ys, zs, f_occs = np.loadtxt(fname).T
    if wrap_around:
        xs_new = []
        ys_new = []
        vals_new = []
        for x, y, z, gap in zip(xs, ys, zs, f_occs):
            if z == -0.5:
                continue
            if x == -0.5:
                xs_new.append(0.5)
                ys_new.append(y)
                vals_new.append(gap)
                if y == -0.5:
                    xs_new.append(0.5)
                    ys_new.append(0.5)
                    vals_new.append(gap)

            if y == -0.5:
                xs_new.append(x)
                ys_new.append(0.5)
                vals_new.append(gap)
            xs_new.append(x)
            ys_new.append(y)
            vals_new.append(gap)

    # sort
    seq = list(zip(xs_new, ys_new))
    idx = sorted(range(len(seq)), key=seq.__getitem__)

    xs_new = np.asarray(xs_new)[idx]
    ys_new = np.asarray(ys_new)[idx]
    vals_new = np.asarray(vals_new)[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    nkx = int(np.sqrt(len(xs_new)) + 0.5)
    nky = int(np.sqrt(len(ys_new)) + 0.5)
    xxx = plt.imshow(vals_new.reshape(nkx, nky), interpolation=interpolation,
                     vmin=vmin, vmax=vmax, cmap=cmap)

    cbar = plt.colorbar(xxx)
    cbar.ax.tick_params(labelsize=15, bottom=False, top=False, left=False,
                        right=True, width=1.5)

    cbar.ax.set_ylabel(r'DOS', fontsize=20, labelpad=10.0)
    cbar.ax.yaxis.set_label_position("right")

    plt.xlabel("$k_{x}$", fontsize=20)
    plt.ylabel("$k_{y}$", fontsize=20)
    ax.tick_params(labelsize=15, width=1.5)

    plt.savefig(fname_save, dpi=300)

plot_fs = plot_fermi_surface
