#! /usr/bin/env python

"""
Projection functions for wannier90 guess.

Original code:
pyWannier90: Wannier90 for PySCF, https://github.com/hungpham2017/pyWannier90
Author:
    Hung Q. Pham <pqh3.14@gmail.com>

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

Modified by
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import numpy as np
import scipy.linalg as la
import pyscf.lib.parameters as param

MR_DIC = {''   : 1, \
          'x'  : 2, 'y'  : 3, 'z'   : 1, \
          'xy' : 5, 'yz' : 3, 'z^2' : 1, 'xz' : 2, 'x2-y2': 4 , \
          'y^3': 7, 'xyz': 5, 'yz^2': 3, 'z^3': 1, 'xz^2' : 2, 'zx^2': 4, 'x^3': 6}

def angle(v1, v2):
    """
    Return the angle (in radiant) between v1 and v2.
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    cosa = v1.dot(v2) / (la.norm(v1) * la.norm(v2))
    return np.arccos(cosa)

def transform(x_vec, z_vec):
    """
    Construct a transformation matrix to transform r_vec to
    the new coordinate system defined by x_vec and z_vec.
    """
    x_vec = np.asarray(x_vec)
    z_vec = np.asarray(z_vec)
    x_vec = x_vec / la.norm(x_vec)
    z_vec = z_vec / la.norm(z_vec)
    assert x_vec.dot(z_vec) == 0
    y_vec = np.cross(z_vec, x_vec)
    new = np.asarray((x_vec, y_vec, z_vec))
    original = np.eye(3)

    tran_matrix = np.empty((3, 3))
    for row in range(3):
        for col in range(3):
            tran_matrix[row, col] = np.cos(angle(original[row], new[col]))
    return tran_matrix

def R_r(r_norm, r=1, zona=1):
    r"""
    Radial functions used to compute \Theta_{l,m_r}(\theta,\phi)
    """
    if r == 1:
        R_r = 2.0 * zona**(1.5) * np.exp(-zona * r_norm)
    elif r == 2:
        R_r = 1.0 / (2.0 * np.sqrt(2.0)) * zona**(1.5) * (2.0 - zona*r_norm) \
                * np.exp(-zona * r_norm * 0.5)
    elif r == 3:
        R_r = np.sqrt(4.0 / 27.0) * zona**(1.5) * (1.0 - 2.0 / 3.0 * zona \
                * r_norm  + 2.0 / 27.0 * (zona**2) * (r_norm**2)) \
                * np.exp(-zona * r_norm / 3.0)
    else:
        raise ValueError
    return R_r

def theta(func, cost, phi):
    r"""
    Basic angular functions (s,p,d,f) used to compute
    \Theta_{l,m_r}(\theta,\phi)
    """
    sint = np.sqrt(1.0 - cost**2)
    if func == 's':
        theta = 1.0 / np.sqrt(4.0 * np.pi) * np.ones(cost.shape[0])
    elif func == 'pz':
        theta = np.sqrt(3.0 / (4.0 * np.pi)) * cost
    elif func == 'px':
        theta = np.sqrt(3.0 / (4.0 * np.pi)) * sint * np.cos(phi)
    elif func == 'py':
        theta = np.sqrt(3.0 / (4.0 * np.pi)) * sint * np.sin(phi)
    elif func == 'dz2':
        theta = np.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * cost**2 - 1.0)
    elif func == 'dxz':
        theta = np.sqrt(15.0 / (4.0 * np.pi)) * sint * cost * np.cos(phi)
    elif func == 'dyz':
        theta = np.sqrt(15.0 / (4.0 * np.pi)) * sint * cost * np.sin(phi)
    elif func == 'dx2-y2':
        theta = np.sqrt(15.0 / (16.0 * np.pi)) * (sint**2) * np.cos(2.0 * phi)
    elif func == 'dxy':
        theta = np.sqrt(15.0 / (16.0 * np.pi)) * (sint**2) * np.sin(2.0 * phi)
    elif func == 'fz3':
        theta = np.sqrt(7.0) / (4.0 * np.sqrt(np.pi)) * \
                (5.0 * cost**3 - 3.0 * cost)
    elif func == 'fxz2':
        theta = np.sqrt(21.0) / (4.0 * np.sqrt(2.0 * np.pi)) \
                * (5.0 * cost**2 - 1.0) * sint * np.cos(phi)
    elif func == 'fyz2':
        theta = np.sqrt(21.0) / (4.0 * np.sqrt(2.0 * np.pi)) \
                * (5.0 * cost**2 - 1.0) * sint * np.sin(phi)
    elif func == 'fz(x2-y2)':
        theta = np.sqrt(105.0) / (4.0 * np.sqrt(np.pi)) \
                * sint**2 * cost * np.cos(2.0 * phi)
    elif func == 'fxyz':
        theta = np.sqrt(105.0) / (4.0 * np.sqrt(np.pi)) \
                * sint**2 * cost * np.sin(2.0 * phi)
    elif func == 'fx(x2-3y2)':
        theta = np.sqrt(35.0) / (4.0 * np.sqrt(2.0 * np.pi)) \
                * sint**3 * (np.cos(phi)**2 - 3.0 * np.sin(phi)**2) \
                * np.cos(phi)
    elif func == 'fy(3x2-y2)':
        theta = np.sqrt(35.0) / (4.0 * np.sqrt(2.0 * np.pi)) \
                * sint**3 * (3.0 * np.cos(phi)**2 - np.sin(phi)**2) \
                * np.sin(phi)
    else:
        raise ValueError
    return theta

def theta_lmr(l, mr, cost, phi):
    r"""
    Compute the value of \Theta_{l,m_r}(\theta,\phi)
    ref: Table 3.1 and 3.2 of Chapter 3, wannier90 User Guide
    """
    assert l in [0, 1, 2, 3, -1, -2, -3, -4, -5]
    assert mr in [1, 2, 3, 4, 5, 6, 7]
    if l == 0: # s
        theta_lmr = theta('s', cost, phi)
    elif l == 1:
        if   mr == 1: # pz
            theta_lmr = theta('pz', cost, phi)
        elif mr == 2: # px
            theta_lmr = theta('px', cost, phi)
        elif mr == 3: # py
            theta_lmr = theta('py', cost, phi)
        else:
            raise ValueError
    elif l == 2:
        if   mr == 1: # dz2
            theta_lmr = theta('dz2', cost, phi)
        elif mr == 2: # dxz
            theta_lmr = theta('dxz', cost, phi)
        elif mr == 3: # dyz
            theta_lmr = theta('dyz', cost, phi)
        elif mr == 4: # dx2-y2
            theta_lmr = theta('dx2-y2', cost, phi)
        elif mr == 5: # dxy
            theta_lmr = theta('dxy', cost, phi)
        else:
            raise ValueError
    elif l == 3:
        if   mr == 1: # fz3
            theta_lmr = theta('fz3', cost, phi)
        elif mr == 2: # fxz2
            theta_lmr = theta('fxz2', cost, phi)
        elif mr == 3: # fyz2
            theta_lmr = theta('fyz2', cost, phi)
        elif mr == 4: # fz(x2-y2)
            theta_lmr = theta('fz(x2-y2)', cost, phi)
        elif mr == 5: # fxyz
            theta_lmr = theta('fxyz', cost, phi)
        elif mr == 6: # fx(x2-3y2)
            theta_lmr = theta('fx(x2-3y2)', cost, phi)
        elif mr == 7: # fy(3x2-y2)
            theta_lmr = theta('fy(3x2-y2)', cost, phi)
        else:
            raise ValueError
    elif l == -1:
        if   mr == 1: # sp-1
            theta_lmr = 1.0 / np.sqrt(2.0) * (theta('s', cost, phi) \
                    + theta('px', cost, phi))
        elif mr == 2: # sp-2
            theta_lmr = 1.0 / np.sqrt(2.0) * (theta('s', cost, phi) \
                    - theta('px', cost, phi))
        else:
            raise ValueError
    elif l == -2:
        if   mr == 1: # sp2-1
            theta_lmr = 1.0 / np.sqrt(3.0) * theta('s', cost, phi) \
                    - 1.0 / np.sqrt(6.0) * theta('px', cost, phi) \
                    + 1.0 / np.sqrt(2.0) * theta('py', cost, phi)
        elif mr == 2: # sp2-2
            theta_lmr = 1.0 / np.sqrt(3.0) * theta('s', cost, phi) \
                    - 1.0 / np.sqrt(6.0) * theta('px', cost, phi) \
                    - 1.0 / np.sqrt(2.0) * theta('py', cost, phi)
        elif mr == 3: # sp2-3
            theta_lmr = 1.0 / np.sqrt(3.0) * theta('s', cost, phi) \
                    + 2.0 / np.sqrt(6.0) * theta('px', cost, phi)
        else:
            raise ValueError
    elif l == -3:
        if   mr == 1: # sp3-1
            theta_lmr = 0.5 * (theta('s', cost, phi) + theta('px', cost, phi) \
                    + theta('py', cost, phi) + theta('pz', cost, phi))
        elif mr == 2: # sp3-2
            theta_lmr = 0.5 * (theta('s', cost, phi) + theta('px', cost, phi) \
                    - theta('py', cost, phi) - theta('pz', cost, phi))
        elif mr == 3: # sp3-3
            theta_lmr = 0.5 * (theta('s', cost, phi) - theta('px', cost, phi) \
                    + theta('py', cost, phi) - theta('pz', cost, phi))
        elif mr == 4: # sp3-4
            theta_lmr = 0.5 * (theta('s', cost, phi) - theta('px', cost, phi) \
                    - theta('py', cost, phi) + theta('pz', cost, phi))
    elif l == -4:
        if   mr == 1: # sp3d-1
            theta_lmr = 1.0 / np.sqrt(3.0) * theta('s', cost, phi) \
                    - 1.0 / np.sqrt(6.0) * theta('px', cost, phi) \
                    + 1.0 / np.sqrt(2.0) * theta('py', cost, phi)
        elif mr == 2: # sp3d-2
            theta_lmr = 1.0 / np.sqrt(3.0) * theta('s', cost, phi) \
                    - 1.0 / np.sqrt(6.0) * theta('px', cost, phi) \
                    - 1.0 / np.sqrt(2.0) * theta('py', cost, phi)
        elif mr == 3: # sp3d-3
            theta_lmr = 1.0 / np.sqrt(3.0) * theta('s', cost, phi) \
                    + 2.0 / np.sqrt(6.0) * theta('px', cost, phi)
        elif mr == 4: # sp3d-4
            theta_lmr = 1.0 / np.sqrt(2.0) * (theta('pz', cost, phi) \
                    + theta('dz2', cost, phi))
        elif mr == 5: # sp3d-5
            theta_lmr = 1.0 / np.sqrt(2.0) * (-theta('pz', cost, phi) \
                    + theta('dz2', cost, phi))
        else:
            raise ValueError
    elif l == -5:
        if   mr == 1: # sp3d2-1
            theta_lmr = 1.0 / np.sqrt(6.0) * theta('s', cost, phi) \
                    - 1.0 / np.sqrt(2.0) * theta('px', cost, phi) \
                    - 1.0 / np.sqrt(12.0) * theta('dz2', cost, phi) \
                    + 0.5 * theta('dx2-y2', cost, phi)
        elif mr == 2: # sp3d2-2
            theta_lmr = 1.0 / np.sqrt(6.0) * theta('s', cost, phi) \
                    + 1.0 / np.sqrt(2.0) * theta('px', cost, phi) \
                    - 1.0 / np.sqrt(12.0) * theta('dz2', cost, phi) \
                    + 0.5 * theta('dx2-y2', cost, phi)
        elif mr == 3: # sp3d2-3
            theta_lmr = 1.0 / np.sqrt(6.0) * theta('s', cost, phi) \
                    - 1.0 / np.sqrt(2.0) * theta('py', cost, phi) \
                    - 1.0 / np.sqrt(12.0) * theta('dz2', cost, phi) \
                    - 0.5 *theta('dx2-y2', cost, phi)
        elif mr == 4: # sp3d2-4
            theta_lmr = 1.0 / np.sqrt(6.0) * theta('s', cost, phi) \
                    + 1.0 / np.sqrt(2.0) * theta('py', cost, phi) \
                    - 1.0 / np.sqrt(12.0) * theta('dz2', cost, phi) \
                    - 0.5 *theta('dx2-y2', cost, phi)
        elif mr == 5: # sp3d2-5
            theta_lmr = 1.0 / np.sqrt(6.0) * theta('s', cost, phi) \
                    - 1.0 / np.sqrt(2.0) * theta('pz', cost, phi) \
                    + 1.0 / np.sqrt(3.0) *theta('dz2', cost, phi)
        elif mr == 6: # sp3d2-6
            theta_lmr = 1.0 / np.sqrt(6.0) * theta('s', cost, phi) \
                    + 1.0 / np.sqrt(2.0) * theta('pz', cost, phi) \
                    + 1.0 / np.sqrt(3.0) *theta('dz2', cost, phi)
        else:
            raise ValueError
    else:
        raise ValueError
    return theta_lmr

def g_r(grids_coor, site, l, mr, r, zona, x_axis=np.array([1.0, 0.0, 0.0]), \
        z_axis=np.array([0.0, 0.0, 1.0]), unit='B'):
    r"""
    Evaluate projection function g(r) or \Theta_{l,m_r}(\theta,\phi) on grid
    ref: Chapter 3, wannier90 User Guide
    Args:
        grids_coor : a grids for the cell of interest
        site       : absolute coordinate (in Borh/Angstrom)
                     of the g(r) in the cell
        l, mr      : l and mr value in the Table 3.1 and 3.2 of the ref

    Returns:
        g_r        : an array (ngrid, value) of g(r)
    """
    from libdmet.utils.misc import cart2sph
    if unit == 'A':
        unit_conv = param.BOHR
    else:
        unit_conv = 1.0
    r_vec = grids_coor - site
    r_vec = np.dot(r_vec, transform(x_axis, z_axis))
    r_norm, theta, phi = cart2sph(r_vec[:, 0], r_vec[:, 1], r_vec[:, 2])
    cost = np.cos(theta)
    return theta_lmr(l, mr, cost, phi) * R_r(r_norm*unit_conv, r=r, zona=zona)

def get_proj_string(cell, idx_all, zaxis=[0, 0, 1], xaxis=[1, 0, 0]):
    """
    Get a list of projection string for wannier90.

    Args:
        cell: cell.
        idx_all: a dic for all orbitals.
        zaxis: zaxis.
        xaxis: xaxis.

    Returns:
        a list: ["f=x, y, z : l, mr : z=z1, z2, z3 : x=x1, x2, x3", ...].
    """
    from libdmet.system.lattice import real2frac
    coords = [real2frac(cell.lattice_vectors(), atom[1]) for atom in cell._atom]
    string = []
    for lab, idx in idx_all.items():
        lab_sp = lab.split()
        atom_id = int(lab_sp[0])
        l = param.ANGULARMAP[lab_sp[-1][1]]
        mr = MR_DIC[lab_sp[-1][2:]]
        string.append("f=%15.10f, %15.10f, %15.10f:l=%s, mr=%s:z=%s, %s, %s:x=%s, %s, %s"\
                %(*coords[atom_id], l, mr, *zaxis, *xaxis))
    return string
