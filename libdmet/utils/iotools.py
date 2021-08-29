#! /usr/bin/env python

"""
I/O routines for crystal structure.

Author:
    Zhi-Hao Cui
    Bo-Xiao Zheng
"""

import os
import sys
from collections import OrderedDict
import numpy as np
import scipy.linalg as la

from pyscf import lib
from pyscf.data.nist import BOHR
from libdmet.utils import logger as log

def build_unitcell(cellsize, atoms, frac=False, center=None):
    import libdmet.system.lattice as lat
    sites = [t[0] for t in atoms]
    if frac:
        sites = map(lambda s: lat.Frac2Real(cellsize, s), sites)
        if center is not None:
            center = lat.Frac2Real(cellsize, center)

    if center is not None:
        T = 0.5 * np.sum(cellsize, axis = 0) - center
        sites = [s + T for s in sites]

    sites_names = [(s, a[1]) for s, a in zip(sites, atoms)]
    return lat.UnitCell(cellsize, sites_names)

def __stat(lst):
    """
    lst: a list of names.
    """
    dic = OrderedDict()
    for idx, name in enumerate(lst):
        if name in dic:
            dic[name].append(idx)
        else:
            dic[name] = [idx]
    return dic

def sc2POSCAR(sc, fout, elements=None, vac=15.0):
    """
    supercell to POSCAR.
    
    Args:
        sc: SuperCell object.
        fout: can be a file object or a string.
        elements: the names of element.
        vac: for low-dimensional system, the vacuum length.
    """
    sites = sc.sites
    name_dict = __stat(sc.names)
    if elements is not None:
        log.eassert(set(elements) == set(name_dict.keys()), "specified elements different from unitcell data")
    else:
        elements = name_dict.keys()
    
    if isinstance(fout, str):
        fwrite = open(fout, 'w')
    else:
        fwrite = fout
    fwrite.write(" ".join(elements))
    fwrite.write("  generated using libdmet.utils.iotools\n")
    fwrite.write("%10.6f\n" % 1)
    latt_vec = np.eye(3) * vac
    latt_vec[:sc.dim, :sc.dim] = sc.size
    for d in range(3):
        fwrite.write("%20.12f%20.12f%20.12f\n" % tuple(latt_vec[d]))
    for key in elements:
        fwrite.write("%4s " % key)
    fwrite.write("\n")
    for key in elements:
        fwrite.write("%4d " % len(name_dict[key]))
    fwrite.write("\n")
    fwrite.write("Cartesian\n")
    for key in elements:
        for s in name_dict[key]:
            site = np.zeros(3)
            site[:sc.dim] = sites[s]
            fwrite.write("%20.12f%20.12f%20.12f\n" % tuple(site))
    if isinstance(fout, str):
        fwrite.close()

def sc2XYZ(sc, fout, elements=None):
    sites = sc.sites
    name_dict = __stat(sc.names)

    if elements is not None:
        log.eassert(set(elements) == set(name_dict.keys()), "specified elements different from unitcell data")
    else:
        elements = name_dict.keys()

    fout.write("%d\n" % sc.nsites)
    fout.write("generated using libdmet.utils.iotools\n")

    for key in elements:
        for s in name_dict[key]:
            site = np.zeros(3)
            site[:sc.dim] = sites[s]
            fout.write("%4s%20.12f%20.12f%20.12f\n" % ((key,) + tuple(site)))

def struct_dump(cellsize, scsize, atoms, fmt, frac=False, center=None, 
                filename=None, elements=None):
    import libdmet.system.lattice as lat
    uc = build_unitcell(cellsize, atoms, frac, center)
    sc = lat.SuperCell(uc, scsize)
    if filename is not None:
        fout = open(filename, "w")
    else:
        fout = sys.stdout

    if fmt.upper() == "POSCAR":
        sc2POSCAR(sc, fout, elements = elements)
    elif fmt.upper() == "XYZ":
        sc2XYZ(sc, fout, elements = elements)
    else:
        log.error("Invalid dump format %s", fmt.upper())
    
    if filename is not None:
        fout.close()

def read_poscar(fname="POSCAR"):
    """
    Read cell structure from a VASP POSCAR file.
    
    Args:
        fname: file name.

    Returns:
        cell: cell, without build, unit in A.
    """
    from pyscf.pbc import gto
    from libdmet.system.lattice import Frac2Real
    with open(fname, 'r') as f:
        lines = f.readlines()

        # 1 line scale factor
        line = lines[1].split()
        assert len(line) == 1
        factor = float(line[0])
        
        # 2-4 line, lattice vector 
        a = np.array([np.fromstring(lines[i], dtype=np.double, sep=' ') \
                for i in range(2, 5)]) * factor
        
        # 5, 6 line, species names and numbers
        sp_names = lines[5].split()
        if all([name.isdigit() for name in sp_names]):
            # 5th line can be number of atoms not names.
            sp_nums = np.fromstring(lines[5], dtype=int, sep=' ')
            sp_names = ["X" for i in range(len(sp_nums))]
            line_no = 6 
        else:
            sp_nums = np.fromstring(lines[6], dtype=int, sep=' ')
            line_no = 7

        # 7, cartisian or fraction or direct
        line = lines[line_no].split()
        assert len(line) == 1
        use_cart = line[0].startswith(('C', 'K', 'c', 'k'))
        line_no += 1
        
        # 8-end, coords
        atom_col = []
        for sp_name, sp_num in zip(sp_names, sp_nums):
            for i in range(sp_num):
		# there may be 4th element for comments or fixation.
                coord = np.array(list(map(float, \
                        lines[line_no].split()[:3])))
                if use_cart:
                    coord *= factor
                else:
                    coord = Frac2Real(a, coord)
                atom_col.append((sp_name, coord))
                line_no += 1
                 
        cell = gto.Cell()
        cell.a = a
        cell.atom = atom_col
        cell.unit = 'A'
        return cell
        
def write_poscar(cell, fname="POSCAR", species=True, cart=False, comment="name"):
    """
    Write cell structure to a VASP POSCAR file.
    
    Args:
        cell: cell
        fname: file name
        species: group the same atom
        cart: write cartesian coords or fractional coords.
        comment: if == 'name', will write the atom name after the coordinates.
    """
    from libdmet.system.lattice import Real2Frac
    if not cell._built:
        log.warn("write_poscar: cell is not initialized.")
        cell = cell.copy()
        cell.basis = {}
        with lib.temporary_env(sys, stderr=open(os.devnull, "w")):
            cell.build()
    atoms = cell._atom
    vec = cell.lattice_vectors() * BOHR
    with open(fname, 'w') as f:
        f.write("POSCAR generated from libdmet \n")
        f.write("1.0 \n")
        f.write("%20.15f %20.15f %20.15f \n"%(vec[0,0], vec[0,1], vec[0,2]))
        f.write("%20.15f %20.15f %20.15f \n"%(vec[1,0], vec[1,1], vec[1,2]))
        f.write("%20.15f %20.15f %20.15f \n"%(vec[2,0], vec[2,1], vec[2,2]))
        if species:
            # species dictonary
            sp_dic = OrderedDict()
            for name, coord in atoms:
                if name in sp_dic:
                    sp_dic[name].append(coord)
                else:
                    sp_dic[name] = [coord]
            
            # write
            for sp_name in sp_dic.keys():
                f.write("%4s " %sp_name)
            f.write("\n")
            for sp_name, coords in sp_dic.items():
                f.write("%4d " %(len(coords)))
            f.write("\n")
            if cart:
                f.write("Cartisian \n")
                for sp_name, coords in sp_dic.items():
                    for coord in coords:
                        if comment == 'name':
                            f.write("%20.15f %20.15f %20.15f # %s\n"\
                                    %(coord[0]*BOHR, coord[1]*BOHR,
                                      coord[2]*BOHR, sp_name))
                        else:
                            f.write("%20.15f %20.15f %20.15f \n"\
                                    %(coord[0]*BOHR, coord[1]*BOHR, coord[2]*BOHR))
            else:
                f.write("Direct \n")
                for sp_name, coords in sp_dic.items():
                    coords = Real2Frac(cell.lattice_vectors(), coords)
                    for coord in coords:
                        if comment == 'name':
                            f.write("%20.15f %20.15f %20.15f # %s\n"\
                                    %(coord[0], coord[1], coord[2], sp_name))
                        else:
                            f.write("%20.15f %20.15f %20.15f \n"\
                                    %(coord[0], coord[1], coord[2]))
            f.write("\n")
        else:
            for atom in atoms:
                f.write("%4s "%atom[0])
            f.write("\n")
            for atom in atoms:
                f.write("%4d "%(1))
            f.write("\n")
            if cart:
                f.write("Cartisian \n")
                for atom in atoms:
                    f.write("%20.15f %20.15f %20.15f \n"\
                            %(atom[1][0]*BOHR, atom[1][1]*BOHR, atom[1][2]*BOHR))
            else:
                f.write("Direct \n")
                for atom in atoms:
                    coord = Real2Frac(cell.lattice_vectors(), atom[1])
                    f.write("%20.15f %20.15f %20.15f \n"\
                            %(coord[0], coord[1], coord[2]))
            f.write("\n")

def cell_plus_imgs(cell, nimgs):
    """
    Create a supercell via nimgs[i] in each +/- direction, as in get_lattice_Ls().
    Note this function differs from :fun:`super_cell` that super_cell only
    stacks the images in + direction.
    This version allows 0 in nimgs.

    Args:
        cell : instance of :class:`Cell`
        nimgs : (3,) array

    Returns:
        supcell : instance of :class:`Cell`
    """
    supcell = cell.copy()
    a = cell.lattice_vectors()
    Ts = lib.cartesian_prod((np.arange(-nimgs[0],nimgs[0]+1),
                             np.arange(-nimgs[1],nimgs[1]+1),
                             np.arange(-nimgs[2],nimgs[2]+1)))
    Ls = np.dot(Ts, a)
    symbs = [atom[0] for atom in cell._atom] * len(Ls)
    coords = Ls.reshape(-1,1,3) + cell.atom_coords()
    supcell.atom = list(zip(symbs, coords.reshape(-1,3)))
    supcell.unit = 'B'
    # ZHC NOTE allow 0 in nimgs
    nimgs = np.array(nimgs)
    nimgs[nimgs == 0] = 1
    supcell.a = np.einsum('i, ij -> ij', nimgs, a)
    supcell.build(False, False, verbose=0)
    supcell.verbose = cell.verbose
    return supcell

def change_cell_shape(cell, vec, origin=np.zeros(3), search_range=[2, 2, 2], tol=1e-6):
    """
    Generate a new cell with new lattice vector and new origin.
    
    Args:
        cell: old cell.
        vec: new lattice vector (unit in Angs.), shape (3, 3).
        origin: new origin (unit in Angs.), default zeros(3).
        search_range: (nx, ny, nz), search new atoms within nearest 
                      [-nx, nx+1] x [-ny, ny+1] x [-nz, nz+1] cells.
        tol: tolerance for atoms within the cells.

    Returns:
        cell_new: new cell, with new lattice vector and atoms.
    """
    from pyscf.data.nist import BOHR
    from libdmet.system.lattice import Real2Frac
    
    vec = np.asarray(vec)
    origin = np.asarray(origin)
    log.eassert(vec.shape == (3, 3), "vec shape %s needs to be (3, 3)", \
            vec.shape)
    vol_new = la.det(vec)
    log.eassert(vol_new > 1e-8, "change_cell_shape: "
            "new lattice vector is not in right-hand convention, "
            "or is singular")
    if not cell._built:
        log.warn("change_cell_shape: cell is not initialized.")
        cell = cell.copy()
        cell.basis = {}
        with lib.temporary_env(sys, stderr=open(os.devnull, "w")):
            cell.build(verbose=0)
    log.info("change_cell_shape: old vectors [in A]:\n%s", \
            cell.lattice_vectors() * BOHR)
    log.info("change_cell_shape: new vectors [in A]:\n%s", vec)

    # natm in the new cell should be an integer
    natm_new = (vol_new / BOHR**3 / cell.vol) * cell.natm
    log.eassert(natm_new - np.round(natm_new) < tol, "number of atoms %s "
            "in the new cell should be an integer.", natm_new)
    natm_new = int(np.round(natm_new))

    cell_new = cell.copy()
    cell_new.a = vec
    vec_bohr = vec / BOHR
    
    # do not show warning.
    with lib.temporary_env(sys, stderr=open(os.devnull, "w")):
        scell = cell_plus_imgs(cell, search_range)
    atoms = scell._atom
    names, coords = zip(*atoms)
    names, coords = np.asarray(names), np.asarray(coords) - (origin / BOHR)

    # find the index of fraction coords with [0, 1)
    coords_frac = Real2Frac(vec_bohr, coords)
    idx = ((coords_frac > -tol) & (coords_frac < 1.0 - tol)).all(axis=1)
    natm = np.sum(idx)
    vol_ratio = vol_new / (cell.vol * BOHR**3)
    log.info("change_cell_shape: old vol [A^3]: %s, new vol [A^3]: %s, "
             "ratio: %s", cell.vol * BOHR**3, vol_new, vol_ratio)
    log.info("change_cell_shape: old natm: %s, new natm: %s, ratio: %s", cell.natm, \
            natm, natm / cell.natm)
    assert natm == natm_new
    if abs(vol_ratio - natm / cell.natm) > tol:
        log.warn("volume change is not consistent with atom number change...")

    atoms_new = list(zip(names[idx], coords[idx] * BOHR))
    cell_new.atom = atoms_new
    cell_new.unit = 'A'
    with lib.temporary_env(sys, stderr=open(os.devnull, "w")):
        cell_new.build()
    return cell_new
