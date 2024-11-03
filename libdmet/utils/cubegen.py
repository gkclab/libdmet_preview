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
#         Peter Koval <koval.peter@gmail.com>
#         Paul J. Robinson <pjrobinson@ucla.edu>
#
# Modified by Zhihao Cui for non-cubic cell.
#

'''
Gaussian cube file format.  Reference:
http://paulbourke.net/dataformats/cube/
http://gaussian.com/cubegen/

The output cube file has the following format

Comment line
Comment line
N_atom Ox Oy Oz         # number of atoms, followed by the coordinates of the origin
N1 vx1 vy1 vz1          # number of grids along each axis, followed by the step size in x/y/z direction.
N2 vx2 vy2 vz2          # ...
N3 vx3 vy3 vz3          # ...
Atom1 Z1 x y z          # Atomic number, charge, and coordinates of the atom
...                     # ...
AtomN ZN x y z          # ...
Data on grids           # (N1*N2) lines of records, each line has N3 elements
'''

import time
import numpy
import numpy as np
import scipy.linalg as la
import pyscf
from pyscf import lib
from pyscf.tools import cubegen
from pyscf.tools.cubegen import density, orbital, mep, RESOLUTION, BOX_MARGIN
from pyscf.pbc.gto import Cell
from pyscf.data.elements import charge

class Cube(cubegen.Cube):
    '''  Read-write of the Gaussian CUBE files  '''
    def __init__(self, mol, nx=80, ny=80, nz=80, resolution=RESOLUTION,
                 margin=BOX_MARGIN, latt_vec=None, boxorig=None, box=None):
        # ZHC FIXME seems there is a small asymmetry when ngrids is small.
        self.mol = mol
        self.coords = None
        if isinstance(self.mol, Cell):
            if latt_vec is None:
                self.latt_vec = mol.lattice_vectors()
            else:
                self.latt_vec = latt_vec
            self.latt_vec_norm = la.norm(self.latt_vec, axis=1)

            if box is None:
                self.box = self.latt_vec * ((margin*2.0 / self.latt_vec_norm) + 1.0)
            else:
                self.box = box
            margin_vec = self.latt_vec * ((margin / self.latt_vec_norm))
            if boxorig is None:
                self.boxorig = -np.sum(margin_vec, axis=0)
            else:
                self.boxorig = np.asarray(boxorig)

            if resolution is not None:
                nx, ny, nz = numpy.ceil(self.latt_vec_norm / resolution).astype(int)

            self.nx = nx
            self.ny = ny
            self.nz = nz
            # .../(nx-1) to get symmetric mesh
            # see also the discussion on https://github.com/sunqm/pyscf/issues/154
            self.xs = numpy.arange(nx) / max(float(nx - 1), 1)
            self.ys = numpy.arange(ny) / max(float(ny - 1), 1)
            self.zs = numpy.arange(nz) / max(float(nz - 1), 1)
            #self.xs = numpy.arange(nx) / float(nx)
            #self.ys = numpy.arange(ny) / float(ny)
            #self.zs = numpy.arange(nz) / float(nz)
        else:
            coord = mol.atom_coords()
            box = numpy.max(coord,axis=0) - numpy.min(coord,axis=0) + margin*2
            self.box = numpy.diag(box)
            self.boxorig = numpy.min(coord,axis=0) - margin
            if resolution is not None:
                nx, ny, nz = numpy.ceil(box / resolution).astype(int)

            self.nx = nx
            self.ny = ny
            self.nz = nz
            # .../(nx-1) to get symmetric mesh
            # see also the discussion on https://github.com/sunqm/pyscf/issues/154
            self.xs = numpy.arange(nx) * (box[0] / (nx - 1))
            self.ys = numpy.arange(ny) * (box[1] / (ny - 1))
            self.zs = numpy.arange(nz) * (box[2] / (nz - 1))

    def get_coords(self) :
        """  Result: set of coordinates to compute a field which is to be stored
        in the file.
        """
        if self.coords is None:
            if isinstance(self.mol, Cell):
                coords = lib.cartesian_prod([self.xs, self.ys, self.zs])
                # frac to cart
                coords = coords.dot(self.box)
            else:
                coords = lib.cartesian_prod([self.xs, self.ys, self.zs])
            coords = numpy.asarray(coords, order='C') + self.boxorig
            self.coords = coords
        return self.coords

    def get_ngrids(self):
        return self.nx * self.ny * self.nz

    def write(self, field, fname, comment=None, header_only=False):
        """  Result: .cube file with the field in the file fname.  """
        if comment is None:
            comment = 'Generic field? Supply the optional argument "comment" to define this line'

        mol = self.mol
        coord = mol.atom_coords()
        with open(fname, 'w') as f:
            f.write(comment+'\n')
            f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
            f.write('%5d' % mol.natm)
            f.write('%12.6f%12.6f%12.6f\n' % tuple(self.boxorig.tolist()))
            if isinstance(self.mol, Cell):
                f.write('%5d%12.6f%12.6f%12.6f\n' % (self.nx, self.box[0,0]/(self.nx), self.box[0,1]/(self.nx),
                    self.box[0,2]/(self.nx)))
                f.write('%5d%12.6f%12.6f%12.6f\n' % (self.ny, self.box[1,0]/(self.ny), self.box[1,1]/(self.ny),
                    self.box[1,2]/(self.ny)))
                f.write('%5d%12.6f%12.6f%12.6f\n' % (self.nz, self.box[2,0]/(self.nz), self.box[2,1]/(self.nz),
                    self.box[2,2]/(self.nz)))
            else:
                f.write('%5d%12.6f%12.6f%12.6f\n' % (self.nx, self.xs[1], 0, 0))
                f.write('%5d%12.6f%12.6f%12.6f\n' % (self.ny, 0, self.ys[1], 0))
                f.write('%5d%12.6f%12.6f%12.6f\n' % (self.nz, 0, 0, self.zs[1]))
            for ia in range(mol.natm):
                #chg = mol.atom_charge(ia)
                chg = charge(mol.atom_symbol(ia))
                f.write('%5d%12.6f'% (chg, chg))
                f.write('%12.6f%12.6f%12.6f\n' % tuple(coord[ia]))

            if not header_only:
                assert field.ndim == 3
                assert field.shape == (self.nx, self.ny, self.nz)
                for ix in range(self.nx):
                    for iy in range(self.ny):
                        #for iz0, iz1 in lib.prange(0, self.nz, 6):
                        #    fmt = '%13.5E' * (iz1-iz0) + '\n'
                        #    f.write(fmt % tuple(field[ix, iy, iz0:iz1]))
                        for iz0, iz1 in lib.prange(0, self.nz, 500):
                            #numpy.savetxt(f, field[ix, iy, iz0:iz1], fmt='%13.6e')
                            fmt = '%13.5E' * (iz1-iz0) #+ '\n'
                            f.write(fmt % tuple(field[ix, iy, iz0:iz1]))

    def write_field(self, field, fname):
        assert field.ndim == 1
        with open(fname, 'a') as f:
            fmt = '%14.6E' * len(field)
            f.write(fmt % tuple(field))

if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.tools import cubegen
    mol = gto.M(atom='''O 0.00000000,  0.000000,  0.000000
                H 0.761561, 0.478993, 0.00000000
                H -0.761561, 0.478993, 0.00000000''', basis='6-31g*')
    mf = scf.RHF(mol).run()
    cubegen.density(mol, 'h2o_den.cube', mf.make_rdm1()) #makes total density
    cubegen.mep(mol, 'h2o_pot.cube', mf.make_rdm1())
    cubegen.orbital(mol, 'h2o_mo1.cube', mf.mo_coeff[:,0])

