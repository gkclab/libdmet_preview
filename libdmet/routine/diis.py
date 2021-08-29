# This file is part of the lattice-dmet program. lattice-dmet is free
# software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software
# Foundation, version 3.
#
# lattice-dmet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with bfint (LICENSE). If not, see http://www.gnu.org/licenses/
#
# Authors:
#    Gerald Knizia, 2012

"""A simple implementation of the DIIS (``direct inversion in the iterative subspace'')
convergence accelerator:
   P.Pulay; Chem. Phys. Lett. 73 393 (1980)
   P.Pulay; J. Comput. Chem. 3 556 (1982)
Note: this technique is ubiquitous in quantum chemistry, but has not found
   that much attention elsewhere. A variant of this restricted to linear
   systems was later re-invented by mathematicans and "GMRes" (generalized
   minimum residual) and is more widely known.
--
C. Gerald Knizia, 2012
"""

import numpy as np
from numpy import *
from scipy import *
from scipy.linalg import *

class FDiisContext:
    def __init__(self, nDim):
        self.MaxDim = nDim
        self.nDim = 0
        self.iNext = 0
        self.DbgPrint = False
        self.NotApplied = True
        self.iVectorAge = np.zeros(self.MaxDim,dtype=int)
    def Reset(self):
        self.nDim = 0
        self.iNext = 0
    def __str__(self):
        if ( self.NotApplied ):
            return " -  -"
        else:
            return "%2i %2i" % (self.nDim, self.iNext)

    def RemoveBadVectors(self,iThis):
        # kick out vectors which are too bad to be useful for extrapolation.
        nDim = self.nDim
        Errs1 = self.Errs[:,:nDim]
        B0 = dot(conj(Errs1.T),Errs1)
        EMin = min(diag(B0))

        iVecs = []
        for i in range(nDim):
            if ( B0[i,i].real <= 1e12 * EMin or i == iThis ):
                iVecs.append(i)
        if ( len(iVecs) != nDim ):
            iVecs = array(iVecs)
            nDim = len(iVecs)
            iThis = list(iVecs).index(iThis)
            self.Amps[:,:nDim] = self.Amps[:,iVecs]
            self.Errs[:,:nDim] = self.Errs[:,iVecs]
            if (self.Othr is not None):
                self.Othr[:,:nDim] = self.Othr[:,iVecs]
            self.iVectorAge[:nDim] = self.iVectorAge[iVecs]
            self.nDim = nDim
            iVecs = list(range(nDim))

    def Apply(self, T_, R_, O_ = None, Skip=None):
        T = T_.flatten()
        R = R_.flatten()

        ContinueIfStarted = True
        if ( dot(conj(R),R) < 1e-30 ): # <- otherwise divide by zero in the B scaling.
            Skip = True; ContinueIfStarted = False
        if (Skip is not None and Skip and (self.nDim == 0 or not ContinueIfStarted)):
            # about the last one: this means 'continue with diis if you
            # ever began with it'. might be required if iterations happen to
            # go in the wrong direction.
            self.NotApplied = True
            if (O_ is not None): return T_, R_, O_, 1.0
            else:               return T_, R_, 1.0
        self.NotApplied = False

        def PrintVec(s,T):
            print ("!x DIIS: %-10s = %s ..." % (s, " ".join(["%12.6f" % o for o in T[:10]])))
        if self.DbgPrint:
            PrintVec("Input T", T)
            PrintVec("Input R", R)
            print ("History:")
            for i in range(self.nDim):
                PrintVec("Amps[%i]" % i, self.Amps[:,i])
            for i in range(self.nDim):
                PrintVec("Errs[%i]" % i, self.Errs[:,i])

        O = None
        if ( O_ is not None ):
            O = O_.flatten()

        if ( self.nDim == 0 ):
            self.Amps = np.zeros((len(T), self.MaxDim),T.dtype)
            self.Errs = np.zeros((len(R), self.MaxDim),R.dtype)
            if (O is not None):
                self.Othr = np.zeros((len(O), self.MaxDim),O.dtype)
            else:
                self.Othr = None
        if ( self.nDim < self.MaxDim ):
            self.nDim += 1
        iThis = self.iNext
        for i in range(self.nDim):
            self.iVectorAge[i] += 1
        self.iVectorAge[iThis] = 0

        self.Amps[:,iThis] = T
        self.Errs[:,iThis] = R
        if (O is not None):
            self.Othr[:,iThis] = O

        self.RemoveBadVectors(iThis)
        nDim = self.nDim

        Errs1 = self.Errs[:,:nDim]
        B0 = dot(conj(Errs1.T),Errs1)
        if self.DbgPrint:
            print ("\n -- DIIS SYSTEM:")
            print ("B0 = \n", B0)

        B = np.zeros((nDim+1,nDim+1),B0.dtype)
        B[:nDim,:nDim] = B0
        rhs = np.zeros((nDim+1))
        fScale = 0
        for i in range(nDim):
            fScale += log(B[i,i].real)
            B[nDim,i] = -1
            B[i,nDim] = -1
        fScale = exp(fScale/nDim)
        B[:nDim,:nDim] /= fScale
        rhs[nDim] = -1.
        B[nDim,nDim] = 0.
        if False:
            print ( "\n -- DIIS SYSTEM:")
            print ( "B = \n", B)
            print ( "RHS = \n", rhs)
            print ( "fScale = %8.2e" % fScale)

        if 1:
            # straight diis
            try:
                c1 = solve(B, rhs)
            except (LinAlgError, e):
                # I don't think that this is supposed to happen here.
                print ("diis: resorted to lstsq...")
                (c1,fitresid,rank,sigma) = lstsq(B, rhs)
        else:
            ew,ev = eigh(B[:-1,:-1])
            c1 = array(list(ev[:,0]) + [ew[0]])
        c = c1[:-1]
        if self.DbgPrint or False:
            print ( "B = \n", B)
            print ( "RHS = \n", rhs)
            print ( "C = \n", c)
            print ( "c1[-1] = %8.2e" % c1[-1])
            print ( "fScale = %8.2e" % fScale)

        c /= sum(c) # might have cut out some weight in overlap truncation.
        #print ("c[iThis] = %8.2e" % c[iThis])

        #print ("output c: %s" % c)
        Tnew = dot(self.Amps[:,:nDim], c[:,newaxis])[:,0]
        Rnew = dot(self.Errs[:,:nDim], c[:,newaxis])[:,0]
        if (O is not None):
            Onew = dot(self.Othr[:,:nDim], c[:,newaxis])[:,0]

        if ( self.nDim < self.MaxDim ):
            self.iNext = self.nDim
        else:
            self.iNext = (iThis + 1) % self.nDim


        if self.DbgPrint:
            PrintVec("Output T", Tnew)
            PrintVec("Output R", Rnew)

        if (O is not None):
            return Tnew.reshape(T_.shape), Rnew.reshape(R_.shape), Onew.reshape(O_.shape), (abs(c1[-1])*fScale)**.5
        else:
            return Tnew.reshape(T_.shape), Rnew.reshape(R_.shape), (abs(c1[-1])*fScale)**.5

  # kate: indent-width 4

    def ApplyBCS(self, Vcor, mu, dVcor, dmu, Skip = False):
        def flat_form(Vcor, mu):
            return array(Vcor[0].flatten().tolist() + Vcor[1].flatten().tolist() + [mu])
        def matrix_form(T):
            Vloc = T[:Vcor[0].size].reshape(Vcor[0].shape)
            Delta = T[Vcor[0].size:-1].reshape(Vcor[1].shape)
            mu = T[-1]
            return [Vloc, Delta], mu

        T = flat_form(Vcor, mu)
        R = flat_form(dVcor, dmu)

        T, R, c0 = self.Apply(T, R, O_ = None, Skip = Skip)
        new_Vcor, new_mu = matrix_form(T)
        new_dVcor, new_dmu = matrix_form(R)

        return new_Vcor, new_mu, new_dVcor, new_dmu, c0


#
# File: diis.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

class DIIS:
# J. Mol. Struct. 114, 31-34
# PCCP, 4, 11
# GEDIIS JCTC, 2, 835
# C2DIIS IJQC, 45, 31
# DIIS try to minimize the change of the input vectors. It rotates the vectors
# to make the error vector as small as possible.
    def __init__(self):
        self.diis_vec_stack = []
        self.threshold = 1e-6
        self.diis_space = 6
        self.diis_start_cycle = 2
        self._head = 0

    def push_vec(self, x):
        self.diis_vec_stack.append(x)
        if self.diis_vec_stack.__len__() > self.diis_space:
            self.diis_vec_stack.pop(0)

    def get_err_vec(self, idx):
        return self.diis_vec_stack[idx+1] - self.diis_vec_stack[idx]

    def get_vec(self, idx):
        return self.diis_vec_stack[idx+1]

    def get_num_diis_vec(self):
        return self.diis_vec_stack.__len__() - 1

    def update(self, x):
        '''use DIIS method to solve Eq.  operator(x) = x.'''
        self.push_vec(x)

        nd = self.get_num_diis_vec()
        if nd < self.diis_start_cycle:
            if self.diis_start_cycle >= self.diis_space:
                self.diis_start_cycle = self.diis_space - 1
            return x

        H = ones((nd+1,nd+1), x.dtype)
        H[-1,-1] = 0
        G = np.zeros(nd+1, x.dtype)
        G[-1] = 1
        for i in range(nd):
            dti = self.get_err_vec(i)
            for j in range(i+1):
                dtj = self.get_err_vec(j)
                H[i,j] = dot(array(dti).flatten(), \
                             array(dtj).flatten())
                H[j,i] = H[i,j].conj()

        # solve  H*x = G
        try:
            c_GH = solve(H, G)
        except linalg.LinAlgError:
          # damp diagonal elements to prevent singular
            for i in range(H.shape[0]):
                H[i,i] = H[i,i] + 1e-8
            c_GH = solve(H, G)

        x = np.zeros_like(x)
        for i, c in enumerate(c_GH[:-1]):
            x += self.get_vec(i) * c
        return x


# error vector = SDF-FDS
# error vector = F_ai ~ (S-SDS)*S^{-1}FDS = FDS - SDFDS ~ FDS-SDF in converge
class SCF_DIIS(DIIS):
    def __init__(self):
        DIIS.__init__(self)
        self.err_vec_stack = []

    def clear_diis_space(self):
        self.diis_vec_stack = []
        self.err_vec_stack = []

    def push_err_vec(self, d, f):
        df = dot(d,f)
        errvec = df.T.conj() - df

        self.err_vec_stack.append(errvec)
        if self.err_vec_stack.__len__() > self.diis_space:
            self.err_vec_stack.pop(0)

    def get_err_vec(self, idx):
        return self.err_vec_stack[idx]

    def get_vec(self, idx):
        return self.diis_vec_stack[idx]

    def get_num_diis_vec(self):
        return self.diis_vec_stack.__len__()

    def update(self, d, f):
        self.push_err_vec(d, f)
        return DIIS.update(self, f)
