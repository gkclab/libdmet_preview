#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Zhi-Hao Cui <zhcui0408@gmail.com>

'''
Unrestricted coupled cluster doubles
'''

import numpy as np
from pyscf.cc import ccsd
from pyscf.cc import ccsd_lambda
from pyscf.cc import uccsd
from pyscf.cc import uccsd_lambda
from libdmet.solver import uccsd_rdm

class UCCD(uccsd.UCCSD):
    def update_amps(self, t1, t2, eris):
        t1, t2 = uccsd.update_amps(self, t1, t2, eris)
        return [np.zeros_like(t1[0]), np.zeros_like(t1[1])], t2

    def kernel(self, t1=None, t2=None, eris=None):
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb

        t1 = [np.zeros((nocca, nvira)), np.zeros((noccb, nvirb))]
        ccsd.CCSD.kernel(self, t1, t2, eris)
        return self.e_corr, self.t1, self.t2

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)

        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        l1 = t1 = [np.zeros((nocca, nvira)), np.zeros((noccb, nvirb))]

        def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
            l1, l2 = uccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
            return [np.zeros_like(l1[0]), np.zeros_like(l1[1])], l2

        self.converged_lambda, self.l1, self.l2 = \
                ccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                   max_cycle=self.max_cycle,
                                   tol=self.conv_tol_normt,
                                   verbose=self.verbose,
                                   fintermediates=uccsd_lambda.make_intermediates, 
                                   fupdate=update_lambda)
        return self.l1, self.l2

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''Un-relaxed 1-particle density matrix in MO space'''
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        l1 = t1 = [np.zeros((nocca, nvira)), np.zeros((noccb, nvirb))]

        if t2 is None: t2 = self.t2
        if l2 is None: l2 = self.l2
        if l2 is None: l2 = self.solve_lambda(t1, t2)

        return uccsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        l1 = t1 = [np.zeros((nocca, nvira)), np.zeros((noccb, nvirb))]
        if t2 is None: t2 = self.t2
        if l2 is None: l2 = self.l2
        if l2 is None: l2 = self.solve_lambda(t1, t2)

        return uccsd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr)
