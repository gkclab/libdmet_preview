#!/usr/bin/env python

"""
Cholesky decomposation of ERI.
part of it is modified from Ankit Mahajan.
part of it is from fcdmft by Xing Zhang and Tianyu Zhu.
"""

import math
import numpy as np
from scipy import linalg as la
from pyscf import lib
from libdmet.utils import logger as log

LINEAR_DEP_THR = 1e-12

# ****************************************************************************
# modified Cholesky for DQMC
# ****************************************************************************

def modified_cholesky(mat, max_error=1e-6):
    """
    modified cholesky for a give matrix
    """
    assert mat.ndim == 2
    size = mat.shape[0]
    diag = np.diag(mat)
    idx = np.argmax(diag)
    delta_max = diag[idx]
    Mapprox = 0.0

    chol_vecs = [mat[idx] / delta_max**0.5]

    max_cycle = size * 2 + 1
    for i in range(max_cycle):
        Mapprox += chol_vecs[i] * chol_vecs[i]
        delta = diag - Mapprox
        idx = np.argmax(np.abs(delta))
        delta_max = abs(delta[idx])
        delta = None

        R = 0.0
        for vec in chol_vecs:
            R += vec[idx] * vec

        chol_vecs.append((mat[idx] - R) / (delta_max)**0.5)
        R = None
        if abs(delta_max) < max_error:
            break
    else:
        log.warn("modified cholesky does not converge ...")
    return np.asarray(chol_vecs)

def modified_cholesky_uhf(mat, max_error=1e-6):
    """
    modified cholesky for a give matrix.
    aa, bb, ab order.
    """
    assert len(mat) == 3
    assert mat[0].ndim == 2
    size = mat[0].shape[0]
    diag = np.hstack((np.diag(mat[0]), np.diag(mat[1])))
    idx = np.argmax(diag)
    delta_max = diag[idx]
    Mapprox = np.zeros_like(diag)

    chol_vecs = [[], []]

    if idx < size:
        chol_vecs[0].append(mat[0][idx] / delta_max**0.5)
        chol_vecs[1].append(mat[2][idx] / delta_max**0.5)
    else:
        chol_vecs[0].append(mat[2].T[idx-size] / delta_max**0.5)
        chol_vecs[1].append(mat[1][idx-size] / delta_max**0.5)

    max_cycle = size * 2 + 1
    for i in range(max_cycle):
        Mapprox[:size] += chol_vecs[0][i] * chol_vecs[0][i]
        Mapprox[size:] += chol_vecs[1][i] * chol_vecs[1][i]
        delta = diag - Mapprox
        idx = np.argmax(np.abs(delta))
        delta_max = abs(delta[idx])
        delta = None

        Ra = 0.0
        Rb = 0.0
        if idx < size:
            for veca, vecb in zip(*chol_vecs):
                Ra += veca[idx] * veca
                Rb += veca[idx] * vecb
            chol_vecs[0].append((mat[0][idx] - Ra) / (delta_max)**0.5)
            chol_vecs[1].append((mat[2][idx] - Rb) / (delta_max)**0.5)
        else:
            for veca, vecb in zip(*chol_vecs):
                Ra += vecb[idx-size] * veca
                Rb += vecb[idx-size] * vecb
            chol_vecs[0].append((mat[2].T[idx-size] - Ra) / (delta_max)**0.5)
            chol_vecs[1].append((mat[1][idx-size] - Rb) / (delta_max)**0.5)

        Ra = Rb = None
        if abs(delta_max) < max_error:
            break
    else:
        log.warn("modified cholesky does not converge ...")
    return np.asarray(chol_vecs)

def get_cderi_rhf(eri, norb, tol=1e-8):
    """
    rhf, ghf ERI decomposation, assume 4-fold symmetry.
    """
    assert eri.ndim == 2
    block_eri = eri
    evecs = modified_cholesky(block_eri, max_error=tol)
    chol = lib.unpack_tril(evecs)
    return chol

def get_cderi_uhf(eri, norb, tol=1e-8):
    """
    uhf ERI decomposation, assume 4-fold symmetry.
    assume aa, bb, ab order.
    """
    assert len(eri) == 3
    assert eri[0].ndim == 2
    evecs = modified_cholesky_uhf(eri, max_error=tol)
    nchol = evecs.shape[-2]
    chol = lib.unpack_tril(evecs.reshape(2*nchol, norb*(norb+1)//2))
    chol = chol.reshape(2, nchol, norb, norb)
    return chol

# ****************************************************************************
# Cholesky from Xing Zhang's code
# ****************************************************************************

def mat_loop_2d(ni,nj, i0=0, istep=1, j0=0, jstep=1):

    '''Loop over the full 2d matrix
    '''

    for i in range(i0, ni, istep):
        for j in range(j0, nj, jstep):
            yield i,j


def get_eri_diag(eri):

    nao = eri.shape[0]
    out = np.empty([nao,nao], dtype=np.double)
    for mu in range(nao):
        for nu in range(nao):
            out[mu,nu] = eri[mu,nu,mu,nu]

    return out


class AOPair():

    def __init__(self,mu,nu):

        self.mu = mu
        self.nu = nu
        self.Bmask = 0
        self.Dmask = 0
        self.eri_diag = -999999.0
        self.eri_off_ioff = 0
        self.Lpq = None


class AOPairs():

    def __init__(self, nao):

        self.nao = nao
        self.n_aopairs = nao*nao
        self.eri_diag = None
        self.sorted_ind = None

        self.data = None

    def init_aopairs(self):

        ni = self.nao
        nj = self.nao

        self.data = np.empty([ni,nj], dtype = object)
        for i,j in mat_loop_2d(ni,nj):
            self.data[i,j] = AOPair(i,j)

    def get_eri_diag(self, eri):

        self.eri_diag = get_eri_diag(eri)
        for i, j in self.ijloop():
            ao_pair = self.get_aopair(i,j)
            ao_pair.eri_diag = self.eri_diag[i,j]

    def print_eri_diag(self):
        for i, j in self.ijloop():
            ao_pair = self.get_aopair(i,j)
            print(i,j, ao_pair.eri_diag)

    def get_aopair(self, i, j):

        return self.data[i,j]


    def ijloop(self):

        for i,j in mat_loop_2d(self.nao, self.nao):
            yield i,j

    def reorder_aopairs(self):

        max_diag_values = np.zeros([self.n_aopairs])
        ind = 0
        for i,j in self.ijloop():
            max_diag_values[ind] = self.eri_diag[i,j]
            ind += 1
        self.sorted_ind = np.argsort(-max_diag_values)

    def sorted_ijloop(self):

        if self.sorted_ind is None:
            self.reorder_aopairs()

        nj = self.nao
        for ind in self.sorted_ind:
            i = ind//nj
            j = ind % nj
            yield i,j

    def sorted_aopairs(self):

        for i, j in self.sorted_ijloop():
            yield self.get_aopair(i,j)



    def make_eri_offdiag(self,eri, p_aopair_ind, q_aopair_ind):

        nij = p_aopair_ind.shape[0]
        nkl = q_aopair_ind.shape[0]
        out = np.empty([nij*nkl], dtype=np.double)
        ind = 0
        for ij in range(nij):
            i = p_aopair_ind[ij,0]
            j = p_aopair_ind[ij,1]
            for kl in range(nkl):
                k = q_aopair_ind[kl,0]
                l = q_aopair_ind[kl,1]
                out[ind] = eri[i,j,k,l]
                ind += 1

        return out, nij, nkl

    def make_eri_ao_aopair(self, eri, kl_ind):

        nao = self.nao
        nao_r = len(kl_ind)
        nao_l = nao * nao
        out =  np.empty([nao_l, nao_r], dtype = np.double)
        ind = 0
        eri = eri.reshape(nao_l, nao, nao)
        for kl in kl_ind:
            k = kl[0]
            l = kl[1]
            out[:,ind] = eri[:,k,l]
            ind += 1

        return out, nao_l, nao_r


class cholesky():

    def __init__(self, eri, tau=1e-8, sigma=1e-2, dimQ=10):

        self.eri = eri
        self.tau = tau
        self.sigma = sigma
        self.dimQ = dimQ

        nao = self.eri.shape[0]
        self.ao_pairs = AOPairs(nao)
        self.ao_pairs.init_aopairs()

    def kernel(self):

        self.step1()
        return self.step2()

    def step1(self):

        self.ao_pairs.get_eri_diag(self.eri)

        it = 0
        while True:

            it += 1
            if it > 200:
                raise RuntimeError("decomposition is likely to fail!!!!")

            Dmax = 0.0
            D = []
            Q = []
            nq = 0
            ioff = 0
            self.ao_pairs.sorted_ind = None
            for ind, ao_pair in enumerate(self.ao_pairs.sorted_aopairs()):

                max_diag = ao_pair.eri_diag
                if max_diag < self.tau:
                    ao_pair.Dmask = 0
                    ao_pair.Lpq = None
                    continue

                if ind == 0: Dmax = max_diag

                i = ao_pair.mu
                j = ao_pair.nu

                ao_pair.Dmask = 1
                D.append((i,j))

                ao_pair.eri_off_ioff = ioff
                ioff += 1

                if nq < self.dimQ:
                    if ao_pair.eri_diag > self.sigma * Dmax:
                        ao_pair.Dmask = 2
                        Q.append((i,j))
                        nq += 1

            if nq == 0: break
            p_aopair_ind = np.asarray(D)
            q_aopair_ind = np.asarray(Q)

            eri_offdiag, nao_ij, nao_kl = self.ao_pairs.make_eri_offdiag(self.eri, p_aopair_ind, q_aopair_ind)
            eri_offdiag = eri_offdiag.reshape((nao_ij, nao_kl))

            tmp = []
            for q in Q:
                i = q[0]
                j = q[1]
                ao_pair = self.ao_pairs.get_aopair(i,j)
                Dq = -999999.0
                if ao_pair.Dmask == 2:
                    Dq = ao_pair.eri_diag
                Dq = float(Dq)
                tmp.append((i, j, 0, Dq))

            tmp1=np.asarray(tmp)[:,-1]
            sorted_q = np.argsort(-tmp1)

            for p in D:
                ao_pair = self.ao_pairs.get_aopair(p[0],p[1])
                if ao_pair.Lpq is None: continue
                ioff = ao_pair.eri_off_ioff
                size = 1

                LL = np.zeros([size, nao_kl], dtype=np.double)
                for ind, q in enumerate(tmp):
                    if q[-1] < -9999: continue
                    ao_pair_q = self.ao_pairs.get_aopair(q[0],q[1])
                    if ao_pair_q.Lpq is None: continue

                    LL[:,ind:ind+1] = np.dot(ao_pair.Lpq, ao_pair_q.Lpq[q[2]:q[2]+1,:].T)
                eri_offdiag[ioff:ioff+size,:] -= LL


            Lpq = np.empty([nao_ij, nq], dtype=np.double)
            iq = 0
            for ind in sorted_q:
                q = tmp[ind]
                if q[-1] < -9999: break

                Mpq = eri_offdiag[:,ind]
                Mqq = -1.0
                ao_pair = self.ao_pairs.get_aopair(q[0],q[1])
                ioff = ao_pair.eri_off_ioff
                q_left = ioff + q[2]
                Mqq = eri_offdiag[q_left,ind]

                Ltmp = np.zeros([nao_ij], dtype=np.double)
                for J in range(0,iq):
                    Ltmp += Lpq[:,J] * Lpq[q_left, J]

                Lpq[:,iq] = (Mpq - Ltmp)/math.sqrt(Mqq)


                ao_pair.Dmask = 1 #remove q from Q
                ao_pair.Bmask = 1


                for p in D:
                    ao_pair = self.ao_pairs.get_aopair(p[0],p[1])
                    ioff = ao_pair.eri_off_ioff
                    size = 1
                    ao_pair.eri_diag -= Lpq[ioff:ioff+size,iq]**2

                iq += 1

            for p in D:
                ao_pair = self.ao_pairs.get_aopair(p[0],p[1])
                ioff = ao_pair.eri_off_ioff
                size = 1
                if ao_pair.Lpq is None:
                    ao_pair.Lpq = Lpq[ioff:ioff+size,:].copy()
                else:
                    ao_pair.Lpq = np.append(ao_pair.Lpq, Lpq[ioff:ioff+size,:], axis=1)


    def step2(self):

        aopair_ind = []
        for i, j in self.ao_pairs.ijloop():
            ao_pair = self.ao_pairs.get_aopair(i,j)
            if ao_pair.Bmask == 1:
                aopair_ind.append((i,j))

        aopair_ind = np.asarray(aopair_ind)

        eri_S, nao_ij, nao_kl = self.ao_pairs.make_eri_offdiag(self.eri, aopair_ind, aopair_ind)
        eri_S = eri_S.reshape(nao_ij,nao_kl)
        eri_S_ao = eri_S

        try:
            L = np.linalg.cholesky(eri_S_ao)
            tag = 'cd'
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(eri_S_ao)
            idx = w > LINEAR_DEP_THR
            L = v[:,idx]/np.sqrt(w[idx])
            tag = 'eig'
            #print(w)

        if tag == 'cd':
            L = np.linalg.inv(L).T

        #print("L.shape = ", L.shape)
        #L = np.insert(L, insert_loc, 0.0, axis = 0)
        #print(L.shape)

        eri, nao_ij, nao_kl = self.ao_pairs.make_eri_ao_aopair(self.eri, aopair_ind)
        eri = eri.reshape(nao_ij, nao_kl)
        cderi = np.dot(eri,L).T

        return cderi

if __name__ == "__main__":
    from pyscf import gto, scf, ao2mo

    mol = gto.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'ccpvdz',
        verbose = 4
    )
    norb = mol.nao_nr()
    myhf = mol.HF()
    myhf.kernel()
    eri = myhf._eri

    eri_s4 = ao2mo.restore(4, eri, norb)
    cderi_s4 = get_cderi_rhf(eri_s4, norb, tol=1e-8)
    print (cderi_s4.shape)

    eri_re = np.einsum('Lpq, Lrs -> pqrs', cderi_s4, cderi_s4)
    eri_ref = ao2mo.restore(1, eri_s4, norb)
    diff = la.norm(eri_re - eri_ref)
    print ("diff", diff)
    assert diff < 1e-7

