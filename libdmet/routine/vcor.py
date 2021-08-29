# define type/symmetry of correlation potential
# potential fitting algorithms using the symmetry
# initial guess

import types
import itertools as it
import numpy as np
import scipy.linalg as la
from libdmet.utils import logger as log
from scipy.optimize import minimize

class Vcor(object):
    def __init__(self):
        self.param = None
        self.value = None
        self.local = True

    def update(self, param):
        self.param = param
        self.value = self.evaluate()

    def islocal(self):
        return self.local
    
    def is_local(self):
        return self.local

    def get(self, i=0, kspace=True):
        log.eassert(self.value is not None, "Vcor not initialized yet")
        if kspace or i == 0:
            return self.value
        else:
            return np.zeros_like(self.value)

    def evaluate(self):
        log.error("function evaulate() is not implemented")

    def gradient(self):
        log.error("function gradient() is not implemented")

    def length(self):
        log.error("function len() is not implemented")

    def assign(self, v0):
        log.eassert(self.islocal(), "This routine is for local vcor only")
        log.eassert(v0.shape == self.gradient().shape[1:], \
            "The correlation potential should have shape %s, rather than %s",
            self.gradient().shape[1:], v0.shape)

        # v_empty
        self.update(np.zeros(self.length()))
        v_empty = self.get()
        v0prime = v0 - v_empty
        param = np.empty(self.length())
        g = self.gradient()
        for i in range(self.length()):
            param[i] = np.sum(g[i] * v0prime) / np.sum(g[i] * g[i])
        self.update(param)
        log.check(la.norm(v0-self.get()) < 1e-7, \
                "symmetrization imposed on initial guess")
        #def fn(x):
        #    self.update(x)
        #    return np.sum((self.get() - v0) ** 2)

        #res = minimize(fn, np.zeros(self.length()), tol = 1e-10)
        #log.check(fn(res.x) < 1e-6, "symmetrization imposed on initial guess")

    def __str__(self):
        return self.evaluate().__str__()

def VcorNonLocal(restricted, bogoliubov, Lat, idx_range=None, \
        bogo_res=False):
    """
    Non-Local correlation potential.

    Args:
        restricted: restricted on v.
        bogoliubov: whether include delta.
        Lat: lattice obj.
        idx_range: index list to add correlation potential.
        bogo_res: if True, delta is restricted so that D = D.T

    Returns:
        vcor: vcor object.
    """
    nscsites = Lat.nscsites
    ncells = Lat.nkpts
    
    if idx_range is None:
        idx_range = list(range(0, nscsites))
    nidx = len(idx_range)
    
    # compute the related cell indices by inversion.
    hermi_list = -np.ones((ncells,), dtype=int)
    weight_list = np.ones((ncells,), dtype=int)
    param_range = [0]
    for R in range(ncells):
        if weight_list[R] == 1:
            idx = Lat.cell_pos2idx(-Lat.cell_idx2pos(R))
            hermi_list[R] = idx
            if idx != R:
                weight_list[R] = 2
                weight_list[idx] = 0
    
    # security checks:
    # elements in weight_list must be 0 or 1 or 2.
    assert np.max(weight_list) <= 2
    # for 1 elements, hermi[R] must be R.
    assert (hermi_list[weight_list==1] == \
            np.arange(ncells)[weight_list==1]).all()
    # the total weight must be ncells
    assert weight_list.sum() == ncells

    nV_1_per_spin = nidx * (nidx + 1) // 2
    nV_2_per_spin = nidx * nidx
    if restricted:
        nV_1 = nV_1_per_spin
        nV_2 = nV_2_per_spin
    else:
        nV_1 = nV_1_per_spin * 2
        nV_2 = nV_2_per_spin * 2
    if bogoliubov:
        if restricted or bogo_res:
            nD_1 = nV_1_per_spin
            nD_2 = nV_2_per_spin
        else:
            nD_1 = nidx * nidx
            nD_2 = nidx * nidx * 2
    else:
        nD_1 = nD_2 = 0

    for R in range(ncells):
        if weight_list[R] == 1:
            param_range.append(param_range[-1] + nV_1 + nD_1)
        elif weight_list[R] == 2:
            param_range.append(param_range[-1] + nV_2 + nD_2)
        else:
            param_range.append(param_range[-1])
    nparam = param_range[-1]

    v = Vcor()
    v.restricted = restricted
    v.bogoliubov = bogoliubov
    v.bogo_res = bogo_res
    v.grad = None
    v.grad_k = None

    if restricted and not bogoliubov:
        def evaluate(self):
            V = np.zeros((1, ncells, nscsites, nscsites))
            for R in range(ncells):
                if weight_list[R] == 1:
                    for idx, (i, j) in enumerate(\
                            it.combinations_with_replacement(idx_range, 2), \
                            start=param_range[R]):
                        V[0, R, i, j] = \
                        V[0, R, j, i] = \
                        self.param[idx]
                elif weight_list[R] == 2:
                    for idx, (i, j) in enumerate(\
                            it.product(idx_range, repeat=2), \
                            start=param_range[R]):
                        V[0, R, i, j] = \
                        V[0, hermi_list[R], j, i] = \
                        self.param[idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nparam, 1, ncells, nscsites, nscsites))
                for R in range(ncells):
                    if weight_list[R] == 1:
                        for idx, (i, j) in enumerate(\
                                it.combinations_with_replacement(idx_range, 2), \
                                start=param_range[R]):
                            g[idx, 0, R, i, j] = \
                            g[idx, 0, R, j, i] = 1
                    elif weight_list[R] == 2:
                        for idx, (i, j) in enumerate(\
                                it.product(idx_range, repeat=2), \
                                start=param_range[R]):
                            g[idx, 0, R, i, j] = \
                            g[idx, 0, hermi_list[R], j, i] = 1
                self.grad = g
                self.grad_k = np.asarray([Lat.R2k(g[i]) for i in \
                        range(g.shape[0])])
            return self.grad

    elif not restricted and not bogoliubov:
        def evaluate(self):
            V = np.zeros((2, ncells, nscsites, nscsites))
            for R in range(ncells):
                if weight_list[R] == 1:
                    for idx, (i, j) in enumerate(\
                            it.combinations_with_replacement(idx_range, 2), \
                            start=param_range[R]):
                        V[0, R, i, j] = \
                        V[0, R, j, i] = \
                        self.param[idx]
                        
                        V[1, R, i, j] = \
                        V[1, R, j, i] = \
                        self.param[idx + nV_1_per_spin]
                elif weight_list[R] == 2:
                    for idx, (i, j) in enumerate(\
                            it.product(idx_range, repeat=2), \
                            start=param_range[R]):
                        V[0, R, i, j] = \
                        V[0, hermi_list[R], j, i] = \
                        self.param[idx]

                        V[1, R, i, j] = \
                        V[1, hermi_list[R], j, i] = \
                        self.param[idx + nV_2_per_spin]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nparam, 2, ncells, nscsites, nscsites))
                for R in range(ncells):
                    if weight_list[R] == 1:
                        for idx, (i, j) in enumerate(\
                                it.combinations_with_replacement(idx_range, 2), \
                                start=param_range[R]):
                            g[idx, 0, R, i, j] = \
                            g[idx, 0, R, j, i] = 1

                            g[idx + nV_1_per_spin, 1, R, i, j] = \
                            g[idx + nV_1_per_spin, 1, R, j, i] = 1
                    elif weight_list[R] == 2:
                        for idx, (i, j) in enumerate(\
                                it.product(idx_range, repeat=2), \
                                start=param_range[R]):
                            g[idx, 0, R, i, j] = \
                            g[idx, 0, hermi_list[R], j, i] = 1

                            g[idx + nV_2_per_spin, 1, R, i, j] = \
                            g[idx + nV_2_per_spin, 1, hermi_list[R], j, i] = 1

                self.grad = g
                self.grad_k = np.asarray([Lat.R2k(g[i]) for i in \
                        range(g.shape[0])])
            return self.grad

    elif restricted and bogoliubov:
        def evaluate(self):
            V = np.zeros((3, ncells, nscsites, nscsites))
            for R in range(ncells):
                if weight_list[R] == 1:
                    for idx, (i, j) in enumerate(\
                            it.combinations_with_replacement(idx_range, 2), \
                            start=param_range[R]):
                        V[0, R, i, j] = \
                        V[0, R, j, i] = \
                        self.param[idx]
                        
                        V[2, R, i, j] = \
                        V[2, R, j, i] = \
                        self.param[idx + nV_1]

                elif weight_list[R] == 2:
                    for idx, (i, j) in enumerate(\
                            it.product(idx_range, repeat=2), \
                            start=param_range[R]):
                        V[0, R, i, j] = \
                        V[0, hermi_list[R], j, i] = \
                        self.param[idx]
                        
                        V[2, R, i, j] = \
                        V[2, hermi_list[R], j, i] = \
                        self.param[idx + nV_2]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nparam, 3, ncells, nscsites, nscsites))
                for R in range(ncells):
                    if weight_list[R] == 1:
                        for idx, (i, j) in enumerate(\
                                it.combinations_with_replacement(idx_range, 2), \
                                start=param_range[R]):
                            g[idx, 0, R, i, j] = \
                            g[idx, 0, R, j, i] = 1
                            
                            g[idx + nV_1, 2, R, i, j] = \
                            g[idx + nV_1, 2, R, j, i] = 1

                    elif weight_list[R] == 2:
                        for idx, (i, j) in enumerate(\
                                it.product(idx_range, repeat=2), \
                                start=param_range[R]):
                            g[idx, 0, R, i, j] = \
                            g[idx, 0, hermi_list[R], j, i] = 1
                            
                            g[idx + nV_2, 2, R, i, j] = \
                            g[idx + nV_2, 2, hermi_list[R], j, i] = 1
                self.grad = g
                self.grad_k = np.asarray([Lat.R2k(g[i]) for i in \
                        range(g.shape[0])])
            return self.grad

    else: # not restricted and bogoliubov
        if bogo_res:
            def evaluate(self):
                V = np.zeros((3, ncells, nscsites, nscsites))
                for R in range(ncells):
                    if weight_list[R] == 1:
                        for idx, (i, j) in enumerate(\
                                it.combinations_with_replacement(idx_range, 2), \
                                start=param_range[R]):
                            V[0, R, i, j] = \
                            V[0, R, j, i] = \
                            self.param[idx]
                            
                            V[1, R, i, j] = \
                            V[1, R, j, i] = \
                            self.param[idx + nV_1_per_spin]

                            V[2, R, i, j] = \
                            V[2, R, j, i] = \
                            self.param[idx + nV_1]

                    elif weight_list[R] == 2:
                        for idx, (i, j) in enumerate(\
                                it.product(idx_range, repeat=2), \
                                start=param_range[R]):
                            V[0, R, i, j] = \
                            V[0, hermi_list[R], j, i] = \
                            self.param[idx]
                            
                            V[1, R, i, j] = \
                            V[1, hermi_list[R], j, i] = \
                            self.param[idx + nV_2_per_spin]
                            
                            V[2, R, i, j] = \
                            V[2, hermi_list[R], j, i] = \
                            self.param[idx + nV_2]
                return V

            def gradient(self):
                if self.grad is None:
                    g = np.zeros((nparam, 3, ncells, nscsites, nscsites))
                    for R in range(ncells):
                        if weight_list[R] == 1:
                            for idx, (i, j) in enumerate(\
                                    it.combinations_with_replacement(idx_range, 2), \
                                    start=param_range[R]):
                                g[idx, 0, R, i, j] = \
                                g[idx, 0, R, j, i] = 1
                                
                                g[idx + nV_1_per_spin, 1, R, i, j] = \
                                g[idx + nV_1_per_spin, 1, R, j, i] = 1
                                
                                g[idx + nV_1, 2, R, i, j] = \
                                g[idx + nV_1, 2, R, j, i] = 1

                        elif weight_list[R] == 2:
                            for idx, (i, j) in enumerate(\
                                    it.product(idx_range, repeat=2), \
                                    start=param_range[R]):
                                g[idx, 0, R, i, j] = \
                                g[idx, 0, hermi_list[R], j, i] = 1
                                
                                g[idx + nV_2_per_spin, 1, R, i, j] = \
                                g[idx + nV_2_per_spin, 1, hermi_list[R], j, i] = 1
                                
                                g[idx + nV_2, 2, R, i, j] = \
                                g[idx + nV_2, 2, hermi_list[R], j, i] = 1
                    self.grad = g
                    self.grad_k = np.asarray([Lat.R2k(g[i]) for i in \
                            range(g.shape[0])])
                return self.grad

        else:
            def evaluate(self):
                V = np.zeros((3, ncells, nscsites, nscsites))
                for R in range(ncells):
                    if weight_list[R] == 1:
                        for idx, (i, j) in enumerate(\
                                it.combinations_with_replacement(idx_range, 2), \
                                start=param_range[R]):
                            V[0, R, i, j] = \
                            V[0, R, j, i] = \
                            self.param[idx]
                            
                            V[1, R, i, j] = \
                            V[1, R, j, i] = \
                            self.param[idx + nV_1_per_spin]
                        
                        for idx, (i, j) in enumerate(\
                                it.product(idx_range, repeat=2),
                                start=param_range[R]+nV_1):
                            V[2, R, i, j] = self.param[idx]

                    elif weight_list[R] == 2:
                        for idx, (i, j) in enumerate(\
                                it.product(idx_range, repeat=2), \
                                start=param_range[R]):
                            V[0, R, i, j] = \
                            V[0, hermi_list[R], j, i] = \
                            self.param[idx]
                            
                            V[1, R, i, j] = \
                            V[1, hermi_list[R], j, i] = \
                            self.param[idx + nV_2_per_spin]
                            
                            V[2, R, i, j] = \
                            self.param[idx + nV_2]
                            
                            V[2, hermi_list[R], i, j] = \
                            self.param[idx + nV_2 + nidx * nidx]
                return V

            def gradient(self):
                if self.grad is None:
                    g = np.zeros((nparam, 3, ncells, nscsites, nscsites))
                    for R in range(ncells):
                        if weight_list[R] == 1:
                            for idx, (i, j) in enumerate(\
                                    it.combinations_with_replacement(idx_range, 2), \
                                    start=param_range[R]):
                                g[idx, 0, R, i, j] = \
                                g[idx, 0, R, j, i] = 1
                                
                                g[idx + nV_1_per_spin, 1, R, i, j] = \
                                g[idx + nV_1_per_spin, 1, R, j, i] = 1
                                
                            for idx, (i, j) in enumerate(\
                                    it.product(idx_range, repeat=2),
                                    start=param_range[R]+nV_1):
                                g[idx, 2, R, i, j] = 1

                        elif weight_list[R] == 2:
                            for idx, (i, j) in enumerate(\
                                    it.product(idx_range, repeat=2), \
                                    start=param_range[R]):
                                g[idx, 0, R, i, j] = \
                                g[idx, 0, hermi_list[R], j, i] = 1
                                
                                g[idx + nV_2_per_spin, 1, R, i, j] = \
                                g[idx + nV_2_per_spin, 1, hermi_list[R], j, i] = 1
                                
                                g[idx + nV_2, 2, R, i, j] = 1
                                g[idx + nV_2 + nidx * nidx, 2, hermi_list[R], i, j] = 1
                    
                    self.grad = g
                    self.grad_k = np.asarray([Lat.R2k(g[i]) for i in \
                            range(g.shape[0])])
                return self.grad
        
    def update(self, param):
        assert len(param) == self.length()
        self.param = param
        self.value = self.evaluate()
        self.value_k = Lat.R2k(self.value)
    
    def get(self, i=0, kspace=True, return_all=False):
        log.eassert(self.value is not None, "Vcor not initialized yet")
        if kspace:
            if return_all:
                return self.value_k
            else:
                return self.value_k[:, i]
        else:
            if return_all:
                return self.value
            else:
                return self.value[:, i]
    
    def assign(self, v0):
        log.eassert(v0.shape == self.gradient().shape[1:], \
            "The correlation potential should have shape %s, rather than %s",
            self.gradient().shape[1:], v0.shape)
        v0prime = v0
        param = np.empty(self.length())
        g = self.gradient()
        for i in range(self.length()):
            param[i] = np.sum(g[i] * v0prime) / np.sum(g[i] * g[i])
        self.update(param)
        log.check(la.norm(v0-self.get(kspace=False, return_all=True)) < 1e-7, \
                "symmetrization imposed on initial guess")
    
    v.local = False
    v.evaluate = types.MethodType(evaluate, v)
    v.gradient = types.MethodType(gradient, v)
    v.length = types.MethodType(lambda self: nparam, v)
    v.update = types.MethodType(update, v)
    v.get = types.MethodType(get, v)
    v.assign = types.MethodType(assign, v)
    return v
