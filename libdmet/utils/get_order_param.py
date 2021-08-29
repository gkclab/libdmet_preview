#! /usr/bin/env python 

"""
Caculate the AFM and d-wave order parameter
from generalized density matrix.

Author:
    Zhi-Hao Cui
"""

import h5py
import numpy as np
import scipy.linalg as la

from libdmet.utils import logger as log

def get_order_param(GRho, idx=[0, 1, 2, 3], return_abs=True):
    """
    Calculate the AF and d-wave SC order parameter.
    Ref: PRB, 93, 035126 (2016).
    Default lattice shape:
    2D Hubbard (0, 1, 2, 3):
        0 1
        2 3
    3-band Hubbard (0, 3, 9, 6):
        3 6
        0 9

    Args:
        GRho: generalized rdm1.
        idx: the index of 2x2 sites.
    Return:
        m_AF: local AFM moment.
        m_SC: d-wave SC order parameter.
    """
    from libdmet.routine.bcs_helper import extractRdm
    rdm1_a, rdm1_b, rdm1_d = extractRdm(GRho)
    mesh = np.ix_(idx, idx)
    rdm1_a = rdm1_a[mesh]
    rdm1_b = rdm1_b[mesh]
    rdm1_d = rdm1_d[mesh]
    
    # 0, 3 alpha, 1, 2 beta
    m0 = 0.5 * (rdm1_a[0, 0] - rdm1_b[0, 0])
    m3 = 0.5 * (rdm1_a[3, 3] - rdm1_b[3, 3])
    m1 = 0.5 * (rdm1_a[1, 1] - rdm1_b[1, 1])
    m2 = 0.5 * (rdm1_a[2, 2] - rdm1_b[2, 2])
    m_AF = 0.25 * (m0 + m3 - m1 - m2)

    factor = 0.5 ** 0.5
    d01 = factor * (rdm1_d[0, 1] + rdm1_d[1, 0])
    d23 = factor * (rdm1_d[2, 3] + rdm1_d[3, 2])
    d02 = factor * (rdm1_d[0, 2] + rdm1_d[2, 0])
    d13 = factor * (rdm1_d[1, 3] + rdm1_d[3, 1])
    m_SC = 0.25 * (d01 + d23 - d02 - d13)
    
    if return_abs:
        m_AF = abs(m_AF)
        m_SC = abs(m_SC)
    return m_AF, m_SC

get_order_param_1band = get_order_param

def get_checkerboard_order(GRho, Lat=None, Cu_idx=None, O_idx=None):
    """
    Get the order parameters of the Checkerboard 1- or 3-band Hubbard model.

    Args:
        GRho: rdm1 (spin, nao, nao) or generalized rdm1 (nso, nso).
        Lat: lattice object. If None, will use 2x2 3-band symmetrized cluster.
        Cu_idx: indices of Cu.
        O_idx : indices of O.

    Returns:
        res: dictionary, including the following:
            m_AFM, m_SC,
            m_AFM_Cu, m_AFM_Cu_list, m_AFM_O, m_AFM_O_list, phase_AFM,
            charge_Cu, charge_O
            m_Cu_Cu, m_Cu_Cu_dic, phase_Cu_Cu_dic, 
            m_nn_O_O, m_nn_O_O_dic, phase_nn_O_O_dic,
            m_Cu_O_dic, m_n_O_O_dic
    """ 
    from libdmet.routine.bcs_helper import extractRdm
    from libdmet.system import lattice
    from libdmet.system.lattice import Frac2Real, Real2Frac, round_to_FUC
    if Lat is None:
        Lat = lattice.Square3BandSymm(1, 1, 1, 1)
    else:
        dist = Lat.neighborDist
        Lat = lattice.LatticeModel(Lat.supercell, np.array([1, 1]))
        Lat.neighborDist = dist
    nao = Lat.nao
    
    GRho = np.asarray(GRho)
    if GRho.shape[-1] == nao:
        assert GRho.ndim == 3
        if GRho.shape[0] == 1:
            rdm1_a = rdm1_b = GRho[0]
        else:
            rdm1_a, rdm1_b = GRho
        rdm1_d = None
    elif GRho.shape[-1] == nao * 2:
        rdm1_a, rdm1_b, rdm1_d = extractRdm(GRho)
    else:
        raise ValueError
    
    if Cu_idx is None:
        Cu_idx = [idx for idx, name in zip(range(nao), Lat.names[:nao]) \
                if name == "Cu" or name == "X"]
    if O_idx is None:
        O_idx = [idx for idx, name in zip(range(nao), Lat.names[:nao]) \
                if name == "O"]
    
    if len(O_idx) != 0: # 3band
        d_dd = Lat.neighborDist[2]
    else: # 1band
        d_dd = Lat.neighborDist[0]

    res = {} # result dict
    
    # total charge and spin density
    charge = np.diag(rdm1_a) + np.diag(rdm1_b)
    spin_density = 0.5 * (np.diag(rdm1_a) - np.diag(rdm1_b))
    res["charge"] = charge
    res["spin_density"] = spin_density
    
    # m_AFM_Cu
    Cu_coords = np.asarray(Lat.sites)[Cu_idx]
    Cu0_idx = np.argmin(la.norm(Cu_coords, axis=1))
    Cu_coords = ((Cu_coords - Cu_coords[Cu0_idx]) / d_dd).astype(int)
    phase_AFM = np.ones(len(Cu_idx), dtype=int)
    phase_AFM[Cu_coords.sum(axis=1) % 2 == 1] = -1
    log.debug(2, "phase_AFM:\n%s", phase_AFM)
    assert phase_AFM.sum() == 0
    phase_FM = np.ones_like(phase_AFM)

    charge_Cu = rdm1_a[Cu_idx, Cu_idx] + rdm1_b[Cu_idx, Cu_idx]
    m_AFM_Cu_list = 0.5 * (rdm1_a[Cu_idx, Cu_idx] - rdm1_b[Cu_idx, Cu_idx])
    m_AFM_Cu = np.average(m_AFM_Cu_list * phase_AFM)
    m_AFM_Cu_abs = abs(m_AFM_Cu)
    m_FM_Cu = np.average(m_AFM_Cu_list * phase_FM)
    m_FM_Cu_abs = abs(m_FM_Cu)
    res["m_AFM_Cu_list"] = m_AFM_Cu_list
    res["m_AFM_Cu"] = m_AFM_Cu
    res["phase_AFM"] = phase_AFM
    res["m_AFM"] = m_AFM_Cu_abs
    res["m_FM_Cu"] = m_FM_Cu
    res["phase_FM"] = phase_FM
    res["m_FM"] = m_FM_Cu_abs
    res["charge_Cu"] = charge_Cu
    log.result("charge (Cu): %s",  charge_Cu)
    log.result("magnet (Cu): %s",  m_AFM_Cu_list)
    log.result("Average m_AFM (Cu): %s",  m_AFM_Cu_abs)
    log.result("Average m_FM  (Cu): %s",  m_FM_Cu_abs)
    
    # m_AFM_O
    if len(O_idx) != 0:
        charge_O = rdm1_a[O_idx, O_idx] + rdm1_b[O_idx, O_idx]
        m_AFM_O_list = 0.5 * (rdm1_a[O_idx, O_idx] - rdm1_b[O_idx, O_idx])
        m_AFM_O = np.average(np.abs(m_AFM_O_list))
        res["m_AFM_O"] = m_AFM_O
        res["m_AFM_O_list"] = m_AFM_O_list
        res["charge_O"] = charge_O
        log.result("charge (O): %s",  charge_O)
        log.result("m_AFM  (O): %s",  m_AFM_O_list)
        log.result("Average m_AFM (O) : %s",  m_AFM_O)
            
    # SC orders
    if rdm1_d is not None:
        def get_vec(s1, s2):
            # round vector to [-0.5, 0.5)
            vec_frac = Real2Frac(Lat.size, Lat.sites[s1] - Lat.sites[s2])
            vec_frac = round_to_FUC(vec_frac, tol=1e-8, wrap_around=True)
            vec = Frac2Real(Lat.size, vec_frac)
            return vec
        
        factor = 0.5 ** 0.5

        # 1. Cu-Cu order
        dd_pairs  = Lat.neighbor(dis=d_dd, sitesA=Cu_idx)
        m_Cu_Cu_dic = {}
        phase_Cu_Cu_dic = {}

        for (i, j) in dd_pairs:
            if (j, i) in m_Cu_Cu_dic:
                m_Cu_Cu_dic[(j, i)] += rdm1_d[i, j] * factor
            else:
                m_Cu_Cu_dic[(i, j)]  = rdm1_d[i, j] * factor
                vec = np.abs(get_vec(i, j))
                if vec[0] > 1e-8 and vec[1] < 1e-8:
                    phase_Cu_Cu_dic[(i, j)] =  1
                elif vec[1] > 1e-8 and vec[0] < 1e-8:
                    phase_Cu_Cu_dic[(i, j)] = -1
                else:
                    raise ValueError
        
        m_Cu_Cu = 0.0
        for (i, j), m in m_Cu_Cu_dic.items():
            m_Cu_Cu += m * phase_Cu_Cu_dic[(i, j)]
        # ZHC NOTE whether to divide the number of Cu?
        #m_Cu_Cu /= float(len(Cu_idx))
        m_Cu_Cu_abs = abs(m_Cu_Cu)
        res["m_Cu_Cu"] = m_Cu_Cu
        res["m_Cu_Cu_dic"] = m_Cu_Cu_dic
        res["phase_Cu_Cu_dic"] = phase_Cu_Cu_dic

        log.result("m_SC (Cu-Cu): %s",  m_Cu_Cu_abs)
        
        res["m_SC"] = abs(m_Cu_Cu)
        
        if len(O_idx) != 0: # 3band
            d_pd  = Lat.neighborDist[0]
            d_pp  = Lat.neighborDist[1]
            d_pp1 = Lat.neighborDist[2]

            pd_pairs  = Lat.neighbor(dis=d_pd,  sitesA=range(nao))
            pp_pairs  = Lat.neighbor(dis=d_pp,  sitesA=range(nao))
            pp1_pairs = Lat.neighbor(dis=d_pp1, sitesA=O_idx)
        
            # 2. next nearest O-O order
            m_nn_O_O_dic = {}
            phase_nn_O_O_dic = {}

            for (i, j) in pp1_pairs:
                if (j, i) in m_nn_O_O_dic:
                    m_nn_O_O_dic[(j, i)] += rdm1_d[i, j] * factor
                else:
                    m_nn_O_O_dic[(i, j)]  = rdm1_d[i, j] * factor
                    vec = np.abs(get_vec(i, j))
                    if vec[0] > 1e-8 and vec[1] < 1e-8:
                        phase_nn_O_O_dic[(i, j)] =  1
                    elif vec[1] > 1e-8 and vec[0] < 1e-8:
                        phase_nn_O_O_dic[(i, j)] = -1
                    else:
                        raise ValueError
            
            m_nn_O_O = 0.0
            for (i, j), m in m_nn_O_O_dic.items():
                m_nn_O_O += m * phase_nn_O_O_dic[(i, j)]
            
            # ZHC NOTE whether to divide the number of Cu?
            #m_nn_O_O /= float(len(Cu_idx))
            m_nn_O_O_abs = abs(m_nn_O_O)
            
            res["m_nn_O_O"] = m_nn_O_O
            res["m_nn_O_O_dic"] = m_nn_O_O_dic
            res["phase_nn_O_O_dic"] = phase_nn_O_O_dic
            res["m_SC"] += m_nn_O_O_abs
            log.result("m_SC (next nearest O-O): %s",  m_nn_O_O_abs)
            log.result("m_SC (total): %s",  res["m_SC"])
            
            # 3. Cu-O order 
            m_Cu_O_dic = {}
            for (i, j) in pd_pairs:
                if (j, i) in m_Cu_O_dic:
                    m_Cu_O_dic[(j, i)] += rdm1_d[i, j] * factor
                else:
                    m_Cu_O_dic[(i, j)]  = rdm1_d[i, j] * factor
            
            # 4. nearest O-O order
            m_n_O_O_dic = {}
            for (i, j) in pp_pairs:
                if (j, i) in m_n_O_O_dic:
                    m_n_O_O_dic[(j, i)] += rdm1_d[i, j] * factor
                else:
                    m_n_O_O_dic[(i, j)]  = rdm1_d[i, j] * factor
            
            res["m_Cu_O_dic"] = m_Cu_O_dic
            res["m_n_O_O_dic"] = m_n_O_O_dic
    
    return res

get_1band_order = get_3band_order = get_checkerboard_order

if __name__ == '__main__':
    import sys
    np.set_printoptions(3, linewidth=1000, suppress=True)
    # program_name, filename, pos, idx
    if len(sys.argv) == 1 :
        fname = './dmet.npy'
        pos = -1
        idx = [0, 1, 2, 3]
    elif len(sys.argv) == 2:
        fname = sys.argv[1]
        pos = -1
        idx = [0, 1, 2, 3]
    elif len(sys.argv) == 3:
        fname = sys.argv[1]
        pos = int(sys.argv[2])
        idx = [0, 1, 2, 3]
    elif len(sys.argv) == 7:
        fname = sys.argv[1]
        pos = int(sys.argv[2])
        idx = tuple(map(int, sys.argv[3:]))
    else:
        raise ValueError

    GRhoImp = np.load(fname)[pos]
    m_AF, m_SC = get_order_param(GRhoImp, idx=idx)
    print ("AF order: %12.6f" % m_AF)
    print ("SC order: %12.6f" % m_SC)
