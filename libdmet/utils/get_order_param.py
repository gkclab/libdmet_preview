#! /usr/bin/env python 

"""
Caculate the AFM and d-wave order parameter
from generalized density matrix.

Author:
    Zhi-Hao Cui
"""

import itertools as it
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
    from libdmet.routine.spinless_helper import extractRdm
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

def get_checkerboard_order(GRho, Lat=None, Cu_idx=None, O_idx=None, tol=1e-8):
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
    from libdmet.routine.spinless_helper import extractRdm
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
        Cu_idx = [idx for idx, name in zip(range(nao), Lat.names[:nao])
                  if name == "Cu" or name == "X"]
    if O_idx is None:
        O_idx = [idx for idx, name in zip(range(nao), Lat.names[:nao])
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
                if vec[0] > tol and vec[1] < tol:
                    phase_Cu_Cu_dic[(i, j)] =  1
                elif vec[1] > tol and vec[0] < tol:
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
                    if vec[0] > tol and vec[1] < tol:
                        phase_nn_O_O_dic[(i, j)] =  1
                    elif vec[1] > tol and vec[0] < tol:
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
            log.result("m_SC (next nearest O-O): %s", m_nn_O_O_abs)
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

def _norm(m_Cu_Cu, norm):
    if norm is None:
        m_Cu_Cu_tot = np.abs(m_Cu_Cu.sum())
    elif norm[0].lower() == 'f':
        m_Cu_Cu = m_Cu_Cu ** 2
        m_Cu_Cu_tot = np.sqrt(m_Cu_Cu.sum())
    elif norm[0].lower() == 'a':
        m_Cu_Cu = np.abs(m_Cu_Cu)
        m_Cu_Cu_tot = m_Cu_Cu.sum()
    else:
        raise ValueError
    return m_Cu_Cu, m_Cu_Cu_tot

def get_order_ab_initio(Lat, rdm1_glob_k, labels, dis_Cu_Cu=[3.0, 4.5],
                        dis_O_O=[3.0, 4.5], rdm1_d=None, s_wave=False,
                        bond_type_Cu=[("Cu", "Cu"), ("Cu1", "Cu2"),
                                      ("Cu3", "Cu4"), ("Cu5", "Cu6"),
                                      ("X", "X")],
                        bond_type_O=[("O", "O"), ("O1", "O1"),
                                     ("O3", "O3"), ("O5", "O5")],
                        tol=1e-8, dis_Cu_O=None, norm='abs'):
    """
    Calculate ab initio pairing order.

    Args:
        norm: 'fro', 'abs', None
    """
    if not s_wave:
        get_order_ab_initio(Lat, rdm1_glob_k, labels, dis_Cu_Cu=dis_Cu_Cu,
                            dis_O_O=dis_O_O, rdm1_d=rdm1_d, s_wave=True,
                            bond_type_Cu=bond_type_Cu, bond_type_O=bond_type_O,
                            tol=tol, norm=norm)
    
    from libdmet.system.lattice import round_to_FUC
    from libdmet_solid.lo import iao, ibo
    from libdmet_solid.utils import max_abs
    log.info("\n" + "-" * 79)
    log.info("Pairing analysis")
    log.info("-" * 79)
    
    nlo = len(labels)
    idx_re = iao.get_idx_to_ao_labels(Lat.cell, minao=None, labels=labels)
    labels = np.asarray(labels)[idx_re]
    
    if rdm1_d is None:
        rdm1_d = np.array(rdm1_glob_k[:, :nlo, nlo:], copy=True)
        rdm1_d = np.array(rdm1_d[:, idx_re][:, :, idx_re], copy=True)
        rdm1_d = Lat.k2R(rdm1_d)[0]
    else:
        rdm1_d = rdm1_d[:nlo, nlo:]
        rdm1_d = rdm1_d[idx_re][:, idx_re]
    
    norm_max = max_abs(rdm1_d)
    norm_tot = la.norm(rdm1_d)
    log.info("max   norm of anormalous part: %15.5f", norm_max)
    log.info("total norm of anormalous part: %15.5f", norm_tot)

    ovlp = np.eye(rdm1_d.shape[-1])
    mol = Lat.cell
    offset = ibo.get_offset(labels)
    if hasattr(Lat, "frac_coords"):
        coords = Lat.frac_coords()
    else:
        coords = Lat.sites

    factor = 0.5 ** 0.5 #* 0.5
    dic = {}

    # First compute Cu-Cu order
    log.info("Cu - Cu order:")
    log.info("-" * 79)
    
    pairs, dis = Lat.get_bond_pairs(mol, length_range=dis_Cu_Cu, unit='A',
                                    allow_pbc=True, nimgs=[1, 1, 1],
                                    bond_type=bond_type_Cu,
                                    triu=True)
    
    log.info("%5s %64s %12s %12s %15s"%("index", "pair", "length", "sign", "value"))
    idx = 0
    
    pairs_new = []
    orders = []
    signs = []
    m_Cu_Cu = 0.0

    for i, (pair, d) in enumerate(zip(pairs, dis)):
        atom_0 = mol._atom[pair[0]][0]
        atom_1 = mol._atom[pair[1]][0]
        idx_0 = offset[pair[0]]
        idx_1 = offset[pair[1]]
        coord_0 = coords[pair[0]]
        coord_1 = coords[pair[1]]
        vec = round_to_FUC(coord_0 - coord_1)
        if s_wave:
            sign = 1
        else:
            if vec[0] > tol and vec[1] < tol:
                sign =  1
            elif vec[1] > tol and vec[0] < tol:
                sign = -1
            else:
                # special treatment for distorted lattice
                if abs(vec[0] - 1.0) < tol:
                    vec[0] = 0.0
                elif abs(vec[1] - 1.0) < tol:
                    vec[1] = 0.0
                if vec[0] > tol and vec[1] < tol:
                    sign =  1
                elif vec[1] > tol and vec[0] < tol:
                    sign = -1
                else:
                    raise ValueError
        
        order = (rdm1_d[idx_0, idx_1] + rdm1_d[idx_1, idx_0].T) * factor
        m_Cu_Cu += order * sign
        log.info("%5d %4s %4s [ %5.3f %5.3f %5.3f ] --%4s %4s [ %5.3f %5.3f %5.3f ]   %10.3f  %10d %15.5f"
                 %(idx, pair[0], atom_0, *coord_0, pair[1], atom_1, *coord_1, d, sign, order.sum()))
        pairs_new.append(pair)
        orders.append(order)
        signs.append(sign)
        idx += 1
     
    m_Cu_Cu, m_Cu_Cu_tot = _norm(m_Cu_Cu, norm)

    lab_0 = labels[idx_0]
    lab_1 = labels[idx_1]

    log.info("-" * 79)
    string = "          "
    for j, labj in enumerate(lab_1):
        string += "%10s"%labj.split()[-1]
    log.info(string)
    for i, labi in enumerate(lab_0):
        string = "%10s"%labi.split()[-1]
        for j, labj in enumerate(lab_1):
            string += "%10.5f"%m_Cu_Cu[i, j]
        log.info(string)
    
    log.info("\ntotal Cu - Cu order = %15.8g \n", m_Cu_Cu_tot)
    
    dic["m_Cu_Cu"] = m_Cu_Cu_tot
    dic["m_Cu_Cu_sum"] = m_Cu_Cu
    dic["m_Cu_Cu_all"] = np.asarray(orders)
    dic["pairs_Cu_Cu"] = np.asarray(pairs_new)
    dic["signs_Cu_Cu"] = np.asarray(signs)

    # Then compute O - O order
    log.info("-" * 79)
    log.info("O - O order:")
    log.info("-" * 79)

    pairs, dis = Lat.get_bond_pairs(mol, length_range=dis_O_O, unit='A',
                                    allow_pbc=True, nimgs=[1, 1, 1],
                                    bond_type=bond_type_O,
                                    triu=True)

    log.info("%5s %64s %12s %12s %15s"%("index", "pair", "length", "sign", "value"))
    idx = 0
    m_O_O = 0.0
    pairs_new = []
    orders = []
    signs = []

    for i, (pair, d) in enumerate(zip(pairs, dis)):
        atom_0 = mol._atom[pair[0]][0]
        atom_1 = mol._atom[pair[1]][0]
        idx_0 = offset[pair[0]]
        idx_1 = offset[pair[1]]
        coord_0 = coords[pair[0]]
        coord_1 = coords[pair[1]]
        vec = round_to_FUC(coord_0 - coord_1)
        if s_wave:
            sign = 1
        else:
            if vec[0] > tol and vec[1] < tol:
                sign =  1
            elif vec[1] > tol and vec[0] < tol:
                sign = -1
            else:
                # special treatment for distorted lattice
                if abs(vec[0] - 1.0) < tol:
                    vec[0] = 0.0
                elif abs(vec[1] - 1.0) < tol:
                    vec[1] = 0.0
                if vec[0] > tol and vec[1] < tol:
                    sign =  1
                elif vec[1] > tol and vec[0] < tol:
                    sign = -1
                else:
                    raise ValueError
        
        order = (rdm1_d[idx_0, idx_1] + rdm1_d[idx_1, idx_0].T) * factor
        m_O_O += order * sign
        log.info("%5d %4s %4s [ %5.3f %5.3f %5.3f ] --%4s %4s [ %5.3f %5.3f %5.3f ]   %10.3f  %10d %15.5f"
                 %(idx, pair[0], atom_0, *coord_0, pair[1], atom_1, *coord_1, d, sign, order.sum()))
        pairs_new.append(pair)
        orders.append(order)
        signs.append(sign)
        idx += 1
    
    m_O_O, m_O_O_tot = _norm(m_O_O, norm)
            
    lab_0 = labels[idx_0]
    lab_1 = labels[idx_1]

    log.info("-" * 79)
    string = "          "
    for j, labj in enumerate(lab_1):
        string += "%10s"%labj.split()[-1]
    log.info(string)
    for i, labi in enumerate(lab_0):
        string = "%10s"%labi.split()[-1]
        for j, labj in enumerate(lab_1):
            string += "%10.5f"%m_O_O[i, j]
        log.info(string)
    
    log.info("\ntotal O - O order = %15.8g", m_O_O_tot)
    
    dic["m_O_O"] = m_O_O_tot
    dic["m_O_O_sum"] = m_O_O
    dic["m_O_O_all"] = np.asarray(orders)
    dic["pairs_O_O"] = np.asarray(pairs_new)
    dic["signs_O_O"] = np.asarray(signs)
    
    m_d_wave = m_Cu_Cu_tot + m_O_O_tot
    log.info("-" * 79)
    if s_wave:
        dic["m_tot"] = dic["m_s_wave"] =m_d_wave
        log.info("\ntotal s-wave order = %15.8g", m_d_wave)
    else:
        dic["m_tot"] = dic["m_d_wave"] =m_d_wave
        log.info("\ntotal d-wave order = %15.8g", m_d_wave)
    
    # finally compute Cu - O order
    if dis_Cu_O is not None and (not s_wave):
        log.info("-" * 79)
        log.info("Cu - O order:")
        log.info("-" * 79)

        bond_type_Cu_O = list(it.product(np.unique(bond_type_Cu), np.unique(bond_type_O)))
        pairs, dis = Lat.get_bond_pairs(mol, length_range=dis_Cu_O, unit='A',
                                        allow_pbc=True, nimgs=[1, 1, 1],
                                        bond_type=bond_type_Cu_O,
                                        triu=True)

        log.info("%5s %64s %12s %12s %15s"%("index", "pair", "length", "sign", "value"))
        idx = 0
        m_Cu_O = 0.0
        pairs_new = []
        orders = []
        signs = []
        
        for i, (pair, d) in enumerate(zip(pairs, dis)):
            atom_0 = mol._atom[pair[0]][0]
            atom_1 = mol._atom[pair[1]][0]
            idx_0 = offset[pair[0]]
            idx_1 = offset[pair[1]]
            coord_0 = coords[pair[0]]
            coord_1 = coords[pair[1]]
            vec = round_to_FUC(coord_0 - coord_1)
            sign = 1

            order = (rdm1_d[idx_0, idx_1] + rdm1_d[idx_1, idx_0].T) * factor
            m_Cu_O += abs(order) * sign
            log.info("%5d %4s %4s [ %5.3f %5.3f %5.3f ] --%4s %4s [ %5.3f %5.3f %5.3f ]   %10.3f  %10d %15.5f"
                     %(idx, pair[0], atom_0, *coord_0, pair[1], atom_1, *coord_1, d, sign, order.sum()))
            pairs_new.append(pair)
            orders.append(order)
            signs.append(sign)
            idx += 1
        
        m_Cu_O, m_Cu_O_tot = _norm(m_Cu_O, norm)
                
        lab_0 = labels[idx_0]
        lab_1 = labels[idx_1]

        log.info("-" * 79)
        string = "          "
        for j, labj in enumerate(lab_1):
            string += "%10s"%labj.split()[-1]
        log.info(string)
        for i, labi in enumerate(lab_0):
            string = "%10s"%labi.split()[-1]
            for j, labj in enumerate(lab_1):
                string += "%10.5f"%m_Cu_O[i, j]
            log.info(string)
        
        log.info("\nCu - O order = %15.8g", m_Cu_O_tot)
        
        dic["m_Cu_O"] = m_Cu_O_tot
        dic["m_Cu_O_sum"] = m_Cu_O
        dic["m_Cu_O_all"] = np.asarray(orders)
        dic["pairs_Cu_O"] = np.asarray(pairs_new)
        dic["signs_Cu_O"] = np.asarray(signs)


    log.info("-" * 79)
    
    return dic

def get_order_ab_initio_cell(Lat, rdm1_glob_k, labels, dis_Cu_Cu=[3.0, 4.5],
                             dis_O_O=[3.0, 4.5], rdm1_d=None, s_wave=False,
                             bond_type_Cu=[("Cu", "Cu"), ("Cu1", "Cu2"),
                                           ("Cu3", "Cu4"), ("Cu5", "Cu6"),
                                           ("X", "X")],
                             bond_type_O=[("O", "O"), ("O1", "O1"),
                                          ("O3", "O3"), ("O5", "O5")],
                             tol=1e-8, dis_Cu_O=None,
                             cell_groups={}, norm='abs'):
    """
    Compute the pairing orders based on the unit-cell.
    cell_groups should be provided as a dictionary, for example
    {0: (0, 1, 2), 1: (3, 4, 5), 2: (6, 7, 8), 3: (9, 10, 11)}
    the cell arrangement is the following:
    1--2
    |  |
    0--3
    """
    from pyscf.data.elements import _std_symbol
    from libdmet.system.lattice import round_to_FUC, round_to_FBZ
    from libdmet_solid.lo import iao, ibo
    from libdmet_solid.utils import max_abs
    log.info("\n" + "-" * 79)
    log.info("Pairing analysis based on unit-cell")
    log.info("-" * 79)
    
    nlo = len(labels)
    idx_re = iao.get_idx_to_ao_labels(Lat.cell, minao=None, labels=labels)
    labels = np.asarray(labels)[idx_re]
    
    if rdm1_d is None:
        rdm1_d = np.array(rdm1_glob_k[:, :nlo, nlo:], copy=True)
        rdm1_d = np.array(rdm1_d[:, idx_re][:, :, idx_re], copy=True)
        rdm1_d = Lat.k2R(rdm1_d)[0]
    else:
        rdm1_d = rdm1_d[:nlo, nlo:]
        rdm1_d = rdm1_d[idx_re][:, idx_re]
    
    norm_max = max_abs(rdm1_d)
    norm_tot = la.norm(rdm1_d)
    log.info("max   norm of anormalous part: %15.5f", norm_max)
    log.info("total norm of anormalous part: %15.5f", norm_tot)

    ovlp = np.eye(rdm1_d.shape[-1])
    mol = Lat.cell
    offset = ibo.get_offset(labels)
    if hasattr(Lat, "frac_coords"):
        coords = Lat.frac_coords()
    else:
        coords = Lat.sites

    dic = {}

    bond_type_Cu_O = list(it.product(np.unique(bond_type_Cu), np.unique(bond_type_O)))
    bond_type_all = list(bond_type_Cu) + list(bond_type_O) + list(bond_type_Cu_O)
    
    pairs, dis = Lat.get_bond_pairs(mol, length_range=[0.0, dis_Cu_Cu[1]], unit='A',
                                    allow_pbc=True, nimgs=[1, 1, 1],
                                    bond_type=bond_type_all,
                                    triu=True)
    
    idx_dis = np.argsort(dis, kind='mergesort')
    dis = np.asarray(dis)[idx_dis]
    pairs = np.asarray(pairs)[idx_dis]
    
    cell_dis = {(0, 1): 1, (0, 2): np.sqrt(2), (0, 3): 1,
                (1, 0): -1, (1, 2): 1, (1, 3): np.sqrt(2),
                (2, 0): np.sqrt(2), (2, 1): -1, (2, 3): -1,
                (3, 0): -1, (3, 1): np.sqrt(2), (3, 2): 1}

    ngroups = len(cell_groups)
    assert ngroups == 4
    pairs_intra = {}
    pairs_inter = {}

    for gp in range(ngroups):
        pairs_intra[gp] = []
    
    for gp1 in range(ngroups):
        for gp2 in range(ngroups):
            if gp1 != gp2 and abs(cell_dis[(gp1, gp2)]) == 1:
                pairs_inter[(gp1, gp2)] = []
    
    idx2cell = {}
    for cell_id, idxs in cell_groups.items():
        for idx in idxs:
            idx2cell[idx] = cell_id

    for i, (pair, d) in enumerate(zip(pairs, dis)):
        atom_0 = mol._atom[pair[0]][0]
        atom_1 = mol._atom[pair[1]][0]
        coord_0 = coords[pair[0]]
        coord_1 = coords[pair[1]]
        vec_raw = coord_1 - coord_0
        vec = round_to_FBZ(vec_raw)
        
        i, j = pair
        idxi = idx2cell[i]
        idxj = idx2cell[j]
        if idxi == idxj:
            pairs_intra[idxi].append(pair)
        else:
            if abs(cell_dis[(idxi, idxj)]) > 1:
                continue
            if abs(vec[0]) < tol:
                if abs(vec[1]) < tol:
                    raise ValueError
                else:
                    if abs(vec[1] + 0.5) < tol:
                        pairs_inter[(idxi, idxj)].append(pair)
                        pairs_inter[(idxj, idxi)].append(pair)
                    else:
                        if vec[1] > 0:
                            pairs_inter[(idxi, idxj)].append(pair)
                        else:
                            pairs_inter[(idxj, idxi)].append(pair)
            else:
                if abs(vec[1]) < tol:
                    if abs(vec[0] + 0.5) < tol:
                        pairs_inter[(idxi, idxj)].append(pair)
                        pairs_inter[(idxj, idxi)].append(pair)
                    else:
                        if vec[0] > 0:
                            pairs_inter[(idxi, idxj)].append(pair)
                        else:
                            pairs_inter[(idxj, idxi)].append(pair)
    
    # intra cell s order 
    log.info("-" * 79)
    log.info("intra cell analysis")
    log.info("-" * 79)
    order_intra = {}
    for i, (cell_id, pairs) in enumerate(pairs_intra.items()):
        log.info("cell_id : %5s", cell_id)
        log.info("%5s %64s %12s %12s %15s"%("index", "pair", "length", "sign", "value"))
        for pair in pairs:
            atom_0 = mol._atom[pair[0]][0]
            atom_1 = mol._atom[pair[1]][0]
            idx_0 = offset[pair[0]]
            idx_1 = offset[pair[1]]
            coord_0 = coords[pair[0]]
            coord_1 = coords[pair[1]]
            vec_raw = coord_1 - coord_0
            vec = round_to_FBZ(vec_raw)
            d = la.norm(vec)

            factor = 1.0 / np.sqrt(2)
            sign = 1.0
            order = (rdm1_d[idx_0, idx_1] + rdm1_d[idx_1, idx_0].T) * factor
            m_Cu_Cu = order * sign
            log.info("%5d %4s %4s [ %5.3f %5.3f %5.3f ] --%4s %4s [ %5.3f %5.3f %5.3f ]   %10.3f  %10d %15.5f"
                     %(idx, pair[0], atom_0, *coord_0, pair[1], atom_1, *coord_1, d, sign, order.sum()))
            idx += 1
            key = (_std_symbol(atom_0), _std_symbol(atom_1))
            key_rev = (_std_symbol(atom_1), _std_symbol(atom_0))
            if key in order_intra:
                order_intra[key] += m_Cu_Cu
            elif key_rev in order_intra:
                order_intra[key_rev] += m_Cu_Cu.T
            else:
                order_intra[key] = m_Cu_Cu
    
    assert len(order_intra) == 2 # only O-O and Cu-O are allowed
    dic["order_intra"] = order_intra

    intra_tot = 0.0
    if norm is None:
        for val in order_intra.values():
            intra_tot += val.sum()
        intra_tot = abs(intra_tot)
    elif norm[0].lower() == 'f':
        for val in order_intra.values():
            intra_tot += (_norm(val, norm=norm)[1])**2
        intra_tot = np.sqrt(intra_tot)
    elif norm[0].lower() == 'a':
        for val in order_intra.values():
            intra_tot += _norm(val, norm=norm)[1]

    dic["order_intra_tot"] = intra_tot
    log.info("-" * 79)
    log.info("total intra-cell order (s-wave) : %15.5f", intra_tot)
    log.info("-" * 79)
   
    # ZHC NOTE
    # inter cell s, p, d waves
    log.info("-" * 79)
    log.info("inter cell analysis")
    log.info("-" * 79)
    for wave in ['s', 'px', 'py', 'd']:
        log.info("--- %s wave ---", wave)
        log.info("-" * 79)
        order_inter = {}
        idx = 0
        for i, ((cell_id1, cell_id2), pairs) in enumerate(pairs_inter.items()):
            log.info("cell_id pair : %5s - %5s", cell_id1, cell_id2)
            log.info("%5s %64s %12s %12s %15s"%("index", "pair", "length", "sign", "value"))
            for pair in pairs:
                atom_0 = mol._atom[pair[0]][0]
                atom_1 = mol._atom[pair[1]][0]
                idx_0 = offset[pair[0]]
                idx_1 = offset[pair[1]]
                coord_0 = coords[pair[0]]
                coord_1 = coords[pair[1]]
                vec_raw = coord_1 - coord_0
                vec = round_to_FUC(vec_raw)
                d = la.norm(vec)
                # special treatment for distorted lattice
                if abs(vec[0] - 1.0) < tol:
                    vec[0] = 0.0
                elif abs(vec[1] - 1.0) < tol:
                    vec[1] = 0.0

                if wave == 's':
                    factor = 0.5 / np.sqrt(2)
                    sign = 1.0
                elif wave == 'px':
                    factor = 0.5
                    if (cell_id1, cell_id2) in [(0, 3), (1, 2)]:
                        sign = 1.0
                    elif (cell_id1, cell_id2) in [(3, 0), (2, 1)]:
                        sign = -1.0
                    else:
                        sign = 0.0
                elif wave == 'py':
                    factor = 0.5
                    if (cell_id1, cell_id2) in [(0, 1), (3, 2)]:
                        sign = 1.0
                    elif (cell_id1, cell_id2) in [(1, 0), (2, 3)]:
                        sign = -1.0
                    else:
                        sign = 0.0
                elif wave == 'd':
                    factor = 0.5 / np.sqrt(2)
                    if (cell_id1, cell_id2) in [(0, 3), (1, 2), (3, 0), (2, 1)]:
                        sign = 1.0
                    elif (cell_id1, cell_id2) in [(0, 1), (3, 2), (1, 0), (2, 3)]:
                        sign = -1.0
                    else:
                        sign = 0.0
                    #if abs(d - 0.5) > tol:
                    #    sign = 0.0
                else:
                    raise ValueError
                order = (rdm1_d[idx_0, idx_1] + rdm1_d[idx_1, idx_0].T) * factor
                m_Cu_Cu = order * sign
                
                log.info("%5d %4s %4s [ %5.3f %5.3f %5.3f ] --%4s %4s [ %5.3f %5.3f %5.3f ]   %10.3f  %10d %15.5f"
                         %(idx, pair[0], atom_0, *coord_0, pair[1], atom_1, *coord_1, d, sign, order.sum()))
                idx += 1
                key = (_std_symbol(atom_0), _std_symbol(atom_1))
                key_rev = (_std_symbol(atom_1), _std_symbol(atom_0))
                if key in order_inter:
                    order_inter[key] += m_Cu_Cu
                elif key_rev in order_inter:
                    order_inter[key_rev] += m_Cu_Cu.T
                else:
                    order_inter[key] = m_Cu_Cu
        
        assert len(order_inter) == 3 # only Cu-Cu, O-O and Cu-O are allowed
        dic["order_inter_%s"%wave] = order_inter
        
        inter_tot = 0.0
        if norm is None:
            for val in order_inter.values():
                inter_tot += val.sum()
            inter_tot = abs(inter_tot)
        elif norm[0].lower() == 'f':
            for val in order_inter.values():
                inter_tot += (_norm(val, norm=norm)[1])**2
            inter_tot = np.sqrt(inter_tot)
        elif norm[0].lower() == 'a':
            for val in order_inter.values():
                inter_tot += _norm(val, norm=norm)[1]

        dic["order_inter_tot_%s"%wave] = inter_tot
        log.info("-" * 79)
        log.info("total inter-cell order (%s-wave) : %15.5f", wave, inter_tot)
        log.info("-" * 79)

    log.info("-" * 79)
    
    return dic

get_ab_initio_order = get_order_ab_initio

get_ab_initio_order_cell = get_order_ab_initio_cell

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
