#! /usr/bin/env python

"""
Test spinless.
"""

def test_bath_offsets():
    import numpy as np

    ncells = 4
    nlo = 5
    nso = nlo * 2
    nbath = nso
    
    latt_val_idx = [0, 1]
    latt_imp_idx = [0, 1, 2, 3]
    
    val_idx = list(latt_val_idx) + [i + nlo for i in latt_val_idx]
    imp_idx = list(latt_imp_idx) + [i + nlo for i in latt_imp_idx]
    imp_idx_bath = val_idx
    env_idx = []
    # boolean mask of virtual in the env_idx
    virt_mask = []
    # boolean mask of alpha orbitals in the env_idx
    alpha_mask = []
    for R in range(ncells):
        for s in range(2):
            for i in range(nlo):
                idx = R * nso + s * nlo + i
                if not idx in imp_idx_bath:
                    env_idx.append(idx)
                    virt_mask.append(idx in imp_idx)
                    alpha_mask.append(s == 0) 
    nimp  = len(imp_idx)
    
    print ("env idx")
    print (env_idx)
    print ("virtual idx")
    print (np.array(env_idx)[virt_mask])
    print ("alpha idx")
    print (np.array(env_idx)[alpha_mask])

def test_unit2emb():
    import numpy as np
    import scipy.linalg as la
    from pyscf import ao2mo
    from libdmet.routine import spinless
    from libdmet.utils.misc import max_abs
    
    nao = 5
    nso = nao * 2
    neo = 13
 
    # 1-fold
    H2_unit = np.random.random((nao, nao, nao, nao))
    
    H2_unit = H2_unit + H2_unit.transpose(1, 0, 2, 3)
    H2_unit = H2_unit + H2_unit.transpose(0, 1, 3, 2)
    H2_unit = H2_unit + H2_unit.transpose(2, 3, 1, 0)
    H2_unit_s4 = ao2mo.restore(4, H2_unit, nao)
    H2_unit_s8 = ao2mo.restore(8, H2_unit, nao)
    
    GH2_unit, _, _ = spinless.transform_H2_local(H2_unit)
    
    # test restore_eri_local
    GH2_unit_s4, _, _ = spinless.transform_H2_local(H2_unit_s4)
    assert max_abs(spinless.restore_eri_local(GH2_unit, nao) \
            - GH2_unit_s4) < 1e-12
    for s in range(GH2_unit.shape[0]):
        assert max_abs(ao2mo.restore(1, GH2_unit_s4[s], nao) \
                - GH2_unit[s]) < 1e-12
    
    GH2_emb_ref = np.zeros((neo, neo, neo, neo))
    GH2_emb_ref[:nao, :nao, :nao, :nao] = GH2_unit[0]
    GH2_emb_ref[nao:nso, nao:nso, nao:nso, nao:nso] = GH2_unit[1]
    GH2_emb_ref[:nao, :nao, nao:nso, nao:nso] = GH2_unit[2]
    GH2_emb_ref[nao:nso, nao:nso, :nao, :nao] = GH2_unit[2].conj().T
    from libdmet.system.integral import check_perm_symm
    check_perm_symm(GH2_emb_ref)
    GH2_emb_ref_s4 = ao2mo.restore(4, GH2_emb_ref, neo)
    
    # test unit2emb
    GH2 = spinless.unit2emb(GH2_unit_s4, neo)
    assert  max_abs(GH2 - GH2_emb_ref_s4) < 1e-12
    assert max_abs(ao2mo.restore(1, GH2, neo) - GH2_emb_ref) < 1e-12

if __name__ == "__main__":
    test_bath_offsets()
    test_unit2emb()
