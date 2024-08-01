#!/usr/bin/env python

import os
import numpy as np
import scipy.linalg as la

from pyscf.pbc import gto as pgto
import pyscf.pbc.scf as pscf
from pyscf.pbc.lib import chkfile

from libdmet.system import lattice
from libdmet.lo import pywannier90
from libdmet.lo import iao
from libdmet.lo import lowdin
from libdmet.basis_transform import make_basis
from libdmet.utils.misc import mdot

def test_pywannier90_CH4_1():
    """
    CH4 minimal basis set 9 bands -> 9 wannier (Cs, 4Csp3, 4Hs)
    """
    import numpy as np
    from pyscf import scf, gto
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf

    cell = pgto.Cell()
    cell.atom = '''
     C                  3.17500000    3.17500000    3.17500000
     H                  2.54626556    2.54626556    2.54626556
     H                  3.80373444    3.80373444    2.54626556
     H                  2.54626556    3.80373444    3.80373444
     H                  3.80373444    2.54626556    3.80373444
    '''
    cell.basis = 'sto-3g'
    cell.a = np.eye(3) * 6.35
    #cell.a = np.eye(3) * 10.0
    cell.precision = 1e-13
    cell.verbose = 5
    cell.build()


    kmesh = [1, 1, 1]
    abs_kpts = cell.make_kpts(kmesh)
    
    chkfname = 'CH4_1.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = pywannier90.load_kmf(chkfname) 
    else:
        kmf = pscf.KRHF(cell, abs_kpts)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-13
        ekpt = kmf.run()
        pywannier90.save_kmf(kmf, chkfname)  # Save the wave function
            
    num_wann = 9
    keywords = \
    '''
    guiding_centres=true
    num_iter = 200
    dis_num_iter = 200
    begin projections
    C:l=0;sp3
    H:l=0
    end projections
    '''

    # wannier run
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords)
    w90.kernel()
    w90.export_unk()
    w90.export_AME()

    # Plotting using Wannier90
    keywords_plot = keywords + 'wannier_plot = True\n wannier_plot_supercell = 1\n'
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords_plot)
    w90.kernel()

def test_pywannier90_CH4_2():
    """
    CH4 321G 4 occ bands -> 4 wannier (4 bonding orbs)
    """
    import numpy as np
    from pyscf import scf, gto
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf

    cell = pgto.Cell()
    cell.atom = '''
     C                  3.17500000    3.17500000    3.17500000
     H                  2.54626556    2.54626556    2.54626556
     H                  3.80373444    3.80373444    2.54626556
     H                  2.54626556    3.80373444    3.80373444
     H                  3.80373444    2.54626556    3.80373444
    '''
    cell.basis = '321g'
    cell.a = np.eye(3) * 6.35
    #cell.a = np.eye(3) * 10.0
    cell.precision = 1e-13
    cell.verbose = 5
    cell.build()


    kmesh = [1, 1, 1]
    abs_kpts = cell.make_kpts(kmesh)
    
    chkfname = 'CH4_2.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = pywannier90.load_kmf(chkfname) 
    else:
        kmf = pscf.KRHF(cell, abs_kpts)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-13
        ekpt = kmf.run()
        pywannier90.save_kmf(kmf, chkfname)  # Save the wave function
            
    num_wann = 4
    keywords = \
    '''
    guiding_centres=true
    num_iter = 200
    dis_num_iter = 200
    begin projections
    C:sp3
    end projections
    exclude_bands : 1,6-%s
    '''%(cell.nao_nr())

    # wannier run
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords)
    w90.kernel()
    w90.export_unk()
    w90.export_AME()

    # Plotting using Wannier90
    keywords_plot = keywords + 'wannier_plot = True\n wannier_plot_supercell = 1\n'
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords_plot)
    w90.kernel()

def test_pywannier90_CH4_3():
    """
    CH4 321G 4 conduction bands -> 4 wannier (anti-bonding orb)
    """
    import numpy as np
    from pyscf import scf, gto
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf

    cell = pgto.Cell()
    cell.atom = '''
     C                  3.17500000    3.17500000    3.17500000
     H                  2.54626556    2.54626556    2.54626556
     H                  3.80373444    3.80373444    2.54626556
     H                  2.54626556    3.80373444    3.80373444
     H                  3.80373444    2.54626556    3.80373444
    '''
    cell.basis = '321g'
    cell.a = np.eye(3) * 6.35
    #cell.a = np.eye(3) * 10.0
    cell.precision = 1e-13
    cell.verbose = 5
    cell.build()


    kmesh = [1, 1, 1]
    abs_kpts = cell.make_kpts(kmesh)
    
    chkfname = 'CH4_3.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = pywannier90.load_kmf(chkfname) 
    else:
        kmf = pscf.KRHF(cell, abs_kpts)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-13
        ekpt = kmf.run()
        pywannier90.save_kmf(kmf, chkfname)  # Save the wave function
            
    num_wann = 4
    keywords = \
    '''
    guiding_centres=true
    num_iter = 400
    dis_num_iter = 200
    begin projections
    C:sp3
    end projections
    exclude_bands : 1-5,10-%s
    '''%(cell.nao_nr())

    # wannier run
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords)
    w90.kernel()
    w90.export_unk()
    w90.export_AME()

    # Plotting using Wannier90
    keywords_plot = keywords + 'wannier_plot = True\n wannier_plot_supercell = 1\n'
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords_plot)
    w90.kernel()

def test_pywannier90_CH4_4():
    """
    CH4 gth-dzv 4 occ + 4 virt bands -> 8 wannier
    """
    import numpy as np
    from pyscf import scf, gto
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf

    cell = pgto.Cell()
    cell.atom = '''
     C                  3.17500000    3.17500000    3.17500000
     H                  2.54626556    2.54626556    2.54626556
     H                  3.80373444    3.80373444    2.54626556
     H                  2.54626556    3.80373444    3.80373444
     H                  3.80373444    2.54626556    3.80373444
    '''
    cell.a = np.eye(3) * 6.35
    #cell.a = np.eye(3) * 10.0
    cell.basis = 'gth-dzv'
    cell.pseudo = 'gth-pade'
    cell.precision = 1e-13
    cell.verbose = 5
    cell.build()


    kmesh = [1, 1, 1]
    abs_kpts = cell.make_kpts(kmesh)
    
    chkfname = 'CH4_4.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = pywannier90.load_kmf(chkfname) 
    else:
        kmf = pscf.KRHF(cell, abs_kpts)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-13
        ekpt = kmf.run()
        pywannier90.save_kmf(kmf, chkfname)  # Save the wave function
            
    num_wann = 8
    keywords = \
    '''
    guiding_centres=true
    num_iter = 400
    dis_num_iter = 200
    begin projections
    C:l=0;l=1
    H:l=0
    end projections
    exclude_bands : 9-%s
    '''%(cell.nao_nr())

    # wannier run
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords)
    w90.kernel()
    w90.export_unk()
    w90.export_AME()

    # Plotting using Wannier90
    keywords_plot = keywords + 'wannier_plot = True\n wannier_plot_supercell = 1\n'
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords_plot)
    w90.kernel()

    """
    1- H s
    2- H s 
    3- H s  
    4- H s 
    5- C sp3 
    6- C sp3 
    7- C sp3 
    8- C sp3 
    """

def test_pywannier90_CH4_5():
    """
    CH4 gth-dzv high virtuals -> 8 wannier
    """
    import numpy as np
    from pyscf import scf, gto
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf

    cell = pgto.Cell()
    cell.atom = '''
     C                  3.17500000    3.17500000    3.17500000
     H                  2.54626556    2.54626556    2.54626556
     H                  3.80373444    3.80373444    2.54626556
     H                  2.54626556    3.80373444    3.80373444
     H                  3.80373444    2.54626556    3.80373444
    '''
    cell.a = np.eye(3) * 6.35
    #cell.a = np.eye(3) * 10.0
    cell.basis = 'gth-dzv'
    cell.pseudo = 'gth-pade'
    cell.precision = 1e-13
    cell.verbose = 5
    cell.build()
    

    kmesh = [1, 1, 1]
    abs_kpts = cell.make_kpts(kmesh)
    
    chkfname = 'CH4_5.chk'
    if os.path.isfile(chkfname):
    #if False:
        kmf = pywannier90.load_kmf(chkfname) 
    else:
        kmf = pscf.KRHF(cell, abs_kpts)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-13
        ekpt = kmf.run()
        pywannier90.save_kmf(kmf, chkfname)  # Save the wave function
            
    num_wann = cell.nao_nr() - 8
    keywords = \
    '''
    guiding_centres=true
    num_iter = 400
    dis_num_iter = 200
    begin projections
    C:sp3
    H:l=0
    end projections
    exclude_bands : 1-8
    '''

    # wannier run
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords)
    w90.kernel()
    w90.export_unk()
    w90.export_AME()

    # Plotting using Wannier90
    keywords_plot = keywords + 'wannier_plot = True\n wannier_plot_supercell = 1\n'
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords_plot)
    w90.kernel()

    """
    1- anti1
    2- anti1
    3- anti1
    4- anti1
    5- anti2
    6- anti2
    7- anti2
    8- anti2
    """

def test_pywannier90_CH4_6():
    """
    CH4 631G 4 occ + 4 virt bands -> 8 wannier
    """
    import numpy as np
    from pyscf import scf, gto
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf

    cell = pgto.Cell()
    cell.atom = '''
     C                  3.17500000    3.17500000    3.17500000
     H                  2.54626556    2.54626556    2.54626556
     H                  3.80373444    3.80373444    2.54626556
     H                  2.54626556    3.80373444    3.80373444
     H                  3.80373444    2.54626556    3.80373444
    '''
    cell.a = np.eye(3) * 6.35
    #cell.a = np.eye(3) * 10.0
    cell.basis = '631g'
    cell.precision = 1e-13
    cell.verbose = 5
    cell.build()


    kmesh = [1, 1, 1]
    abs_kpts = cell.make_kpts(kmesh)
    
    chkfname = 'CH4_6.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = pywannier90.load_kmf(chkfname) 
    else:
        kmf = pscf.KRHF(cell, abs_kpts)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-13
        ekpt = kmf.run()
        pywannier90.save_kmf(kmf, chkfname)  # Save the wave function
            
    num_wann = 8
    keywords = \
    '''
    guiding_centres=true
    num_iter = 400
    dis_num_iter = 200
    begin projections
    C:l=0;l=1
    H:l=0
    end projections
    exclude_bands : 1,10-%s
    '''%(cell.nao_nr())

    # wannier run
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords)
    w90.kernel()
    w90.export_unk()
    w90.export_AME()

    # Plotting using Wannier90
    keywords_plot = keywords + 'wannier_plot = True\n wannier_plot_supercell = 1\n'
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords_plot)
    w90.kernel()

    """
    1- C sp3
    2- C sp3
    3- C sp3
    4- C sp3
    5- H s
    6- H s
    7- H s
    8- H s
    """

def test_pywannier90_hchain_1():
    """
    H chain, 1 occ + 1 virt -> 2 wannier
    """
    cell = pgto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    #cell.atom = ''' H 0.0      0.0      0.75
    #                H 0.0      0.0      2.25 '''
    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    cell.basis = '321g'
    #cell.basis = 'gth-dzv'
    #cell.pseudo = 'gth-pade'
    cell.verbose = 4
    cell.precision = 1e-10
    cell.build(unit='Angstrom')
    
    # lattice class
    kmesh = [1, 1, 1]  
    Lat = lattice.Lattice(cell, kmesh)
    nscsites = Lat.nscsites
    kpts_abs = Lat.kpts_abs
    #kpts_abs = cell.make_kpts(kmesh)
   
    kmf = pscf.KRHF(cell, kpts_abs)
    kmf = kmf.density_fit()
    kmf.conv_tol = 1e-10
    kmf.kernel()
    
    num_wann = 2
    keywords = \
    '''
    num_iter = 200
    begin projections
    H:l=0
    end projections
    exclude_bands : 3-%s
    '''%(cell.nao_nr())

    # wannier run
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords=keywords)
    #w90.use_bloch_phases = True
    #w90.use_scdm = True
    w90.kernel()
    w90.export_unk()
    w90.export_AME()

    # Plotting using Wannier90
    keywords_plot = keywords + \
            'wannier_plot = True\n wannier_plot_supercell = 1\n'
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords=keywords_plot)
    #w90.use_bloch_phases = True
    w90.kernel()

def test_pywannier90_hchain_2():
    """
    H chain, 2 high virt -> 2 wannier
    """
    cell = pgto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''

    #cell.atom = ''' H 0.0      0.0      0.75
    #                H 0.0      0.0      2.25 '''
    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    cell.basis = '321g'
    #cell.basis = 'gth-dzv'
    #cell.pseudo = 'gth-pade'
    cell.verbose = 4
    cell.precision = 1e-15
    cell.build(unit='Angstrom')
    
    # Latice class
    kmesh = [1, 1, 5]  
    Lat = lattice.Lattice(cell, kmesh)
    nscsites = Lat.nscsites
    kpts_abs = Lat.kpts_abs
    #kpts_abs = cell.make_kpts(kmesh)
   
    chkfname = 'hchain_2.chk'
    kmf = pscf.KRHF(cell, kpts_abs)
    kmf = kmf.density_fit()
    kmf.conv_tol = 1e-15
    kmf.kernel()

    num_wann = 2
    keywords = \
    '''
    num_iter = 400
    begin projections
    H:l=0
    end projections
    exclude_bands : 1-2
    '''

    # wannier run
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords)
    #w90.use_bloch_phases = True
    #w90.guiding_centres = False
    w90.kernel()
    
    C_ao_mo = np.asarray(w90.mo_coeff)[:, :, w90.band_included_list]
    C_mo_lo = make_basis.tile_u_matrix(np.array(w90.U_matrix.transpose(2, 0, 1), \
        order='C'), u_virt=None, u_core=None)
    C_ao_lo = make_basis.multiply_basis(C_ao_mo, C_mo_lo)

    # plot
    from libdmet.utils.plot import plot_orb_k_all
    plot_orb_k_all(cell, 'wann_virt', C_ao_lo, kpts_abs, \
            nx=100, ny=100, nz=100, resolution=None, margin=5.0)
    #w90.export_unk()
    #w90.export_AME()

    ## Plotting using Wannier90
    #keywords_plot = keywords + 'wannier_plot = True\n wannier_plot_supercell = 1\n'
    #w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords_plot)
    ##w90.use_bloch_phases = True
    #w90.kernel()

def test_lowdin_k():
    from pyscf import lo
    np.set_printoptions(3, linewidth=1000)

    cell = pgto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''

    cell.atom = ''' H 0.      0.      0.
                    H 0.      0.      1.5 '''
    #cell.basis = 'sto3g'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 4
    cell.precision = 1e-15
    cell.build(unit='Angstrom')
   
    # lattice class
    kmesh = [1, 1, 3]  
    Lat = lattice.Lattice(cell, kmesh)
    nscsites = Lat.nscsites
    kpts_abs = Lat.kpts_abs

    mf = pscf.KRHF(cell, kpts_abs)
    mf = mf.density_fit()
    mf.conv_tol = 1e-15
    mf.kernel()
    
    C_ao_lo = lowdin.lowdin_k(mf, method='meta_lowdin')
    dm_lo_k = make_basis.transform_rdm1_to_lo(mf.make_rdm1(), C_ao_lo, mf.get_ovlp())
    dm_lo_R = Lat.k2R(dm_lo_k)
    dm_lo_R_full = Lat.expand(dm_lo_R, dense=True)
    
    # supercell calculation
    scell = Lat.bigcell
    mf = pscf.KRHF(scell)
    mf = mf.density_fit()
    mf.conv_tol = 1e-15
    mf.kernel()

    C = lo.orth_ao(scell, method='meta_lowdin')
    mo = np.linalg.solve(C, mf.mo_coeff[0]) # MO is C_{lo, mo}
    dm = mf.make_rdm1(np.asarray([mo]), mf.mo_occ)[0]
    diff = np.max(np.abs(dm - dm_lo_R_full))
    print("max diff between dm_lo kpts and gamma rdm: %s"%diff)
    assert(diff < 1e-7)

def test_iao():
    np.set_printoptions(3, linewidth=1000)

    cell = pgto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''

    cell.atom = ''' H 0.      0.      0.
                    H 0.      0.      1.5 '''
    cell.basis = '321G'
    cell.verbose = 4
    cell.precision = 1e-15
    cell.build(unit='Angstrom')
   
    # lattice class
    kmesh = [1, 1, 3]  
    Lat = lattice.Lattice(cell, kmesh)
    nscsites = Lat.nscsites
    kpts_abs = Lat.kpts_abs
    nkpts = Lat.nkpts

    chkfname = 'hchain_iao_k.chk'
    if os.path.isfile(chkfname):
    #if False:
        kmf = pscf.KRHF(cell, kpts_abs)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-15
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = pscf.KRHF(cell, kpts_abs)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-15
        kmf.chkfile = chkfname
        kmf.kernel()
    
    S = kmf.get_ovlp()
    mo_coeff_occ = [kmf.mo_coeff[k][:, kmf.mo_occ[k]>0] for k in range(nkpts)]
    
    # IAO
    C_ao_lo_val = iao.iao(cell, mo_coeff_occ, kpts=kpts_abs)
    C_ao_lo_val = iao.vec_lowdin_k(C_ao_lo_val, S)
    nval = C_ao_lo_val.shape[1]

    # IAO virtual
    C_ao_lo_virt = iao.get_iao_virt(cell, C_ao_lo_val, S)
    C_ao_lo_virt = iao.vec_lowdin_k(C_ao_lo_virt, S)

    # plot
    #from libdmet.utils.plot import plot_orb_k_all
    #plot_orb_k_all(cell, 'iao_val', C_ao_lo_val, kpts_abs, nx=100, ny=100, nz=100, resolution=None, margin=5.0)
    #plot_orb_k_all(cell, 'iao_virt', C_ao_lo_virt, kpts_abs, nx=100, ny=100, nz=100, resolution=None, margin=5.0)

    # check orthonormal
    assert(iao.check_orthonormal(C_ao_lo_val, S)) 
    assert(iao.check_orthonormal(C_ao_lo_virt, S)) 
    
    # check orthogonal
    assert(iao.check_orthogonal_two_sets(C_ao_lo_virt, C_ao_lo_val, S)) 
    
    # check density matrix
    C_ao_lo = make_basis.tile_C_ao_iao(C_ao_lo_val, C_ao_lo_virt)
    assert(iao.check_orthonormal(C_ao_lo, S)) 
    
    dm_ao_ao = kmf.make_rdm1()
    dm_lo_lo = make_basis.transform_rdm1_to_lo(dm_ao_ao, C_ao_lo, S)
    dm_R = Lat.k2R(dm_lo_lo)
    
    # check the trace
    assert(np.abs(np.trace(dm_R[0]) - cell.nelectron) < 1e-13)
    assert(np.abs(np.trace(dm_R[0][:nval, :nval]) - cell.nelectron) < 1e-13)

def test_iao_supercell():
    np.set_printoptions(3, linewidth=1000)

    cell = pgto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''

    cell.atom = ''' H 0.      0.      0.
                    H 0.      0.      1.5 '''
    cell.basis = '321G'
    cell.verbose = 4
    cell.precision = 1e-15
    cell.build(unit='Angstrom')
   
    # lattice class
    kmesh = [1, 1, 3]  
    Lat = lattice.Lattice(cell, kmesh)
    nscsites = Lat.nscsites
    kpts_abs = Lat.kpts_abs

    # supercell calculation
    scell = Lat.bigcell
    chkfname = 'hchain_iao_supercell.chk'
    if os.path.isfile(chkfname):
        kmf = pscf.KRHF(scell)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-15
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = pscf.KRHF(scell)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-15
        kmf.chkfile = chkfname
        kmf.kernel()
    mo_occ = kmf.mo_coeff[0][:,kmf.mo_occ[0]>0]
    a = iao.iao(scell, mo_occ)

    # Orthogonalize IAO
    from pyscf import lo
    a = lo.vec_lowdin(a, kmf.get_ovlp()[0])
    
    S1 = kmf.get_ovlp()
    C = a
    CCdS = mdot(C, C.conj().T, S1[0])

    cell_b2 = cell.copy()
    cell_b2.basis = 'MINAO'
    cell_b2.build(unit='Angstrom')
    Lat_b2 = lattice.Lattice(cell_b2, kmesh)

    scell_b2 = Lat_b2.bigcell

    b1_labels = scell.ao_labels()
    b2_labels = scell_b2.ao_labels()
    virt_idx = [i for i, label in enumerate(b1_labels) if (not label in b2_labels)]

    Q = np.eye(CCdS.shape[0]) - CCdS

    Q_virt = Q[:, virt_idx]
    Q_orth = lo.vec_lowdin(Q_virt, kmf.get_ovlp()[0])
    
    C = np.hstack((a, Q_orth))
    
    C_inv = C.conj().T.dot(S1[0])
    dm_LO = mdot(C_inv, kmf.make_rdm1()[0], C_inv.conj().T)
    print (dm_LO)

def test_plot_orb_k():
    np.set_printoptions(3, linewidth=1000)

    cell = pgto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''

    cell.atom = ''' H 0.      0.      0.
                    H 0.      0.      1.5 '''
    cell.basis = '321G'
    cell.verbose = 4
    cell.precision = 1e-15
    cell.build(unit='Angstrom')
   
    # lattice class
    kmesh = [1, 1, 3]  
    Lat = lattice.Lattice(cell, kmesh)
    nscsites = Lat.nscsites
    kpts_abs = Lat.kpts_abs
    nkpts = Lat.nkpts

    chkfname = 'hchain_iao_k.chk'
    if os.path.isfile(chkfname):
    #if False:
        kmf = pscf.KRHF(cell, kpts_abs)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-15
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = pscf.KRHF(cell, kpts_abs)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-15
        kmf.chkfile = chkfname
        kmf.kernel()
    
    num_wann = 2
    keywords = \
    '''
    guiding_centres=true
    num_iter = 400
    dis_num_iter = 200
    begin projections
    H:l=0
    end projections
    exclude_bands : 1-2
    '''

    # wannier run
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords)
    w90.kernel()
    w90.export_unk()
    w90.export_AME()

    C_ao_mo = np.asarray(w90.mo_coeff_kpts)[:, :, w90.band_included_list] 
    C_lo_mo = make_basis.tile_u_matrix(w90.U_matrix)
    C_ao_lo = make_basis.get_C_ao_lo(C_ao_mo, C_lo_mo)
    
    # plot
    from libdmet.utils.plot import plot_orb_k_all
    plot_orb_k_all(cell, 'wann_virt', C_ao_lo, kpts_abs, nx=100, ny=100, nz=100, resolution=None, margin=5.0)
    #plot_orb_k_all(cell, 'wann_virt', C_ao_lo_virt, kpts_abs, nx=100, ny=100, nz=100, resolution=None, margin=5.0)

    # Plotting using Wannier90
    #keywords_plot = keywords + 'wannier_plot = True\n wannier_plot_supercell = 1\n'
    #w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords_plot)
    #w90.kernel()
    

if __name__ == '__main__':
    #test_pywannier90_CH4_5()
    #test_pywannier90_hchain_1()
    test_pywannier90_hchain_2()
    #test_lowdin_k()
    #test_iao()
    #test_iao_supercell()
    #test_plot_orb_k()
