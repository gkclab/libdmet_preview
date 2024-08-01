#! /usr/bin/env python

"""
Test vcor.
"""

def test_vcor_local():
    import numpy as np
    import scipy.linalg as la
    from libdmet.dmet.Hubbard import VcorLocal
    from libdmet.utils import logger as log
    
    log.result("Test resctricted potential")
    vcor = VcorLocal(True, False, 4)
    vcor.update(np.asarray([2,1,0,-1,3,4,2,1,2,3]))
    log.result("Vcor:\n%s", vcor.get())
    log.result("Gradient:\n%s", vcor.gradient())

    log.result("Test unresctricted potential")
    vcor = VcorLocal(False, False, 2)
    vcor.update(np.asarray([2,1,0,-1,3,4]))
    log.result("Vcor:\n%s", vcor.get())
    log.result("Gradient:\n%s", vcor.gradient())

    log.result("Test unresctricted Bogoliubov potential")
    vcor = VcorLocal(False, True, 2)
    vcor.update(np.asarray([1,2,3,4,5,6,7,8,9,10]))
    log.result("Vcor:\n%s", vcor.get())
    log.result("Gradient:\n%s", vcor.gradient())

def test_vcor_non_local():
    import numpy as np
    from pyscf.pbc import gto
    
    from libdmet.system import lattice
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet
    from libdmet.utils.misc import max_abs

    log.verbose = "DEBUG1"
    np.set_printoptions(4, linewidth=1000, suppress=True)
    
    cell = lattice.HChain()
    cell.basis = 'sto3g'
    cell.verbose = 4
    cell.precision = 1e-10
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 4]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nscsites = Lat.nao
    nkpts = Lat.nkpts
    
    # RHF vcor
    print ("Test RHF vcor")
    restricted = True
    bogoliubov = False
    vcor = dmet.VcorNonLocal(restricted, bogoliubov, Lat)
    z_mat = np.random.random((1, nkpts, nscsites, nscsites))
    vcor.assign(z_mat)
    
    vcor_mat = vcor.get(kspace=False, return_all=True)
    vcor_mat_k = vcor.get(kspace=True, return_all=True)
    print ("vcor_mat")
    print (vcor_mat)

    assert np.allclose(vcor_mat[0, 0], vcor_mat[0, 0].T)
    assert max_abs(vcor_mat_k[0, 0].imag) < 1e-12
    for k in range(nkpts):
        assert np.allclose(vcor_mat_k[0, k], vcor_mat_k[0, k].conj().T)

    vcor_grad = vcor.gradient()
    print ("vcor_grad")
    print (vcor_grad)

    # UHF vcor
    print ("Test UHF vcor")
    restricted = False
    bogoliubov = False
    vcor = dmet.VcorNonLocal(restricted, bogoliubov, Lat)
    z_mat = np.random.random((2, nkpts, nscsites, nscsites))
    vcor.assign(z_mat)
    
    vcor_mat = vcor.get(kspace=False, return_all=True)
    vcor_mat_k = vcor.get(kspace=True, return_all=True)
    print ("vcor_mat")
    print (vcor_mat)

    assert np.allclose(vcor_mat[0, 0], vcor_mat[0, 0].T)
    assert np.allclose(vcor_mat[1, 0], vcor_mat[1, 0].T)
    assert max_abs(vcor_mat_k[0, 0].imag) < 1e-12
    assert max_abs(vcor_mat_k[1, 0].imag) < 1e-12
    for k in range(nkpts):
        assert np.allclose(vcor_mat_k[0, k], vcor_mat_k[0, k].conj().T)
        assert np.allclose(vcor_mat_k[1, k], vcor_mat_k[1, k].conj().T)

    vcor_grad = vcor.gradient()
    print ("vcor_grad")
    print (vcor_grad)
    
    # RBCS vcor
    print ("Test RBCS vcor")
    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nscsites = Lat.nao
    nkpts = Lat.nkpts
    restricted = True
    bogoliubov = True
    
    vcor = dmet.VcorNonLocal(restricted, bogoliubov, Lat)
    z_mat = np.random.random((3, nkpts, nscsites, nscsites))
    vcor.assign(z_mat)
    
    vcor_mat = vcor.get(kspace=False, return_all=True)
    vcor_mat_k = vcor.get(kspace=True, return_all=True)
    print ("vcor_mat")
    print (vcor_mat)

    assert np.allclose(vcor_mat[0, 0], vcor_mat[0, 0].T)
    assert np.allclose(vcor_mat[2, 0], vcor_mat[2, 0].T)
    assert max_abs(vcor_mat_k[0, 0].imag) < 1e-12
    for k in range(nkpts):
        assert np.allclose(vcor_mat_k[0, k], vcor_mat_k[0, k].conj().T)
        assert np.allclose(vcor_mat_k[2, k], vcor_mat_k[2, k].conj().T)

    vcor_grad = vcor.gradient()
    print ("vcor_grad")
    print (vcor_grad)
    
    # UBCS res
    print ("Test UBCS res")
    restricted = False
    bogoliubov = True
    bogo_res = True
    
    vcor = dmet.VcorNonLocal(restricted, bogoliubov, Lat, bogo_res=bogo_res)
    z_mat = np.random.random((3, nkpts, nscsites, nscsites))
    vcor.assign(z_mat)
    
    vcor_mat = vcor.get(kspace=False, return_all=True)
    vcor_mat_k = vcor.get(kspace=True, return_all=True)
    print ("vcor_mat")
    print (vcor_mat)

    assert np.allclose(vcor_mat[0, 0], vcor_mat[0, 0].T)
    assert np.allclose(vcor_mat[1, 0], vcor_mat[1, 0].T)
    assert np.allclose(vcor_mat[2, 0], vcor_mat[2, 0].T)
    assert max_abs(vcor_mat_k[0, 0].imag) < 1e-12
    for k in range(nkpts):
        assert np.allclose(vcor_mat_k[0, k], vcor_mat_k[0, k].conj().T)
        assert np.allclose(vcor_mat_k[1, k], vcor_mat_k[1, k].conj().T)
        assert np.allclose(vcor_mat_k[2, k], vcor_mat_k[2, k].conj().T)

    vcor_grad = vcor.gradient()
    print ("vcor_grad")
    print (vcor_grad)
    
    # UBCS unres
    print ("Test UBCS unres")
    restricted = False
    bogoliubov = True
    bogo_res = False
    
    vcor = dmet.VcorNonLocal(restricted, bogoliubov, Lat, bogo_res=bogo_res)
    z_mat = np.random.random((3, nkpts, nscsites, nscsites))
    vcor.assign(z_mat)
    
    vcor_mat = vcor.get(kspace=False, return_all=True)
    vcor_mat_k = vcor.get(kspace=True, return_all=True)
    print ("vcor_mat")
    print (vcor_mat)

    assert np.allclose(vcor_mat[0, 0], vcor_mat[0, 0].T)
    assert np.allclose(vcor_mat[1, 0], vcor_mat[1, 0].T)
    assert max_abs(vcor_mat_k[0, 0].imag) < 1e-12
    for k in range(nkpts):
        assert np.allclose(vcor_mat_k[0, k], vcor_mat_k[0, k].conj().T)
        assert np.allclose(vcor_mat_k[1, k], vcor_mat_k[1, k].conj().T)

    vcor_grad = vcor.gradient()
    print ("vcor_grad")
    print (vcor_grad)

def test_vcor_init_guess():
    import numpy as np
    from libdmet.dmet.Hubbard import AFInitGuess
    from libdmet.utils import max_abs
    
    np.set_printoptions(3, linewidth=1000, suppress=True)
    ImpSize = [2, 2]
    U = 4
    Filling = 0.5
    bogo = False
    
    vcor = AFInitGuess(ImpSize, U, Filling, polar=None, bogoliubov=bogo, rand=0.0, 
                       subA=None, subB=None, bogo_res=False, d_wave=False)
    print (vcor.get())
    
    # minus polar
    vcor = AFInitGuess(ImpSize, U, Filling, polar=-U*Filling**2, bogoliubov=bogo, rand=0.0, 
                       subA=None, subB=None, bogo_res=False, d_wave=False)
    print (vcor.get())
    
    bogo = True
    rand = 0.1
    vcor = AFInitGuess(ImpSize, U, Filling, polar=-U*Filling**2, bogoliubov=bogo, rand=0.1, 
                       subA=None, subB=None, bogo_res=False, d_wave=False, trace_zero=True)
    print (vcor.get())
    
    bogo = True
    bogo_res = True
    rand = 0.1
    vcor = AFInitGuess(ImpSize, U, Filling, polar=-U*Filling**2, bogoliubov=bogo, rand=0.1, 
                       subA=None, subB=None, bogo_res=bogo_res, d_wave=False, trace_zero=True)
    print (vcor.get())
    
    # d wave guess 2x2
    bogo = True
    bogo_res = True
    rand = 0.1
    d_wave = True
    vcor = AFInitGuess(ImpSize, U, Filling, polar=None, bogoliubov=bogo, rand=0.1, 
                       subA=None, subB=None, bogo_res=bogo_res, d_wave=d_wave, trace_zero=True)
    print (vcor.get())
    
    # d wave guess 4x4
    bogo = True
    bogo_res = True
    rand = 0.1
    d_wave = True
    vcor = AFInitGuess([4, 4], U, Filling, polar=None, bogoliubov=bogo, rand=0.1, 
                       subA=None, subB=None, bogo_res=bogo_res, d_wave=d_wave, trace_zero=True)
    print (vcor.get())

    # d wave guess 4x2, 2x2 sublattice
    bogo = True
    bogo_res = True
    rand = 0.1
    d_wave = True
    vcor = AFInitGuess([4, 2], U, Filling, polar=None, bogoliubov=bogo, rand=0.1, 
                       subA=[0, 3], subB=[1, 2], bogo_res=bogo_res, d_wave=d_wave, trace_zero=True)
    print (vcor.get())

def test_vcor_bogo():
    import numpy as np
    from libdmet.dmet.Hubbard import VcorSymmBogo
    from libdmet.utils import max_abs
    
    np.set_printoptions(3, linewidth=1000, suppress=True)
    ImpSize = [2, 2]
    U = 4
    Filling = 0.5
    bogo = False
    
    nscsites = np.prod(ImpSize)
    Ca = [np.eye(nscsites)]
    Cb = [np.eye(nscsites)]

    vcor = VcorSymmBogo(restricted=False, bogoliubov=True, nscsites=nscsites, 
                        Ca=Ca, Cb=Cb, idx_range=None, bogo_res=True)
    
    print ("length")
    print (vcor.length())
    assert vcor.length() == nscsites * (nscsites+1) // 2
    print ("vcor")
    print (vcor.get())
    
    print ("gradient")
    print (vcor.gradient())

def t_vcor_kpoints():
    import numpy as np
    from pyscf.pbc import gto
    
    from libdmet.system import lattice
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet
    from libdmet.utils.misc import max_abs

    log.verbose = "DEBUG1"
    np.set_printoptions(4, linewidth=1000, suppress=True)
    
    cell = lattice.HChain()
    cell.basis = 'sto3g'
    cell.verbose = 4
    cell.precision = 1e-10
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 4]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nscsites = Lat.nao
    nkpts = Lat.nkpts
    
    # RHF vcor
    print ("Test RHF vcor")
    restricted = True
    bogoliubov = False
    vcor = dmet.VcorKpoints(restricted, bogoliubov, Lat)
    z_mat = np.random.random((nkpts, 2, nscsites, nscsites)) + np.random.random((nkpts, 2, nscsites, nscsites)) * 1j
    vcor.assign(z_mat)
    
    for k in range(nkpts):
        print (k)
        print (vcor.get(k))

if __name__ == "__main__":
    t_vcor_kpoints()
    test_vcor_init_guess()
    test_vcor_bogo
    test_vcor_local()
    test_vcor_non_local()
