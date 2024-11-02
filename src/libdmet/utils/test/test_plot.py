#! /usr/bin/env python

"""
Test plot.
"""

def test_plot_smooth():
    """
    Test plot_smooth.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from libdmet.utils import plot_smooth

    x = [0.0, 0.5, 2.0, 4.0, 5.0, 7.0, 8.0]
    y = np.sin(x) + (np.random.random(len(x)) - 0.5) * 0.1

    x_plot, y_plot1 = plot_smooth(x, y, smooth=0.1, do_plot=False)
    x_plot, y_plot2 = plot_smooth(x, y, smooth=1e-2, n0left=20, remove_neg=True)
    plt.savefig("plot_smooth.png")
    x_plot, y_plot2 = plot_smooth(x, y, smooth=1e-2, n0right=30)

def test_dos():
    """
    Test pdos routine and plot.
    """
    import numpy as np
    import scipy.linalg as la
    from libdmet.utils.plot import get_dos, plot_dos
    from libdmet.utils.misc import max_abs

    # not use Xwindow as backend
    import matplotlib
    matplotlib.use('Agg')

    # fake MO and C_lo_mo
    mo_energy = np.asarray([[-2.0, -2.0, 0.0, 0.1, 0.5, 7.0], \
                            [-2.0, -0.1, 0.0, 0.1, 0.5, 6.5],
                            [-2.0, -0.1, 0.0, 0.1, 0.5, 6.5]])
    nkpts, nmo = mo_energy.shape
    C_lo_mo = np.random.random((nkpts, nmo, nmo))
    for k in range(nkpts):
        C_lo_mo[k] = la.qr(C_lo_mo[k])[0]
    idx_dic = {"%s"%(i): [i] for i in range(nmo)}
    color_dic = {"%s"%(i): 'C%s'%(i) for i in range(nmo)}
    color_dic_select = {"%s"%(i): 'C%s'%(i) for i in range(2)}
    elist, pdos = get_dos(mo_energy, ndos=201, mo_coeff=C_lo_mo, sigma=0.1)
    elist_spin, pdos_spin = get_dos(mo_energy[None], ndos=201, \
            mo_coeff=C_lo_mo[None], sigma=0.1)
    assert max_abs(elist - elist_spin) < 1e-10
    assert pdos_spin.ndim == pdos.ndim + 1
    assert max_abs(elist - elist_spin) < 1e-10
    plot_dos(elist, pdos.sum(axis=0), idx_dic=None, text='test_total', fig_name='test.pdf')

    plot_dos(elist, pdos, idx_dic=idx_dic, text='test')

    plot_dos(elist, np.asarray((pdos, pdos * 0.7)), \
            idx_dic=idx_dic, color_dic=color_dic, fig_name='pdos_uhf.pdf')

    plot_dos(elist, np.asarray((pdos, pdos * 0.7)), \
            idx_dic=idx_dic, color_dic=None, \
            fig_name='pdos_uhf_no_color_dic.pdf')

    plot_dos(elist, np.asarray((pdos, pdos * 0.7)), \
            idx_dic=idx_dic, color_dic=color_dic_select, \
            fig_name='pdos_uhf_select_color_dic.pdf')

    elist_spin, pdos_spin = get_dos(mo_energy[None], ndos=201, \
            mo_coeff=C_lo_mo[None], sigma=0.1, e_fermi=0.0)

    plot_dos(elist_spin, np.asarray((pdos_spin[0], pdos_spin[0])), \
            idx_dic=idx_dic, color_dic=color_dic_select, \
            fig_name='pdos_e_fermi.pdf')

def test_cube():
    import os
    import numpy as np
    import scipy.linalg as la

    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft, cc

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.basis_transform import eri_transform

    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet
    from libdmet.utils.misc import max_abs, mdot
    from libdmet.utils.plot import plot_orb_k_all, plot_density_k

    log.verbose = "DEBUG1"
    np.set_printoptions(4, linewidth=1000, suppress=True)

    ### ************************************************************
    ### System settings
    ### ************************************************************

    cell = gto.Cell()
    cell.a = ''' 10.0     0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts

    exxdiv = None

    ### ************************************************************
    ### DMET settings
    ### ************************************************************

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = False
    int_bath = True
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = np.inf

    # DMET SCF control
    MaxIter = 1
    u_tol = 1.0e-6
    E_tol = 1.0e-6
    iter_tol = 4

    # DIIS
    adiis = lib.diis.DIIS()
    adiis.space = 4
    diis_start = 4
    dc = dmet.FDiisContext(adiis.space)
    trace_start = 3

    # solver and mu fit
    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12)
    solver = FCI
    nelec_tol = 5.0e-6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    imp_fit = False
    emb_fit_iter = 100 # embedding fitting
    full_fit_iter = 0

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((2, nscsites, nscsites))
    vcor.assign(z_mat)

    ### ************************************************************
    ### SCF Mean-field calculation
    ### ************************************************************

    log.section("\nSolving SCF mean-field problem\n")

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()
        assert(kmf.converged)


    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************

    log.section("\nPre-process, orbital localization and subspace partition\n")
    # IAO guess
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, \
            kmf, minao='minao', full_return=True, max_ovlp=True)

    # Wannier orbitals
    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo)

    ### ************************************************************
    ### DMET procedure
    ### ************************************************************

    # DMET main loop
    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None
    if load_frecord:
        dmet.SolveImpHam_with_fitting.load("./frecord")

    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)

        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)
        rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
        Lat.update_Ham(rho*2.0, rdm1_lo_k=res["rho_k"]*2.0)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=True, int_bath=int_bath)

        # plot LOs or MOs
        latt_vec = cell.lattice_vectors()
        latt_vec[0,0] = 3.0
        latt_vec[1,1] = 3.0
        plot_orb_k_all(cell, 'iao_val', C_ao_iao_val, kpts, nx=51, ny=51,
                       nz=51, resolution=None, margin=5.0, latt_vec=latt_vec,
                       boxorig=[5.0, 5.0, -3.0])
        plot_orb_k_all(cell, 'iao_virt', C_ao_iao_virt[None], kpts, nx=51, ny=51,
                       nz=51, resolution=None, margin=5.0, latt_vec=latt_vec,
                       boxorig=[5.0, 5.0, -3.0])
        #plot_orb_k_all(cell, 'mo_val', C_ao_mo, kpts, nx=50, ny=50, nz=50, \
        #        resolution=None, margin=3.0,  latt_vec=latt_vec, \
        #        boxorig=[5.0, 5.0, -3.0])

        # plot rdm1_lo_R0
        scell = Lat.bigcell
        scell.verbose = 4
        basis_k = Lat.R2k_basis(basis)
        rdm1_lo_R0 = rho[0, 0] * 2.0
        C_ao_lo0 = Lat.k2R_basis(C_ao_lo)
        C_ao_lo0_full = C_ao_lo0.reshape((nkpts*nscsites, -1))
        rdm1_sc = mdot(C_ao_lo0_full, rdm1_lo_R0, C_ao_lo0_full.conj().T)
        plot_density_k(scell, 'rdm1_sc.cube', rdm1_sc[None], kpts_abs=[[0.0, 0.0, 0.0]],
                       nx=51, ny=51, nz=51, resolution=None, margin=3.0)

        # plot bath orbital density
        C_ao_emb_k = make_basis.multiply_basis(C_ao_lo[None], basis_k)
        C_ao_emb_R = Lat.k2R_basis(C_ao_emb_k).reshape((1, nkpts*nscsites, -1))
        C_bath = C_ao_emb_R[:, :, -nval:]
        dm_bath = (C_bath[0].dot(C_bath[0].conj().T))[None]

        plot_density_k(scell, 'rho.cube', dm_bath, kpts_abs=[[0.0, 0.0, 0.0]],
                       nx=51, ny=51, nz=51, resolution=None, margin=0.0)
        break

def test_plot_bands():
    """
    Plot band structure with PDOS of the 3-band Hubbard model.
    """
    import os
    import numpy as np
    import scipy.linalg as la

    import libdmet.utils.logger as log
    import libdmet.dmet.abinitioBCS as dmet
    from libdmet.routine import mfd
    from libdmet.utils import get_dos
    from libdmet.utils import get_kdis
    from libdmet.utils import plot_bands
    from libdmet.utils import max_abs

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    matplotlib.use('Agg')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    log.verbose = "DEBUG0"
    np.set_printoptions(3, linewidth=1000, suppress=True)

    doping = 0.0
    Filling = (5.-doping) / 6
    Filling_Cu = (1.0 - doping) / 2.0

    LatSize = (10, 10)
    ImpSize = (1, 1)

    Lat = dmet.Square3BandSymm(*(LatSize + ImpSize))
    nao = Lat.nscsites
    nkpts = Lat.nkpts
    Ham = dmet.Hubbard3band_ref(Lat, "Hanke", min_model=False, ignore_intercell=False)
    Lat.setHam(Ham)
    vcor = dmet.vcor_zeros(False, True, Lat.supercell.nsites)

    # initial guess for HF
    nCu_tot = np.prod(LatSize) * 4 # 4 is number of Cu site per 2x2 cell
    nO_tot = np.prod(LatSize) * 8
    nao_tot = nao * nkpts
    nelec_half = np.prod(LatSize) * 20 # 20 electron per cell
    nelec_half_Cu = np.prod(LatSize) * 4
    nelec_half_O = np.prod(LatSize) * 16

    x_dop = 0.0
    nelec_dop = int(np.round(x_dop * nCu_tot))
    if nelec_dop % 2 == 1:
        diff_l = abs(nelec_dop - 1 - x_dop * nCu_tot)
        diff_r = abs(nelec_dop + 1 - x_dop * nCu_tot)
        if diff_l < diff_r:
            nelec_dop = nelec_dop - 1
        else:
            nelec_dop = nelec_dop + 1
    x_dop = nelec_dop / float(nCu_tot)

    Filling = (nelec_half - nelec_dop) / (nao_tot * 2.0)
    if nelec_dop >= 0: # hole doping
        Filling_Cu = (nelec_half_Cu) / (nCu_tot * 2.0)
        Filling_O = (nelec_half_O - nelec_dop) / (nO_tot * 2.0)
    else: # electron doping
        Filling_Cu = (nelec_half_Cu - nelec_dop) / (nCu_tot * 2.0)
        Filling_O = (nelec_half_O) / (nO_tot * 2.0)

    log.info("doping x = %s", x_dop)
    log.info("nelec_half = %s", nelec_half)
    log.info("nelec_dop = %s", nelec_dop)

    polar = 0.5
    fCu_a = Filling_Cu * (1.0 - polar)
    fCu_b = Filling_Cu * (1.0 + polar)
    fO = Filling_O

    dm0_a = np.diag([fCu_a, fO, fO, fCu_b, fO, fO, fCu_a, fO, fO, fCu_b, fO, fO])
    dm0_b = np.diag([fCu_b, fO, fO, fCu_a, fO, fO, fCu_b, fO, fO, fCu_a, fO, fO])
    dm0 = np.zeros((2, nkpts, nao, nao))
    dm0[0] = dm0_a
    dm0[1] = dm0_b

    # HF calcs
    if os.path.exists("./dm0.npy"):
        dm0 = np.load("dm0.npy")
    rho, mu, E, res = mfd.HF(Lat, vcor, Filling, False, mu0=0.0, \
            beta=np.inf, ires = True, scf=True, dm0=dm0)
    np.save("dm0.npy", res["rho_k"])
    rdm1_a, rdm1_b = rho[:, 0]
    m_AFM = 0.25 * (abs(rdm1_a[0, 0] - rdm1_b[0, 0]) + abs(rdm1_a[3, 3] - rdm1_b[3, 3]) \
                   +abs(rdm1_a[6, 6] - rdm1_b[6, 6]) + abs(rdm1_a[9, 9] - rdm1_b[9, 9]))
    log.result("m_AFM = %s", m_AFM)


    # *****************************************
    # Plot bands and pdos
    # *****************************************

    mo_occ = (np.abs(res["mo_occ"] - 1.0) < 1e-6)
    gap = res["gap"]
    ew = res["e"]
    ev = res["coef"]
    # shift by VBM
    vbm = ew[mo_occ].max()
    ew -= vbm

    # K-path:
    # G (0, 0) -> X (0.5, 0.0) -> S(0.5, 0.5) -> G(0, 0)
    # GX, XS, SG
    kpts = Lat.kpts_scaled
    kpts[kpts == -0.5] = 0.5

    GX_idx = np.where((kpts[:, 0] >= 0.0)  & (kpts[:, 1] == 0))[0][:-1]
    XS_idx = np.where((kpts[:, 0] == 0.5)  & (kpts[:, 1] >= 0))[0][:-1]
    SG_idx = np.where((kpts[:, 0] >=  0.0) & (kpts[:, 1] == kpts[:, 0]))[0][::-1]

    log.result("G-X indices:\n%s", GX_idx)
    log.result("G-X kpts:\n%s", kpts[GX_idx])
    log.result("X-S indices:\n%s", XS_idx)
    log.result("X-S kpts:\n%s", kpts[XS_idx])
    log.result("S-G indices:\n%s", SG_idx)
    log.result("S-G kpts:\n%s", kpts[SG_idx])

    kpath_idx = np.hstack((GX_idx, XS_idx, SG_idx))
    kpts_bands = kpts[kpath_idx]

    kpts_sp = np.array([[0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.0],
                        [0.5, 0.5, 0.0],
                        [0.0, 0.0, 0.0]])

    kdis = np.diff(kpts_bands, axis=0)
    kdis = np.hstack((0.0, np.cumsum(la.norm(kdis, axis=1))))
    G_idx = np.where((kpts_bands[:, 0] == 0.0) & (kpts_bands[:, 1] == 0))[0][0]
    X_idx = np.where((kpts_bands[:, 0] == 0.5) & (kpts_bands[:, 1] == 0))[0][0]
    S_idx = np.where((kpts_bands[:, 0] == 0.5) & (kpts_bands[:, 1] == 0.5))[0][0]
    sp_points_idx = np.hstack((G_idx, X_idx, S_idx, -1))
    sp_points = kdis[sp_points_idx]

    latt_vec = Lat.cell.lattice_vectors()

    kdis_new, kdis_sp_new = get_kdis(kpts_bands, kpts_sp)
    assert max_abs(kdis - kdis_new) < 1e-10
    assert max_abs(kdis_sp_new - sp_points) < 1e-10
    kdis_new, kdis_sp_new = get_kdis(kpts_bands, kpts_sp, latt_vec=latt_vec)
    #assert max_abs(kdis - kdis_new) < 1e-10
    #assert max_abs(kdis_sp_new - sp_points) < 1e-10

    sigma = 0.01
    ew_bands = (ew[0, kpath_idx])
    ev_bands = ev[0, kpath_idx]

    ev_bands_O_sq = la.norm(ev_bands[:, [1, 2, 4, 5, 7, 8, 10, 11]], axis=1) ** 2
    ev_bands_Cu_sq = la.norm(ev_bands[:, [0, 3, 6, 9]], axis=1) ** 2
    ev_bands_Cu_percent = ev_bands_Cu_sq #/ (ev_bands_Cu_sq + ev_bands_O_sq)
    ev_bands_O_percent = ev_bands_O_sq #/ (ev_bands_Cu_sq + ev_bands_O_sq)

    mo_energy_min = ew.min()
    mo_energy_max = ew.max()
    margin = max(10 * sigma, 0.05 * (mo_energy_max - mo_energy_min)) # margin
    emin = -8.5
    emax = 4.5
    nbands = ew_bands.shape[-1]

    fig, ax = plt.subplots(figsize=(6, 5), sharey=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 2])
    ax = plt.gca()

    """
    Color mapped bands:
    """
    plt.subplot(gs[0])
    plt.tick_params(labelsize=20, bottom=False, top=False, left=True, right=True, width=1.5)
    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    bands = plot_bands(ax, kdis, ew_bands, weights=ev_bands_Cu_percent, \
            cmap='coolwarm', linewidth=4)

    for p in sp_points[1:3]:
        # special lines grids
        plt.plot([p, p], [emin, emax], '--', color='lightgrey', linewidth=2.0, zorder=-10)

    plt.xticks(sp_points, [r'$\Gamma$', 'X', 'M', r'$\Gamma$'])
    plt.axis(xmin=0, xmax=sp_points[-1], ymin=-8.5, ymax=4.5)
    plt.yticks(np.arange(-8, 6, 2))
    plt.xlabel(r"$\mathbf{k}$", fontsize=20)
    plt.ylabel(r"energy [eV]", fontsize=20)

    plt.text(0.03, 0.95, 'HF', horizontalalignment='left', \
            verticalalignment='center', transform=ax.transAxes, fontsize=20)

    """
    PDOS:
    """

    Es, pdos = get_dos(ew, ndos=3001, sigma=0.05, mo_coeff=ev)
    pdos = pdos.sum(axis=0)
    np.save("elist-hf.npy", Es)
    np.save("pdos-hf.npy", pdos)

    DOS_Cu = pdos[[0, 3, 6, 9], :]
    DOS_O = pdos[[1, 2, 4, 5, 7, 8, 10, 11], :]
    DOS_Cu = np.sum(DOS_Cu, axis=0)
    DOS_O = np.sum(DOS_O, axis=0)

    plt.subplot(gs[1])
    ax = plt.gca()
    ax.set_facecolor('white') # background color

    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tick_params(labelsize=20, bottom='off', top='off', left=True, right=True, width=1.5)
    plt.xlabel("PDOS", fontsize=20, labelpad=7)
    plt.xticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.xlim(0.0, 7.5)
    plt.ylim(emin, emax)
    plt.yticks(np.arange(-8, 6, 2))

    #try:
    #    cmap = matplotlib.cm.get_cmap('coolwarm')
    #except AttributeError:
    cmap = plt.get_cmap("coolwarm")
    rgba1 = cmap(0.95)
    rgba2 = cmap(0.05)

    plt.plot(DOS_Cu, Es, label='Cu $3d$', linewidth=2, color=rgba1, linestyle='dashed')
    plt.plot(DOS_O,  Es, label='O   $2p$', linewidth=2, color=rgba2)

    """
    Color bar:
    """
    cbar = fig.colorbar(bands, ax=ax, aspect=35, pad=0.1, ticks=[0.025, 0.975])
    cbar.outline.set_linewidth(1.5)
    cbar.ax.set_yticklabels(['O', 'Cu'])
    cbar.ax.tick_params(labelsize=20, bottom=True, top=True, left=False,
            right=False, width=1.5)

    plt.subplots_adjust(left=0.15, bottom=0.165, right=0.94, top=0.97, wspace=0.1, hspace=0.0)
    #plt.show()
    plt.tight_layout()
    plt.savefig("./bands-dos-HF.eps", dpi=300)

def test_plot_elf_mol():
    import numpy as np
    from pyscf import gto
    from libdmet.utils import cubegen, plot_elf
    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = gto.M(atom='O 0 0 0; H 0.9584 0 0; H -0.2391543829355199 0.9280817534693432 0',
                basis='ccpvdz', verbose=4)
    myhf = mol.RKS()
    myhf.xc = 'b3lyp'
    myhf.grids.level = 5
    myhf.kernel()

    nx = 51
    ny = 51
    nz = 51
    dm = myhf.make_rdm1()
    cc = cubegen.Cube(mol, nx, ny, nz)
    plot_elf(mol, cc, dm, max_memory=100)

    dm = np.asarray((dm, dm)) * 0.5
    plot_elf(mol, cc, dm, max_memory=100, spin_average=False)

def test_plot_elf_pbc():
    import numpy as np
    from pyscf.pbc import gto, scf
    from libdmet.utils import cubegen, plot_elf

    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    from matplotlib import ticker, cm

    np.set_printoptions(3, linewidth=1000, suppress=True)

    cell = gto.Cell(a=np.eye(3) * 15.0,
                    atom='O 0 0 0; H 0.9584 0 0; H -0.2391543829355199 0.9280817534693432 0',
                    basis='sto6g', verbose=4)
    myhf = scf.RHF(cell).density_fit()
    myhf.kernel()

    nx = 51
    ny = 51
    nz = 1
    dm = myhf.make_rdm1()

    cc = cubegen.Cube(cell, nx, ny, nz, latt_vec=np.eye(3) * 2.0)
    cc.boxorig = np.asarray((-4, -4, -4.0))
    cc.box = np.eye(3)
    cc.box[0, 0] = cc.box[1, 1] = cc.box[2, 2] = 9.0

    coords = cc.get_coords()
    coords[:, 2] = 0.0

    dm = np.asarray((dm, dm)) * 0.5
    plot_elf(cell, cc, dm, max_memory=100)

    with h5py.File("ELF.h5", 'r') as f:
        coords = np.asarray(f["coords"])
        elf = np.asarray(f["elf_0"])
        nx, ny, nz = np.asarray(f["n"])
        coords_x = coords[:, 0].reshape((nx, ny))
        coords_y = coords[:, 1].reshape((nx, ny))
        elf = elf.reshape((nx, ny))

        fig, ax = plt.subplots(figsize=(6, 6))
        CS = ax.contourf(coords_x, coords_y, elf, cmap='Blues')
        plt.scatter([0.0], [0.0], marker='o', linewidths=2, linestyle='--',
                s=3000, facecolor="None",
                edgecolors='black', alpha=0.2)
        plt.scatter([1.811114], [0.0], marker='o', linewidths=2, linestyle='--',
                s=1000, facecolor="None",
                edgecolors='black', alpha=0.2)
        plt.scatter([-0.451936], [1.753820], marker='o', linewidths=2,
                linestyle='--',
                s=1000, facecolor="None",
                edgecolors='black', alpha=0.2)
        plt.plot([0.0, -0.451936], [0.0, 1.753820], marker=None, color='black',
                linewidth=4, linestyle='-', alpha=0.2)
        plt.plot([0.0, 1.811114], [0.0, 0.0], marker=None, color='black',
                linewidth=4, linestyle='-', alpha=0.2)
        plt.savefig("elf.png", dpi=300)

        rho = np.asarray(f["rho_0"])
        rho = rho.reshape((nx, ny))

        fig, ax = plt.subplots(figsize=(6, 6))
        CS = ax.contourf(coords_x, coords_y, rho, cmap='Blues',
                         locator=ticker.LogLocator(base=1.001))
        plt.scatter([0.0], [0.0], marker='o', linewidths=2, linestyle='--',
                s=3000, facecolor="None",
                edgecolors='black', alpha=0.2)
        plt.scatter([1.811114], [0.0], marker='o', linewidths=2, linestyle='--',
                s=1000, facecolor="None",
                edgecolors='black', alpha=0.2)
        plt.scatter([-0.451936], [1.753820], marker='o', linewidths=2,
                linestyle='--',
                s=1000, facecolor="None",
                edgecolors='black', alpha=0.2)
        plt.plot([0.0, -0.451936], [0.0, 1.753820], marker=None, color='black',
                linewidth=4, linestyle='-', alpha=0.2)
        plt.plot([0.0, 1.811114], [0.0, 0.0], marker=None, color='black',
                linewidth=4, linestyle='-', alpha=0.2)
        plt.savefig("rho.png", dpi=300)

    # restricted
    dm = dm[0] * 2.0
    plot_elf(cell, cc, dm, max_memory=100)

    # k-point sampling
    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 0.0      0.0      0.75
                    H 0.0      0.0      2.25 '''
    cell.basis = '321g'
    cell.verbose = 5
    cell.build(unit='Angstrom')

    kpts = cell.make_kpts([1, 1, 3])
    myhf = scf.KRHF(cell, kpts).density_fit()
    myhf.kernel()

    nx = 51
    ny = 51
    nz = 51
    dm = myhf.make_rdm1()

    dm = np.asarray((dm, dm)) * 0.5
    cc = cubegen.Cube(cell, nx, ny, nz, latt_vec=np.eye(3) * 2.0)
    plot_elf(cell, cc, dm, max_memory=100, kpts=kpts, spin_average=False)

def test_spin_corr_func():
    import numpy as np
    from libdmet.utils.plot import eval_spin_corr_func_lo
    def eval_spin_corr_func_lo_ref(rdm1_lo, rdm2_lo, idx1, idx2):
        rdm1_a, rdm1_b = rdm1_lo
        rdm2_aa, rdm2_ab, rdm2_bb = rdm2_lo
        norb = rdm1_a.shape[-1]
        mesh = np.ix_(idx1, idx1, idx2, idx2)

        delta = np.eye(norb)
        rdm1_a_delta = np.einsum('ij, kl -> ikjl', rdm1_a, delta)
        rdm1_b_delta = np.einsum('ij, kl -> ikjl', rdm1_b, delta)
        rdm1_tmp = rdm1_a_delta + rdm1_b_delta

        Az_iijj = 0.25 * (rdm1_tmp + rdm2_aa + rdm2_bb - rdm2_ab - rdm2_ab.transpose((2, 3, 0, 1)))
        Axy_iijj = 0.5 * (rdm1_tmp - rdm2_ab.transpose(0, 3, 1, 2) - rdm2_ab.transpose(1, 2, 0, 3))
        cf = np.einsum('iijj->', Az_iijj[mesh] + Axy_iijj[mesh])
        return cf

    norb = 6
    rdm1 = np.random.random((2, norb, norb))
    rdm2 = np.random.random((3, norb, norb, norb, norb))
    idx1 = [0, 1]
    idx2 = [2, 3, 5]

    cf_ref = eval_spin_corr_func_lo_ref(rdm1.copy(), rdm2.copy(), idx1, idx2)
    cf = eval_spin_corr_func_lo(rdm1, rdm2, idx1, idx2)
    assert abs(cf - cf_ref) < 1e-12

def test_plot_density_matrix():
    import os
    import numpy as np
    import scipy.linalg as la

    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft, cc

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis

    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet

    from libdmet.routine import spinless
    from libdmet.utils import plot

    log.verbose = "DEBUG1"
    np.set_printoptions(4, linewidth=1000, suppress=True)

    ### ************************************************************
    ### System settings
    ### ************************************************************

    cell = lattice.HPlane(shift=[1.0, 1.0, 5.0])
    cell.basis = 'STO6G'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')

    kmesh = [2, 2, 1]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts

    exxdiv = None

    ### ************************************************************
    ### DMET settings
    ### ************************************************************

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = False
    int_bath = True
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = np.inf

    # DMET SCF control
    MaxIter = 1
    u_tol = 1.0e-6
    E_tol = 1.0e-6
    iter_tol = 4

    # DIIS
    adiis = lib.diis.DIIS()
    adiis.space = 4
    diis_start = 4
    dc = dmet.FDiisContext(adiis.space)
    trace_start = 3

    # solver and mu fit
    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12)
    solver = FCI
    nelec_tol = 5.0e-6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    imp_fit = False
    emb_fit_iter = 100 # embedding fitting
    full_fit_iter = 0

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((2, nscsites, nscsites))
    vcor.assign(z_mat)

    ### ************************************************************
    ### SCF Mean-field calculation
    ### ************************************************************

    log.section("\nSolving SCF mean-field problem\n")

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        dm0 = kmf.get_init_guess()
        idx_a = [0, 3]
        idx_b = [1, 2]
        dm0[0, :, idx_a, idx_a] *= 2.0
        dm0[0, :, idx_b, idx_b]  = 0.0
        dm0[1, :, idx_a, idx_a]  = 0.0
        dm0[1, :, idx_b, idx_b] *= 2.0
        kmf.kernel(dm0=dm0)
        assert(kmf.converged)

    Lat.analyze(kmf)
    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************

    log.section("\nPre-process, orbital localization and subspace partition\n")
    # IAO guess
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, \
            kmf, minao='minao', full_return=True, max_ovlp=True)

    # Wannier orbitals
    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo)

    ovlp = np.asarray(kmf.get_ovlp())
    rdm1 = np.asarray(kmf.make_rdm1())
    Grdm1 = spinless.transform_rdm1_k(rdm1, ovlp, compact=False)
    Govlp = spinless.combine_mo_coeff_k(ovlp)

    C_sao_slo = spinless.combine_mo_coeff_k(C_ao_iao)
    Grdm1_lo = make_basis.transform_rdm1_to_lo(Grdm1, C_sao_slo, Govlp)
    Grdm1_lo_R = Lat.k2R(Grdm1_lo)

    rdm1_emb = Grdm1_lo_R[0]
    rdm1_D = np.asarray([[  0.    ,  0.0074, -0.0074, -0.    ],
                         [  0.1248,  0.    ,  0.    , -0.1248],
                         [ -0.1248, -0.    , -0.    ,  0.1248],
                         [  0.    , -0.0074,  0.0074, -0.    ]])


    rdm1_emb[:4, 4:] = rdm1_D
    rdm1_emb[4:, :4] = rdm1_D.conj().T

    basis = Lat.k2R_basis(C_sao_slo)
    basis_a, basis_b = spinless.separate_basis(basis)

    neo = rdm1_emb.shape[-1]
    nso = nao * 2
    rdm1_full = basis_a.reshape(nkpts*nao, neo) @ rdm1_emb @ basis_b.reshape(nkpts*nao, neo).T

    R0 = np.asarray(Lat.bigcell._atom[3][1])
    plot.plot_density_matrix_k(Lat.bigcell, "test.cube", rdm1_full[None],
                               kpts_abs=np.zeros((1, 3)), R0=R0, nx=201, ny=201, nz=101,
                               resolution=None, margin=0.0, latt_vec=None, boxorig=None,
                               box=None, skip_calc=False, coord0_idx=None)

def test_get_fermi_surface():
    import numpy as np
    from libdmet.utils import logger as log
    from libdmet.utils.misc import max_abs
    from libdmet.system import lattice
    from libdmet.dmet import Hubbard as dmet
    log.verbose = "DEBUG2"

    # Lattice settings
    LatSize = [80, 80]
    ImpSize = [2, 2]
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize)

    ImpSize_uc = [1, 1]
    Lat_uc = dmet.SquareLattice(*(LatSize+ImpSize_uc))

    # Hamiltonian
    U = 0.0
    Filling = 0.6 / 2.0
    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        ntotal = int(np.round(ntotal))
        log.warning("rounded total number of electrons to integer %d", ntotal)
        Filling=float(ntotal) / np.prod(LatSize)

    Ham = dmet.Ham(Lat, U)
    restricted = False
    bogoliubov = False
    use_hcore_as_emb_ham = True
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)


    vcor = dmet.PMInitGuess(ImpSize, U, Filling)
    vcor_mat = vcor.get()
    vcor_mat *= 0.0
    vcor.assign(vcor_mat)

    rho, mu, ires = dmet.HartreeFock(Lat, vcor, Filling, mu0=None, beta=1000.0, \
            ires=True, scf=True, dm0=None, conv_tol=1e-9)

    mo_coeff = ires["coef"]
    mo_energy = ires['e']

    from libdmet.utils import plot
    kpts_scaled, fs = plot.get_fermi_surface(mo_coeff[0], mo_energy[0], Lat, mu=mu, sigma=0.1,
                                             latt_uc=Lat_uc, fname="fs.txt")

    plot.plot_fs(fname="fs.txt")

if __name__ == "__main__":
    test_get_fermi_surface()
    test_plot_density_matrix()
    test_spin_corr_func()
    test_plot_elf_pbc()
    test_plot_elf_mol()
    test_plot_bands()
    test_cube()
    test_plot_smooth()
    test_dos()
