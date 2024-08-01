#! /usr/bin/env python

def test_lattice_plot():
    import numpy as np
    from libdmet.utils import lattice_plot
    from libdmet.system import lattice
    import matplotlib
    matplotlib.use('Agg')

    Lat = lattice.Square3BandSymm(1, 1, 1, 1)

    latt_plt = lattice_plot.LatticePlot(Lat)
    latt_plt.plot_lattice(noframe=True)
    latt_plt.plot_atoms(rad_list=[1.0, 0.2, 0.2] * 4, \
            color_dic={'Cu': 'gold', 'O': 'C3'})
    latt_plt.plot_spins(m_list=[0.3, 0.0, 0.0, -0.25, -0.0, -0.0001, \
            0.3, 0.0, 0.0, -0.35, 0.0, 0.0])
    latt_plt.plot_text([0.0, 0.0], "test")
    latt_plt.plot_d_orb([4.0, 4.0], direct='down')
    latt_plt.plot_d_orb([0.0, 4.0], direct='up')
    latt_plt.plot_p_orb([3.0, 4.0], direct='right', phase=["+", "-"])
    latt_plt.plot_p_orb([2.0, 4.0], direct='down', phase=["-", "+"])
    latt_plt.plot_bond([1.0, 1.0], [2.0, 1.0], val=-0.01)
    latt_plt.plot_bond([1.0, 1.0], [1.0, 3.0], val=+0.02)
    latt_plt.savefig("latt.png")

def plot_3band_order(res, pairing="Cu-Cu", transparent=True, **kwargs):
    """
    Plot order parameter of the 3band model in a 2x2 cluster.

    Args:
        res: result dict from the get_3band_order
        pairing: "Cu-Cu", "O-O", "Cu-O"

    Returns:
        latt_plt: LatticePlot object.
    """
    #
    #        4O        15O2
    #         |          |
    #  14O2 - 3Cu - 5O - 6Cu - 7O
    #         |          |
    #        2O         8O
    #         |          |
    # - 1O - 0Cu - 11O - 9Cu -12O2
    #  (0,1)  |          |
    #        13O2       10O (3, 0)
    #                    |

    from libdmet.system import lattice
    from libdmet.utils import lattice_plot
    from libdmet.utils.lattice_plot import COLORS
    Lat = lattice.Square3BandSymm(1, 1, 1, 1)

    O2_names = ["O2", "O2", "O2", "O2"]
    O2_coords = [[4.0, 1.0], [1.0, 0.0], [0.0, 3.0], [3.0, 4.0]]
    O2_charges = 2.0 - res["charge"][[1, 4, 7, 10]]
    O2_spins = res["m_AFM_O_list"][::2]

    latt_plt = lattice_plot.LatticePlot(Lat)
    latt_plt.plot_lattice(**kwargs)

    # plot hole density
    latt_plt.plot_atoms(rad_list=(2.0 - res["charge"]),
                        color_dic={'Cu': COLORS['gold'], 'O': 'C3'})
    latt_plt.plot_atoms(rad_list=O2_charges, names=O2_names, coords=O2_coords,
                        color_dic={'O2': COLORS["red2"]}, edgecolor=COLORS["red-gray"])

    # plot spin
    #latt_plt.plot_spins(m_list=res["spin_density"])
    #latt_plt.plot_spins(m_list=O2_spins, coords=O2_coords, color=COLORS["red-gray"])

    # plot pairing
    if "m_Cu_Cu_dic" in res:
        if pairing == "Cu-Cu":
            latt_plt.plot_pairings(pair_dic=res["m_Cu_Cu_dic"],
                                   transparent=transparent)
        elif pairing == "O-O":
            latt_plt.plot_pairings(pair_dic=res["m_nn_O_O_dic"], bond_max=2.01,
                                   transparent=transparent)
        elif pairing == "Cu-O":
            latt_plt.plot_pairings(pair_dic=res["m_Cu_O_dic"], bond_max=2.01,
                                   transparent=transparent)
            latt_plt.plot_pairings(pair_dic=res["m_n_O_O_dic"], bond_max=2.01,
                                   transparent=transparent)
    return latt_plt

def test_plot_3band_order():
    import os
    import numpy as np
    from collections import OrderedDict

    from libdmet.system import lattice
    from libdmet.utils import lattice_plot
    from libdmet.utils import get_order_param as order
    import matplotlib
    matplotlib.use('Agg')

    GRho_file = os.path.dirname(os.path.realpath(__file__)) + "/GRho_3band"
    GRho = np.load(GRho_file)
    res = order.get_3band_order(GRho)

    #print (res)

    #exit()
    res["m_Cu_Cu_dic"] = {(0, 3): 0.01406,
                          (0, 9): -0.01546,
                          (3, 6): -0.01243,
                          (6, 9): 0.01070}

    latt_plt = plot_3band_order(res, pairing='Cu-Cu', xleft=0, xright=4,
                                yleft=0, yright=4)
    #latt_plt = lattice_plot.plot_3band_order(res, pairing='O-O')
    #latt_plt = lattice_plot.plot_3band_order(res, pairing='Cu-O')

    #latt_plt.show()
    latt_plt.savefig("pairing.png")

if __name__ == "__main__":
    test_plot_3band_order()
    test_lattice_plot()
