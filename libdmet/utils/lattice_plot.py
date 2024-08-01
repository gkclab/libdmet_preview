#! /usr/bin/env python

"""
lattice_plot.py
A python module for plotting lattice model.

Author:
    Zhihao Cui <zhcui0408@gmail.com>
"""

import os, sys
import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# set font to 42 for Type2 fonts:
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype'] = 42

COLORS = \
       {"red"   : "#E57B7B",
        "blue"  : "#76ABD1",
        "gray"  : "#252525",
        "green" : "#75C175",
        "purple": "#B99CD4",
        "red2"  : "#E88889",
        "red-gray": "#683D3D",
        "gold": "#F9A602"}

class LatticePlot(object):
    def __init__(self, latt, fig=None, ax=None, linewidth=1.5):
        """
        LatticePlot: a class for plotting 1D and 2D lattice model.

        Args:
            latt: lattice object.
            fig: fig object.
            ax: ax object.
            linewidth: line width.
        """
        self.latt = latt
        self.fig = fig
        self.ax = ax
        self.linewidth = linewidth

    def show(self):
        """
        Show the figure to window.
        """
        plt.show()

    def savefig(self, fname, dpi=400, tight_layout=True, *args):
        """
        Save the figure to a file.

        Args:
            fname: file name
            dpi: 400
            bbox_inches: tight
        """
        if tight_layout:
            plt.tight_layout()
        plt.savefig(fname, dpi=dpi, *args)
    
    def plot_lattice(self, figsize=(4.8, 4.8), **kwargs):
        """
        Create a canvas for the lattice.

        Kwargs:
            xleft, xright, yleft, yright: depend on lattice vector
            noframe: False
            framewidth: self.linewidth
            facecolor: "white"
            show_xticks: False
            show_yticks: False
        """
        if self.fig is None or self.ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            self.fig = fig
            self.ax = ax

        # determine the xlim and ylim
        # ZHC TODO non-orthogonal lattice
        latt_size = np.diag(self.latt.size)
        latt_coords = np.array(self.latt.sites).T
        xmin = np.min(latt_coords[0])
        xmax = np.max(latt_coords[0])
        ymin = np.min(latt_coords[1])
        ymax = np.max(latt_coords[1])
        self.xleft  = kwargs.get("xleft",  xmin - 0.15 * latt_size[0])
        self.xright = kwargs.get("xright", xmax + 0.15 * latt_size[0])
        self.yleft  = kwargs.get("yleft",  ymin - 0.15 * latt_size[1])
        self.yright = kwargs.get("yright", ymax + 0.15 * latt_size[1])

        plt.xlim(self.xleft, self.xright)
        plt.ylim(self.yleft, self.yright)
        
        # facecolor
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor(kwargs.get("facecolor", "white")) # background color

        # do not show tick and labels
        ax.axes.get_xaxis().set_visible(kwargs.get("show_xticks", False)) 
        ax.axes.get_yaxis().set_visible(kwargs.get("show_yticks", False))

        if kwargs.get("noframe", False):
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            framewidth = kwargs.get("framewidth", self.linewidth)
            ax.spines['left'].set_linewidth(framewidth)
            ax.spines['right'].set_linewidth(framewidth)
            ax.spines['top'].set_linewidth(framewidth)
            ax.spines['bottom'].set_linewidth(framewidth)
    
    def plot_atom(self, coord, rad, color, edgecolor='black', 
                  linewidth=None, **kwargs):
        """
        Plot a single atom.

        Args:
            coord: [x, y] position of the atom
            rad: radius of atom
            color: color
            edgecolor: black
            linewidth: self.linewidth
        """
        if linewidth is None:
            linewidth = self.linewidth
        plt.scatter(coord[0], coord[1], c=color, s=np.sqrt(1000 * rad) * 20,
                    edgecolor=edgecolor, linewidths=linewidth, **kwargs)
    
    def plot_spin(self, coord, m, factor=4.0, color='black', width=0.05,
                  head_width=0.16, head_length=0.13, **kwargs):
        """
        Plot a single spin arrow.

        Args:
            coord: [x, y] position of the atom
            factor: scaling factor
            color: color
            width: 0.05
            head_width: 0.16
            head_length: 0.13
        """
        dx = 0.0
        dy = 0.5 * factor * m
        coord = np.array(coord)
        coord[1] -= 0.5 * dy
        plt.arrow(coord[0], coord[1], dx, dy, width=width,
                  head_width=head_width, head_length=head_length,
                  length_includes_head=False, color=color, **kwargs)
    
    def plot_name(self, coord, name, fontsize=15, **kwargs):
        """
        Plot a name at coord.
        """
        return plt.text(coord[0], coord[1], name, horizontalalignment='center', 
                        verticalalignment='center', fontsize=fontsize, **kwargs)
    
    plot_text = plot_name

    def plot_p_orb(self, coord, direct='up', phase=["+", "-"], width=0.4,
                   height=0.2, fontsize={"+": 16, "-": 24},
                   color={"+": COLORS["red"], "-": COLORS["blue"]}, **kwargs):
        """
        Plot a p orbital at coord.
        
        Args:
            color: dict, +: blue, -: red.
        """
        if direct == 'up':
            coord0 = [coord[0], coord[1] + width * 0.54]
            coord1 = [coord[0], coord[1] - width * 0.54]
            self.plot_name(coord0, phase[0], fontsize=fontsize[phase[0]])
            self.plot_name(coord1, phase[1], fontsize=fontsize[phase[1]])
            fc_list = [color[p] for p in phase]
            angle = 90
        elif direct == 'down':
            coord0 = [coord[0], coord[1] + width * 0.54]
            coord1 = [coord[0], coord[1] - width * 0.54]
            self.plot_name(coord0, phase[1], fontsize=fontsize[phase[1]])
            self.plot_name(coord1, phase[0], fontsize=fontsize[phase[0]])
            fc_list = [color[p] for p in phase][::-1]
            angle = 90
        elif direct == 'left':
            coord0 = [coord[0] + width * 0.54, coord[1]]
            coord1 = [coord[0] - width * 0.54, coord[1]]
            self.plot_name(coord1, phase[0], fontsize=fontsize[phase[0]])
            self.plot_name(coord0, phase[1], fontsize=fontsize[phase[1]])
            fc_list = [color[p] for p in phase][::-1]
            angle = 0
        elif direct == 'right':
            coord0 = [coord[0] + width * 0.54, coord[1]]
            coord1 = [coord[0] - width * 0.54, coord[1]]
            self.plot_name(coord1, phase[1], fontsize=fontsize[phase[1]])
            self.plot_name(coord0, phase[0], fontsize=fontsize[phase[0]])
            fc_list = [color[p] for p in phase]
            angle = 0
        else:
            raise ValueError
         
        ellipse1 = Ellipse(xy=coord0, width=width, height=height,
                           edgecolor=COLORS["gray"], fc=fc_list[0], lw=1.5, angle=angle)
        ellipse2 = Ellipse(xy=coord1, width=width, height=height,
                           edgecolor=COLORS["gray"], fc=fc_list[1], lw=1.5, angle=angle)
        self.ax.add_patch(ellipse1)
        self.ax.add_patch(ellipse2)

    def plot_d_orb(self, coord, direct='up', **kwargs):
        """
        Plot a d orbital at coord.
        """
        if direct == 'up':
            self.plot_p_orb(coord, direct='up', phase=["+", "+"])
            self.plot_p_orb(coord, direct='left', phase=["-", "-"])
        else:
            self.plot_p_orb(coord, direct='up', phase=["-", "-"])
            self.plot_p_orb(coord, direct='left', phase=["+", "+"])
    
    def plot_bond(self, coord0, coord1, val, color_list=None, zorder=None,
                  transparent=True, **kwargs):
        """
        Plot a bond between coord0 and coord1.

        Args:
            coord0
            coord1
            val: width of bond
            color_list: can be two colors, depend on the sign of val.
                        val >=0 color_list[0], val < 0 color_list[1].
            zorder: zorder of bond
        """
        x, y = zip(coord0, coord1)
        if transparent:
            alpha = 0.65
            if color_list is None:
                color_list = ["C2", "C4"]
        else:
            alpha = 1.0
            if color_list is None:
                color_list = [COLORS["green"], COLORS["purple"]]
        if zorder is None:
            zorder = -int(val < 0)
        plt.plot(x, y, color=color_list[int(val < 0)], linestyle='-',
                 linewidth=abs(val)*1000, alpha=alpha, zorder=zorder, **kwargs)
    
    """
    The following functions are for plotting a list of atoms, spins, pairings.
    """
    def plot_atoms(self, rad_list, color_dic, coords=None, names=None, **kwargs):
        """
        Plot all atoms in the lattice.

        Args:
            rad_list: a list of radius of atoms
            color_dic: dictionary of colors of species
            coords: coordinates of atoms, if None will use the positions 
                    from lattice
            names: names of atoms, if None will use the names from lattice
        
        Kwargs:
            edgecolor: black
            linewidth: self.linewidth
        """
        if names is None:
            names = self.latt.names
        assert len(names) == len(rad_list)
        if coords is None:
            coords = self.latt.sites
        assert len(coords) == len(names)
        spec_names = np.unique(names)
        #assert len(spec_names) == len(color_dic)
        
        for i, name in enumerate(names):
            self.plot_atom(coords[i], rad_list[i], color_dic[name], **kwargs)
    
    def plot_spins(self, m_list, coords=None, **kwargs):
        """
        Plot all spins in the lattice.

        Args:
            m_list: a list of radius of atoms
            coords: coords of atoms
        """
        if coords is None:
            coords = self.latt.sites
        assert len(coords) == len(m_list)
        
        for i in range(len(m_list)):
            self.plot_spin(coords[i], m_list[i], **kwargs)
    
    def plot_pairings(self, pair_dic, bond_max=np.inf, bond_min=0.0,
                      cross_boundary=True, cross_box=False, **kwargs):
        """
        Plot all pairings.

        Args:
            pair_dic: dict, {(i, j): val}, where i, j are indices, val are
                      pairing value
            bond_max: max length for a bond
            bond_min: min length for a bond
            cross_boundary: search bond across the boundary.
            cross_box: search bond beyond the plotting box.
        """
        latt_vec = self.latt.size
        for (i, j), val in pair_dic.items():
            coord0 = self.latt.site_idx2pos(i)
            coord1 = self.latt.site_idx2pos(j)
            if (la.norm(coord1 - coord0) <= bond_max) and \
               (la.norm(coord1 - coord0) >= bond_min):
                self.plot_bond(coord0, coord1, val, **kwargs)
            if cross_boundary:
                for m in [-1, 0, 1]:
                    for n in [-1, 0, 1]:
                        if m == 0 and n == 0:
                            continue
                        shift = m * latt_vec[0] + n * latt_vec[1]
                        coord0p = coord0 + shift
                        coord1p = coord1 + shift
                        if cross_box:
                            if (la.norm(coord1 - coord0p) <= bond_max) and \
                               (la.norm(coord1 - coord0p) >= bond_min):
                                self.plot_bond(coord0p, coord1, val, **kwargs)
                        elif coord0p[0] >= self.xleft and \
                             coord0p[0] < self.xright and \
                             coord0p[1] >= self.yleft and \
                             coord0p[1] < self.yright:
                            if (la.norm(coord1 - coord0p) <= bond_max) and \
                               (la.norm(coord1 - coord0p) >= bond_min):
                                self.plot_bond(coord0p, coord1, val, **kwargs)

                        if cross_box:
                            if (la.norm(coord1p - coord0) <= bond_max) and \
                               (la.norm(coord1p - coord0) >= bond_min):
                                self.plot_bond(coord0, coord1p, val, **kwargs)
                        elif coord1p[0] >= self.xleft and \
                           coord1p[0] < self.xright and \
                           coord1p[1] >= self.yleft and \
                           coord1p[1] < self.yright:
                            if (la.norm(coord1p - coord0) <= bond_max) and \
                               (la.norm(coord1p - coord0) >= bond_min):
                                self.plot_bond(coord0, coord1p, val, **kwargs)

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
    Lat = lattice.Square3BandSymm(1, 1, 1, 1)
    
    O2_names = ["O2", "O2", "O2", "O2"]
    O2_coords = [[4.0, 1.0], [1.0, 0.0], [0.0, 3.0], [3.0, 4.0]]
    O2_charges = 2.0 - res["charge"][[1, 4, 7, 10]]
    O2_spins = res["m_AFM_O_list"][::2]
    
    latt_plt = LatticePlot(Lat)
    latt_plt.plot_lattice(**kwargs)
    
    # plot hole density
    latt_plt.plot_atoms(rad_list=(2.0 - res["charge"]), 
                        color_dic={'Cu': COLORS['gold'], 'O': 'C3'})
    latt_plt.plot_atoms(rad_list=O2_charges, names=O2_names, coords=O2_coords,
                        color_dic={'O2': COLORS["red2"]}, edgecolor=COLORS["red-gray"])
    
    # plot spin
    latt_plt.plot_spins(m_list=res["spin_density"])
    latt_plt.plot_spins(m_list=O2_spins, coords=O2_coords, color=COLORS["red-gray"])
    
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

if __name__ == '__main__':
    import numpy as np
    from libdmet.utils import lattice_plot
    from libdmet.system import lattice
    
    Lat = lattice.Square3BandSymm(1, 1, 1, 1)

    latt_plt = lattice_plot.LatticePlot(Lat)
    latt_plt.plot_lattice()
    latt_plt.plot_atoms(rad_list=[1.0, 0.2, 0.2] * 4, 
                        color_dic={'Cu': 'gold', 'O': 'C3'})
    latt_plt.plot_spins(m_list=[0.3, 0.0, 0.0, -0.25, -0.0, -0.0001,
                        0.3, 0.0, 0.0, -0.35, 0.0, 0.0])
    latt_plt.plot_text([0.0, 0.0], "test")
    latt_plt.plot_d_orb([4.0, 4.0], direct='down')
    latt_plt.plot_p_orb([3.0, 4.0], direct='left', phase=["+", "-"])
    latt_plt.plot_bond([1.0, 1.0], [2.0, 1.0], val=-0.01)
    latt_plt.plot_bond([1.0, 1.0], [1.0, 3.0], val=+0.02)
    
    latt_plt.show()
    #latt_plt.savefig("latt.png")
