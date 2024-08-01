#! /usr/bin/env python

import sys
import numpy as np
import scipy.linalg as la
from libdmet.utils.misc import readlines_find
import matplotlib.pyplot as plt
from scipy import stats

def extrapolate_M(fname, start=1, dw_tol=1e-4):
    lines, M_num = readlines_find('Sweep Energy', fname)
    M_lines = [lines[num] for num in M_num]

    print ("%10s %20s %20s %5s"%("M", "discarded weight", "Energy", "Used"))
    dws = []
    Ms = []
    Es = []

    n = 0
    M_old = int(M_lines[0].split()[2])
    for i, line in enumerate(M_lines[1:]):
        line_sp = line.split()
        line_old_sp = M_lines[i].split()
        M_new = int(line_sp[2])
        dw = float(line_sp[10])
        dw_old = float(line_old_sp[10])
        E_old = float(line_old_sp[-1])
        if M_new != M_old or float(dw) == 0.0:
            if dw_old < dw_tol and n >= start:
                Ms.append(M_old)
                dws.append(dw_old)
                Es.append(E_old)
                used = True
            else:
                used = False
            print ("%10d %20.12f %20.12f %5s"%(M_old, dw_old, E_old, used))
            if float(dw) == 0.0: # last twodot calc
                break
            M_old = M_new
            n += 1

    # if calc is not done
    if i == (len(M_lines)-2):
        Ms.append(M_new)
        dws.append(dw)
        Es.append(float(line_sp[-1]))
        used = True
        print ("%10d %20.12f %20.12f %5s"%(Ms[-1], dws[-1], Es[-1], used))

    slope, intercept, r_value, p_value, std_err = stats.linregress(dws, Es)
    E_ext = intercept

    print ("Extrapolated Energy:")
    print (E_ext)
    print ("Extrapolated Error:")
    print (E_ext - Es[-1])
    print ("r value")
    print (r_value)

    if r_value < 0.9:
        print ("WARNING! r value is far from 1.0")

    #fit = np.polyfit(dws, Es, 1)
    #fit_fn = np.poly1d(fit)
    #x = np.linspace(0, np.max(dws), 100)
    #plt.plot(x, fit_fn(x), '--k', linewidth=2, color='black')
    #plt.show()
    return E_ext

if __name__ == '__main__':
    if len(sys.argv) > 1 :
        fname = sys.argv[1]
        if len(sys.argv) > 2:
            start = int(sys.argv[2])
        else:
            start = 1
    else:
        fname = './dmrg.out'
    extrapolate_M(fname, start=start)
