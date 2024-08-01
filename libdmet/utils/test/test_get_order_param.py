#! /usr/bin/env python

def test_1band_order():
    import numpy as np
    from libdmet.utils import get_order_param as order
    from libdmet.system.lattice import SquareLattice
    
    GRho = np.array([[ 0.364,  0.168,  0.168,  0.049, -0.   ,  0.002, -0.002,  0.   ], 
     		     [ 0.168,  0.436,  0.044,  0.168,  0.004, -0.   ,  0.   , -0.004],
     		     [ 0.168,  0.044,  0.436,  0.168, -0.004,  0.   , -0.   ,  0.004],
     		     [ 0.049,  0.168,  0.168,  0.364,  0.   , -0.002,  0.002, -0.   ],
                     [-0.   ,  0.004, -0.004,  0.   ,  0.564, -0.168, -0.168, -0.044],
                     [ 0.002, -0.   ,  0.   , -0.002, -0.168,  0.636, -0.049, -0.168],
                     [-0.002,  0.   , -0.   ,  0.002, -0.168, -0.049,  0.636, -0.168],
                     [ 0.   , -0.004,  0.004, -0.   , -0.044, -0.168, -0.168,  0.564]])
    m_AFM, m_SC = order.get_order_param_1band(GRho)
    assert abs(m_AFM - 0.036) < 1e-10
    assert abs(m_SC - 0.004242640687119286) < 1e-10
    
    Lat = SquareLattice(2, 2, 2, 2)
    res = order.get_checkerboard_order(GRho, Lat, Cu_idx=[3, 1, 0, 2])
    assert abs(res["m_AFM"] - 0.036) < 1e-10
    assert abs(res["m_SC"] - 0.004242640687119286 * 4) < 1e-10

def test_3band_order():
    import os
    import numpy as np
    from libdmet.utils import get_order_param as order
    from libdmet.utils import logger as log
    log.verbose = 'DEBUG2'

    GRho_file = os.path.dirname(os.path.realpath(__file__)) + "/GRho_3band"
    GRho = np.load(GRho_file)
    res = order.get_3band_order(GRho)
    print ("AFM order: %12.6f" % res["m_AFM"])
    print ("SC  order: %12.6f" % res["m_SC"])
    assert abs(res["m_AFM"] - 0.23514104386522472) < 1e-10
    assert abs(res["m_Cu_Cu"] - 0.0012881272484919128 * 4) < 1e-10
    
    rho_file = os.path.dirname(os.path.realpath(__file__)) + "/rho_3band"
    rho = np.load(rho_file)
    res = order.get_3band_order(rho)
    print ("AFM order: %12.6f" % res["m_AFM"])
    assert abs(res["m_AFM"] - 0.23514104386522472) < 1e-10
    
def test_ab_initio_order():
    import os
    import numpy as np
    from libdmet.utils import get_order_param as order
    from libdmet.utils import logger as log
    from libdmet.system import lattice
    log.verbose = 'DEBUG2'

    GRho_file = os.path.dirname(os.path.realpath(__file__)) + "/GRho_3band"
    GRho = np.load(GRho_file)
    #res = order.get_3band_order(GRho)
    
    Lat = lattice.Square3BandSymm(1, 1, 1, 1)
    
    res = order.get_order_ab_initio(Lat, None, labels=Lat.cell.ao_labels(),
                                    rdm1_d=GRho, dis_Cu_Cu=[1.9, 2.1],
                                    dis_O_O=[1.9, 2.1], dis_Cu_O=[0.9, 1.1])
    
    print ("SC  order: %12.6f" % res["m_d_wave"])

    assert abs(res["m_Cu_Cu"] - 0.0012881272484919128 * 4) < 1e-10
    assert abs(res["m_d_wave"] - 0.006261066439120363) < 1e-10
    
    res = order.get_order_ab_initio(Lat, None, labels=Lat.cell.ao_labels(),
                                    rdm1_d=GRho, dis_Cu_Cu=[1.9, 2.1],
                                    dis_O_O=[1.9, 2.1], s_wave=True)
    
    print ("SC  order: %12.6f" % res["m_s_wave"])
    assert abs(res["m_s_wave"] - 0.00056264929) < 1e-10
    
    print ("*" * 79)
    cell_groups = {0 : (0, 1, 2), 1 : (3, 4, 5), 2 : (6, 7, 8), 3 : (9, 10, 11)}
    res = order.get_order_ab_initio_cell(Lat, None, labels=Lat.cell.ao_labels(),
                                         rdm1_d=GRho, dis_Cu_Cu=[1.9, 2.1],
                                         dis_O_O=[1.9, 2.1], dis_Cu_O=[0.9, 1.1],
                                         cell_groups=cell_groups)

if __name__ == "__main__":
    test_ab_initio_order()
    test_3band_order()
    test_1band_order()
