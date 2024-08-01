#! /usr/bin/env python

def test_visualize():
    import sys
    import libdmet.dmet.HubbardBCS as dmet
    from libdmet.routine.localizer import visualize_bath
    import matplotlib
    matplotlib.use('Agg')
    
    LatSize = (18, 18)
    ImpSize = (2, 2)
    U = 6.0 
    Filling = 0.8 / 2.0 
    Mu = U * Filling # initial guess of global Mu
    
    Lat = dmet.SquareLattice(*(LatSize + ImpSize))
    Ham = dmet.Ham(Lat, U)
    Lat.setHam(Ham)
    vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand=0.0)
    GRho, Mu = dmet.HartreeFockBogoliubov(Lat, vcor, Filling, Mu, thrnelec=1e-7)
    
    visualize_bath(Lat, LatSize, GRho, localize_bath='pm', spin=0, bath_index=None) 
    visualize_bath(Lat, LatSize, GRho, localize_bath='scdm', spin=0, bath_index=None) 
    sys.modules.pop("libdmet.dmet.Hubbard", None)

if __name__ == "__main__":
    test_visualize()
