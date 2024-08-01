#! /usr/bin/env python

def test_get_vertex():
    from libdmet.dmet import quad_fit

    pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]
    print ("points: %s" %pts)
    x, y = zip(*pts)
    a, b, c, status = quad_fit.get_parabola_vertex(x, y)
    print (a, b, c)
    assert status
    assert abs(a - 1.0) < 1e-12
    assert abs(b - 0.0) < 1e-12
    assert abs(c - 0.0) < 1e-12

    pts = [(0.0, 0.0), (1.0, -1.0), (2.0, -4.0)]
    print ("points: %s" %pts)
    x, y = zip(*pts)
    a, b, c, status = quad_fit.get_parabola_vertex(x, y)
    print (a, b, c)
    assert status
    assert abs(a + 1.0) < 1e-12
    assert abs(b - 0.0) < 1e-12
    assert abs(c - 0.0) < 1e-12

    pts = [(0.0, 1.0), (1.0, 1.0), (2.0, 4.0)]
    print ("points: %s" %pts)
    x, y = zip(*pts)
    a, b, c, status = quad_fit.get_parabola_vertex(x, y)
    print (a, b, c)
    assert status
    assert abs(-b/(2.0 * a) - 0.5) < 1e-12
    assert abs(c - 1.0) < 1e-12

    pts = [(0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    print ("points: %s" %pts)
    x, y = zip(*pts)
    a, b, c, status = quad_fit.get_parabola_vertex(x, y)
    print (a, b, c)
    assert not status

    pts = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    print ("points: %s" %pts)
    x, y = zip(*pts)
    a, b, c, status = quad_fit.get_parabola_vertex(x, y)
    print (a, b, c)
    assert not status

def test_quad_fit():
    import numpy as np
    from libdmet.dmet import quad_fit

    # 1. normal cases
    x1, y1 = 6.035156250e-01, 1.508072e+00
    x2, y2 = 5.432540625e-01, 1.488130e+00
    x3, y3 = 4.890186562e-01, 1.465602e+00

    x = np.array([x1, x2, x3])
    y = np.array([y1, y2, y3])
    mu = x
    dnelecs = y

    mu_new, status = quad_fit.quad_fit(mu, dnelecs, tol=1e-12)
    assert mu_new < x3
    assert status

    mu_new, status = quad_fit.quad_fit(mu, dnelecs-1.48, tol=1e-12)
    assert x3 < mu_new < x2
    assert status

    mu_new, status = quad_fit.quad_fit(mu, dnelecs-1.488130e+00, tol=1e-12)
    assert abs(mu_new - x2) < 1e-10
    assert status

    mu_new, status = quad_fit.quad_fit(mu, dnelecs-1.49, tol=1e-12)
    assert x2 < mu_new < x1
    assert status

    mu_new, status = quad_fit.quad_fit(mu, dnelecs-1.51, tol=1e-12)
    assert mu_new > x1
    assert status

    # 2. bad cases

    x1, y1 = 6.035156250e-01, 1.508072e+00
    x2, y2 = 5.432540625e-01, 1.588130e+00
    x3, y3 = 4.890186562e-01, 1.465602e+00

    x = np.array([x1, x2, x3])
    y = np.array([y1, y2, y3])
    mu = x
    dnelecs = y

    mu_new, status = quad_fit.quad_fit(mu, dnelecs-1.465602e+00, tol=1e-12)
    assert status
    assert abs(mu_new - x3) < 1e-10

    # 2.1 not monotonic
    mu_new, status = quad_fit.quad_fit(mu, dnelecs, tol=1e-12)
    assert mu_new < x3
    assert status

    # 2.2 complex root
    mu_new, status = quad_fit.quad_fit(mu, dnelecs-1.6, tol=1e-12)
    assert not status
    assert mu_new == 0

    # 2.3 duplicate points
    x1, y1 = 5.432540625e-01, 1.508072e+00
    x2, y2 = 5.432540625e-01, 1.588130e+00
    x3, y3 = 4.890186562e-01, 1.465602e+00

    x = np.array([x1, x2, x3])
    y = np.array([y1, y2, y3])
    mu = x
    dnelecs = y

    mu_new, status = quad_fit.quad_fit(mu, dnelecs-1.465602e+00, tol=1e-12)
    assert not status
    assert mu_new == 0

    # 2.4 cannot find root within proper range
    x = np.array([0.97, 0.46, 0.49])
    y = np.array([0.33, 0.12, 0.95])
    mu = x
    dnelecs = y

    mu_new, status = quad_fit.quad_fit(mu, dnelecs-1.8602e+00, tol=1e-12)
    assert not status
    assert mu_new == 0

def test_quad_fit_mu():
    import numpy as np
    from libdmet.dmet import quad_fit

    x1, y1 = 0.0, 1.585691128394
    x2, y2 = 0.1, 1.592874942297
    x3, y3 = 0.1, 1.592874942297

    x = np.array([x1, x2, x3])
    y = np.array([y1, y2, y3])
    mus = x
    nelecs = y

    # duplicate linear
    mu_new = quad_fit.quad_fit_mu(mus, nelecs, 1.70 * 0.5, 10.0)
    a = (y2 - y1) / (x2 - x1)
    b = y1
    assert abs(a * mu_new + b - 1.70) < 1e-12

    # duplicate larger than trust step
    mu_new = quad_fit.quad_fit_mu(mus, nelecs, 1.70 * 0.5, 0.1)
    assert abs(mu_new - 0.2) < 1e-12

    # not monotonic
    x1, y1 = 6.035156250e-01, 1.508072e+00
    x2, y2 = 5.432540625e-01, 1.588130e+00
    x3, y3 = 4.890186562e-01, 1.465602e+00

    x = np.array([x1, x2, x3])
    y = np.array([y1, y2, y3])
    mus = x
    nelecs = y

    mu_new = quad_fit.quad_fit_mu(mus, nelecs, 1.60 * 0.5, 10.0)
    #assert abs(mu_new - 0.490018656200) < 1e-12
    assert abs(mu_new - 10.489018656200) < 1e-12

if __name__ == "__main__":
    test_quad_fit()
    test_get_vertex()
    test_quad_fit_mu()
