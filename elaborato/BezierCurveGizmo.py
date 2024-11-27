import matplotlib.pyplot as plt
import numpy as np
import pygismo as gs
from matplotlib.widgets import Slider

def select_points():
    plt.figure()
    plt.plot()  # Example plot
    plt.title('Click to select points, press Enter to finish')
    # Use ginput to select points
    points = plt.ginput(n=-1, timeout=0)
    plt.close()
    return points

def create_basis( degree = 2):
    knot_array = np.zeros(degree + 1)
    knot_array = np.append(knot_array, np.ones(degree + 1))
    kv = gs.nurbs.gsKnotVector(np.array(knot_array), degree)
    basis = gs.nurbs.gsBSplineBasis(kv)
    return basis



#construct curve
def construct_curve():
    pts = select_points()
    basis = create_basis(len(pts) - 1)
    coefs = np.zeros((len(pts), 2))
    coefs[:, 0] = np.array([pt[0] for pt in pts])
    coefs[:, 1] = np.array([pt[1] for pt in pts])

    return gs.nurbs.gsBSpline(basis, coefs)


def plot_curve(curve):
    N = 100
    x = np.linspace(0, 1, N)
    x = np.matrix(np.meshgrid(x))
    y = curve.eval(x)

    plt.plot(y[0, :], y[1, :], label='Bezier Curve')
    plt.scatter(curve.coefs()[:, 0], curve.coefs()[:, 1], color='red', label='Control Points')
    plt.plot(curve.coefs()[:, 0], curve.coefs()[:, 1], linestyle='--', color='gray', label='Control Polygon')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()

curve = construct_curve()
plot_curve(curve)
