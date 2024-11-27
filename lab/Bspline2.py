import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pygismo as gs  # Import pygismo

def select_points():
    plt.figure()
    plt.plot()  # Example plot
    plt.title('Click to select points, press Enter to finish')
    # Use ginput to select points
    points = plt.ginput(n=-1, timeout=0)
    plt.close()
    return points

pts = select_points()
print('The selected points are: ', pts)
coefs = np.zeros((len(pts), 2))
coefs[:, 0] = np.array([pt[0] for pt in pts])
coefs[:, 1] = np.array([pt[1] for pt in pts])

# Create a knot vector based on the number of control points
degree = 3
num_knots = len(pts) + degree + 1
kv = np.zeros(num_knots)
kv[degree:num_knots-degree] = np.linspace(0, 1, num_knots - 2*degree)
kv[num_knots-degree:] = 1

kv = gs.nurbs.gsKnotVector(kv, degree)
basis = gs.nurbs.gsBSplineBasis(kv)

print('The knots of the basis are:\n', basis.knots(0).get())
print('The size of the basis is: ', basis.size())

curve = gs.nurbs.gsBSpline(basis, coefs)

N = 100
x = np.linspace(0, 1, N)
x = np.matrix(np.meshgrid(x))
y = curve.eval(x)

plt.plot(y[0, :], y[1, :])
plt.scatter(curve.coefs()[:, 0], curve.coefs()[:, 1])

# Plot the convex hull
hull = ConvexHull(coefs)

# Fill the convex hull area with a color
plt.fill(coefs[hull.vertices, 0], coefs[hull.vertices, 1], 'lightblue', alpha=0.3)

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()












