import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pygismo as gs

def create_basis(degree_a = 2, degree_b = 2):
    if degree_a == degree_b:
        degree = degree_a

        knot_array = np.zeros(degree + 1)
        knot_array = np.append(knot_array, np.ones(degree + 1))
        kv = gs.nurbs.gsKnotVector(np.array(knot_array), degree)

        tens_basis = gs.nurbs.gsTensorBSplineBasis2(kv, kv)

    else:
        knot_array_a = np.zeros(degree_a + 1)
        knot_array_a = np.append(knot_array_a, np.ones(degree_a + 1))
        kv_u = gs.nurbs.gsKnotVector(np.array(knot_array_a), degree_a)

        knot_array_b = np.zeros(degree_b + 1)
        knot_array_b = np.append(knot_array_b, np.ones(degree_b + 1))
        kv_v = gs.nurbs.gsKnotVector(np.array(knot_array_b), degree_b)

        tens_basis = gs.nurbs.gsTensorBSplineBasis2(kv_u, kv_v)

    return tens_basis

#basis computations
def compute_basis_evals(params, basis, N=100, M=100):
    ZZ = []
    for i in range(0, basis.size()):
        eval = basis.evalSingle(i, params)
        ZZ.append(eval.reshape((N, M)))
    return pts, ZZ

# Control point definition
def define_control_poins(basis_size, dims=None):
    coefs = np.zeros((basis_size, 3))

    if np.sqrt(basis_size) == int(np.sqrt(basis_size)):
        x = np.linspace(0, 1, int(np.sqrt(basis_size)))
        X, Y = np.meshgrid(x, x)
    else:
        x = np.linspace(0, 1, dims[0])
        y = np.linspace(0, 1, dims[1])
        X, Y = np.meshgrid(x, y)
    coefs[:,0] = X.flatten()
    coefs[:,1] = Y.flatten()
    
    for i in range(basis_size):
        if np.random.rand() > 0.5:
            coefs[i, 2] = int(np.random.rand()*10 - 5)

    return coefs

# Basis plotting
def plot_basis(x, y, z, basis):
    fig = plt.figure()
    ax = fig.add_subplot(projection ='3d')
    print(basis.size())
    for i in range(basis.size()):
        ax.plot_surface(x, y, z[i], cmap=cm.coolwarm)
    plt.show()

# Patch plotting
def plot_bezier_patch(x, y, z, surf, params, N=100, M=100):
    s = surf.eval(params)

    z = s[2,:].reshape((N,M))

    fig = plt.figure()
    ax = fig.add_subplot(projection ='3d')
    ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    ax.scatter(surf.coefs()[:,0],surf.coefs()[:,1],surf.coefs()[:,2])
    plt.show()


if __name__=='__main__':
    basis_a_deg = 2
    basis_b_deg = 2
    tbasis = create_basis(basis_a_deg, basis_b_deg)

    N = 100
    M = 100
    x_vals = np.linspace(0, 1, N)
    y_vals = np.linspace(0, 1, M)
    XX, YY = np.meshgrid(x_vals, y_vals, indexing='xy')
    pts = np.stack((XX.flatten(),YY.flatten()))

    pts, ZZ = compute_basis_evals(pts, tbasis)

    plot_basis(XX, YY, ZZ, tbasis)

    coefs = define_control_poins(tbasis.size(), dims=(basis_b_deg + 1, basis_b_deg + 1))
    surf = gs.nurbs.gsTensorBSpline2(tbasis,coefs)
    plot_bezier_patch(XX, YY, ZZ, surf, pts)