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
        basis = gs.nurbs.gsBSplineBasis(kv)

        tens_basis = gs.nurbs.gsTensorBSplineBasis2(basis, basis)

    else:
        knot_array_a = np.zeros(degree_a + 1)
        knot_array_a = np.append(knot_array_a, np.ones(degree_a + 1))
        kv_u = gs.nurbs.gsKnotVector(np.array(knot_array_a), degree_a)
        basis_u = gs.nurbs.gsBSplineBasis(kv_u)

        knot_array_b = np.zeros(degree_b + 1)
        knot_array_b = np.append(knot_array_b, np.ones(degree_b + 1))
        kv_v = gs.nurbs.gsKnotVector(np.array(knot_array_b), degree_b)
        basis_v = gs.nurbs.gsBSplineBasis(kv_v)

        tens_basis = gs.nurbs.gsTensorBSplineBasis2(basis_u, basis_v)

    return tens_basis

#basis computations
def compute_basis_evals(basis, N=100, M=100):
    x_vals = np.linspace(0, 1, N)
    y_vals = np.linspace(0, 1, M)
    XX, YY = np.meshgrid(x_vals, y_vals, indexing='xy')
    pts = np.stack((XX.flatten(),YY.flatten()))
    ZZ = []
    for i in range(0, 4):
        eval = basis.evalSingle(i, pts)
        ZZ.append(eval.reshape((N, M)))
    return pts, XX, YY, ZZ

if __name__=='__main__':
    tbasis_a = create_basis(2, 2)

    pts, XX, YY, ZZ = compute_basis_evals(tbasis_a)

    fig = plt.figure()
    ax = fig.add_subplot(projection ='3d')
    for i in range(tbasis_a.size()):
        ax.plot_surface(XX,YY,ZZ[i],cmap=cm.coolwarm)
    plt.show()

    coefs = np.zeros((tbasis_a.size(),3))
    n = np.sqrt(tbasis_a.size())

    x = np.linspace(0,1,int(n))
    X,Y = np.meshgrid(x,x)
    coefs[:,0] = X.flatten()
    coefs[:,1] = Y.flatten()

    coefs[2,2] = -1
    coefs[3,2] = 1
    coefs[4,2] = 10
    coefs[5,2] = 1

    surf = gs.nurbs.gsTensorBSpline2(tbasis_a,coefs)
    s = surf.eval(pts)

    ZZ = s[2,:].reshape((N,M))

    fig = plt.figure()
    ax = fig.add_subplot(projection ='3d')
    ax.plot_surface(XX,YY,ZZ,cmap=cm.coolwarm)
    ax.scatter(surf.coefs()[:,0],surf.coefs()[:,1],surf.coefs()[:,2])
    plt.show()