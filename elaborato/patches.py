import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pygismo as gs

def create_bezier_basis(degree_a = 2, degree_b = 2):
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

def generate_knots(degree, num_basis_functions, clamp=True, custom_knots=None):
    # Calculate the total number of knots
    num_knots = num_basis_functions + degree + 2
    # Number of interior knots
    inner_knots = num_knots - 2 * (degree + 1)
    
    if custom_knots is not None and len(custom_knots) == inner_knots:
        # Custom knots allow dynamic updates
        knot_vector = np.concatenate([
            np.zeros(degree + 1),
            custom_knots,
            np.ones(degree + 1)
        ])
    elif clamp:
        # Start with clamped knots at the start
        knot_vector = np.zeros(degree + 1)
        # Add the interior knots
        if inner_knots > 0:
            knot_vector = np.append(knot_vector, np.linspace(0, 1, inner_knots + 2)[1:-1])
        # Add clamped knots at the end
        knot_vector = np.append(knot_vector, np.ones(degree + 1))
    else:
        # Uniformly spaced knots for unclamped
        knot_vector = np.linspace(0, 1, num_knots)
    
    return knot_vector


def create_bspline_basis(degree_a=2, degree_b=2, num_basis_function_a=2, num_basis_function_b=2, clamp_a=True, clamp_b=True, custom_a=None, custom_b=None):
    knot_vector_u = generate_knots(degree_a, num_basis_function_a, clamp_a, custom_a)
    knot_vector_v = generate_knots(degree_b, num_basis_function_b, clamp_b, custom_b)
    kv_u = gs.nurbs.gsKnotVector(np.array(knot_vector_u), degree_a)
    kv_v = gs.nurbs.gsKnotVector(np.array(knot_vector_v), degree_b)
    basis = gs.nurbs.gsTensorBSplineBasis2(kv_u, kv_v)
    return basis, kv_u, kv_v

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
def plot_patch(x, y, z, surf, params, N=100, M=100):
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
    # tbasis = create_bezier_basis(basis_a_deg, basis_b_deg)
    tbasis, ku, kv = create_bspline_basis(basis_a_deg, basis_b_deg, clamp_a=True, clamp_b=False)

    N = 100
    M = 100
    x_vals = np.linspace(0, 1, N)
    y_vals = np.linspace(0, 1, M)
    XX, YY = np.meshgrid(x_vals, y_vals, indexing='xy')
    pts = np.stack((XX.flatten(),YY.flatten()))

    pts, ZZ = compute_basis_evals(pts, tbasis, N, M)

    # plot_basis(XX, YY, ZZ, tbasis)

    coefs = define_control_poins(tbasis.size(), dims=(basis_a_deg + 1, basis_b_deg + 1))
    surf = gs.nurbs.gsTensorBSpline2(tbasis, coefs)
    # plot_patch(XX, YY, ZZ, surf, pts, N, M)

    gs.io.gsWriteParaview(tbasis,"elaborato/pw_out/basis",100000)
    gs.io.gsWriteParaview(surf,"elaborato/pw_out/surf",100000)