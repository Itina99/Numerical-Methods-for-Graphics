import numpy as np


import matplotlib.pyplot as plt
from matplotlib import cm
import pygismo as gs  # Import pygism
from matplotlib.widgets import Slider

import random

def plot_bspline_basis(degree):
    # Define the number of control points
    num_control_points = random.randint(2, 10)
    
    # Create a uniform knot vector with degree control
    # The first and last degree+1 knots are repeated, and internal knots are uniformly spaced
    internal_knots = np.linspace(0, 5, num_control_points - degree)  # Evenly distributed internal knots
    knots = np.concatenate((np.zeros(degree), internal_knots, np.full(degree, 5)))  # Repeated boundary knots
    
    # Create the knot vector for the B-spline basis
    kv = gs.nurbs.gsKnotVector(knots, degree)
    
    # Create the B-spline basis functions
    basis = gs.nurbs.gsBSplineBasis(kv)
    
    # Prepare the x values for plotting
    N = 100
    x_vals = np.linspace(0, 5, N)
    x_input = np.matrix(np.meshgrid(x_vals))
    
    # Evaluate each basis function over the x range
    evals = np.zeros((basis.size(), N))
    for i in range(basis.size()):
        evals[i, :] = basis.evalSingle(i, x_input)
    
    # Plotting
    ax.clear()
    for i in range(basis.size()):
        ax.plot(x_vals, evals[i, :], label=f'Basis function {i+1}')
    
    # Enhance plot aesthetics
    ax.set_title(f"B-spline Basis Functions (Degree {degree})")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\varphi$")
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.canvas.draw()

# Create the initial plot
fig, ax = plt.subplots(figsize=(10, 6))

# Initial degree to start with
initial_degree = 2
plot_bspline_basis(initial_degree)

# Add a slider to the plot for the degree control
ax_slider = plt.axes([0.15, 0.01, 0.7, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Degree', 1, 5, valinit=initial_degree, valstep=1)

# Define the update function for the slider
def update(val):
    degree = int(slider.val)
    plot_bspline_basis(degree)

slider.on_changed(update)

plt.show()

print("Committest")
