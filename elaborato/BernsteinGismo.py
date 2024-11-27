import matplotlib.pyplot as plt
import numpy as np
import pygismo as gs
from matplotlib.widgets import Slider

#plotting


def create_basis( degree = 2):
    knot_array = np.zeros(degree + 1)
    knot_array = np.append(knot_array, np.ones(degree + 1))
    kv = gs.nurbs.gsKnotVector(np.array(knot_array), degree)
    basis = gs.nurbs.gsBSplineBasis(kv)
    return basis

#basis computations
def compute_basis_evals(basis, N=100):
    x_vals = np.linspace(0, 1, N)
    x_input = np.matrix(np.meshgrid(x_vals))
    evals = np.zeros((basis.size(), N))
    for i in range(0, basis.size()):
        evals[i, :] = basis.evalSingle(i, x_input)
    return x_vals, evals


def plot_basis_functions():
    # Initial setup
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    initial_degree = 2
    basis = create_basis(initial_degree)
    x_vals, evals = compute_basis_evals(basis)
    
    # Plot the initial basis functions
    lines = []
    for i in range(0, basis.size()):
        line, = ax.plot(x_vals, evals[i, :], label=f'Basis function {i+1}')
        lines.append(line)
    
    ax.set_title('Bernstein polynomials')
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$\varphi$')
    
    # Add slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Degree', 1, 10, valinit=initial_degree, valstep=1)
    
    def update(val):
        # Update basis and evaluations
        degree = int(slider.val)
        basis = create_basis(degree)
        x_vals, evals = compute_basis_evals(basis)
        
        # Adjust number of lines dynamically
        while len(lines) < basis.size():
            line, = ax.plot([], [])
            lines.append(line)
        while len(lines) > basis.size():
            line = lines.pop()
            line.remove()
        
        # Update line data
        for i, line in enumerate(lines):
            line.set_xdata(x_vals)
            line.set_ydata(evals[i, :])
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()



plot_basis_functions()


