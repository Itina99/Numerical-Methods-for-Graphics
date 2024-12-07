import matplotlib.pyplot as plt
import numpy as np
import pygismo as gs
from matplotlib.widgets import Slider

def generate_knots(degree, num_basis_functions, clamp=True):
    # Calculate the total number of knots
    num_knots = num_basis_functions + degree + 2
    # Number of interior knots
    inner_knots = num_knots - 2 * (degree + 1)
    
    if clamp:
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
        

def create_basis(degree=2, num_basis_functions=6, clamp=True):
    knot_vector = generate_knots(degree, num_basis_functions, clamp)
    kv = gs.nurbs.gsKnotVector(np.array(knot_vector), degree)
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

# Interactive plot function
def plot():
    # Initial setup
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)  # Leave more space for the sliders
    basis = create_basis()
    x_vals, evals = compute_basis_evals(basis)

    # Plot the initial basis functions
    lines = []
    for i in range(0, basis.size()):
        line, = ax.plot(x_vals, evals[i, :], label=f'Basis function {i+1}')
        lines.append(line)
    
    ax.set_title('B-spline basis functions')
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$\varphi$')

    # Add sliders
    ax_slider_functions = plt.axes([0.25, 0.15, 0.65, 0.03])
    sliderfunctions = Slider(ax_slider_functions, 'Num basis functions', 1, 10, valinit=6, valstep=1)
    
    ax_slider_degree = plt.axes([0.25, 0.1, 0.65, 0.03])
    sliderdegree = Slider(ax_slider_degree, 'Degree', 1, 10, valinit=2, valstep=1)

    def update():
        # Update basis and evaluations
        num_basis_functions = int(sliderfunctions.val)
        degree = int(sliderdegree.val)
        basis = create_basis(degree, num_basis_functions)
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
            line.set_data(x_vals, evals[i, :])
        
        fig.canvas.draw_idle()

    sliderfunctions.on_changed(lambda val: update())
    sliderdegree.on_changed(lambda val: update())
    plt.show()
                             
    

if __name__ == "__main__":
    plot()
