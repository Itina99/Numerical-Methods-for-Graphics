import matplotlib.pyplot as plt
import numpy as np
import pygismo as gs
from matplotlib.widgets import Slider

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

def create_basis(degree=2, num_basis_functions=6, clamp=True, custom_knots=None):
    knot_vector = generate_knots(degree, num_basis_functions, clamp, custom_knots)
    kv = gs.nurbs.gsKnotVector(np.array(knot_vector), degree)
    basis = gs.nurbs.gsBSplineBasis(kv)
    return basis, knot_vector

# Basis computations
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
    plt.subplots_adjust(bottom=0.4)  # Leave more space for the sliders

    # Default values
    degree = 2
    num_basis_functions = 6
    basis, knot_vector = create_basis(degree=degree, num_basis_functions=num_basis_functions)
    x_vals, evals = compute_basis_evals(basis)

    # Plot the initial basis functions
    lines = []
    for i in range(0, basis.size()):
        line, = ax.plot(x_vals, evals[i, :], label=f'Basis function {i+1}')
        lines.append(line)
    
    # Plot the knot vector
    knot_lines = ax.vlines(knot_vector, ymin=0, ymax=1, colors='gray', linestyles='dashed', label='Knots')

    ax.set_title('B-spline basis functions')
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$\varphi$')

    # Add sliders
    ax_slider_functions = plt.axes([0.25, 0.25, 0.65, 0.03])
    sliderfunctions = Slider(ax_slider_functions, 'Num basis functions', 1, 10, valinit=num_basis_functions, valstep=1)
    
    ax_slider_degree = plt.axes([0.25, 0.2, 0.65, 0.03])
    sliderdegree = Slider(ax_slider_degree, 'Degree', 1, 10, valinit=degree, valstep=1)

    # Custom knot sliders
    inner_knots = len(knot_vector) - 2 * (degree + 1)
    custom_knot_sliders = []
    custom_knot_axes = []

    def create_custom_knots_sliders():
        """Create the custom knot sliders based on the current knot vector"""
        for ax in custom_knot_axes:
            ax.clear()  # Remove previous sliders and axes
        custom_knot_sliders.clear()  # Clear previous sliders

        for i in range(inner_knots):
            ax_slider = plt.axes([0.25, 0.15 - i * 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
            slider = Slider(ax_slider, f'Knot {i + 1}', 0.01, 0.99, valinit=knot_vector[degree + 1 + i])
            custom_knot_sliders.append(slider)
            custom_knot_axes.append(ax_slider)

    # Initial custom knot sliders
    create_custom_knots_sliders()

    def update():
        # Update basis, knot vector, and evaluations
        num_basis_functions = int(sliderfunctions.val)
        degree = int(sliderdegree.val)
        
        # Update custom knots
        custom_knots = np.array([slider.val for slider in custom_knot_sliders])
        basis, knot_vector = create_basis(degree, num_basis_functions, custom_knots=custom_knots)
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

        # Update knot markers
        for kline in ax.collections:
            kline.remove()  # Clear old knots
        ax.vlines(knot_vector, ymin=0, ymax=1, colors='gray', linestyles='dashed')

        # Recreate custom knot sliders
        create_custom_knots_sliders()

        fig.canvas.draw_idle()

    # Connect sliders to update function
    sliderfunctions.on_changed(lambda val: update())
    sliderdegree.on_changed(lambda val: update())
    for slider in custom_knot_sliders:
        slider.on_changed(lambda val: update())

    plt.show()
                             
if __name__ == "__main__":
    plot()
