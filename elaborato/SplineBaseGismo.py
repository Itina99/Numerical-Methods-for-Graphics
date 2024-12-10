import matplotlib.pyplot as plt
import numpy as np
import pygismo as gs
from matplotlib.widgets import Slider

def generate_knots(degree, num_basis_functions, clamp=True, custom_knots=None):
    # Compute the total number of knots
    num_knots = degree + num_basis_functions + 1
    inner_knots = num_knots - 2 if not clamp else num_knots - 2 * (degree + 1)

    if custom_knots is not None and len(custom_knots) == inner_knots:
        # Use custom knots directly if provided
        if clamp:
            knot_vector = np.concatenate([
                np.zeros(degree + 1),  # Clamped start
                custom_knots,         # Inner knots
                np.ones(degree + 1)   # Clamped end
            ])
        else:
            knot_vector = np.concatenate([
                [0],                  # Start
                custom_knots,         # Inner knots
                [1]                   # End
            ])
    elif clamp:
        # Default clamped knot vector
        knot_vector = np.concatenate([
            np.zeros(degree + 1),
            np.linspace(0, 1, inner_knots + 2)[1:-1],
            np.ones(degree + 1)
        ])
    else:
        # Default unclamped knot vector
        knot_vector = np.concatenate([
            [0],
            np.linspace(0, 1, inner_knots + 2)[1:-1],
            [1]
        ])

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
def plot_degree_functions_sliders(clamped=True):    
    # Initial setup
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Leave more space for the sliders

    # Default values
    degree = 2
    num_basis_functions = 6
    basis, knot_vector = create_basis(degree=degree, num_basis_functions=num_basis_functions, clamp=clamped)
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
    sliderfunctions = Slider(ax_slider_functions, 'Num basis functions', 1, 10, valinit=num_basis_functions, valstep=1)
    
    ax_slider_degree = plt.axes([0.25, 0.1, 0.65, 0.03])
    sliderdegree = Slider(ax_slider_degree, 'Degree', 1, 10, valinit=degree, valstep=1)

    def update():
        # Update basis, knot vector, and evaluations
        num_basis_functions = int(sliderfunctions.val)
        degree = int(sliderdegree.val)
        
        basis, knot_vector = create_basis(degree, num_basis_functions,clamp=clamped)
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

    # Connect sliders to update function
    sliderfunctions.on_changed(lambda val: update())
    sliderdegree.on_changed(lambda val: update())

    plt.show()


def plot_with_knot_sliders(clamped=True):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.4)  # Leave more space for the sliders

    # User input for initial degree and number of basis functions
    degree = int(input("Degree: "))
    num_basis_functions = int(input("Number of basis functions: "))

    # Initial basis and knot vector
    basis, knot_vector = create_basis(degree=degree, num_basis_functions=num_basis_functions, clamp=clamped)
    x_vals, evals = compute_basis_evals(basis)

    # Plot the initial basis functions
    lines = []
    for i in range(basis.size()):
        line, = ax.plot(x_vals, evals[i, :], label=f'Basis function {i + 1}')
        lines.append(line)

    # Initial dotted lines for inner knots
    inner_knots_start = degree + 1 if clamped else 1
    inner_knots_end = -(degree + 1) if clamped else -1
    inner_knot_positions = knot_vector[inner_knots_start:inner_knots_end]
    
    ax.vlines(
        inner_knot_positions,
        ymin=0, ymax=1, colors='gray', linestyles='dotted', label='Knots'
    )

    ax.set_title('B-spline basis functions')
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$\varphi$')

    # Add sliders for custom knots
    custom_knot_axes = []
    custom_knot_sliders = []
    inner_knots = len(inner_knot_positions)
    for i in range(inner_knots):
        ax_slider = plt.axes([0.25, 0.25 - i * 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, f'Knot {i + 1}', 0.01, 0.99, valinit=inner_knot_positions[i])
        custom_knot_sliders.append(slider)
        custom_knot_axes.append(ax_slider)

    def update():
        # Collect current slider values for the inner knots
        custom_knots = np.array([slider.val for slider in custom_knot_sliders])

        # Ensure the custom knots are sorted
        custom_knots.sort()

        # Rebuild the knot vector for unclamped configuration
        if not clamped:
            updated_knot_vector = np.concatenate([
                [0],                  # Start
                custom_knots,         # Inner knots (exactly from sliders)
                [1]                   # End
            ])
        else:
            updated_knot_vector = np.concatenate([
                [0] * (degree + 1),   # Clamped start
                custom_knots,         # Inner knots
                [1] * (degree + 1)    # Clamped end
            ])

        # Update basis and evaluations
        try:
            basis, _ = create_basis(degree, num_basis_functions, custom_knots=custom_knots, clamp=clamped)
        except Exception as e:
            print(f"Error in basis creation: {e}")
            return

        x_vals, evals = compute_basis_evals(basis)

        # Update basis function plots
        for i, line in enumerate(lines):
            line.set_data(x_vals, evals[i, :])
        for i in range(len(lines), basis.size()):
            line, = ax.plot(x_vals, evals[i, :], label=f'Basis function {i + 1}')
            lines.append(line)
        while len(lines) > basis.size():
            line = lines.pop()
            line.remove()

        # Update inner knot lines
        for collection in list(ax.collections):
            collection.remove()  # Safely remove collections from the plot

        ax.vlines(
            updated_knot_vector[1:-1],  # Exclude the first and last knots
            ymin=0, ymax=1, colors='gray', linestyles='dotted', label='Knots'
        )

        # Refresh the figure
        fig.canvas.draw_idle()



    # Connect sliders to the update function
    for slider in custom_knot_sliders:
        slider.on_changed(lambda val, slider=slider: update())

    plt.show()




                             
if __name__ == "__main__":
    plot_with_knot_sliders(clamped=False)
    plot_with_knot_sliders(clamped=True)
    plot_degree_functions_sliders(clamped=False)
    plot_degree_functions_sliders()
