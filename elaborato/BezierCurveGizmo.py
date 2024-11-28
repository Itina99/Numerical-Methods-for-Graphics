import matplotlib.pyplot as plt
import numpy as np
import pygismo as gs
from matplotlib.widgets import Slider

def select_points():
    plt.figure()
    plt.plot()  # Example plot
    plt.title('Click to select points, press Enter to finish')
    # Use ginput to select points
    pts = plt.ginput(n=-1, timeout=0)
    plt.close()

    coefs = np.zeros((len(pts), 2))
    coefs[:, 0] = np.array([pt[0] for pt in pts])
    coefs[:, 1] = np.array([pt[1] for pt in pts])

    return coefs , len(pts)-1



class CurveEditor:
    def __init__(self, coefs, degree):
        self.coefs = np.array(coefs)  # Control points
        self.degree = degree          # Degree of the curve
        self.curve = self.construct_curve()  # Initial curve
        self.dragging_point = None
        self.fig, self.ax = plt.subplots()
        self.plot()  # Plot the curve and control points
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        plt.show()

    def construct_curve(self):
        """Reconstruct the curve using the current control points."""
        knot_array = np.zeros(self.degree + 1)
        knot_array = np.append(knot_array, np.ones(self.degree + 1))
        kv = gs.nurbs.gsKnotVector(np.array(knot_array), self.degree)
        basis = gs.nurbs.gsBSplineBasis(kv)
        return gs.nurbs.gsBSpline(basis, self.coefs)
    
    def set_axes_limits(self):
        """Set fixed limits for the axes based on the initial control points."""
        margin = 0.1  # Margin around the control points
        x_min, x_max = self.coefs[:, 0].min(), self.coefs[:, 0].max()
        y_min, y_max = self.coefs[:, 1].min(), self.coefs[:, 1].max()

        # Set limits with a margin
        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_max + margin)

    def construct_curve(self):
        """Reconstruct the curve using the current control points."""
        # Create knot vector
        knot_array = np.zeros(self.degree + 1)
        knot_array = np.append(knot_array, np.ones(self.degree + 1))
        kv = gs.nurbs.gsKnotVector(np.array(knot_array), self.degree)
        basis = gs.nurbs.gsBSplineBasis(kv)
        return gs.nurbs.gsBSpline(basis, self.coefs)

    def plot(self):
        """Plot the curve, control points, and control polygon."""
        self.ax.clear()

        # Restore fixed axes limits
        self.set_axes_limits()

        # Evaluate the curve
        N = 100
        x = np.linspace(0, 1, N)
        x = np.matrix(np.meshgrid(x))
        y = self.curve.eval(x)

        # Plot the curve
        self.ax.plot(y[0, :], y[1, :], label='Bezier Curve')

        # Plot control points and polygon
        self.control_points, = self.ax.plot(self.coefs[:, 0], self.coefs[:, 1], 'ro', label='Control Points', picker=5)
        self.control_polygon, = self.ax.plot(self.coefs[:, 0], self.coefs[:, 1], '--', color='gray', label='Control Polygon')

        self.ax.legend()
        self.ax.set_xlabel(r'$x$')
        self.ax.set_ylabel(r'$y$')

        plt.draw()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        # Initialize a variable to store the closest point's index and distance
        min_distance = float('inf')  # Set initially to infinity
        closest_point = None

        # Iterate over all control points
        for i, (x, y) in enumerate(self.coefs):
            distance = np.hypot(event.xdata - x, event.ydata - y)  # Compute Euclidean distance
            if distance < min_distance and distance < 0.05:  # Update if closer and within tolerance
                min_distance = distance
                closest_point = i

        if closest_point is not None:
            self.dragging_point = closest_point

    def on_release(self, event):
        self.dragging_point = None  # Clear the dragging state

    def on_motion(self, event):
        if event.inaxes != self.ax or self.dragging_point is None:
            return
        # Update the position of the dragged point
        self.coefs[self.dragging_point] = [event.xdata, event.ydata]
        self.curve = self.construct_curve()  # Reconstruct the curve
        self.plot()  # Redraw the plot


if __name__ == '__main__':
    # Select points interactivel
    control_points, degree = select_points()

    # Initialize the editor
    editor = CurveEditor(control_points, degree)

