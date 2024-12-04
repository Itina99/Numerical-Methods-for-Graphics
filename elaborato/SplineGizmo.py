import matplotlib.pyplot as plt
import numpy as np
import pygismo as gs

def select_points():
    """
    Allows the user to select control points for the spline via mouse clicks.
    Returns the selected control points and a default degree.
    """
    plt.figure()
    plt.plot()  # Example plot
    plt.title('Click to select points, press Enter to finish')
    pts = plt.ginput(n=-1, timeout=0)
    plt.close()

    coefs = np.zeros((len(pts), 2))
    coefs[:, 0] = np.array([pt[0] for pt in pts])
    coefs[:, 1] = np.array([pt[1] for pt in pts])

    return coefs, 3  # Default to cubic degree


class SplineEditor:
    def __init__(self, coefs, degree):
        self.coefs = np.array(coefs)  # Control points
        self.degree = degree          # Degree of the spline
        self.knots = self.generate_knots()  # Knot vector
        self.curve = self.construct_curve()  # Initial spline
        self.dragging_point = None
        self.fig, self.ax = plt.subplots()
        self.plot()  # Plot the curve and control points
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        plt.show()

    def generate_knots(self, open_curve=True):
        """
        Create a uniform knot vector for a B-spline curve.
        
        Parameters:
            open_curve (bool): Whether the curve is open or closed. Default is True (open curve).
            
        Returns:
            np.array: The uniform knot vector.
        """
        num_control_points = self.coefs.shape[0]  # Number of control points
        if num_control_points < self.degree + 1:
            raise ValueError("Number of control points must be at least degree + 1.")
        
        num_knots = num_control_points + self.degree + 1  # Total number of knots
        
        if open_curve:
            # Open uniform knot vector: clamped at the ends
            knot_vector = (
                [0] * (self.degree + 1) +  # Clamped start
                list(range(1, num_knots - 2 * (self.degree + 1) + 1)) +  # Uniform interior
                [num_knots - 2 * (self.degree + 1) + 1] * (self.degree + 1)  # Clamped end
            )
        else:
            # Closed uniform knot vector: periodic (wraps around)
            knot_vector = list(range(num_knots - self.degree))

        print(f"Generated knot vector: {knot_vector}")  # Debugging statement
        return np.array(knot_vector)

    def construct_curve(self):
        """
        Construct the spline using the control points, degree, and knot vector.
        
        Returns:
            gs.nurbs.gsBSpline: Constructed spline.
        """
        kv = gs.nurbs.gsKnotVector(self.knots, self.degree)
        basis = gs.nurbs.gsBSplineBasis(kv)
        return gs.nurbs.gsBSpline(basis, self.coefs)

    def set_axes_limits(self):
        margin = 0.1
        x_min, x_max = self.coefs[:, 0].min(), self.coefs[:, 0].max()
        y_min, y_max = self.coefs[:, 1].min(), self.coefs[:, 1].max()
        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_max + margin)

    def plot(self, highlight_range=None):
        """Plot the curve and control points."""
        self.ax.clear()
        self.set_axes_limits()

        # Evaluate the curve
        N = 100  # Number of points on the curve
        t_vals = np.linspace(self.knots[self.degree], self.knots[-self.degree - 1], N)  # Parameter domain
        evaluated_points = np.array([self.curve.eval(np.array([t])) for t in t_vals]).squeeze()

        # Plot the curve
        if evaluated_points.ndim == 2:
            self.ax.plot(evaluated_points[:, 0], evaluated_points[:, 1], label='B-Spline Curve')
        else:
            print("Error: Evaluated points are not 2D. Check curve evaluation.")

        # Highlight the affected area if specified
        if highlight_range is not None:
            print(f"Highlight range: {highlight_range}")  # Debugging statement
            highlight_t_vals = np.linspace(highlight_range[0], highlight_range[1], N)
            highlight_points = np.array([self.curve.eval(np.array([t])) for t in highlight_t_vals]).squeeze()
            if highlight_points.ndim == 2:
                self.ax.plot(highlight_points[:, 0], highlight_points[:, 1], color='red', label='Affected Area')

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

        min_distance = float('inf')
        closest_point = None

        for i, (x, y) in enumerate(self.coefs):
            distance = np.hypot(x - event.xdata, y - event.ydata)
            if distance < min_distance:
                min_distance = distance
                closest_point = i

        if closest_point is not None:
            self.dragging_point = closest_point

    def on_release(self, event):
        self.dragging_point = None

    def on_motion(self, event):
        if event.inaxes != self.ax or self.dragging_point is None:
            return
        self.coefs[self.dragging_point] = [event.xdata, event.ydata]
        self.curve = self.construct_curve()
        self.visualize_locality(self.dragging_point)
        self.plot()

    def visualize_locality(self, modified_point_index):
        start = self.knots[modified_point_index]
        end = self.knots[modified_point_index + self.degree + 1]
        t_min = start / self.knots[-1]
        t_max = end / self.knots[-1]
        highlight_range = (t_min, t_max)

        self.plot(highlight_range=highlight_range)


if __name__ == '__main__':
    coefs, degree = select_points()
    SplineEditor(coefs, degree)
