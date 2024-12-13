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
        self.curvature_ = self.curvature()
        self.dragging_point = None
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
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
        self.ax1.set_xlim(x_min - margin, x_max + margin)
        self.ax1.set_ylim(y_min - margin, y_max + margin)

    def plot(self, affected_point=None):
        """Plot the curve and control points, highlighting a specific range if provided."""
        self.ax1.clear()
        self.ax2.clear()
        self.set_axes_limits()

        # Define the full range of parameter values based on the knots (excluding the clamped values)
        N = 100  # Number of evaluation points
        t_vals = np.linspace(self.knots[self.degree], self.knots[-self.degree - 1], N)
        evaluated_points = np.array([self.curve.eval(np.array([t])) for t in t_vals]).squeeze()

        # Plot the full spline curve in blue
        self.ax1.plot(evaluated_points[:, 0], evaluated_points[:, 1], label='B-Spline Curve', color='blue')   

        # Highlight the affected section of the curve if a control point is specified
        if affected_point is not None:
            """start = max(0, affected_point - self.degree)
            end = min(self.coefs.shape[0], affected_point + self.degree + 1)
            affected_coefs = self.coefs[start:end]
            
            # Plot the affected polygon
            self.ax1.fill(affected_coefs[:, 0], affected_coefs[:, 1], color='red', alpha=0.3, label='Affected Control Polygon')"""

            start = self.knots[affected_point]
            end = self.knots[affected_point + self.degree + 1]
            c_vals = [i for i in t_vals if i >= start and i <= end]
            affected_points = np.array([self.curve.eval(np.array([c])) for c in c_vals]).squeeze()
            self.ax1.plot(affected_points[:, 0], affected_points[:, 1], label='Affected curve', color='purple', linewidth='3')

        self.ax2.plot(t_vals, self.curvature_, label='Curvature')
        
        # Plot control points and control polygon
        self.control_points, = self.ax1.plot(self.coefs[:, 0], self.coefs[:, 1], 'ro', label='Control Points', picker=5)
        self.control_polygon, = self.ax1.plot(self.coefs[:, 0], self.coefs[:, 1], '--', color='gray', label='Control Polygon')

        # Add the legend and labels
        self.ax1.legend()
        self.ax1.set_xlabel(r'$x$')
        self.ax1.set_ylabel(r'$y$')

        self.ax2.legend()
        self.ax2.set_xlabel(r'$x$')
        self.ax2.set_ylabel(r'$y$')

        # Redraw the plot
        plt.draw()

    def on_press(self, event):
        if event.inaxes != self.ax1:
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
        if event.inaxes != self.ax1 or self.dragging_point is None:
            return
        self.coefs[self.dragging_point] = [event.xdata, event.ydata]
        self.curve = self.construct_curve()
        self.curvature_ = self.curvature() 
        self.plot(affected_point=self.dragging_point)


    """def visualize_locality(self, modified_point_index):
        start = self.knots[modified_point_index]
        end = self.knots[modified_point_index + self.degree + 1]
        t_min = start / self.knots[-1]
        t_max = end / self.knots[-1]
        highlight_range = (t_min, t_max)

        self.plot(highlight_range=highlight_range)"""

    def curvature(self):
        N = 100
        x = np.linspace(self.knots[self.degree], self.knots[-self.degree - 1], N)
        x = np.matrix(np.meshgrid(x))

        a = self.curve.deriv(x)
        b = self.curve.deriv2(x)

        return np.linalg.norm(np.cross(a, b, axis=0), axis=0)/np.linalg.norm(a, axis=0)**3


if __name__ == '__main__':
    coefs, degree = select_points()
    SplineEditor(coefs, degree)
