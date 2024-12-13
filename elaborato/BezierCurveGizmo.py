import matplotlib.pyplot as plt
import numpy as np
import pygismo as gs
from matplotlib.widgets import Slider
import math

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
        self.curvature_ = self.curvature()
        self.dragging_point = None
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
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
        self.ax1.set_xlim(x_min - margin, x_max + margin)
        self.ax1.set_ylim(y_min - margin, y_max + margin)

    def plot(self):
        """Plot the curve, control points, and control polygon."""
        self.ax1.clear()
        self.ax2.clear()

        # Restore fixed axes limits
        self.set_axes_limits()

        # Evaluate the curve
        N = 100
        x_space = np.linspace(0, 1, N)
        x = np.matrix(np.meshgrid(x_space))
        y = self.curve.eval(x)

        # Plot the curve
        self.ax1.plot(y[0, :], y[1, :], label='Bezier Curve')
        print(self.curvature_)
        self.ax2.plot(x_space, self.curvature_, label='Curvature')

        # Plot control points and polygon
        self.control_points, = self.ax1.plot(self.coefs[:, 0], self.coefs[:, 1], 'ro', label='Control Points', picker=5)
        self.control_polygon, = self.ax1.plot(self.coefs[:, 0], self.coefs[:, 1], '--', color='gray', label='Control Polygon')

        self.ax1.legend()
        self.ax1.set_xlabel(r'$x$')
        self.ax1.set_ylabel(r'$y$')

        self.ax2.legend()
        self.ax2.set_xlabel(r'$x$')
        self.ax2.set_ylabel(r'$y$')

        plt.draw()

    def on_press(self, event):
        if event.inaxes != self.ax1:
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
        if event.inaxes != self.ax1 or self.dragging_point is None:
            return
        # Update the position of the dragged point
        self.coefs[self.dragging_point] = [event.xdata, event.ydata]
        self.curve = self.construct_curve()  # Reconstruct the curve
        self.curvature_ = self.curvature()
        self.plot()  # Redraw the plot

    def linear_precision(self):
    # Codice per la proprietà di linear precision
        """
            la curva generata dal poligono di controllo
            i cui vertici sono allineati su un segmento di retta e fra loro equidistanti
            è il segmento di retta compreso tra i punti di controllo con t che rappresenta
            la lunghezza dell'arco sul segmento.
        """ 
        degrees = [2, 3, 4]
        basis = []
        ctl_points = []
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 10))
        axes = [ax1, ax2, ax3]
        for idx, d in enumerate(degrees):
            knot_array = np.zeros(d + 1)
            knot_array = np.append(knot_array, np.ones(d + 1))
            kv = gs.nurbs.gsKnotVector(np.array(knot_array), d)
            basis.append(gs.nurbs.gsBSplineBasis(kv))

            # Verifica dell'equazione per ciascuna base creata
            t = np.random.random()
            res = 0

            for i in range(basis[idx].size()):
                res += i/d * basis[idx].evalSingle(i, np.array([t]))
            print(np.isclose(res[0][0], t))
        
        # Creazione di più esempi sullo stesso plot
            # Nessun controllo su dove si disegnano i punti, tanto devono essere allineati ed equidistanti
            xs = np.linspace(np.random.random() * d, 1 + np.random.random() * d, d+1)
            ys = np.linspace(np.random.random() * d, 1 + np.random.random() * d, d+1)
            ctl_points.append(np.vstack((xs, ys)).T)
        
        # Calcolo le curve
        N = 100
        x = np.linspace(0, 1, N)
        x = np.matrix(np.meshgrid(x))
        for i, ax in enumerate(axes):
            curve =  gs.nurbs.gsBSpline(basis[i], ctl_points[i]) 
            y = curve.eval(x)
            ax.plot(y[0, :], y[1, :], color='purple', linewidth='3', label=f'Bezier Curve degree {degrees[i]}')
            ax.plot(ctl_points[i][:, 0], ctl_points[i][:, 1], 'ro', label='Control Points', picker=5)
            ax.plot(ctl_points[i][:, 0], ctl_points[i][:, 1], '-.', color='yellow', label='Control Polygon')
            ax.legend()
        plt.show()

    def curvature(self):
        N = 100
        x = np.linspace(0, 1, N)
        x = np.matrix(np.meshgrid(x))

        a = self.curve.deriv(x)
        b = self.curve.deriv2(x)

        return np.linalg.norm(np.cross(a, b, axis=0), axis=0)/np.linalg.norm(a, axis=0)**3

def bezier_3d(deg, curv=False):
    """
        Permette di disegnare una curva di Bezier nello spazio
    """
    # Creo la base di bernstein corrispondente al grado in input
    knot_array = np.zeros(deg + 1)
    knot_array = np.append(knot_array, np.ones(deg+ 1))
    kv = gs.nurbs.gsKnotVector(np.array(knot_array), deg)
    basis = gs.nurbs.gsBSplineBasis(kv)
    
    # Prendo dei punti casuali nello spazio 3d
    x_points = np.random.uniform(0, 10, deg + 1)
    y_points = np.random.uniform(0, 10, deg + 1)
    z_points = np.random.uniform(0, 10, deg + 1)
    points = np.vstack((x_points, y_points, z_points)).T
    print(points)

    # Calcolo la curva e la valuto nello spazio
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    curve = gs.nurbs.gsBSpline(basis, points)
    N = 100
    x_space = np.linspace(0, 1, N)
    x = np.matrix(np.meshgrid(x_space))
    y = curve.eval(x)
    ax.plot(y[0, :], y[1, :], y[2, :], color='purple', linewidth='3')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'ro', label='Control Points', picker=5)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], '-.', color='red', label='Control Polygon')

    if curv:
        ax = fig.add_subplot(1, 2, 2)
        c = curvature(curve, x)

        ax.plot(x_space, c, color='purple', linewidth='3', label='Curvature')

    plt.show()

def curvature(curve, param_space):
    a = curve.deriv(param_space)
    b = curve.deriv2(param_space)

    curvature = np.linalg.norm(np.cross(a, b, axis=0), axis=0)/np.linalg.norm(a, axis=0)**3

    return curvature

"""
In cantiere, una funzione per calcolare la torsione, necessita di derivate terze
def torsion(curve, param_sapce):

    a = curve.deriv()
    b = curve.deriv2()
"""    

"""
def delta(ctl_points, deg, idx):
    if deg == 0:
        return ctl_points[idx]
    else:
        return delta(ctl_points, deg-1, idx+1) - delta(ctl_points, deg -1, idx)
"""

"""
Funzione d'emergenza per il calcolo della derivata; equivalente a curve.deriv
def ith_derivative(pts, curve, i, param_range):
    basis = curve.basis()
    degree = curve.degree()

    sum = 0
    for j in range(degree - i + 1):
        sum += np.matmul(delta(pts, i, j).reshape(2, 1), basis.evalSingle(j, param_range))
    return (math.factorial(degree)/math.factorial(degree-i)) * sum 
"""

"""
# Funzione d'emergenza per il calcolo della derivata prima; equivalente a curve.deriv
def st_derivative(points, curve, param):
    basis = curve.basis()
    
    sum = 0
    for i in range(2):
        sum += np.matmul((points[i+1, :] - points[i, :]).reshape(2, 1), basis.evalSingle(i, param))
    return 2 * sum 
"""

if __name__ == '__main__':
    #Select points interactively
    control_points, degree = select_points()

    # Initialize the editor
    editor = CurveEditor(control_points, degree)
    editor.linear_precision()

    bezier_3d(5, curv=True)
