import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Function that implements de Casteljau algorithm to compute a point on a Bezier curve
def deCasteljau(control_pts, t):
    n = len(control_pts) - 1
    points = np.array(control_pts, dtype=float)
    for k in range(n):
        for i in range(n - k):
            points[i] = (1 - t) * points[i] + t * points[i + 1]
    return points[0]

# Example usage of ginput
def select_points():
    plt.figure()
    plt.plot()  # Example plot
    plt.title('Click to select points, press Enter to finish')
    # Use ginput to select points
    points = plt.ginput(n=-1, timeout=0)
    plt.close()
    return points

def compute_segments(control_pts, t):
    segments = []
    points = control_pts
    while len(points) > 1:
        new_points = []
        for i in range(len(points) - 1):
            x = (1 - t) * points[i][0] + t * points[i + 1][0]
            y = (1 - t) * points[i][1] + t * points[i + 1][1]
            new_points.append((x, y))
        segments.append(new_points)
        points = new_points
    return segments

# Function to plot the Bezier curve and control polygon
def plot_bezier(control_pts):
    n = len(control_pts) - 1
    x = [control_pts[i][0] for i in range(n + 1)]
    y = [control_pts[i][1] for i in range(n + 1)]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    control_polygon, = plt.plot(x, y, 'ro-', label='Control Polygon')
    de_casteljau_point, = plt.plot([], [], 'bo', label='de Casteljau Point')
    de_casteljau_segments, = plt.plot([], [], 'g--', label='de Casteljau Segments')

    # Plot the Bezier curve
    t_values = np.linspace(0, 1, 100)
    bezier_points = [deCasteljau(control_pts, t) for t in t_values]
    bezier_x = [p[0] for p in bezier_points]
    bezier_y = [p[1] for p in bezier_points]
    bezier_line, = plt.plot(bezier_x, bezier_y, label='Bezier Curve')

    plt.title('Bezier Curve with de Casteljau Algorithm')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # Slider for t
    ax_t = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    t_slider = Slider(ax_t, 't', 0.0, 1.0, valinit=0.0)

    # Update function for the slider
    def update(val):
        t = t_slider.val
        point = deCasteljau(control_pts, t)
        de_casteljau_point.set_data(point[0], point[1])

        segments = compute_segments(control_pts, t)
        segment_x = []
        segment_y = []
        for segment in segments:
            segment_x.extend([p[0] for p in segment] + [None])
            segment_y.extend([p[1] for p in segment] + [None])
        de_casteljau_segments.set_data(segment_x, segment_y)

        fig.canvas.draw_idle()

    t_slider.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    control_points = select_points()
    plot_bezier(control_points)