import matplotlib.pyplot as plt
import bernstein as b
import numpy as np  

# Fuction to plot Bezier curves from control points with control polygon
def plot_bezier(control_pts):
    n = len(control_pts) - 1
    t = np.linspace(0, 1, 101)
    x = [sum(b.bernstein(i, n, t[j]) * control_pts[i][0] for i in range(n + 1)) for j in range(101)]
    y = [sum(b.bernstein(i, n, t[j]) * control_pts[i][1] for i in range(n + 1)) for j in range(101)]
    
    # Plot Bezier curve
    plt.plot(x, y, label='Bezier Curve')
    
    # Plot control polygon
    control_x, control_y = zip(*control_pts)
    plt.plot(control_x, control_y, 'r--', label='Control Polygon')
    plt.scatter(control_x, control_y, color='red')
    
    plt.title('Bezier Curve with Control Polygon')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Example usage of ginput
def select_points():
    plt.figure()
    plt.plot()  # Example plot
    plt.title('Click to select points, press Enter to finish')
    # Use ginput to select points
    points = plt.ginput(n=-1, timeout=0)
    plt.close()
    return points

#function to compute nth derivative of Bezier curve
def bezier_nth_derivative(control_pts, deg, n):
    if n == 0:
        return control_pts
    else:
        n -= 1
        n = len(control_pts) - 1
        for i in range(n-deg):
            # Compute derivative with delta
            control_pts[i] = (n-deg) * (control_pts[i+1] - control_pts[i])
        return bezier_nth_derivative(control_pts, deg, n)

#function to compute the curvature of Bezier curve
def bezier_curvature(control_pts):
    n = len(control_pts) - 1
    t = np.linspace(0, 1, 101)
    x = [sum(b.bernstein(i, n, t[j]) * control_pts[i][0] for i in range(n + 1)) for j in range(101)]
    y = [sum(b.bernstein(i, n, t[j]) * control_pts[i][1] for i in range(n + 1)) for j in range(101)]
    dx = np.gradient(x)
    ddx = np.gradient(dx)
    dy = np.gradient(y)
    ddy = np.gradient(dy)
    curvature = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5
    return curvature

#function to plot the curvature of Bezier curve
def plot_curvature(control_pts):
    curvature = bezier_curvature(control_pts)
    plt.plot(curvature)
    plt.title('Curvature of Bezier Curve')
    plt.xlabel('t')
    plt.ylabel('Curvature')
    plt.show()
    return curvature

#function to compute curvature in extreme points of t parameter (t=0 and t=1)
def curvature_extreme_points(control_pts):
    n = len(control_pts) - 1
    x = [control_pts[i][0] for i in range(n + 1)]
    y = [control_pts[i][1] for i in range(n + 1)]
    dx = np.gradient(x)
    ddx = np.gradient(dx)
    dy = np.gradient(y)
    ddy = np.gradient(dy)
    curvature = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5
    return curvature


#draw the Bezier curve with control polygon
def draw_bezier():
    control_pts = select_points()
    plot_bezier(control_pts)
    k = plot_curvature(control_pts)
    print(k[0], k[-1])

if __name__ == "__main__":
    draw_bezier()
    

