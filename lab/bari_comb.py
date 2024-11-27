import numpy as np
import matplotlib.pyplot as plt

def barycentric_coords(A, B, C, P):
    # Vertices of the triangle
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    P = np.array(P)

    # Calculate the area of the triangle using cross product
    # This is a scalar in 2D
    area_ABC = np.cross(B - A, C - A)

    # Barycentric coordinates lambda_1, lambda_2, lambda_3
    lambda_1 = np.cross(B - P, C - P) / area_ABC
    lambda_2 = np.cross(C - P, A - P) / area_ABC
    lambda_3 = np.cross(A - P, B - P) / area_ABC

    return lambda_1, lambda_2, lambda_3

# Define the triangle vertices
A = np.array([0, 0])
B = np.array([1, 1])
C = np.array([0.5, 2])

# Define the point inside the triangle
P = np.array([0.2, 0.3])

l1, l2, l3 = barycentric_coords(A, B, C, P)

print('Barycentric coordinates of P:', l1, l2, l3)

""" # Plot the triangle
plt.figure(figsize=(6,6))
triangle = plt.Polygon([A, B, C], fill=None, edgecolor='b', linewidth=2)
plt.gca().add_patch(triangle)

# Plot vertices
plt.plot(*A, 'ro')  # Vertex A
plt.text(A[0], A[1], '  A', fontsize=12)
plt.plot(*B, 'ro')  # Vertex B
plt.text(B[0], B[1], '  B', fontsize=12)
plt.plot(*C, 'ro')  # Vertex C
plt.text(C[0], C[1], '  C', fontsize=12)

# Plot point P
plt.plot(*P, 'go')  # Point P
plt.text(P[0], P[1], '  P', fontsize=12)

# Set limits and show plot
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.title('Triangle with Point P')
plt.show() """
