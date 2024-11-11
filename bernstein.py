import numpy as np
import matplotlib.pyplot as plt

#factorial fiunction
def factorial(n):  
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


#binomial coefficient function
def binomial(n, i):
    return factorial(n) / (factorial(i) * factorial(n-i))

#bernstein polynomial function
def bernstein(i, n, t):
    return binomial(n, i) * (t ** i) * ((1 - t) ** (n - i))

# Plotting the Bernstein polynomials
def plot_bernstein(n):
    t = np.linspace(0, 1, 100)
    for i in range(n + 1):
        plt.plot(t, bernstein(i, n, t), label=f'B_{i},{n}(t)')
    plt.title(f'Bernstein Polynomials of degree {n}')
    plt.xlabel('t')
    plt.ylabel('B_{i,n}(t)')
    plt.legend()
    plt.show()

# Function to evaluate the Bernstein polynomial at a point t
def evaluate_bernstein(n, t):
    return np.array([bernstein(i, n, t) for i in range(n + 1)])

# Example usage
if __name__ == "__main__":
    plot_bernstein(10)
    check = True
    for i in np.linspace(0, 1, 101):
        if not np.isclose(np.sum(evaluate_bernstein(10, i)), 1):
            check = False
            break
        
    print(f"Sum of Bernstein polynomials is 1: {check}")