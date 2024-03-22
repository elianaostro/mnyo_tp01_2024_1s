import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return 0.3**abs(x) * np.sin(4*x) - np.tanh(2*x) +2

# Generate x values
x_values = np.linspace(-4, 4, 400)

# Generate y values
y_values = f(x_values)

# Plot the graph
plt.plot(x_values, y_values, label='$f(x) = x^2$')

# Add labels and title
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of $f(x) = x^2$')

# Add grid
plt.grid(True)

# Add legend
plt.legend()

# Show the plot
plt.show()