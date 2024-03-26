import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

# Step 1: Read CSV file
file_path = "mnyo_ground_truth.csv"
data = pd.read_csv(file_path, header=None, delimiter=' ', names=['x', 'y'])

file_path_mediciones = "mnyo_mediciones.csv"
data_mediciones = pd.read_csv(file_path_mediciones, header=None, delimiter=' ', names=['x', 'y'])

# Read CSV file
file_path_mediciones2 = "mnyo_mediciones2.csv"
data_mediciones2 = pd.read_csv(file_path_mediciones2, header=None, delimiter=' ', names=['x', 'y'])

# Calculate cumulative distance as 't'
data_mediciones['t'] = np.arange(1, len(data_mediciones) + 1)

# Calculate cumulative distance as 't'
data_mediciones2['t'] = np.arange(1, len(data_mediciones2) + 1)

# Interpolate using cubic spline with respect to 't'
cs_x = CubicSpline(data_mediciones['t'], data_mediciones['x'])
cs_y = CubicSpline(data_mediciones['t'], data_mediciones['y'])

cs_x2 = CubicSpline(data_mediciones2['t'], data_mediciones2['x'])
cs_y2 = CubicSpline(data_mediciones2['t'], data_mediciones2['y'])

# Generate new t-values for smoother plot
t_new = np.linspace(data_mediciones['t'].min(), data_mediciones['t'].max(), 1000)
t_new2 = np.linspace(data_mediciones2['t'].min(), data_mediciones2['t'].max(), 1000)

# Obtain interpolated x and y values for the new t-values
x_new = cs_x(t_new)
y_new = cs_y(t_new)

x_new2 = cs_x2(t_new2)
y_new2 = cs_y2(t_new2)



# Step 2: Plot the trajectory
plt.figure(figsize=(8, 6))
plt.plot(data['x'], data['y'], linestyle='--')
plt.plot(x_new, y_new, 'r-', label='Cubic Splines Trajectory 1', linestyle='-')
plt.plot(x_new2, y_new2, 'g-', label='Cubic Splines Trajectory 2', linestyle='-')
plt.title('Trajectory Plot')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()


