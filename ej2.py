import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

def read_csv(file_path):
    data = pd.read_csv(file_path, header=None, delimiter=' ', names=['x', 'y'])
    return data

def calculate_cumulative_distance(data):
    data['t'] = np.arange(1, len(data) + 1)

def interpolate_cubic_spline(data):
    cs_x = CubicSpline(data['t'], data['x'])
    cs_y = CubicSpline(data['t'], data['y'])
    t_new = np.linspace(data['t'].min(), data['t'].max(), 1000)
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    return x_new, y_new

def newton_raphson_doble_variable(f1, f2, x0, y0, tol=1e-6, max_iter=1000): 
         # p(n-1)=p(n)-(jacobiano*-1)(p(n)) * f(p(n)) 
         # f1, f2: funciones f1(x, y), f2(x, y) 
  
         # Calculo el jacobiano 
         def jacobiano(x, y): 
             j11 = (f1(x + tol, y) - f1(x, y)) / tol 
             j12 = (f1(x, y + tol) - f1(x, y)) / tol 
             j21 = (f2(x + tol, y) - f2(x, y)) / tol 
             j22 = (f2(x, y + tol) - f2(x, y)) / tol 
             return np.array([[j11, j12], [j21, j22]]) 
  
         for _ in range(max_iter): 
             j_inv = np.linalg.inv(jacobiano(x0, y0)) 
             f = np.array([f1(x0, y0), f2(x0, y0)]) 
             p = np.array([x0, y0]) - j_inv @ f 
             if np.linalg.norm(p - np.array([x0, y0])) < tol: 
                 return x0 
             x0, y0 = p 
         return None 
  

def find_intersection_point(interpolated_trajectory_v1_x, interpolated_trajectory_v2_x, interpolated_trajectory_v1_y, interpolated_trajectory_v2_y):
    def f1(x, y): 
         return interpolated_trajectory_v1_x(x) - interpolated_trajectory_v2_x(y) 
  
    def f2(x, y):
         return interpolated_trajectory_v1_y(x) - interpolated_trajectory_v2_y(y) 

    t_intersect = newton_raphson_doble_variable(f1, f2, 0, 0)

    m1_x1_intersect = interpolated_trajectory_v1_x(t_intersect) 
    m1_x2_intersect = interpolated_trajectory_v1_y(t_intersect)

    return m1_x1_intersect, m1_x2_intersect
     

def plot_trajectory(data, x_new, y_new, x_new2, y_new2, m1_x1_intersect, m1_x2_intersect):
    plt.figure(figsize=(8, 6))
    plt.plot(data['x'], data['y'], label='Ground Truth', linestyle='--')
    plt.plot(x_new, y_new, 'r-', label='Trayectoria 1 Spline Cubico', linestyle='-')
    plt.plot(x_new2, y_new2, 'g-', label='Trayectoria 2 Spline Cubico', linestyle='-')
    plt.scatter(m1_x1_intersect, m1_x2_intersect, color='black', label='Punto de Interseccion')
    plt.title('Trayectorias Interpoladas con Splines Cubicos')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()




def main():

    file_path = "mnyo_ground_truth.csv"
    data = pd.read_csv(file_path, header=None, delimiter=' ', names=['x', 'y'])
    file_path_mediciones = "mnyo_mediciones.csv"
    data_mediciones = pd.read_csv(file_path_mediciones, header=None, delimiter=' ', names=['x', 'y'])
    file_path_mediciones2 = "mnyo_mediciones2.csv"
    data_mediciones2 = pd.read_csv(file_path_mediciones2, header=None, delimiter=' ', names=['x', 'y'])

    calculate_cumulative_distance(data_mediciones)
    calculate_cumulative_distance(data_mediciones2)

    x_new, y_new = interpolate_cubic_spline(data_mediciones)
    x_new2, y_new2 = interpolate_cubic_spline(data_mediciones2)

    interpolated_trajectory_v1_x = CubicSpline(data_mediciones.index, data_mediciones["x"])
    interpolated_trajectory_v1_y = CubicSpline(data_mediciones.index, data_mediciones["y"])

    interpolated_trajectory_v2_x = CubicSpline(data_mediciones2.index, data_mediciones2["x"])
    interpolated_trajectory_v2_y = CubicSpline(data_mediciones2.index, data_mediciones2["y"])

    m1_x1_intersect, m1_x2_intersect = find_intersection_point(
                    CubicSpline(data_mediciones.index, data_mediciones["x"]), 
                    CubicSpline(data_mediciones2.index, data_mediciones2["x"]), 
                    CubicSpline(data_mediciones.index, data_mediciones["y"]), 
                    CubicSpline(data_mediciones2.index, data_mediciones2["y"])
                                                              )
    
    plot_trajectory(data, x_new, y_new, x_new2, y_new2, m1_x1_intersect, m1_x2_intersect)

       
if __name__ == '__main__':
    main()