# Author: Nicolas Sanchez
# Characterization via Energy Minimization
# Test: Uniaxial test

import os,sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.Elements as EL 
import src.Material as Mat
import src.Functions as Fun
import src.Solvers as Sol

import jax.numpy as jnp
from jax import jit
import numpy as np
import pyvista as pv
import jax
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import optax
import jaxopt
import plotly.graph_objs as go
import plotly.io as pio

## Mesh SET
mesh_file = 'Uniaxial_Caracterization/input/cubo_10_x_10.msh'
mesh = pv.read(mesh_file)  #Orgiinal File
mesh.clear_data()
points_total,connectivity_total,bc_drichlet_cells,bc_drichlet_nodes,bc_neumann_cells,bc_neumann_nodes = Fun.Hex_Reader(mesh_file,plot = False)


## Material SET
constant = [0.03,3.77]  ## Constantes originales
material = Mat.Delphino(constant,100) ## 100 == penalty for incompresibility  
malla = EL.Hexs(material, points_total,connectivity_total)

## Displacement SET
displacement = np.load('Uniaxial_Caracterization/input/Cube_10x10.npy')[-1,:,:]

### Optimization 
@jit
def loss_function(params):
    global displacement
    energy_internal = malla.PSI(displacement,params)
    return jnp.abs(energy_internal  - 0.331)

@jit
def Jacobian(params):
    J = jax.jacrev(loss_function)(params)
    return J


Energy = malla.PSI(displacement, constant)


initial_variable = jnp.array([0.025,3.8])


solver = Sol.optimizers(loss_function, Jacobian, 1e-3)

params , loss_values, path = solver.LBFGS(initial_variable,4000, tolerancia = 1e-16)
#params , loss_values, path = solver.adam(initial_variable,4000,tolerancia= 1e-8)


# Convert path to a numpy array for easier plotting
path = jnp.array(path)


print(params)
# Create a grid for plotting the loss function surface
x = jnp.linspace(params[0]/2, params[0]*1.5, 400)  # Adjust range based on your problem
y = jnp.linspace(params[1]/2, params[1]*1.5, 400)  # Adjust range based on your problem
X, Y = jnp.meshgrid(x, y)
Z = jnp.array([[loss_function((xi, yi)) for xi in x] for yi in y])

# Plot the surface and the optimization path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.8, cmap='inferno')

# Extract the x, y, z coordinates of the path
path_x = path[:, 0]
path_y = path[:, 1]
path_z = jnp.array([loss_function(params) for params in path])

# Plot the optimization path
ax.plot(path_x[-1], path_y[-1], path_z[-1], color='b', marker='o', markersize=8)
ax.plot(path_x, path_y, path_z, color='r', marker='o', markersize=3, label='Optimization Path')
ax.set_xlabel('Variable 1')
ax.set_ylabel('Variable 2')
ax.set_zlabel('Loss Function')
ax.legend()

plt.show()


plt.plot(loss_values)
plt.grid()
# plt.yscale('log')
plt.show()

# Create the plotly 3D surface plot
surface = go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.8)
path_trace = go.Scatter3d(x=path_x, y=path_y, z=path_z, mode='lines+markers', line=dict(color='red', width=5), marker=dict(size=5))
optimum_point = go.Scatter3d(x=[path_x[-1]], y=[path_y[-1]], z=[path_z[-1]], mode='markers', line=dict(color='blue', width=7), marker=dict(size=9))


# Define the layout
layout = go.Layout(
    title='Optimization Path on Loss Function Surface',
    scene=dict(
        xaxis=dict(title='X', showgrid=True),
        yaxis=dict(title='Y', showgrid=True),
        zaxis=dict(title='Loss Function', showgrid=True)
    ),
    showlegend=False
)

# Create the figure
fig = go.Figure(data=[surface, path_trace,optimum_point], layout=layout)

# Save the interactive plot as an HTML file
pio.write_html(fig, 'interactive_optimization_path.html')

# To open the plot later, you can simply open the 'interactive_optimization_path.html' file in a web browser.


            
