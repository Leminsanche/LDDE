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


initial_variable = jnp.array([0.01,4])


solver = Sol.optimizers(loss_function, Jacobian, 1e-6)

params, path = solver.CMAES(initial_variable, lower_bounds= [0.005 , 1], upper_bounds= [0.05, 5])

print(params)

# Convert path to a numpy array for easier plotting
path = jnp.array(path)


# Create a grid for plotting the loss function surface
x = jnp.linspace(params[0]/2, params[0]*1.5, 200)  # Adjust range based on your problem
y = jnp.linspace(params[1]/2, params[1]*1.5, 200)  # Adjust range based on your problem
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
ax.scatter(path_x, path_y, path_z, color='r', marker='o', label='Population Evaluations')
ax.set_xlabel('Variable 1')
ax.set_ylabel('Variable 2')
ax.set_zlabel('Loss Function')
ax.legend()

plt.show()
