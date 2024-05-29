## Import libraires
import jax
import optax
from jax.scipy.special import logsumexp
from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv
import Elements as EL
import Material as mat
from Functions import *
from jax import random
import numpy as np
from tqdm import tqdm
from jax import config
import time
# config.update("jax_enable_x32", True)
key = random.key(0)


#Mesh SET

mesh_file = 'Testing/Meshes/cubo_10_x_10.msh'
mesh = pv.read(mesh_file)  #Orgiinal File
mesh.clear_data()

points_total,connectivity_total,bc_drichlet_cells,bc_drichlet_nodes,bc_neumann_cells,bc_neumann_nodes = Hex_Reader(mesh_file,plot = False)



constant = [0.03,3.77]
material = mat.Delphino(constant,100)
malla = EL.Hexs(material, points_total,connectivity_total)

#Boundary_conditions

X0_index = jnp.where(points_total[:, 0] == 0)[0]
Y0_index = jnp.where(points_total[:, 1] == 0)[0]
Z0_index = jnp.where(points_total[:, 2] == 0)[0]
X1_index = jnp.where(points_total[:, 0] == 1)[0]


### Optimization 
@jit
def loss_function(disp):
    constant = material.constants
    energy_internal = malla.PSI(disp,constant)
    return energy_internal  

@jit
def u(disp,imp):
    disp = disp.at[X0_index,0].set(0)
    disp = disp.at[Y0_index,1].set(0)
    disp = disp.at[Z0_index,2].set(0)
    disp = disp.at[X1_index,0].set(imp)
    return disp


@jit
def Jacobian(disp):
    J = jax.jacrev(loss_function)(disp)
    return J




disp_0  = jnp.zeros_like(points_total)
print('Initial Energy', loss_function(disp_0))


start_learning_rate = 1e-2
optimizer = optax.adam(start_learning_rate)
disp_Adam = u(disp_0,0.01)
opt_state = optimizer.init(disp_Adam)


mesh_def = mesh.copy()
mesh_def.save('cube_views/Cubo10x10_'+str(0)  + '.vtu')

Energys_Adam  = []

inicio = time.time()
imp_final = 1
disps = jnp.linspace(0.01,imp_final,40)

for it, desp in enumerate(disps):
  for i in tqdm(range(250)):
    grads = Jacobian(disp_Adam)
    updates, opt_state = optimizer.update(grads, opt_state)
    disp_Adam = optax.apply_updates(disp_Adam, updates)
    disp_Adam = u(disp_Adam,desp)
    loss_value = loss_function(disp_Adam)
    Energys_Adam.append(loss_value)


  print('Desplazamientos: ', desp,'Energia Potencial',loss_value)
  mesh_def.points = mesh.points+ np.array(disp_Adam)
  name = 'cube_views/Cubo10x10_'+str(it+1)  + '.vtu'
  mesh_def.save(name)

fin = time.time()
print('Execution time',fin-inicio)
print('Adam Energy Result',loss_function(disp_Adam)) 

disp_FEM  = np.load('Testing/Displacement_Tests/Cube_10x10.npy')[-1,:,:]
L2_norm = np.linalg.norm(disp_Adam- disp_FEM, axis = 1)
print('L2 Norm with FEM result: ', np.linalg.norm(disp_Adam- disp_FEM))


plt.plot(Energys_Adam, label = 'Loss function Adam Optimizer')
plt.title('Potential Energy Optimization')
plt.ylabel('Potential Energy')
plt.xlabel('Number Iterations')
plt.yscale('log')
plt.grid()
plt.savefig('Resultados_cubo/Loss_plot.pdf')
plt.show()

mesh_def['Displacement'] = disp_Adam
mesh_def['L2 Norm Displacement'] = L2_norm 

plt.hist(L2_norm)
plt.title('Histograma error nodal')
plt.ylabel('Numero de Nodos')
plt.xlabel('Normal L2')
plt.savefig('Resultados_cubo/Histograma_error_Cubo.pdf')


mesh_def.plot(show_edges=True)
mesh_def.save('Resultados_cubo/Malla_final.vtk')
