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
config.update("jax_enable_x64", True)
key = random.key(0)



#Mesh SET

mesh_file = 'Testing/Meshes/cubo.msh'
mesh = pv.read(mesh_file)  #Orgiinal File
mesh.clear_data()

points_total,connectivity_total,bc_drichlet_cells,bc_drichlet_nodes,bc_neumann_cells,bc_neumann_nodes = Hex_Reader(mesh_file,
                                                                                                        neumann_bc=["X_1"],plot = False)


#Neumann condition 
bc_neumann_cells['X_1'] = [bc_neumann_cells['X_1'], jnp.array([4.2/200,0,0]) ]


#Dritchlet condition 
X_0_index = jnp.where(points_total[:, 0] == 0)[0]
Y_0_index = jnp.where(points_total[:, 1] == 0)[0]
Z_0_index = jnp.where(points_total[:, 2] == 0)[0]


X_0_index = X_0_index*3 + 0 
Y_0_index = Y_0_index*3 + 1
Z_0_index = Z_0_index*3 + 2 

indices_known = jnp.concatenate((X_0_index,Y_0_index,Z_0_index))
index_disp_inc = jnp.delete(jnp.arange(points_total.shape[0]*3),indices_known) 


disp_0  = jnp.zeros_like(points_total).reshape((-1,1))
disp_0 = disp_0.at[X_0_index].set(0)
disp_0 = disp_0.at[Y_0_index].set(0)
disp_0 = disp_0.at[Z_0_index].set(0)



disp_inc = jnp.zeros((index_disp_inc.shape[0],1))

@jit
def u(disp_in):
    global disp_0
    disp_0 = disp_0.at[index_disp_inc].set(disp_in)
    return disp_0.reshape((-1,3))


@jit
def loss_function(disp_in):
    disp = u(disp_in)
    constant = material.constants
    energia_externa = malla.External_Energy(disp, Neumann_BC= bc_neumann_cells)
    energy_internal = malla.PSI(disp,constant)
    E_pot = energy_internal - energia_externa 

    return E_pot  




@jit
def Jacobian(disp):
    J = jax.jacrev(loss_function)(disp)
    #J = jax.grad(loss_2)(disp)
    
    return J



constant = [0.03,3.77]
material = mat.Delphino(constant,100)
malla = EL.Hexs(material, points_total,connectivity_total)


start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)

disp_Adam = disp_inc

opt_state = optimizer.init(disp_Adam)
Energys_Adam  = []
for _ in tqdm(range(2000)):
  grads = Jacobian(disp_Adam)
  updates, opt_state = optimizer.update(grads, opt_state)
  disp_Adam = optax.apply_updates(disp_Adam, updates)
  Energys_Adam.append(loss_function(disp_Adam))

  # print(_)
  
print('Adam Energy Result',loss_function(disp_Adam)) 
print(u(disp_Adam))
# print('Energia externa: ',malla.External_Energy(u(disp_Adam), Neumann_BC= bc_neumann_cells))
# print('Energia interna: ',malla.PSI(u(disp_Adam),constant))



change_state_plot(mesh,u(disp_Adam))