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
config.update("jax_enable_x64", True)
key = random.key(0)


#Mesh SET
mesh_file = '/home/nicolas/Escritorio/Low-Dimension-Deep-Energy/Testing/Meshes/Biaxial_COARSE.msh'
mesh = pv.read(mesh_file)  #Orgiinal File
mesh.clear_data()

points_total,connectivity_total,bc_drichlet_cells,bc_drichlet_nodes,bc_neumann_cells,bc_neumann_nodes = Hex_Reader(mesh_file,plot = False)



constant = [0.03,3.77]
material = mat.Delphino(constant,100)
malla = EL.Hexs(material, points_total,connectivity_total)

#Boundary_conditions
Sym_Y_index = jnp.where(points_total[:, 1] == 0)[0]
Sym_X_index = jnp.where(points_total[:, 0] == 0)[0]
Sym_Z_index = jnp.where(points_total[:, 2] == 0)[0]
agarreY_index = jnp.where(points_total[:, 1] == 20)[0]
agarreX_index = jnp.where(points_total[:, 0] == 20)[0]


Sym_Y_index   =  Sym_Y_index*3 + 1   
Sym_X_index   =  Sym_X_index*3 + 0   
Sym_Z_index   =  Sym_Z_index*3 + 2   
agarreY_index =  agarreY_index*3 + 1
agarreX_index =  agarreX_index*3 + 0

indices_known = jnp.concatenate((Sym_Y_index,Sym_X_index,agarreY_index,agarreX_index))
index_disp_inc = jnp.delete(jnp.arange(points_total.shape[0]*3),indices_known) 

#### Initialization Variables
disp_0  = jnp.zeros_like(points_total).reshape(-1,1)    #Displacement Field column form start from 0
disp_inc = jnp.zeros((index_disp_inc.shape[0],1))   #unknow varibales start form 0


### Optimization  Functions
@jit
def loss_function(disp):
    disp_f = u(disp)
    constant = material.constants
    energy_internal = malla.PSI(disp_f,constant)
    return energy_internal

@jit
def u(disp):
    global disp_0
    disp_0 = disp_0.at[agarreY_index].set(desp)
    disp_0 = disp_0.at[agarreX_index].set(desp)
    disp_0 = disp_0.at[index_disp_inc].set(disp)
    return disp_0.reshape(-1,3)



@jit
def Jacobian(disp):
    J = jax.jacrev(loss_function)(disp)
    return J


# print('Initial Energy', loss_function(disp_inc))   #Start whitout BC


start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)
disp_Adam = disp_inc
opt_state = optimizer.init(disp_Adam)


mesh_def = mesh.copy()
mesh_def.save('Biaxial_views/Biaxial_'+str(0)  + '.vtu')  ### Initial condition
Energys_Adam  = []


inicio = time.time()
imp_final = 5
disps = jnp.linspace(0.001,imp_final,2)




for it, desp in enumerate(disps):






  # for i in tqdm(range(150)):
  #   grads = Jacobian(disp_Adam)
  #   updates, opt_state = optimizer.update(grads, opt_state)
  #   disp_Adam = optax.apply_updates(disp_Adam, updates)
  #   loss_value = loss_function(disp_Adam,desp)
  #   Energys_Adam.append(loss_value)


  # opt_state = optimizer.init(disp_Adam)
#   print('Desplazamientos: ', desp,'Energia Potencial',loss_value)
    mesh_def.points = mesh.points+ np.array(disp_0.reshape(-1,3))
    name = 'Biaxial_views/Biaxial_'+str(it+1)  + '.vtu'
    mesh_def.save(name)

fin = time.time()
print('Execution time',fin-inicio)
print('Adam Energy Result',loss_function(disp_Adam)) 
