import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv
import Elements as EL
import Material as mat
from Functions import *
from jax import random
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

key = random.key(0)



## import mesh
mesh_file = 'Testing/Meshes/Biaxial_COARSE.msh'
mesh = pv.read(mesh_file)  #Orgiinal File
mesh.clear_data()

points_total,connectivity_total,bc_drichlet_cells,bc_drichlet_nodes,bc_neumann_cells,bc_neumann_nodes = Hex_Reader(mesh_file,plot = False)
#####


## Till the moment Only Works with perpendicular directions
# bc_drichlet_cells["X_1"] = [bc_drichlet_cells["X_1"], jnp.array([1,0,0]) ]
# bc_neumann_cells["X_1"] = [bc_neumann_cells["X_1"], jnp.array([1,0,0]) ]
# bc_neumann_cells['Extremo_Sup'] = [bc_neumann_cells['Extremo_Sup'], jnp.array([0,0,1]) ]


## displacement from FEM software (Vulcan)
disp_file  ='/home/nicolas/Escritorio/Low-Dimension-Deep-Energy/Testing/Displacement_Tests/Biaxial_Coarse.npy'
disp = np.load(disp_file)[-1,:,:]  # The order of this file is [time, node, dim]



mesh_def = mesh.copy()
mesh_def.points += disp



constant = [30e-3,3.77]
material = mat.Delphino(constant,100)

malla = EL.Hexs(material, points_total,connectivity_total)



#### Stress Tensor Test
Tensor_Cauchy = Result_Tensor(malla.Cauchy(disp),malla.nodes_repeated)
# print('Cauchy Tensor: \n',Tensor_Cauchy.shape)


##### Internal Forces ######

Internal_Forces = Result_Vector(malla.Internal_Force(disp),malla.nodes_repeated)
# print('Internal Forces: \n',Internal_Forces.shape)


mesh_def['Tensor de Cauchy'] = Tensor_Cauchy
mesh_def['displacement'] = disp
mesh_def['Internal Force'] = Internal_Forces

save_name = disp_file.split('/')[-1].split('.')[0] + '.vtk'
mesh_def.save(save_name)
# mesh_def['Tensor de Cauchy ZZ'] = Tensor_Cauchy[0,:,2,2]
# mesh_def.set_active_scalars('Tensor de Cauchy ZZ')


##### Volumetric Energy Test #####
print('Energy Initial State', malla.PSI(jnp.zeros_like(disp),constant))

print('Energy Final State',malla.PSI(disp,constant))
# print('Energy Final State',malla.PSI(disp).dtype)

#mesh_def.plot()