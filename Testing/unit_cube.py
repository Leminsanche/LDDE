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
mesh_file = 'Testing/cubo.msh'
mesh = pv.read('Testing/cubo.msh')  #Orgiinal File
mesh.clear_data()

points_total,connectivity_total,bc_drichlet_cells,bc_neumann_cells = Hex_Reader(mesh_file,
                                                                                drichlet_bc= ["X_0"],plot = False)

## Till the moment Only Works with perpendicular directions
# bc_drichlet_cells['Extremo_Sup'] = [bc_drichlet_cells['Extremo_Sup'], jnp.array([0,0,1]) ]
# bc_neumann_cells['Extremo_Sup'] = [bc_neumann_cells['Extremo_Sup'], jnp.array([0,0,1]) ]

## displacement from FEM software (Vulcan)
disp_file  ='Testing/Displacement_Tests/Cubo_Traccion_Z.npy'
disp = np.load(disp_file)[-1,:,:]  # The order of this file is [time, node, dim]



mesh_def = mesh.copy()
mesh_def.points += disp



constant = [0.03,3.77]
material = mat.Delphino(constant,100)

malla = EL.Hexs(material, points_total,connectivity_total)



#### Deformation Gradiente Test
#print(malla.f(mesh_def.points)[-1,-1,:,:])
#f_1 = malla.f(mesh_def.points)[-1,-1,:,:]


#### Stress Tensor Test
Tensor_Cauchy = malla.Cauchy(disp)
#print(Tensor_Cauchy[-1,-1,:,:])


##### Internal Forces ######
Internal_Forces = malla.Internal_Force(disp)


### External Energy Test
# External_energy  =  malla.External_Energy(disp, Dritchlet_BC = bc_drichlet_cells)
# print('External Energy',External_energy)

mesh_def['Tensor de Cauchy'] = Tensor_Cauchy[0,:,:,:]
mesh_def['displacement'] = disp
mesh_def['Internal Force'] = Internal_Forces

save_name = disp_file.split('/')[-1].split('.')[0] + '.vtk'
mesh_def.save(save_name)
# mesh_def['Tensor de Cauchy ZZ'] = Tensor_Cauchy[0,:,2,2]
# mesh_def.set_active_scalars('Tensor de Cauchy ZZ')


print('Energy Initial State', malla.PSI(jnp.zeros_like(disp),constant))

print('Energy Final State',malla.PSI(disp,constant))
# print('Energy Final State',malla.PSI(disp).dtype)

#mesh_def.plot()