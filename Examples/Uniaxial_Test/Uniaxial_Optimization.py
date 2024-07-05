import sys
import os

# # Agregar el directorio 'src' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))


import Elements as EL
import Material as mat
import Functions as Fun
import Solvers as sol
import pyvista as  pv
import jax.numpy as jnp
import time
import numpy as np 
import jax
import matplotlib.pyplot as plt
import itertools

from jax import config, random, jit
config.update("jax_enable_x64", True)
key = random.key(0)

mesh_file = 'Examples/Uniaxial_Test/cubo_10_x_10.msh'
mesh = pv.read(mesh_file)  #Orgiinal File
mesh.clear_data()

points_total,connectivity_total,bc_drichlet_cells,bc_drichlet_nodes,bc_neumann_cells,bc_neumann_nodes = Fun.Hex_Reader(mesh_file,plot = False)


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

### Initial Displacment
disp_0  = jnp.zeros_like(points_total)
print('Initial Energy', loss_function(disp_0))

mesh_def = mesh.copy()
mesh_def.save('Examples/Uniaxial_Test/Result/Cubo10x10_'+str(0)  + '.vtu')

Energys_Adam  = []

inicio = time.time()
imp_final = 1
disps = jnp.linspace(0.01,imp_final,2)


### Solver ####
solver_adam = sol.optimizers(loss_function, u,Jacobian, 1e-2)

for it, disp in enumerate(disps):

    disp_final, Energy = solver_adam.adam(disp_0)
    disp_0 = u(disp_final,disp)

    Energys_Adam.append(Energy)
    print('Desplazamientos: ', disp,'Energia Potencial',Energy[-1])
    mesh_def.points = mesh.points+ np.array(disp_0)
    name = 'Examples/Uniaxial_Test/Result/Cubo10x10_'+str(it+1)  + '.vtu'
    mesh_def.save(name)

fin = time.time()
print('Execution time',fin-inicio)

###### PLOTS ########
Energy_F = list(itertools.chain.from_iterable(Energys_Adam))

plt.plot(Energy_F, label = 'Loss function Adam Optimizer')
plt.title('Potential Energy Optimization')
plt.ylabel('Potential Energy')
plt.xlabel('Number Iterations')
plt.yscale('log')
plt.grid()
plt.savefig('Examples/Uniaxial_Test/Loss_plot.pdf')
plt.show()

