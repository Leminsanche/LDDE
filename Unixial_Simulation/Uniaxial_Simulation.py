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
from jax import config
config.update("jax_enable_x64", True)

## Mesh SET
mesh_file = 'Unixial_Simulation/input/cubo_10_x_10.msh'
mesh = pv.read(mesh_file)  #Orgiinal File
mesh.clear_data()
mesh_info = Fun.Hex_Reader(mesh_file,drichlet_bc=['X_0','Y_0','Z_0','X_1'],plot = False)

points_total,connectivity_total,bc_drichlet_cells,bc_drichlet_nodes,bc_neumann_cells,bc_neumann_nodes = mesh_info


## Material SET
constant = [0.03,3.77] #Random Values to initialize  ## Constantes originales [0.03,3.77]
material = Mat.Delphino(constant,100) ## 100 == penalty for incompresibility  
malla = EL.Hexs(material, points_total,connectivity_total)


####
disp_real = np.load('Unixial_Simulation/input/cubo_10_disp.npy')[-1,:,:]
print('Valor FEM',malla.PSI(disp_real,constant))
####

# ## Boundary Conditions 
# X0_index = bc_drichlet_nodes['X_0']
# Y0_index = bc_drichlet_nodes['Y_0']
# Z0_index = bc_drichlet_nodes['Z_0']
# X1_index = bc_drichlet_nodes['X_1']


### Optimization Problem ###
class Problem():
    def __init__(self,imp, mesh_problem, material_problem, dritchlet_nodes):
        self.imposed_displacement = imp
        self.malla = mesh_problem
        self.material = material_problem
        self.dritchlet_nodes = dritchlet_nodes


    #@jit
    def loss_function(self,displacement):
        constantes = self.material.constants
        displacement = self.u(displacement, self.imposed_displacement)
        energy_internal = self.malla.PSI(displacement,constantes)
        return energy_internal 

    #@jit
    def u(self,disp,imp):
        disp = disp.at[self.dritchlet_nodes['X_0'],0].set(0.0)
        disp = disp.at[self.dritchlet_nodes['Y_0'],1].set(0.0)
        disp = disp.at[self.dritchlet_nodes['Z_0'],2].set(0.0)
        disp = disp.at[self.dritchlet_nodes['X_1'],0].set(imp)
        return disp

    #@jit
    def Jacobian(self,displacement):
        J = jax.jacrev(self.loss_function)(displacement)
        return J

mesh_def = mesh.copy()
mesh_def.save('Unixial_Simulation/Result/Cubo10x10_'+str(0)+'.vtu')
disp_0  = jnp.zeros_like(points_total)


Our_Problem = Problem(0.0, malla, material, bc_drichlet_nodes)
print('Initial energy',Our_Problem.loss_function(disp_0))
displacmenet_steps = jnp.linspace(0.0,1.0,20)[1:] 

for it, step in enumerate(displacmenet_steps):
    Our_Problem = Problem(step, malla, material, bc_drichlet_nodes)

    jacobian = jax.jit(Our_Problem.Jacobian)
    loss = jax.jit(Our_Problem.loss_function)
    solver = Sol.optimizers(loss, jacobian, 1e-3)

    params , loss_values, path = solver.adam(disp_0,4000,tolerancia= 1e-8)
    #params , loss_values, path = solver.LBFGS(disp_0,4000, tolerancia = 1e-3)
    disp_0 = params

    print('Displacement: ', step, '   Energy: ',loss_values[-1])

    mesh_def.points = mesh.points+ np.array(Our_Problem.u(disp_0,step))
    name = 'Unixial_Simulation/Result/Cubo10x10_'+str(it+1)  + '.vtu'
    mesh_def.save(name)







            
