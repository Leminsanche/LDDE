from Functions import *
import jax.numpy as jnp
# File to test mesh reader function
# The goal it's read de .msh file extract boundary conditions cells and points 
# and select between dritchlet, neuman or both 
# Here are tow case cube.msh is a cube mesh with one element and biaxial2.msh who is a complex mesh

# file = 'Testing/Meshes/Biaxial2.msh'
# file = 'Testing/Meshes/cubo.msh'
file = 'Testing/Meshes/cube_2x2.msh'

points_total,connectivity_total,bc_drichlet_cells,bc_drichlet_nodes,bc_neumann_cells,bc_neumann_nodes = Hex_Reader(file, 
                                                                                               drichlet_bc= ["X_0","Y_0","Z_0"], 
                                                                                               neumann_bc = ["X_1"]
                                                                                               , plot = True)

# points_total,connectivity_total,bc_drichlet_cells,bc_neumann_cells = Hex_Reader(file, 
#                                                                                      drichlet_bc= ["SymY","SymX","SymZ"], 
#                                                                                      neumann_bc = ["agarreY","agarreX"]
#                                                                                      , plot = True)

print(bc_drichlet_cells,'\n \n',bc_drichlet_nodes)
######## Definition Boundary condition #########
# for key in bc_drichlet_cells:
#     print(key, type(bc_drichlet_cells[key]))

# jnp.append(bc_drichlet_cells['SymY'],jnp.array([0,1,0])) #Direction of the dritchlet condition 
# # bc_drichlet_cells['SymX'], jnp.array([1,0,0])
# # bc_drichlet_cells['SymZ'], jnp.array([0,0,1])

# bc_drichlet_cells['SymY'] = [bc_drichlet_cells['SymY'],jnp.array([0,1,0]) ]  ## Cells Direction Boundary
# bc_drichlet_cells['SymX'] = [bc_drichlet_cells['SymY'],jnp.array([1,0,0]) ]  ## Cells Direction Boundary
# bc_drichlet_cells['SymZ'] = [bc_drichlet_cells['SymY'],jnp.array([0,0,1]) ]  ## Cells Direction Boundary


# bc_neumann_cells['agarreY'] = [bc_neumann_cells['agarreY'],jnp.array([0,0,1]) ] ## Cells, Force
# bc_neumann_cells['agarreX'] = [bc_neumann_cells['agarreX'],jnp.array([0,0,1]) ] ## Celld, Force


