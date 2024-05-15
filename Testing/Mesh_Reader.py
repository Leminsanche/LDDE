from Functions import *
import jax.numpy as jnp
# File to test mesh reader function
# The goal it's read de .msh file extract boundary conditions cells and points 
# and select between dritchlet, neuman or both 
# Here are tow case cube.msh is a cube mesh with one element and biaxial2.msh who is a complex mesh

file = 'Testing/Meshes/Biaxial2.msh'
#file = 'Testing/unit_cube.msh'

# points_total,connectivity_total,bc_drichlet_cells,bc_neumann_cells = Hex_Reader(file, 
#                                                                                      drichlet_bc= ["Extremo_Inf","Sym_x","Sym_y"], neumann_bc = ["Extremo_Sup"]
#                                                                                     , plot = True)

points_total,connectivity_total,bc_drichlet_cells,bc_neumann_cells = Hex_Reader(file, 
                                                                                     drichlet_bc= ["SymY","SymX","SymZ"], 
                                                                                     neumann_bc = ["agarreY","agarreX"]
                                                                                     , plot = True)


######## Definition Boundary condition #########
# for key in bc_drichlet_cells:
#     print(key, type(bc_drichlet_cells[key]))

jnp.append(bc_drichlet_cells['SymY'],jnp.array([0,1,0])) #Direction of the dritchlet condition 
# bc_drichlet_cells['SymX'], jnp.array([1,0,0])
# bc_drichlet_cells['SymZ'], jnp.array([0,0,1])

bc_drichlet_cells['SymY'] = [bc_drichlet_cells['SymY'],jnp.array([0,1,0]) ]  ## Cells Direction Boundary
bc_drichlet_cells['SymX'] = [bc_drichlet_cells['SymY'],jnp.array([1,0,0]) ]  ## Cells Direction Boundary
bc_drichlet_cells['SymZ'] = [bc_drichlet_cells['SymY'],jnp.array([0,0,1]) ]  ## Cells Direction Boundary


bc_neumann_cells['agarreY'] = [bc_neumann_cells['agarreY'],jnp.array([0,0,1]) ] ## Cells, Force
bc_neumann_cells['agarreX'] = [bc_neumann_cells['agarreX'],jnp.array([0,0,1]) ] ## Celld, Force


