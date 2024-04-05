import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv
import Elements as EL
import Material as mat
from jax import random
import numpy as np
key = random.key(0)



## import mesh

mesh = pv.read('/home/nicolas/Escritorio/Codigos/JAX/Unit_Tests/unit_cube.msh')  #Orgiinal File
mesh.clear_data()

## displacement from FEM software (Vulcan)
disp = np.load('/home/nicolas/Escritorio/Codigos/JAX/Unit_Tests/cube_displacements.npy')[-1,:,:]  # The orden of this file is [time, node, dim]
mesh_def = mesh.copy()
mesh_def.points += disp

pl = pv.Plotter(shape=(1, 2))


pl.subplot(0, 0)
pl.add_text("Original State", font_size=30)
pl.add_mesh(mesh, show_edges=True, color='lightblue',opacity = 1)


pl.subplot(0, 1)
pl.add_text("Deformed State", font_size=30)
pl.add_mesh(mesh_def, show_edges=True, color='lightblue')


# Display the window
#pl.show()



material = mat.Delphino_incompressible([30.0E-3,3.77,0])

malla = EL.energy(material, mesh.points, mesh.cells_dict[12])
print('Energy Initial State', malla.PSI(mesh.points))

print('Energy Final State',malla.PSI(mesh_def.points))
