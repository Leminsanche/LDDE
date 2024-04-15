import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv
import Elements as EL
import Material as mat
from jax import random
import numpy as np
from jax import config
config.update("jax_enable_x64", True)


key = random.key(0)



## import mesh

mesh = pv.read('Testing/unit_cube.msh')  #Orgiinal File
mesh.clear_data()

## displacement from FEM software (Vulcan)
disp = np.load('Testing/Desplazamientos_cubo.npy')[-1,:,:]  # The orden of this file is [time, node, dim]
mesh_def = mesh.copy()
mesh_def.points += disp

# pl = pv.Plotter(shape=(1, 2))


# pl.subplot(0, 0)
# pl.add_text("Original State", font_size=30)
# pl.add_mesh(mesh, show_edges=True, color='lightblue',opacity = 1)


# pl.subplot(0, 1)
# pl.add_text("Deformed State", font_size=30)
# pl.add_mesh(mesh_def, show_edges=True, color='lightblue')


# # # Display the window
# pl.show()



material = mat.Delphino([30.0E-3,3.77,100])

malla = EL.Hexs(material, mesh.points, mesh.cells_dict[12])

Tensor_Cauchy = malla.Cauchy(disp)

# print(np.linalg.eig(Tensor_Cauchy))

mesh_def['Tensor de Cauchy'] = Tensor_Cauchy[0,:,:,:]
mesh_def['displacement'] = disp
mesh_def.save('cubo.vtk')
mesh_def['Tensor de Cauchy ZZ'] = Tensor_Cauchy[0,:,2,2]
mesh_def.set_active_scalars('Tensor de Cauchy ZZ')


print('Energy Initial State', malla.PSI(jnp.zeros_like(disp)))

print('Energy Final State',malla.PSI(disp))
# print('Energy Final State',malla.PSI(disp).dtype)

mesh_def.plot()