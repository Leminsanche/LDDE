import meshio
import pyvista as pv
import numpy as np
import jax.numpy as jnp
from jax import jit
def Hex_Reader(mesh_file, drichlet_bc= [], neumann_bc = [], plot = False):
    """
    Function to read gmsh file and extract boundary contidions
    cells and connectivity

    WORKS ONLY FOR HEX MESH WITH BOUNDARIES WITH QUAD ELEMENTS
    """
    mesh = meshio.read(mesh_file)
    element_type = mesh.cells_dict    ## Dict with element information {'type element': array([connectivity]) }
    physical_names = mesh.field_data  ## Dict with physical names {'physical Name: array([Number Etiquete, Dimension])'}
    physical_data_bc = mesh.get_cell_data("gmsh:physical", 'quad') ## Extract only 2D elements

    #bc_drichlet_cells,bc_neumann_cells = None, None #Initializing variable

    ### Mesh total nodes and conectivity
    connectivity_total = mesh.cells_dict['hexahedron']
    points_total = mesh.points

    if drichlet_bc is not None:
        bc_drichlet_cells = {}
        bc_drichlet_nodes = {}
        for i in drichlet_bc:
            physical_tag = physical_names[i][0]
            element_index = np.where(physical_data_bc == physical_tag)[0]#[0]
            #print(element_index)
            unique_nodes = np.unique(element_type['quad'][element_index])
            bc_drichlet_cells[i] = element_type['quad'][element_index]#element_index
            bc_drichlet_nodes[i] = unique_nodes#element_index

    #print(bc_drichlet_cells)

    if neumann_bc is not None:
        bc_neumann_cells = {}
        bc_neumann_nodes = {}
        for i in neumann_bc:
            physical_tag = physical_names[i][0]
            element_index = np.where(physical_data_bc == physical_tag)[0]#[0]
            unique_nodes = np.unique(element_type['quad'][element_index])
            bc_neumann_cells[i] = element_type['quad'][element_index]#element_index
            bc_neumann_nodes[i] = unique_nodes




    if plot == True:


        plotter = pv.Plotter()
        malla_pv = pv.read(mesh_file)
        malla_pv.clear_data()
        plotter.add_mesh(malla_pv,color = 'cyan', show_edges=True,opacity = 1) 

        for i in bc_drichlet_cells:
            aux = 4* np.ones(bc_drichlet_cells[i].shape[0])
            aux = aux.reshape(-1,1).astype(int)
            conectivity = bc_drichlet_cells[i]#element_type['quad'][bc_drichlet_cells[i]]
            cells = np.array(np.hstack((aux,conectivity))) ## Conectivity pyvista format
            grid = pv.UnstructuredGrid(cells, np.array([pv.CellType.QUAD for i in range(aux.shape[0])]), points_total)
            #plotter.add_mesh(grid,color=np.random.rand(3), show_edges = True) 
            plotter.add_point_labels(grid.cell_centers().points[0],['Dritchlet BC ' + i], always_visible=True, font_size=20)

        for i in bc_neumann_cells:
            aux = 4* np.ones(bc_neumann_cells[i].shape[0])
            aux = aux.reshape(-1,1).astype(int)
            conectivity = bc_neumann_cells[i]#element_type['quad'][bc_neumann_cells[i]]
            cells = np.array(np.hstack((aux,conectivity))) ## Conectivity pyvista format
            grid = pv.UnstructuredGrid(cells,np.array([pv.CellType.QUAD for i in range(aux.shape[0])]), points_total)
            #plotter.add_mesh(grid,color=np.random.rand(3), show_edges = True) 
            plotter.add_point_labels(grid.cell_centers().points[0],['Neumann BC ' + i], always_visible=True, font_size=20)
          

        plotter.add_title('Boundary Conditions')   
        plotter.add_axes()      
        plotter.show()
        


    return points_total,connectivity_total,bc_drichlet_cells,bc_drichlet_nodes,bc_neumann_cells,bc_neumann_nodes


def Result_Tensor(Original_array,Coincident_nodes):
    New_array  = jnp.zeros((len(Coincident_nodes.keys()),Original_array.shape[-2],Original_array.shape[-1]))
    for i in Coincident_nodes:
        tensor_indexs = jnp.array(Coincident_nodes[i])
        aux  = jnp.mean(Original_array[tensor_indexs[:,0],tensor_indexs[:,1],:,:],axis = 0)
        New_array  = New_array.at[i].set(aux)
    return New_array

def Result_Vector(Original_array,Coincident_nodes, avg = False):
    New_array  = jnp.zeros((len(Coincident_nodes.keys()),Original_array.shape[-1]))
    for i in Coincident_nodes:
        tensor_indexs = jnp.array(Coincident_nodes[i])
        if avg == True:
            aux  = jnp.mean(Original_array[tensor_indexs[:,0],tensor_indexs[:,1],:],axis = 0)
        else:
            aux  = jnp.sum(Original_array[tensor_indexs[:,0],tensor_indexs[:,1],:],axis = 0)
        New_array  = New_array.at[i].set(aux)
    return New_array

def Result_Scalar(Original_array,Coincident_nodes, avg = False):
    New_array  = jnp.zeros((len(Coincident_nodes.keys())))
    for it,i in enumerate(Coincident_nodes):
        tensor_indexs = jnp.array(Coincident_nodes[i])
        if avg == True:
            aux  = jnp.mean(Original_array[tensor_indexs[:,0],tensor_indexs[:,1]])
        else:
            aux  = jnp.sum(Original_array[tensor_indexs[:,0],tensor_indexs[:,1]])
        New_array  = New_array.at[it].set(aux)
    return New_array

def Coincident_nodes(array):
    """
    Some functiones in this code work with the dimensions (#Element, #nodes per element, dim 1, dim 2)
    Example Deformation Gradient (#element, #nodes per element, 3,3)

    For this reazon each element have a 8 values, this generate for the coincident nodes to much information
    for this reason the results in coincident nodes are averaged

    This functon obtai wich nodes are coincident and of wich element belong
    """
    nodes_repeated = {}
    unique_values = sorted(set(val for sublist in array for val in sublist))
    
    for i in unique_values:
        nodes_repeated[i] = []

    for i, ielem in enumerate(array):
        for it, nodo in enumerate(ielem):
            nodes_repeated[nodo].append([i, it])

    return nodes_repeated

def change_state_plot(mesh, disp):
    mesh_def = mesh.copy()
    mesh_def.points = mesh_def.points + np.array(disp)

    pl = pv.Plotter(shape=(1, 2))


    pl.subplot(0, 0)
    pl.add_text("Original State", font_size=30)
    pl.add_mesh(mesh, show_edges=True, color='lightblue',opacity = 1)


    pl.subplot(0, 1)
    pl.add_text("Deformed State", font_size=30)
    pl.add_mesh(mesh_def, show_edges=True, color='lightblue')


    # # Display the window
    pl.show()


@jit
def extract_column(array, direction):
    index_direction_1 = jnp.argmin(direction)
    index_direction_2 = jnp.argmin(direction[::-1]) + 2
    indices = jnp.array([index_direction_1, index_direction_2], dtype= 'int64')
    return array[:, indices]