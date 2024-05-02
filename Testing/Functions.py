import meshio
import pyvista as pv
import numpy as np

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
        for i in drichlet_bc:
            physical_tag = physical_names[i][0]
            element_index = np.where(physical_data_bc == physical_tag)[0]#[0]
            bc_drichlet_cells[i] = element_type['quad'][element_index]#element_index

    #print(bc_drichlet_cells)

    if neumann_bc is not None:
        bc_neumann_cells = {}
        for i in neumann_bc:
            physical_tag = physical_names[i][0]
            element_index = np.where(physical_data_bc == physical_tag)[0]#[0]
            bc_neumann_cells[i] = element_type['quad'][element_index]#element_index




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
        


    return points_total,connectivity_total,bc_drichlet_cells,bc_neumann_cells