"""
@author : Nicolás Sánchez

This Code is the alpha Version of the library who contains the way to integrate energy inside elements
To the date only Hexaedric element it's implemented
The reference is Bonet Nonlinear Continuum Mechanics for Finite Element Analysis
"""


import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv
import jax
import numpy as np
import Functions as fun
# from jax import config
# config.update("jax_enable_x64", True)

class Hexs():

    def __init__(self, material,nodes, conn):
        """
        Hexs(material,nodes, conn)
        Parameters: material class of Material.py who have the constitutive model for energy (ex: Delphino_incompresible([coinstant]))
                    nodos: Nodos de la malla
                    conn: array de conectividades de la malla

        """
        self.nodes_or = nodes
        self.conn = conn
        self.material = material
        self.nodes = nodes[conn]
        self.nnodes = 8
        self.nodes_repeated = self.Coincident_nodes()


        puntos_iso = jnp.array([[-1,-1,-1],
                               [ 1,-1,-1],
                               [ 1, 1,-1],
                               [-1, 1,-1],
                               [-1,-1, 1],
                               [ 1,-1, 1],
                               [ 1, 1, 1],
                               [-1, 1, 1] ])


        self.gauss_points = jnp.array([[-1/3**0.5, -1/3**0.5, -1/3**0.5],
                                      [ 1/3**0.5, -1/3**0.5, -1/3**0.5],
                                      [ 1/3**0.5,  1/3**0.5, -1/3**0.5],
                                      [-1/3**0.5,  1/3**0.5, -1/3**0.5],
                                      [-1/3**0.5, -1/3**0.5,  1/3**0.5],
                                      [ 1/3**0.5, -1/3**0.5,  1/3**0.5],
                                      [ 1/3**0.5,  1/3**0.5,  1/3**0.5],
                                      [-1/3**0.5,  1/3**0.5,  1/3**0.5],
                                      ])

        self.der_N_X_gp = [self.der_N_X(i) for i in self.gauss_points]
        self.der_N_X_gp = jnp.array(self.der_N_X_gp).transpose((1,0,2,3))

        self.der_N_X_esquinas = [self.der_N_X(i) for i in puntos_iso]
        self.der_N_X_esquinas = np.array(self.der_N_X_esquinas).transpose((1,0,2,3))

        # self.der_N_x_gp = [self.der_N_x(i) for i in self.gauss_points]
        # self.der_N_x_gp = jnp.array(self.der_N_x_gp).transpose((1,0,2,3))


    def Coincident_nodes(self):
        """
        Some functiones in this code work with the dimensions (#Element, #nodes per element, dim 1, dim 2)
        Example Deformation Gradient (#element, #nodes per element, 3,3)

        For this reazon each element have a 8 values, this generate for the coincident nodes to much information
        for this reason the results in coincident nodes are averaged

        This functon obtai wich nodes are coincident and of wich element belong
        """
        nodes_repeated = {}

        for i in range(len(self.nodes_or)):
            nodes_repeated[i] = []

        for i, ielem in enumerate(self.conn):
            for it, nodo in enumerate(ielem):
                nodes_repeated[nodo].append([i, it])

        return nodes_repeated


    def _get_nodes(self, x):
        return x[self.conn,:]

    def x_def(self,disp):

        xn = self.nodes_or + disp

        return xn


    def N_func(self, xi):
        xi0 = xi[0]
        xi1 = xi[1]
        xi2 = xi[2]
        N1 = (1.0 - xi0)*(1.0 - xi1)*(1.0 - xi2)/8.0
        N2 = (1.0 + xi0)*(1.0 - xi1)*(1.0 - xi2)/8.0
        N3 = (1.0 + xi0)*(1.0 + xi1)*(1.0 - xi2)/8.0
        N4 = (1.0 - xi0)*(1.0 + xi1)*(1.0 - xi2)/8.0
        N5 = (1.0 - xi0)*(1.0 - xi1)*(1.0 + xi2)/8.0
        N6 = (1.0 + xi0)*(1.0 - xi1)*(1.0 + xi2)/8.0
        N7 = (1.0 + xi0)*(1.0 + xi1)*(1.0 + xi2)/8.0
        N8 = (1.0 - xi0)*(1.0 + xi1)*(1.0 + xi2)/8.0
        return jnp.array([N1, N2, N3, N4, N5, N6, N7, N8])

    def der_N_fun(self, xi):
        xi0 = xi[0]
        xi1 = xi[1]
        xi2 = xi[2]
        return jnp.array([[  -(1.0 - xi1)*(1.0 - xi2)/8.0, -(1.0 - xi0)*(1.0 - xi2)/8.0, -(1.0 - xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 - xi1)*(1.0 - xi2)/8.0, -(1.0 + xi0)*(1.0 - xi2)/8.0, -(1.0 + xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 + xi1)*(1.0 - xi2)/8.0,  (1.0 + xi0)*(1.0 - xi2)/8.0, -(1.0 + xi0)*(1.0 + xi1)/8.0],
                         [  -(1.0 + xi1)*(1.0 - xi2)/8.0,  (1.0 - xi0)*(1.0 - xi2)/8.0, -(1.0 - xi0)*(1.0 + xi1)/8.0],
                         [  -(1.0 - xi1)*(1.0 + xi2)/8.0, -(1.0 - xi0)*(1.0 + xi2)/8.0,  (1.0 - xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 - xi1)*(1.0 + xi2)/8.0, -(1.0 + xi0)*(1.0 + xi2)/8.0,  (1.0 + xi0)*(1.0 - xi1)/8.0],
                         [   (1.0 + xi1)*(1.0 + xi2)/8.0,  (1.0 + xi0)*(1.0 + xi2)/8.0,  (1.0 + xi0)*(1.0 + xi1)/8.0],
                         [  -(1.0 + xi1)*(1.0 + xi2)/8.0,  (1.0 - xi0)*(1.0 + xi2)/8.0,  (1.0 - xi0)*(1.0 + xi1)/8.0],
                         ])



    def der_X_xi(self, xi):  # 7.6b
        return jnp.einsum('...ai,...aj -> ...ij ', self.nodes, self.der_N_fun(xi))

    def der_N_X(self, xi):  # 7.6b
        temp = self.der_X_xi(xi).transpose(0,2,1)
        inv_der_X_xi = np.linalg.inv(temp)
        out = np.matmul(inv_der_X_xi, self.der_N_fun(xi).T).transpose(0,2,1)####
        return out

    def der_x_xi(self, x, xi):  # 7.11a
        return jnp.einsum('...ai,aj -> ...ij', x, self.der_N_fun(xi))

    def der_x_xi_vec(self,x, xi):  # 7.11a
        """
        Vectorized function for der_x_xi
        """
        return jnp.einsum('...ai,...aj -> ...ij', x, jax.vmap(self.der_N_fun)(xi))

    def der_N_x(self, x, xi):  # 7.11b
        temp = self.der_x_xi(x, xi).transpose(0,2,1)
        inv_der_x_xi = jnp.linalg.inv(temp)

        return jnp.matmul(inv_der_x_xi,self.der_N_fun(xi).T).transpose(0,2,1)


    def f_gauss(self, x_n):  # gradiente de deformacion -- 7.5
        #print("disp", x_n)
        """
        f_gauss(self, x_n): Function to calculate deformation gradiente in gauss points

        Parameters
        x_n: array with nodal coordinates deformated state

        Return
        F array dimensions (a,8,3,3) a: number of number of element, 8 deformation gradientes as 3x3 matrix
        """
        x = self._get_nodes(x_n)
        #print(f"xn: {x}")
        #print(self.der_N_X_gp)
        F = jnp.einsum('eai,exaj->exij', x, self.der_N_X_gp)
        #print("test", F)
        return F

    def f(self, x_n):  # gradiente de deformacion -- 7.5
        #print("disp", x_n)
        x = self._get_nodes(x_n)

        F = jnp.einsum('eai,exaj->exij', x, self.der_N_X_esquinas)

        #print("test", F)
        return F

    def Cauchy_Green_rigth(self,x_n):
        F = self.f_gauss(x_n)
        F_T = jnp.moveaxis(F,-1,-2)
        C = jnp.matmul(F_T,F)

        return C

    def psi(self,disp):
        """
        This work with contitutive models in tensorial way
        DON'T USE
        """

        x_n = self.x_def(disp)
        C =self.Cauchy_Green_rigth(x_n)

        temp = self.material.psi(C)
        return temp

    def psi_jax(self,disp,constantes):

        x_n = self.x_def(disp)
        C =self.Cauchy_Green_rigth(x_n)

        temp = jax.vmap(jax.vmap(self.material.psi,in_axes=[0, None]), in_axes=[0, None])(C,constantes)
        return temp

    def S_jax(self,disp):
        x_n = self.x_def(disp)
        C =self.Cauchy_Green_rigth(x_n)
        # C_inv = np.linalg.inv(C)
        # J = np.linalg.det(C)
        #print(J)
        temp = jax.vmap(jax.vmap(jax.jacobian(self.material.psi,argnums= 0), in_axes=[0, None]), in_axes=[0, None])(C,self.material.constants)
        return 2*temp #+ self.material.constants[-1]*(J[:, :, None, None]-1)*C_inv
    

    def P_first(self,disp):
        S = self.S_jax(disp)
        x_n = self.x_def(disp)
        F = self.f(x_n)

        P = jnp.einsum('naik,nakj->naij',F,S)
        return P 


    def Cauchy(self,disp):
        S = self.S_jax(disp)
        x_n = self.x_def(disp)
        F = self.f(x_n)
        J_inv = 1/jnp.linalg.det(F)
        # print(J_inv.shape)
        # print(S.shape)
        # print(F.shape)
        sigma = jnp.einsum('naik,nakl,najl->naij',F,S,F)

        return sigma * J_inv[:, :, np.newaxis, np.newaxis]

    def Internal_Force(self, disp):  #Based on equation 7.15 Bonet
        """
        The Output Should Be one Force per Node
        """
        x_n = self.x_def(disp)
        xn = self._get_nodes(x_n)

        Cauchy = self.Cauchy(disp)
        Jacob = self.der_x_xi_vec(xn,self.gauss_points)
        det_Jacob = jnp.linalg.det(Jacob)
        grad_shape_def = jax.vmap(self.der_N_x,(None,0))(xn,self.gauss_points)### Puede Traer Error
        grad_shape_def = jnp.transpose(grad_shape_def,(1,0,2,3))
        #grad_shape_def = jnp.array([self.der_N_x(xn,i) for  i in self.gauss_points]).transpose((1,0,2,3)) #less numeric error than vmap
        aux = jnp.einsum('...aij , ...akj ->...aki',Cauchy,grad_shape_def)
        Internal_Forces = jnp.einsum('...aij,a -> ...ij',aux,det_Jacob)

        return Internal_Forces


    def PSI(self,disp, constantes):
        """
        PSI(self,x_n): Function to calculate volume integral of element

        THIS CODE HAVE TO BE OPTIMIZED

        Parameters
        x_n: Deformed Coordinates

        Return
        e_t: Array (1 element)
        """

        x_n = self.x_def(disp)
        x = self._get_nodes(x_n)

        temp = self.psi_jax(disp,constantes)

        aux = jnp.linalg.det(self.der_x_xi_vec(x, self.gauss_points))
        e_t = jnp.dot(temp,aux)


        return jnp.sum(e_t)


    def External_Energy(self,disp,Dritchlet_BC = None, Neumann_BC = None):
        """
        This Functions Calculate energy froce generate in due to boundary conditions
        This External Force can be produce due to Dritchlet or Neumann Boundary conditions
        In this function Both cases are evaluted in case for Dritchlet conditions it's necesary calculate internal forces
        As this function is in Hex class works only for QUAD borders

        As This Code works with the codes in Functions.py so Dritchlet_BC it's a dict with {Name_BC: array([connectivity Quad element 1],...,
                                                                                                           [connectivity Quad element n])}

        If it's more than one BC this dict will have more element with the same shape

        Inputs
        Dritchlet_BC = array with the information of cells and conectivity for every BC | Dim (#Surfaces, 4)
        Neuman_BC = array with the information of cells and conectivity for every BC | Dim (#Surfaces, 4)
        Direction = array who indicate the perpendicular direction of boundary conditions

        """
        Dritchlet_Energy = 0
        Neumann_Energy = 0

        x_n = self.x_def(disp)

        if Dritchlet_BC is not None: #This Could Interfere with JIT but IDK
            #Internal_Traccion = fun.Result_Vector(self.Internal_Force(disp),self.nodes_repeated)
            Internal_Traccion = fun.Result_Tensor(self.Cauchy(disp),self.nodes_repeated)
            #Internal_Traccion = fun.Result_Tensor(self.P_first(disp),self.nodes_repeated)

            # e_ext = jnp.einsum('ij,ij -> i',Internal_Traccion,disp)
            # print(e_ext.shape)

            for key in Dritchlet_BC:
                cells = Dritchlet_BC[key][0]
                # print(nodes_repited)
                direction = Dritchlet_BC[key][1]

                Internal_Traccion = jnp.einsum('nij,k->nk',Internal_Traccion,direction)
                #print(Internal_Traccion.shape)
                Traccions = Internal_Traccion[cells]
                #print(Traccions)
                displacement_bc = disp[cells]

                e_ext = jnp.einsum('ijk,ijk -> ij',Traccions,displacement_bc)
                # index_direction = jnp.argmax(direction)# index_direction = jnp.nonzero(direction)[0][0]
                # indices = jnp.arange(self.nodes_or.shape[1])
                # mask = indices != index_direction
                # nodes = self.nodes_or[:,mask]#jnp.delete(self.nodes_or,index_direction, axis = 1)
                # nodes_def = x_n[:,mask]#jnp.delete(x_n,index_direction, axis = 1)
                
                nodes = fun.extract_column(self.nodes_or,direction)
                nodes_def = fun.extract_column(x_n,direction)
                bc = Quads(nodes,cells)

                N_fun = jax.vmap(bc.N_func)(bc.puntos_iso)
                J = jax.vmap(bc.der_x_xi, in_axes=[None, 0])(bc._get_nodes(nodes_def),bc.gauss_points)#.reshape(4,2,2) #Jacobiano
                # print(J)
                cross_prod_J = jnp.cross(J[:,:, :, 0], J[:,:, :, 1]).reshape(-1,1)
                cross_prod_J_norm = jnp.linalg.norm(cross_prod_J,axis = 1)
                #cross_prod_J_norm = jnp.abs(cross_prod_J)
                # print(cross_prod_J_norm.shape)

                e_iso = jnp.einsum('ij , ia -> aj',N_fun,e_ext.transpose((1,0)))
                #print(e_iso)
                #Integral = jnp.einsum('ai,ai',e_iso,cross_prod_J_norm)
                Integral = jnp.vdot(e_iso,cross_prod_J_norm.reshape(e_iso.shape))
                #Integral = jnp.sum(e_iso)
                #print(Integral)
                Dritchlet_Energy += Integral

        if Neumann_BC is not None:

            for key in Neumann_BC:
                cells = Neumann_BC[key][0]
                Force = Neumann_BC[key][1]
                displacement_bc = disp[cells]
                Forces = jnp.ones_like(displacement_bc)*Force
                e_ext = jnp.einsum('ijk,ijk -> ij',Forces,displacement_bc)
                index_direction = jnp.nonzero(Force)[0][0]
                nodes = jnp.delete(self.nodes_or,index_direction, axis = 1)
                nodes_def = jnp.delete(x_n,index_direction, axis = 1)
                # print(index_direction)
                # print(self.nodes_or)
                bc = Quads(nodes,cells)

                N_fun = jax.vmap(bc.N_func)(bc.gauss_points)
                J = jax.vmap(bc.der_x_xi, in_axes=[None, 0])(bc._get_nodes(nodes_def),bc.gauss_points)#.reshape(4,2,2) #Jacobiano

                cross_prod_J_norm = jnp.linalg.norm(jnp.cross(J[:,:, 0, :], J[:,:, 1, :]).reshape(-1,1),axis = 1)
                aux = jnp.einsum('ij , ai -> aj',N_fun,e_ext)
                Integral = jnp.vdot(aux,cross_prod_J_norm)
                Neumann_Energy += Integral

        return Dritchlet_Energy + Neumann_Energy





################################### QUAD ELEMENT #################################

class Quads:
    def __init__(self, nodes, conn):
        self.nodes_or = nodes
        self.conn = conn
        self.nodes = nodes[conn]
        self.nodes_repeated = self.Coincident_nodes()
        #print(self.nodes.shape)
        self.nnodes = 4
        self.puntos_iso = jnp.array([[-1,-1],
                               [ 1,-1],
                               [ 1, 1],
                               [-1, 1]])

        self.gauss_points = jnp.array([[-1/3**0.5, -1/3**0.5],
                                      [ 1/3**0.5, -1/3**0.5],
                                      [ 1/3**0.5,  1/3**0.5],
                                      [-1/3**0.5,  1/3**0.5]
                                      ])

        self.der_N_X_esquinas = [self.der_N_X(i) for i in self.puntos_iso]
        self.der_N_X_esquinas = jnp.array(self.der_N_X_esquinas).transpose((1,0,2,3))

    def _get_nodes(self, x):
        return x[self.conn,:]
    
    def Coincident_nodes(self):
        """
        Some functiones in this code work with the dimensions (#Element, #nodes per element, dim 1, dim 2)
        Example Deformation Gradient (#element, #nodes per element, 3,3)

        For this reazon each element have a 8 values, this generate for the coincident nodes to much information
        for this reason the results in coincident nodes are averaged

        This functon obtai wich nodes are coincident and of wich element belong
        """
        nodes_repeated = {}

        for i in range(len(self.nodes_or)):
            nodes_repeated[i] = []

        for i, ielem in enumerate(self.conn):
            for it, nodo in enumerate(ielem):
                nodes_repeated[nodo].append([i, it])

        return nodes_repeated


    def x_def(self,disp):

        xn = self.nodes_or + disp

        return xn


    def N_func(self, xi):
        xi0 = xi[0]
        xi1 = xi[1]
        N1 = (1.0 - xi0)*(1.0 - xi1)/4.0
        N2 = (1.0 + xi0)*(1.0 - xi1)/4.0
        N3 = (1.0 + xi0)*(1.0 + xi1)/4.0
        N4 = (1.0 - xi0)*(1.0 + xi1)/4.0
        return jnp.array([N1, N2, N3, N4])

    def der_N_fun(self, xi):
        xi0 = xi[0]
        xi1 = xi[1]
        return jnp.array([[  -(1.0 - xi1)/4.0  , -(1.0 - xi0)/4.0],
                         [   (1.0 - xi1)/4.0  , -(1.0 + xi0)/4.0],
                         [   (1.0 + xi1)/4.0  ,  (1.0 + xi0)/4.0],
                         [  -(1.0 + xi1)/4.0  ,  (1.0 - xi0)/4.0]
                         ])


    def der_X_xi(self, xi):  # 7.6b
        aux = jnp.einsum('...ai,aj', self.nodes, self.der_N_fun(xi))
        #print(aux.shape)
        #aux = jnp.einsum('...ai,aj -> ...ji', self.nodes, self.der_N_fun(xi))
        #print(aux)
        return aux

    def der_N_X(self, xi):  # 7.6b
        temp = self.der_X_xi(xi).transpose(0,2,1)
        inv_der_X_xi = jnp.linalg.inv(temp)
        out = jnp.matmul(inv_der_X_xi, self.der_N_fun(xi).T).transpose(0,2,1)
        ##print(out.shape)
        return out

    def der_x_xi(self, x, xi):  # 7.11a
        #print(self.der_N_fun(xi).shape)
        return jnp.einsum('...ai,aj', x, self.der_N_fun(xi))

    def der_N_x(self, x, xi):  # 7.11b
        temp = self.der_x_xi(x, xi).transpose(0,2,1)
        inv_der_x_xi = jnp.linalg.inv(temp)

        return jnp.matmul(inv_der_x_xi,self.der_N_fun(xi).T).transpose(0,2,1)

    def f(self, x_n):  # gradiente de deformacion -- 7.5
        #print("disp", x_n)
        x = self._get_nodes(x_n)
        Fs = []

        F = jnp.einsum('eai,exaj->exij', x, self.der_N_X_esquinas)

        #print("test", F)
        return F

    def der_x_xi_vec(self,x, xi):  # 7.11a
        """
        Vectorized function for der_x_xi
        """
        return jnp.einsum('...ai,aj -> ...ij', x, jax.vmap(self.der_N_fun)(xi))

    def External_Energy(self,disp,Force = 0, direction = jnp.array([0,0,1])):
        Force_Vec = direction * Force
        nodal_energy = jnp.einsum('naj,j -> na',disp[self.conn],Force_Vec)
        print(nodal_energy.shape)

        return print('AA')


    def PSI(self,disp, constantes):
        """
        PSI(self,x_n): Function to calculate volume integral of element

        THIS CODE HAVE TO BE OPTIMIZED

        Parameters
        x_n: Deformed Coordinates

        Return
        e_t: Array (1 element)
        """
        x_n = self.x_def(disp)
        x = self._get_nodes(x_n)

        temp = self.psi_jax(disp,constantes)

        aux = jnp.linalg.det(self.der_x_xi_vec(x, self.gauss_points))
        e_t = jnp.dot(temp,aux)


        return e_t