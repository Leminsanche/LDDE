"""
@author : Nicolás Sánchez

This Code is the alpha of the library who contains the way to integrate energy inside elements
To the date only Hexaedric element it's implemented
The reference is Bonet Nonlinear Continuum Mechanics for Finite Element Analysis
"""


import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv
import jax 
import numpy as np
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
        return jnp.einsum('...ai,aj', self.nodes, self.der_N_fun(xi))

    def der_N_X(self, xi):  # 7.6b
        temp = self.der_X_xi(xi).transpose(0,2,1)
        inv_der_X_xi = np.linalg.inv(temp)
        out = np.matmul(inv_der_X_xi, self.der_N_fun(xi).T).transpose(0,2,1)
        ##print(out.shape)
        return out

    def der_x_xi(self, x, xi):  # 7.11a
        return jnp.einsum('...ai,aj', x, self.der_N_fun(xi))
    
    def der_x_xi_vec(self,x, xi):  # 7.11a
        """
        Vectorized function for der_x_xi
        """
        return jnp.einsum('...ai,...aj', x, jax.vmap(self.der_N_fun)(xi))

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
        
        F = np.einsum('eai,exaj->exij', x, self.der_N_X_esquinas)
    
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
    
    def psi_jax(self,disp):
        
        x_n = self.x_def(disp)
        C =self.Cauchy_Green_rigth(x_n)

        temp = jax.vmap(jax.vmap(self.material.psi))(C)
        return temp
    
    def S_jax(self,disp):
        x_n = self.x_def(disp)
        C =self.Cauchy_Green_rigth(x_n)
        temp = jax.vmap(jax.vmap(jax.jacobian(self.material.psi)))(C)
        return 2*temp
    
    def Cauchy(self,disp):
        S = self.S_jax(disp)
        x_n = self.x_def(disp)
        F = self.f(x_n)
        #print(F)
        J_inv = 1/jnp.linalg.det(F)
        # print(J_inv.shape)
        # print(S.shape)
        # print(F.shape)
        sigma = jnp.einsum('naik,nakl,najl->naij',F,S,F)
        
        return sigma #* J_inv[:, :, np.newaxis, np.newaxis]
    
    def PSI(self,disp):
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

        temp = self.psi_jax(disp)

        aux = jnp.linalg.det(self.der_x_xi_vec(x, self.gauss_points))
        e_t = jnp.dot(temp,aux)


        return e_t