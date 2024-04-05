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

class energy():

    def __init__(self, material,nodes, conn):
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
        return jnp.einsum('...ai,aj', self.nodes, jax.jit(self.der_N_fun)(xi))

    def der_N_X(self, xi):  # 7.6b
        temp = self.der_X_xi(xi).transpose(0,2,1)
        inv_der_X_xi = np.linalg.inv(temp)
        out = np.matmul(inv_der_X_xi, self.der_N_fun(xi).T).transpose(0,2,1)
        ##print(out.shape)
        return out

    def der_x_xi(self, x, xi):  # 7.11a
        return jnp.einsum('...ai,...aj', x, jax.jit(self.der_N_fun)(xi))

    def der_N_x(self, x, xi):  # 7.11b
        temp = self.der_x_xi(x, xi).transpose(0,2,1)
        inv_der_x_xi = jnp.linalg.inv(temp)
        
        return jnp.matmul(inv_der_x_xi,jax.jit(self.der_N_fun)(xi).T).transpose(0,2,1)
    
    def f_gauss(self, x_n):  # gradiente de deformacion -- 7.5
        #print("disp", x_n)
        x = self._get_nodes(x_n)
        #print(f"xn: {x}")
        #print(self.der_N_X_gp)
        F = jnp.einsum('eai,exaj->exij', x, self.der_N_X_gp)
        #print("test", F)
        return F
    
    def f(self, x_n):  # gradiente de deformacion -- 7.5
        #print("disp", x_n)
        x = self._get_nodes(x_n)
        Fs = []
        
        F = np.einsum('eai,exaj->exij', x, self.der_N_X_esquinas)
    
        #print("test", F)
        return F
    

    def PSI(self,x_n):
        # x_n nuevas coordenadas
        #print(x_n)
        x = jax.jit(self._get_nodes)(x_n)
        F = jax.jit(self.f_gauss)(x_n)
        # print(F.shape)
        temp = self.material.psi(F)
        # print(temp.shape)
        micro = []
        for it, gp in enumerate(self.gauss_points):
            #print(x.shape)
            aux = jnp.linalg.det(jax.jit(self.der_x_xi)(x, gp))
            micro.append(aux)
        e_t = jnp.dot(temp,jnp.array(micro))
        # print(jnp.array(micro).sum())
        # # print(e_t)
        # #print(micro)
        # print(x_n)
        return e_t