"""
@author : Nicolás Sánchez

This code have all Constitutive models implemented
"""


import jax.numpy as jnp

# Superior Class who have more important functions
class Material:
     

    def Cauchy_Green_rigth(self,F):
        F_T = jnp.moveaxis(F,-1,-2)
        C = jnp.matmul(F_T,F)

        return C
    

# Sub class for consitutive models
class Neo_hooke_incompressible(Material):

    def __init__(self, constantes):
        self.mu , self.kappa = constantes 

    def psi(self,f):
        C = self.Cauchy_Green_rigth(f)
        I1,I2,I3 = jnp.einsum('...ii',C), 0.5 * (jnp.einsum('...ii',C)**2 - jnp.einsum('...ii',jnp.matmul(C,C))), jnp.linalg.det(C)
        J = I3**0.5
        aux = I3**(-1/3)
        psi = (self.mu/2)*(I1-3) + 0.5* self.kappa * (J - 1)**2
        return psi
    
class Delphino_incompressible(Material):

    def __init__(self, constantes):
        self.c1 , self.c2, self.kappa = constantes

    def psi(self,f):
        C = self.Cauchy_Green_rigth(f)
        I1,I2,I3 = jnp.einsum('...ii',C), 0.5 * (jnp.einsum('...ii',C)**2 - jnp.einsum('...ii',jnp.matmul(C,C))), jnp.linalg.det(C)
        J = I3**0.5
        aux = I3**(-1/3)
        psi = (self.c1/self.c2) * (jnp.e**(self.c2*0.5*(I1*aux-3)) - 1 ) + 0.5* self.kappa * (J - 1)**2
        return psi
    



