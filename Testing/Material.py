"""
@author : Nicolás Sánchez

This code have all Constitutive models implemented
"""


import jax.numpy as jnp
import jax

# classes for consitutive models
class Neo_hooke_incompressible():
    """
    This Code Calculate the psi for Neo-hooke but in tensorial way
    DON'T USE
    """

    def __init__(self, constantes):
        self.mu , self.kappa = constantes 

    def psi(self,C):
        I1,I2,I3 = jnp.einsum('...ii',C), 0.5 * (jnp.einsum('...ii',C)**2 - jnp.einsum('...ii',jnp.matmul(C,C))), jnp.linalg.det(C)
        J = I3**0.5
        aux = I3**(-1/3)
        psi = (self.mu/2)*(I1-3) + 0.5* self.kappa * (J - 1)**2
        return psi
    
class Delphino_incompressible():
    """
    This Code Calculate the psi for Neo-hooke but in tensorial way
    DON'T USE
    """

    def __init__(self, constantes):
        self.c1 , self.c2, self.kappa = constantes

    def psi(self,C):
        I1,I2,I3 = jnp.einsum('...ii',C), 0.5 * (jnp.einsum('...ii',C)**2 - jnp.einsum('...ii',jnp.matmul(C,C))), jnp.linalg.det(C)
        J = I3**0.5
        aux = I3**(-1/3)
        psi = (self.c1/self.c2) * (jnp.e**(self.c2*0.5*(I1*aux-3)) - 1 ) + 0.5* self.kappa * (J - 1)**2
        return psi
    
 ####################################################################################################################################################
 #To implement model we only have to write the normal equation who works with a normal tensor like F
 # JAX and the code implemented in element works to convert this to multidimensional function like (Number elements, number nodes, 3,3)
    
class Neo_hooke():

    def __init__(self, constantes):
        self.mu , self.kappa = constantes 

    def psi(self,C):
        I1 = jnp.trace(C)
        psi = (self.mu/2)*(I1-3)
        return psi
    
class Delphino():

    def __init__(self, constantes):
        self.c1 , self.c2, self.kappa = constantes

    def psi(self,C):
        I1 = jnp.trace(C)
        I3 = jnp.linalg.det(C)
        J = I3**0.5
        aux = I3**(-1/3)
        psi = (self.c1/self.c2) * (jnp.e**(self.c2*0.5*(I1*aux-3)) - 1 ) +  0.5*self.kappa * (J - 1)**2
        return psi




