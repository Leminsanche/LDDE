"""
@author : Nicolás Sánchez

This code have all Solvers Implemented to minimize Energy
"""


import jax.numpy as jnp
import jax
from jax import jit
from tqdm import tqdm
import optax


class optimizers():

    def __init__(self, loss_function, displacament_function, learning_rate):
        self.loss_function = loss_function
        self. learning_rate = learning_rate


    @jit
    def Jacobian(self, disp):
        J = jax.jacrev(self.loss_function)(disp)
        return J
    
    def extract_zeros(arr):
        mask = (arr == 0)  # Crear una máscara booleana donde los elementos son 0
        return arr[mask]   # Usar la máscara para seleccionar elementos que son 0

    @jit
    def adam(self,initial_disp,iterations = 150, patience = 5):
        """
        function to solve the problem with adam optimizator 
        this function uses early stoping as regularizer
        """


        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(initial_disp)
        final_displacement = initial_disp
        loss_values = jnp.zeros(iterations)

        for i in tqdm(range(iterations)):
            grads = self.Jacobian(final_displacement)
            updates, opt_state = optimizer.update(grads, opt_state)
            final_displacement = optax.apply_updates(final_displacement, updates)
            loss_value = self.loss_function(final_displacement)
            loss_values = loss_values.at[i].set(loss_value)


        return final_displacement, self.extract_zeros(loss_values)
      # for i in tqdm(range(150)):




