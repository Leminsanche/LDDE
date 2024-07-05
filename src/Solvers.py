"""
@author : Nicolás Sánchez

This code have all Solvers Implemented to minimize Energy
"""


import jax.numpy as jnp
import jax
from jax import jit
from tqdm import tqdm
import optax
import jaxopt


class optimizers():

    def __init__(self, loss_function,jacob, learning_rate,displacament_function = None):
        self.loss_function = loss_function
        self. learning_rate = learning_rate
        self.displacement_function = displacament_function
        self.jacob = jacob

    def early_stopping(self,loss_value, best_loss, patience_counter,patience):

        stop_criteria = False

        if loss_value < best_loss:
            best_loss = loss_value
            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= patience:
            stop_criteria = True

        return best_loss, patience_counter, stop_criteria


    def adam_earlyStopping(self,initial_disp,iterations = 1000, patience_number = 50):
        """
        function to solve the problem with adam optimizator 
        this function uses early stoping as regularizer

        first initialize the optimizer, define the learning rate
        """


        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(initial_disp)
        final_displacement = initial_disp
        loss_values = []
        
        best_loss = 1000
        patience_counter = 0

        for i in range(iterations):
            grads = self.jacob(final_displacement)
            updates, opt_state = optimizer.update(grads, opt_state)
            final_displacement = optax.apply_updates(final_displacement, updates)
            loss_value = self.loss_function(final_displacement)
            loss_values.append(loss_value)
            
        
            ####### Early Stopping ########
            best_loss, patience_counter, stop_criteria = self.early_stopping(loss_value, best_loss, patience_counter,patience_number)

            if stop_criteria:
                print(f'Early stopping at iteration {i}')
                break
            
        return final_displacement, loss_values
    
    def adam(self, initial_variable, iterations = 1000, tolerancia = 0.001):

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(initial_variable)
        final_variable = initial_variable
        loss_values = []
        path = [initial_variable]
                
        for i in tqdm(range(iterations)):
            grads = self.jacob(final_variable)
            updates, opt_state = optimizer.update(grads, opt_state)
            final_variable = optax.apply_updates(final_variable, updates)
            loss_value = self.loss_function(final_variable)
            loss_values.append(loss_value)
            path.append(final_variable)
            if jnp.linalg.norm(path[-2] - path[-1] ) < tolerancia:
                break
            
        
        return final_variable , loss_values, path
    
    def LBFGS(self,initial_variable, iterations = 1000, tolerancia = 0.01):

        solver = jaxopt.LBFGS(fun=self.loss_function, maxiter = iterations, tol= tolerancia)
        state = solver.init_state(initial_variable)
        params = initial_variable
        path = [initial_variable]
        loss_values = []

        for _ in tqdm(range(solver.maxiter)):
            params, state = solver.update(params, state)
            path.append(params)
            loss_values.append(state.error)
            if jnp.linalg.norm(path[-2] - path[-1] ) < solver.tol:
                 break

        return params , loss_values, path




