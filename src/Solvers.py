"""
@author : Nicolás Sánchez

This code have all Solvers Implemented to minimize Energy
"""
import numpy as np

import jax.numpy as jnp
import jax
from jax import jit
from tqdm import tqdm
import optax
import jaxopt

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback

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


    def adam_earlyStopping(self,initial_disp,iterations = 1000, patience_number = 50, imposed_disp = None):
        """
        function to solve the problem with adam optimizator 
        this function uses early stoping as regularizer

        first initialize the optimizer, define the learning rate
        """


        optimizer = optax.adam(self.learning_rate)
        initial_disp = self.displacement_function(initial_disp, imposed_disp)
        opt_state = optimizer.init(initial_disp)
        final_displacement = initial_disp
        loss_values = []
        
        best_loss = 1000
        patience_counter = 0

        for i in range(iterations):
            grads = self.jacob(final_displacement)
            updates, opt_state = optimizer.update(grads, opt_state)
            final_displacement = optax.apply_updates(final_displacement, updates)
            final_displacement = self.displacement_function(final_displacement, imposed_disp)
            loss_value = self.loss_function(final_displacement)
            loss_values.append(loss_value)
            
        
            ####### Early Stopping ########
            best_loss, patience_counter, stop_criteria = self.early_stopping(loss_value, best_loss, 
                                                                             patience_counter,patience_number)

            if stop_criteria:
                print(f'Early stopping at iteration {i}')
                break
            
        return final_displacement, loss_values
    

    # def adam_disp(self,opt_variables,imposed_disp, iterations = 250):
    ###############################
    ##### DEPRECATED FUNCTION #####
    ############################## 
    #     optimizer = optax.adam(self.learning_rate)
    #     opt_variables = self.displacement_function(opt_variables,imposed_disp)
    #     opt_state = optimizer.init(opt_variables)
    #     loss_values = []
    #     for _ in tqdm(range(iterations)):
    #         grads = self.jacob(opt_variables)
    #         updates, opt_state = optimizer.update(grads, opt_state)
    #         opt_variables = optax.apply_updates(opt_variables, updates)
    #         opt_variables = self.displacement_function(opt_variables,imposed_disp)
    #         loss_value = self.loss_function(opt_variables)
    #         loss_values.append(loss_value)
    #     return opt_variables, loss_values
    
    def adam(self, initial_variable, iterations = 1000, tolerancia = 0.001, imposed_disp = None):

        optimizer = optax.adam(self.learning_rate)

        if imposed_disp != None and self.displacement_function != None: ## Displacement Solver Flag
            opt_variable = self.displacement_function(initial_variable,imposed_disp)
        else: 
            opt_variable = initial_variable


        opt_state = optimizer.init(opt_variable)

        loss_values = [self.loss_function(opt_variable)]
        path = [opt_variable]
                
        for _ in tqdm(range(iterations)):
            grads = self.jacob(opt_variable)
            updates, opt_state = optimizer.update(grads, opt_state)
            opt_variable = optax.apply_updates(opt_variable, updates)

            if imposed_disp != None and self.displacement_function != None: ## Displacement Solver Flag
                opt_variable = self.displacement_function(opt_variable,imposed_disp)
            loss_value = self.loss_function(opt_variable)
            loss_values.append(loss_value)
            path.append(opt_variable)
            if jnp.linalg.norm((loss_values[-2]) - loss_values[-1] ) < tolerancia:
                 break
            
        
        return opt_variable , loss_values, path
    
    def LBFGS(self,initial_variable, iterations = 1000, tolerancia = 0.01, imposed_disp = None):

        solver = jaxopt.LBFGS(fun=self.loss_function, maxiter = iterations, tol= tolerancia)

        if imposed_disp != None and self.displacement_function != None: ## Displacement Solver Flag
            opt_variable = self.displacement_function(initial_variable,imposed_disp)

        else:
            opt_variable = initial_variable


        state = solver.init_state(opt_variable)
        path = [opt_variable]
        loss_values = [self.loss_function(opt_variable)]

        for _ in tqdm(range(solver.maxiter)):
            opt_variable, state = solver.update(opt_variable, state)

            if imposed_disp != None and self.displacement_function != None: ## Displacement Solver Flag
                opt_variable = self.displacement_function(opt_variable,imposed_disp)

            path.append(opt_variable)
            loss_values.append(self.loss_function(opt_variable))
            if jnp.linalg.norm(loss_values[-2] - loss_values[-1] ) < solver.tol:
                 break

        return opt_variable , loss_values, path

    def CMAES(self,initial_variable,lower_bounds = None, upper_bounds = None ,iterations = 1000, tolerancia = 0.01, verbose_output = False):

        initial_variable = np.array(initial_variable)

        if lower_bounds == None and upper_bounds == None:
            lower_bounds = initial_variable * 0.5
            upper_bounds = initial_variable * 1.5

        problem = Population_Based_Problem(initial_variable, self.loss_function,lower_bounds, upper_bounds)
        algorithm = CMAES(x0=initial_variable,
                 sigma=0.5,
                 restarts=2,
                 tolfun= tolerancia,
                 tolx= tolerancia,
                 restart_from_best=True,
                 bipop=True)
        
        res = minimize(problem,
                    algorithm,
                    ('n_iter', iterations),
                    callback=MyCallback(),
                    seed=1,
                    verbose= verbose_output)
        
        best_solutions = res.algorithm.callback.data["best"]

        

        return res.X, best_solutions


################################### Evolution Based Algoritms ###################################################
class Population_Based_Problem(ElementwiseProblem):
    def __init__(self,Initial_Variable, loss_function,lower_bounds, upper_bounds, **kwargs):
        
        self.Initial_Variable = Initial_Variable
        self.Modelo = loss_function
        self.num_const = len(Initial_Variable)
        super().__init__(n_var=self.num_const,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=lower_bounds,
                         xu=upper_bounds )

    def _evaluate(self, x, out, *args, **kwargs):
        x = jnp.array(x)
        aux = self.Modelo(x)
        aux = np.array(aux)
        f1 = aux
        #g1 = x[0] > 0
        out["F"] = f1
        #out["G"] = g1

class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("X")[np.argmin(algorithm.pop.get("F"))])


