import jax.numpy as jnp
import jaxopt
from jax import grad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the optimization problem: a simple quadratic function
# def objective_function(params):
#     x, y = params
#     return (x - 1) ** 2 + (y - 2) ** 2
def objective_function(params):
    x, y = params
    return 100 * (y - 3*x**2)**2 + (1 - x)**2

# Gradient of the objective function
objective_grad = grad(objective_function)

# Define the initial guess for the parameters
initial_params = jnp.array([-1.0, -2.0])

# Create a list to store the optimization path
path = []

# Custom solver run function to track the path
def custom_solver_run(solver, init_params):
    state = solver.init_state(init_params)
    params = init_params
    for _ in range(solver.maxiter):
        params, state = solver.update(params, state)
        path.append(params)
        if state.error < solver.tol:
             print(state.error, solver.tol)
             break
    return params, state

# Create the L-BFGS solver
solver = jaxopt.LBFGS(fun=objective_function, maxiter=500)

# Run the optimization and collect the path
final_params, final_state = custom_solver_run(solver, initial_params)
print(final_params)
# Convert path to a numpy array for easier plotting
path = jnp.array(path)

# Create a grid for plotting the objective function surface
x = jnp.linspace(-2, 2, 400)
y = jnp.linspace(-1, 3, 400)
X, Y = jnp.meshgrid(x, y)
Z = objective_function((X, Y))

# Plot the surface and the optimization path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')

# Extract the x, y, z coordinates of the path
path_x = path[:, 0]
path_y = path[:, 1]
path_z = jnp.array([objective_function(params) for params in path])

# Plot the optimization path
ax.plot(path_x, path_y, path_z, color='r', marker='o', markersize=5, label='Optimization Path')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Objective Function')
ax.legend()

plt.show()
