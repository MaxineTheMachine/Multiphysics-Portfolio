import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation import system

def objective(load):
    r = system.solve_static_response(load[0])
    return -abs(r["neff"][0] - target_neff)

from scipy.optimize import minimize

X = np.array([[5e-4], [1.5e-3]])
y = np.array([objective(x) for x in X])

for iteration in range(10):

    gp.fit(X, y)

    def acquisition(x):
        mu, std = gp.predict(x.reshape(1, -1), return_std=True)
        return -(mu + 2 * std)  # UCB

    res = minimize(acquisition, x0=[1e-3], bounds=[(0, 2e-3)])
    next_x = res.x

    next_y = objective(next_x)

    X = np.vstack([X, next_x])
    y = np.append(y, next_y)

    print(f"Iteration {iteration}: load={next_x[0]:.2e}")


# visualizations: 

# Objective v Iteration

plt.figure()
plt.plot(objective_values, "o-")
plt.xlabel("Iteration")
plt.ylabel("|n_eff − target|")
plt.title("Bayesian Optimization Convergence")
plt.grid(True)
plt.show()


# Selection v Iterations

plt.figure()
plt.plot(selected_loads * 1e3, "o-")
plt.xlabel("Iteration")
plt.ylabel("Selected Load (mN/m)")
plt.title("Bayesian Optimization Load Trajectory")
plt.grid(True)
plt.show()

