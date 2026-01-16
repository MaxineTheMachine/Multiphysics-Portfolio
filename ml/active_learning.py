import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.system import build_default_system

system = build_default_system() #delete later !!!!!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#plot intermediate GP states
monitor_gaussian_process = True

# Initial dataset (small!)
loads = np.linspace(0, 2e-3, 5)
records = []

# Sweep load
loads = np.linspace(0, 2e-3, 5)
results = system.optical_response_vs_load(loads)

for load in loads:
    r = system.solve_static_response(load)
    records.append({"load": load, "n_eff": r["neff"][0]})

df = pd.DataFrame(records)

# Candidate pool
candidate_loads = np.linspace(0, 2e-3, 200).reshape(-1, 1)

for iteration in range(10):

    X = df[["load"]].values
    y = df["n_eff"].values

    gp = GaussianProcessRegressor(
        kernel=RBF(1e-4) + WhiteKernel(1e-8),
        normalize_y=True
    )
    gp.fit(X, y)

    _, std = gp.predict(candidate_loads, return_std=True)
    idx = np.argmax(std)

    next_load = candidate_loads[idx, 0]
    r = system.solve_static_response(next_load)

    df.loc[len(df)] = {
        "load": next_load,
        "n_eff": r["neff"][0]
    }

    print(f"Iteration {iteration}: sampled load={next_load:.2e}")





# visualizations after each iteration

    import matplotlib.pyplot as plt

    def plot_gp_state(gp, df, iteration, candidate_loads): # IMPELEMNT MEEEEEEE! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        X_train = df[["load"]].values
        y_train = df["n_eff"].values

        mu, std = gp.predict(candidate_loads, return_std=True)

        plt.figure()
        plt.fill_between(
            candidate_loads.flatten() * 1e3,
            mu - 2 * std,
            mu + 2 * std,
            alpha=0.3,
            label="±2σ uncertainty"
        )
        plt.plot(candidate_loads * 1e3, mu, label="GP mean")
        plt.plot(X_train * 1e3, y_train, "ko", label="Samples")

        plt.xlabel("Load (mN/m)")
        plt.ylabel("Effective Index n_eff")
        plt.title(f"Active Learning Iteration {iteration}")
        plt.legend()
        plt.grid(True)
        plt.show()


    # error v Number of Samples
    if monitor_gaussian_process:
        plot_gp_state(gp, df, iteration, candidate_loads)
    errors = []

    # after retraining GP each iteration
    y_pred = gp.predict(candidate_loads)
    true_vals = [
        system.solve_static_response(l[0])["neff"][0]
        for l in candidate_loads[::20]
    ]

    errors.append(np.mean(np.abs(y_pred[::20] - np.asarray(true_vals))))

plt.figure()
plt.plot(errors, "o-")
plt.xlabel("Active Learning Iteration")
plt.ylabel("Mean Absolute Error")
plt.title("Active Learning Convergence")
plt.grid(True)
plt.show()
print("Final MAE:", errors[-1])