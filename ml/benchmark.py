#Explicit objective

# Quantify and compare: Wall-clock latency

# Prediction error between:
# Full coupled MEMS–photonics simulation
# Trained ML surrogate model

# INCLUDED IN THIS FILE:
# Loading a pre-trained surrogate model
# Running a single-point and batch prediction
# Running the true solver for the same inputs
# Timing both paths
# Printing or plotting speedup and error

"""
Benchmark ML surrogate vs full multiphysics simulation.

Compares:
- Runtime
- Prediction accuracy
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import joblib

from simulation.system import build_default_system

# 1: Load System and Model

# Load full simulation system
system = build_default_system()

# Load trained surrogate
model = joblib.load("ml/models/gp_surrogate.joblib")

# 2: Define Test Inputs

loads = np.array([
    [5e-4],
    [1e-3],
    [1.5e-3],
])

# Benchmark Full Simulation

sim_times = []
sim_outputs = []

for load in loads:
    t0 = time.time()
    r = system.solve_static_response(load[0])
    sim_times.append(time.time() - t0)
    sim_outputs.append(r["neff"][0])

sim_times = np.array(sim_times)
sim_outputs = np.array(sim_outputs)

#4: Benchmark Surrogate Model

t0 = time.time()
ml_outputs = model.predict(loads)
ml_time = time.time() - t0

#5: Compare results

#runtime
print("Simulation runtime per evaluation (s):")
print(sim_times)

print(f"\nML surrogate batch runtime (s): {ml_time:.6f}")
print(f"Average simulation runtime (s): {sim_times.mean():.4f}")
print(f"Speedup factor: {sim_times.mean() / ml_time:.1f}x")

#Accuracy
error = ml_outputs - sim_outputs

print("\nPrediction error (Δn_eff):")
for i, load in enumerate(loads):
    print(
        f"Load={load[0]*1e3:.2f} mN/m | "
        f"Error={error[i]:.2e}"
    )

# mean absolute error
print(f"Mean absolute error: {np.mean(np.abs(error)):.2e}")

#visualizations:
import matplotlib.pyplot as plt

plt.figure()
plt.plot(loads.flatten() * 1e3, sim_outputs, "o-", label="Simulation")
plt.plot(loads.flatten() * 1e3, ml_outputs, "x--", label="ML surrogate")
plt.xlabel("Load (mN/m)")
plt.ylabel("Effective index n_eff")
plt.title("Simulation vs ML Surrogate")
plt.legend()
plt.grid(True)
plt.show()


