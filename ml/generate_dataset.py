import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from simulation.system import build_default_system

system = build_default_system()

loads = np.linspace(0, 2e-3, 100)
records = []

for load in loads:
    r = system.solve_static_response(load)
    records.append({
        "load": load,
        "z_shift": r["z_shift"],
        "n_eff": r["neff"][0],
    })

df = pd.DataFrame(records)
df.to_csv("data/mems_photonic_dataset.csv", index=False)
