"""
Train surrogate models for MEMS–photonics system.

Models:
- Linear regression
- Gaussian process
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------

df = pd.read_csv("data/mems_photonic_dataset.csv")

# Inputs and targets
X = df[["load"]].values          # shape (N, 1)
y = df["n_eff"].values           # shape (N,)

# ------------------------------------------------------------------
# Train / test split
# ------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# Baseline model: Linear regression
# ------------------------------------------------------------------

lin_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

lin_model.fit(X_train, y_train)

y_pred_lin = lin_model.predict(X_test)

print("Linear model:")
print(f"  MAE: {mean_absolute_error(y_test, y_pred_lin):.2e}")
print(f"  R² : {r2_score(y_test, y_pred_lin):.4f}")

# ------------------------------------------------------------------
# Gaussian process surrogate
# ------------------------------------------------------------------

kernel = RBF(length_scale=1e-4) + WhiteKernel(noise_level=1e-8)

gp_model = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True
)

gp_model.fit(X_train, y_train)

y_pred_gp = gp_model.predict(X_test)

print("\nGaussian Process model:")
print(f"  MAE: {mean_absolute_error(y_test, y_pred_gp):.2e}")
print(f"  R² : {r2_score(y_test, y_pred_gp):.4f}")

# ------------------------------------------------------------------
# Save best model
# ------------------------------------------------------------------

# Use the script's directory so the save location is independent of the current working directory
script_dir = Path(__file__).resolve().parent
models_dir = script_dir / "models"
print(f"Saving models to: {models_dir} (cwd: {Path.cwd()})")
try:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gp_model, str(models_dir / "gp_surrogate.joblib"), compress=3)
    print(f"\nSaved GP surrogate to {models_dir / 'gp_surrogate.joblib'}")
    # Also save the linear baseline for comparison
    joblib.dump(lin_model, str(models_dir / "lin_surrogate.joblib"), compress=3)
    print(f"Saved linear baseline to {models_dir / 'lin_surrogate.joblib'}")
except Exception as e:
    print(f"Failed to save models to {models_dir}: {e}")
    print("Directory exists:", models_dir.exists())
    print("Directory is dir:", models_dir.is_dir())
    raise
