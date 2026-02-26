# ==============================================================
# QSAR vs ML Small-Data Benchmark
# Reproducible Public Version (Zenodo)
# ==============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import xgboost as xgb
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================
# 1. Load dataset
# =============================
df = pd.read_csv("dataset_standard.csv")

print("Dataset shape:", df.shape)

if "Log Ki" not in df.columns:
    raise ValueError("Column 'Log Ki' not found.")

y = df["Log Ki"]
X_full = df.drop(columns=["Log Ki"])

selected_descriptors = ["Mi", "nCs", "nHDon"]
X_selected = X_full[selected_descriptors]

cv = LeaveOneOut()

# =============================
# 2. Define models
# =============================
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, max_iter=10000),
    "SVR_RBF": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=5, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=3),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbosity=0)
}

# =============================
# 3. Classical model
# =============================
pipeline_classical = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

y_pred_classical = cross_val_predict(
    pipeline_classical, X_selected, y, cv=cv)

r2_classical = r2_score(y, y_pred_classical)

# =============================
# 4. Full-set evaluation
# =============================
results_full = {}
predictions_full = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    y_pred = cross_val_predict(pipeline, X_full, y, cv=cv)
    r2 = r2_score(y, y_pred)

    results_full[name] = r2
    predictions_full[name] = y_pred

best_model_name = max(results_full, key=results_full.get)
best_model = models[best_model_name]
real_r2 = results_full[best_model_name]
y_pred_best_full = predictions_full[best_model_name]

# =============================
# 5. QSAR metrics
# =============================
def qsar_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    y_mean = np.mean(y_true)
    press = np.sum(residuals**2)
    tss = np.sum((y_true - y_mean)**2)
    q2_f1 = 1 - (press / tss)
    q2_f2 = 1 - (press / np.sum((y_pred - y_mean)**2))
    return mae, rmse, q2_f1, q2_f2

mae_classical, rmse_classical, q2f1_classical, q2f2_classical = \
    qsar_metrics(y, y_pred_classical)

mae_full, rmse_full, q2f1_full, q2f2_full = \
    qsar_metrics(y, y_pred_best_full)

# =============================
# 6. Y-randomization
# =============================
n_permutations = 1000
random_r2_scores = []

pipeline_best_full = Pipeline([
    ("scaler", StandardScaler()),
    ("model", best_model)
])

for _ in range(n_permutations):
    y_random = np.random.permutation(y)
    y_pred_random = cross_val_predict(
        pipeline_best_full, X_full, y_random, cv=cv)
    random_r2_scores.append(r2_score(y_random, y_pred_random))

random_r2_scores = np.array(random_r2_scores)

z_score = (real_r2 - np.mean(random_r2_scores)) / np.std(random_r2_scores)

# =============================
# 7. Statistical robustness
# =============================
errors_classical = np.abs(y - y_pred_classical)
errors_full = np.abs(y - y_pred_best_full)

t_stat, p_value_t = stats.ttest_rel(errors_classical, errors_full)
w_stat, p_value_w = stats.wilcoxon(errors_classical, errors_full)

# =============================
# 8. Print final results
# =============================
print("\n===== FINAL RESULTS =====")
print("Classical R2:", round(r2_classical,4))
print("Best Model:", best_model_name)
print("Best Model R2:", round(real_r2,4))
print("Classical MAE, RMSE:", round(mae_classical,4), round(rmse_classical,4))
print("Full MAE, RMSE:", round(mae_full,4), round(rmse_full,4))
print("Z-score:", round(z_score,4))
print("t-test p:", round(p_value_t,4))
print("Wilcoxon p:", round(p_value_w,4))
