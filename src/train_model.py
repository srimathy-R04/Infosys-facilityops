
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


# 1. Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "maintenance_data"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

# 2. Dataset column names
cols = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

# 3. Load NASA datasets
train = pd.read_csv(
    DATA / "train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=cols
)

test = pd.read_csv(
    DATA / "test_FD001.txt",
    sep=r"\s+",
    header=None,
    names=cols
)

actual_rul = pd.read_csv(
    DATA / "RUL_FD001.txt",
    header=None
).iloc[:, 0].to_numpy()

print("Datasets loaded successfully")
print("Training shape:", train.shape)
print("Testing shape:", test.shape)


# 4. Create RUL labels for training data
max_cycles = train.groupby("unit")["cycle"].max()

train["RUL"] = (
    train["unit"].map(max_cycles) - train["cycle"]
)


# 5. Select useful sensors
sensor_cols = [f"s{i}" for i in range(1, 22)]

useful_sensors = [
    sensor for sensor in sensor_cols
    if train[sensor].std() > 0.01
]

features = ["cycle", "op1", "op2", "op3"] + useful_sensors

print("\nUseful sensors:", useful_sensors)
print("Number of useful sensors:", len(useful_sensors))


# 6. Split by engine to avoid data leakage
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        train[features],
        train["RUL"],
        groups=train["unit"]
    )
)

X_train = train.iloc[train_idx][features]
y_train = train.iloc[train_idx]["RUL"]

X_val = train.iloc[val_idx][features]
y_val = train.iloc[val_idx]["RUL"]


# 7. Train Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# 8. Evaluate on unseen engines
val_predictions = np.maximum(
    model.predict(X_val), 0
)

mae = mean_absolute_error(y_val, val_predictions)

rmse = np.sqrt(
    mean_squared_error(y_val, val_predictions)
)

print("\nValidation Results")
print("------------------")
print("MAE:", round(mae, 2), "cycles")
print("RMSE:", round(rmse, 2), "cycles")


# 9. Train final model using all training engines
final_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

final_model.fit(train[features], train["RUL"])


# 10. Get the latest reading for each test engine
test_latest = (
    test.sort_values("cycle")
    .groupby("unit")
    .tail(1)
    .sort_values("unit")
    .copy()
)

X_test = test_latest[features]

predicted_rul = np.maximum(
    final_model.predict(X_test), 0
)

# 11. Compare predictions with actual test RUL
test_mae = mean_absolute_error(
    actual_rul, predicted_rul
)

test_rmse = np.sqrt(
    mean_squared_error(actual_rul, predicted_rul)
)

print("\nNASA Test Results")
print("-----------------")
print("Test MAE:", round(test_mae, 2), "cycles")
print("Test RMSE:", round(test_rmse, 2), "cycles")


# 12. Display engine predictions
results = pd.DataFrame({
    "engine": test_latest["unit"].to_numpy(),
    "actual_RUL": actual_rul,
    "predicted_RUL": np.round(predicted_rul, 1)
})

print("\nSample predictions:")
print(results.head(10).to_string(index=False))


# 13. Save trained model and feature information
joblib.dump(
    {
        "model": final_model,
        "features": features
    },
    MODEL_DIR / "cmapss_rf.joblib"
)

results.to_csv(
    MODEL_DIR / "cmapss_predictions.csv",
    index=False
)

print("\nModel saved successfully!")
print("Location:", MODEL_DIR / "cmapss_rf.joblib")
