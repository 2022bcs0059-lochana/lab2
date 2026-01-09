import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Print metrics (IMPORTANT)
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Create output directories if they don't exist
os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)


# Save model
joblib.dump(model, "outputs/model/model.pkl")

# Save metrics
metrics = {
    "MSE": mse,
    "R2": r2
}

with open("outputs/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
