import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data/winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

# Feature selection using correlation
corr = df.corr()["quality"].abs()
selected_features = corr[corr > 0.2].index.drop("quality")
X_sel = df[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = Ridge(alpha=1.0)
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
