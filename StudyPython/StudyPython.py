#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import os
print(os.getcwd())
compatabilty_pairs = "data/compatibility_pairs.csv"

# Load dataset
df = pd.read_csv(compatabilty_pairs)

# Select features (independent variables)
features = [
    "skill_match_score",
    "skill_complementarity_score",
    "network_value_a_to_b",
    "network_value_b_to_a",
    "career_alignment_score",
    "experience_gap",
    "industry_match",
    "geographic_score",
    "seniority_match"
]

X = df[features]
y = df["compatibility_score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for regression stability)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
r2 = r2_score(y_test, y_pred)
print(f"R² score: {r2:.4f}")

# Show coefficients
coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})
print("\nModel coefficients:")
print(coefficients)
