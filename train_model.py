import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample training dataset
data = {
    "temperature": [25, 30, 35, 28, 33, 26, 31, 29],
    "humidity": [60, 65, 70, 55, 75, 50, 68, 62],
    "soil_type": ["loamy", "sandy", "clay", "loamy", "silty", "sandy", "clay", "silty"],
    "pest": ["none", "aphid", "borer", "aphid", "none", "borer", "aphid", "none"],
    "crop": ["wheat", "rice", "maize", "wheat", "rice", "maize", "wheat", "rice"]
}

df = pd.DataFrame(data)

# Encode categorical data
df = pd.get_dummies(df)

# Features and target
X = df.drop(columns=["crop_rice", "crop_wheat", "crop_maize"], errors='ignore')
y = df["crop_wheat"]  # You can build one-vs-all models per crop

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "backend/models/crop_model.pkl")
print("✅ Model trained and saved!")
