import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("dataset.csv")

# To remove unnecessary columns 
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Printing column names for debugging
print("Columns in dataset:", df.columns)

# check if 'TenYearCHD' is in dataset
if "TenYearCHD" not in df.columns:
    raise ValueError("Error: 'TenYearCHD' column not in dataset!")

# to define features and target
X = df.drop(columns=["TenYearCHD"])  # <<<Features
y = df["TenYearCHD"]                 # <<<Target variable

# Here spliting the  data into training (80%)/testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Here we train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions \/\/


y_pred = model.predict(X_test)

# Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

# Show classification report

print("Classification Report:")

print(classification_report(y_test, y_pred))


# Save model

with open("model.pkl", "wb") as f:
    
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")

