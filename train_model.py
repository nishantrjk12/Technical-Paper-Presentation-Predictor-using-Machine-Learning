
# Step 2: Train the ML model on the generated dataset
# Uses Random Forest Classifier - simple and effective for this kind of problem

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# ---- Load the dataset ----
df = pd.read_csv('dataset.csv')

print("Dataset loaded. Shape:", df.shape)
print("\nCategory counts:")
print(df['Category'].value_counts())

# ---- Separate features (X) and target (y) ----
X = df.drop('Category', axis=1)
y = df['Category']

# ---- Split into training and testing sets (80/20) ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---- Train a Random Forest model ----
# n_estimators=50 keeps it simple and fast enough for a student project
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

print("\nModel training complete!")

# ---- Evaluate the model ----
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy: {:.2f}%".format(accuracy * 100))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
print(cm)
print("(Rows = Actual, Columns = Predicted)")
print("Labels order: Low, Medium, High")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# ---- Save the trained model to a file ----
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl — ready for the UI!")
