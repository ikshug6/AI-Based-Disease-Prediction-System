
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned datasets
training = pd.read_csv("Training_clean.csv")
testing = pd.read_csv("Testing_clean.csv")

# Separate symptoms (input) from disease labels (output)
X_train = training.drop("prognosis", axis=1)
y_train = training["prognosis"]

X_test = testing.drop("prognosis", axis=1)
y_test = testing["prognosis"]

# Create the machine-learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate on the supplied testing dataset
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nModel training completed!")
print("Number of symptom features:", X_train.shape[1])
print("Number of disease classes:", y_train.nunique())
print(f"Test dataset accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model and feature names
with open("disease_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("symptom_columns.pkl", "wb") as file:
    pickle.dump(X_train.columns.tolist(), file)

print("\nModel saved as disease_model.pkl")
print("Symptom columns saved as symptom_columns.pkl")
