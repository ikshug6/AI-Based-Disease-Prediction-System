
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the cleaned dataset
data = pd.read_csv("Training_clean.csv")

# Remove repeated symptom patterns
# Each symptom pattern has one disease label in this dataset
data = data.drop_duplicates()

print("Original records: 4920")
print("Unique records after removing duplicates:", len(data))

# Separate inputs and target
X = data.drop(columns=["prognosis"])
y = data["prognosis"]

# Split the unique records
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nTraining on unique records...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_valid)

print("\nUnique-record evaluation completed!")
print("Training records:", len(X_train))
print("Validation records:", len(X_valid))
print(f"Validation accuracy: {accuracy_score(y_valid, predictions) * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_valid, predictions, zero_division=0))
