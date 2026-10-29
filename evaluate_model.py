
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the cleaned training dataset
data = pd.read_csv("Training_clean.csv")

# Separate symptoms and disease labels
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train a fresh model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Training evaluation model...")
model.fit(X_train, y_train)

# Evaluate on held-out validation records
predictions = model.predict(X_valid)
accuracy = accuracy_score(y_valid, predictions)

print("\nEvaluation completed!")
print("Training records:", len(X_train))
print("Validation records:", len(X_valid))
print(f"Validation accuracy: {accuracy * 100:.2f}%")
print("\nRemember: this measures performance on this dataset,")
print("not clinical accuracy on real patients.")
