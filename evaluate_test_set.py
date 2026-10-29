
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load both cleaned datasets
training = pd.read_csv("Training_clean.csv")
testing = pd.read_csv("Testing_clean.csv")

# Remove duplicate training rows
training = training.drop_duplicates()

# Separate symptoms and labels
X_train = training.drop(columns=["prognosis"])
y_train = training["prognosis"]

X_test = testing.drop(columns=["prognosis"])
y_test = testing["prognosis"]

# Ensure the test features match the training features
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Train using unique training records only
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate on the separate supplied test dataset
predictions = model.predict(X_test)

print("SEPARATE TEST SET EVALUATION")
print("----------------------------")
print("Unique training records:", len(training))
print("Test records:", len(testing))
print(
    f"Test accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%"
)
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
