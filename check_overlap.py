
import pandas as pd

# Load the datasets
training = pd.read_csv("Training_clean.csv")
testing = pd.read_csv("Testing_clean.csv")

# Keep unique training symptom patterns
train_unique = training.drop_duplicates()

# Compare symptom patterns only, excluding disease labels
X_train = train_unique.drop(columns=["prognosis"])
X_test = testing.drop(columns=["prognosis"])

# Align feature columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Convert each symptom pattern to a tuple for comparison
train_patterns = set(map(tuple, X_train.to_numpy()))
test_patterns = list(map(tuple, X_test.to_numpy()))

overlap_count = sum(
    pattern in train_patterns for pattern in test_patterns
)

print("TRAIN-TEST OVERLAP CHECK")
print("------------------------")
print("Unique training patterns:", len(train_patterns))
print("Test records:", len(test_patterns))
print("Test patterns also in training:", overlap_count)
print(
    "Test patterns not found in training:",
    len(test_patterns) - overlap_count
)
