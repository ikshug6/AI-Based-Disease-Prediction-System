
import pandas as pd

# Load the cleaned training dataset
data = pd.read_csv("Training_clean.csv")

# Check for completely identical rows
duplicate_rows = data.duplicated().sum()

# Check repeated symptom patterns, ignoring the disease label
symptoms = data.drop(columns=["prognosis"])
duplicate_symptom_patterns = symptoms.duplicated().sum()

# Check whether identical symptom patterns have different labels
combined = data.groupby(
    list(symptoms.columns),
    dropna=False
)["prognosis"].nunique()

conflicting_patterns = (combined > 1).sum()

print("DUPLICATE DATA CHECK")
print("--------------------")
print("Total training records:", len(data))
print("Completely duplicated rows:", duplicate_rows)
print("Repeated symptom patterns:", duplicate_symptom_patterns)
print("Patterns associated with multiple disease labels:", conflicting_patterns)
