
import pandas as pd

data = pd.read_csv("Training_clean.csv")

# Keep one copy of each complete symptom pattern
unique_data = data.drop_duplicates()

# Count unique patterns for each disease
counts = unique_data["prognosis"].value_counts().sort_values()

print("UNIQUE PATTERNS PER DISEASE")
print("---------------------------")
print(counts.to_string())

print("\nTotal disease classes:", len(counts))
print("Diseases with fewer than 5 unique patterns:", (counts < 5).sum())
print("Minimum unique patterns for one disease:", counts.min())
print("Maximum unique patterns for one disease:", counts.max())
