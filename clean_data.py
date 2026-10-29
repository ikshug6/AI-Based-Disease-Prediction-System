
import pandas as pd

# Load the original datasets
training = pd.read_csv("Training.csv")
testing = pd.read_csv("Testing.csv")

# Remove columns that are completely empty
training = training.dropna(axis=1, how="all")
testing = testing.dropna(axis=1, how="all")

# Remove duplicate column names if any were created
training = training.loc[:, ~training.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]

# Save the cleaned datasets
training.to_csv("Training_clean.csv", index=False)
testing.to_csv("Testing_clean.csv", index=False)

print("Dataset cleaning completed!")
print("Training dataset shape:", training.shape)
print("Testing dataset shape:", testing.shape)
print("Training missing values:", training.isnull().sum().sum())
print("Testing missing values:", testing.isnull().sum().sum())
print("Disease label column exists:", "prognosis" in training.columns)
