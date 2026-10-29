
import pandas as pd
import os

print("Checking dataset files...")

print("Training.csv exists:", os.path.exists("Training.csv"))
print("Testing.csv exists:", os.path.exists("Testing.csv"))

training = pd.read_csv("Training.csv")
testing = pd.read_csv("Testing.csv")

print("\nTraining dataset shape:", training.shape)
print("Testing dataset shape:", testing.shape)

print("\nFirst five rows of training data:")
print(training.head())

print("\nColumn names:")
print(training.columns.tolist())

print("\nMissing values:", training.isnull().sum().sum())
