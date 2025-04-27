import pandas as pd

# Load a small portion first (the full dataset is too large to load entirely)
df = pd.read_csv('training_set_VU_DM.csv', nrows=50000)

# Basic info
print(df.shape)
print(df.info())

# Check missing values
print(df.isnull().sum())

# Describe numerical features
print(df.describe())

# Target variables distribution
print(df['click_bool'].value_counts(normalize=True))
print(df['booking_bool'].value_counts(normalize=True))
