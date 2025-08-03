import pandas as pd

# Load the datasets
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

# Add a new column 'label' to each
fake['label'] = 'FAKE'
true['label'] = 'REAL'

# Select only required columns
fake = fake[['text', 'label']]
true = true[['text', 'label']]

# Combine the two dataframes
combined = pd.concat([fake, true])

# Shuffle the rows
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to new CSV
combined.to_csv('news.csv', index=False)

print("Combined CSV saved as 'news.csv'")
