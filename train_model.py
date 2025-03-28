# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
csv_path = "data/crime_data.csv"
df = pd.read_csv(csv_path)

# Clean and rename columns
df.columns = df.columns.str.strip()
df.rename(columns={
    'Year': 'year',
    'District': 'district',
    'Crime Category': 'crime_category',
    'Crime Severity': 'crime_severity',
    'Number of Cases': 'num_cases'
}, inplace=True)

# Select features and target
y = df['num_cases']
X = df[['year', 'district', 'crime_category', 'crime_severity']]

# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['district', 'crime_category', 'crime_severity'], drop_first=True)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open('models/model.pkl', 'wb'))
print("✅ Model trained and saved at models/model.pkl")
print(f"✅ Model Test Accuracy: {model.score(X_test, y_test):.2f}")
