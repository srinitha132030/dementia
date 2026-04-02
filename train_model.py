import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_excel("data\\oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx")

# Clean columns
df.columns = df.columns.str.strip()

# Encode
le = LabelEncoder()
df['M/F'] = le.fit_transform(df['M/F'])
df['Group'] = le.fit_transform(df['Group'])

# Drop unnecessary
df.drop(['Subject ID', 'MRI ID', 'Hand'], axis=1, inplace=True)

# Split
X = df.drop("Group", axis=1)
y = df["Group"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save files
joblib.dump(model, "dementia_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Files created!")