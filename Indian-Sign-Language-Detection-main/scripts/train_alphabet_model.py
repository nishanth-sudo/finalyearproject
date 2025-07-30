import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels
import joblib

# === Step 1: Load CSV and Assign Column Names ===

# Load your combined CSV (no header)
df = pd.read_csv('combined_data.csv', header=None)

# Set column names: f0 to f125 for features, and 'label' for the last column
num_features = df.shape[1] - 1  # 127 columns: 0–125 are features, 126 is label
column_names = [f'f{i}' for i in range(num_features)] + ['label']
df.columns = column_names

# === Step 2: Clean Labels ===

# Keep only valid single uppercase letters A–Z
df = df[df['label'].str.fullmatch(r'[A-Z]')]
df = df.reset_index(drop=True)

# === Step 3: Split Features and Labels ===

X = df.drop('label', axis=1)
y = df['label']

# Encode labels (A–Z → 0–25)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Step 4: Train-Test Split ===

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Step 5: Train Model ===

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# === Step 6: Evaluate ===

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n Model trained with accuracy: {acc*100:.2f}%\n")

# Handle only the labels present in y_test/y_pred
label_indices = unique_labels(y_test, y_pred)
print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    labels=label_indices,
    target_names=le.inverse_transform(label_indices)
))

# === Step 7: Save Model and Encoder ===

joblib.dump((model, le), 'gesture_model.pkl')
print("Model and label encoder saved as gesture_model.pkl")
