from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import joblib
from Dataset import load_data

# 1. Load Data
df = load_data()

# 2. Feature Engineering
features = ['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target = 'Machine failure'

df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
df['wear_rate'] = df['Tool wear [min]'] / (df['Rotational speed [rpm]'] + 1)

all_features = features + ['temp_diff', 'power', 'wear_rate']

X = df[all_features]
y = df[target]

# 3. Split Data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale Data (Crucial for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 


# 5. Train Random Forest Model

print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,       
    max_depth=10,           
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)

# Generate RF predictions
y_pred_rf = rf_model.predict(X_test)

# Print Feature Importances
print("\n=== Random Forest Feature Importances ===")
importances = pd.Series(
    rf_model.feature_importances_, index=all_features
).sort_values(ascending=False)
print(importances)


# 6. Train Logistic Regression Model

print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)

# Generate LR predictions
y_pred_lr = lr_model.predict(X_test)


# 7. Evaluate and Compare Models

target_labels = ['Healthy', 'Needs Replacement']

print("\n" + "="*50)
print("📊 MODEL COMPARISON RESULTS")
print("="*50)

print("\n--- Random Forest Performance ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=target_labels))

print("\n--- Logistic Regression Performance ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=target_labels))

# 8. Save the preferred model to file (Saving RF here)
print("\nSaving Random Forest model & scaler to model.pkl...")
artifacts = {
    'model': rf_model,
    'scaler': scaler,
    'feature_names': all_features
}
joblib.dump(artifacts, 'model.pkl')
print("Training complete!")