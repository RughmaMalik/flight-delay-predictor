# ----------------
# 0. Import libraries
# ----------------
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter

# ----------------
# 1. Load dataset
# ----------------
print("\nLoading dataset...")
df = pd.read_csv("../data/flights_dataset.csv")

print("\n===== DATA OVERVIEW =====")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isna().sum())

# ----------------
# 2. Remove cancelled flights
# ----------------
df = df[df["cancelled"] == 0]

# ----------------
# 3. Create target variable
# ----------------
df["delay"] = (df["dep_delay"] > 15).astype(int)

print("\n===== AFTER CLEANING =====")
print("Shape:", df.shape)

print("\nTarget distribution:")
print(df["delay"].value_counts(normalize=True))

# ----------------
# 4. Feature selection
# ----------------
features = [
    'month',
    'day_of_month',
    'day_of_week',
    'op_unique_carrier',
    'origin',
    'dest',
    'crs_dep_time',
    'distance',
]

X = df[features].copy()
y = df["delay"]

print("Final feature columns:", X.columns.tolist())

# ----------------
# 5. Handle missing values (NUMERIC ONLY)
# ----------------
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

num_imputer = SimpleImputer(strategy="median")
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

# ----------------
# 6. Label encode categorical variables
# ----------------
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print("\n===== FEATURE CHECK =====")
print("Numeric columns:", list(numeric_cols))
print("Categorical columns:", list(categorical_cols))

# ----------------
# 7. Train/Test split
# ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------
# 8. Handle class imbalance using RandomOverSampler
# ----------------
ros = RandomOverSampler(random_state=42)

X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print("\nClass distribution BEFORE oversampling:")
print(Counter(y_train))

print("\nClass distribution AFTER oversampling:")
print(Counter(y_train_resampled))

# ----------------
# 9. Feature scaling (AFTER oversampling)
# ----------------
scaler = StandardScaler()

X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

print("\nClass balance in training set:")
print(Counter(y_train_resampled))

# ----------------
# 10. Define Models
# ----------------

models = { 
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    
    "ANN": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=2500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42
    )
}

# ----------------
# 11. Evaluate model
# ----------------

best_model_name = None
best_model_score = 0 
best_model_obj = None

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name} with RandomOverSampling...")
    
    model.fit(X_train_resampled, y_train_resampled)
    trained_models[name] = model
    
    y_pred = model.predict(X_test)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0)
    })

    if f1_score(y_test, y_pred, zero_division=0) > best_model_score:
        best_model_score = f1_score(y_test, y_pred, zero_division=0)
        best_model_name = name
        best_model_obj = model

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-Score", ascending=False)

print("\n===== MODEL COMPARISON =====")
print(results_df)

best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test)

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        y_pred_best,
        target_names=["On-Time", "Delayed"]
    )
)

cm_best = confusion_matrix(y_test, y_pred_best)
print("\n===== BEST MODEL CONFUSION MATRIX =====")
print(cm_best)

best_model_params_df = pd.DataFrame(
    best_model.get_params().items(),
    columns=["Parameter", "Value"]
)

print("\n===== BEST MODEL PARAMETERS =====")
print(best_model_params_df)

best_model_params_df.to_csv("../data/best_model_parameters.csv", index=False)

# ----------------
# 12. Save model and preprocessing artifacts
# ----------------

print("\nSaving model and preprocessing artifacts...")

os.makedirs("../models", exist_ok=True)

# Save best model
joblib.dump(best_model, "../models/flight_delay_model.pkl")

# Save scaler
joblib.dump(scaler, "../models/scaler.pkl")

# Save label encoders
joblib.dump(label_encoders["op_unique_carrier"], "../models/le_carrier.pkl")
joblib.dump(label_encoders["origin"], "../models/le_origin.pkl")
joblib.dump(label_encoders["dest"], "../models/le_dest.pkl")

print("Models saved successfully.")