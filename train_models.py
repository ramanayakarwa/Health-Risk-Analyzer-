import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import sqlite3
import json
import datetime
import pickle
import os

# ------------------ Ensure Models Folder Exists ------------------
os.makedirs("models", exist_ok=True)

# ------------------ SQLite Logging ------------------
def log_model_results(disease_name, acc, model_type="RandomForest", features=None):
    
    
    
    conn = sqlite3.connect("models/model_results.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS training_logs (
        disease TEXT,
        model_type TEXT,
        accuracy REAL,
        n_features INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    c.execute("INSERT INTO training_logs (disease, model_type, accuracy, n_features) VALUES (?, ?, ?, ?)",
              (disease_name, model_type, acc, len(features) if features else 0))
    conn.commit()
    conn.close()

# ------------------ Feature Importance ------------------
def show_feature_importance(model, X, y, disease_name):
    results = permutation_importance(model, X, y, scoring='accuracy', n_repeats=5, random_state=42)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': results.importances_mean
    }).sort_values(by='Importance', ascending=False)
    importance_df.to_csv(f"models/{disease_name.lower()}_feature_importance.csv", index=False)
    print(f" Top Features for {disease_name}:")
    print(importance_df.head(5))
    return importance_df

# ------------------ Model Training Function ------------------
def train_save_rf_model(X, y, disease_name):
    print(f"\n--- TRAINING {disease_name.upper()} MODEL ---")
    
    # Handle imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                        test_size=0.2, random_state=42, stratify=y_resampled)

    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ {disease_name} Best Accuracy: {acc:.4f}")
    print("Best Params:", grid.best_params_)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    pickle.dump((best_model, X.columns.tolist()), open(f"models/{disease_name.lower()}_rf_model.pkl", "wb"))
    log_model_results(disease_name, acc, "RandomForest_GridSearch", X.columns.tolist())
    show_feature_importance(best_model, X_test, y_test, disease_name)

    # Save summary JSON
    summary = {
        "model": disease_name,
        "accuracy": acc,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_params": grid.best_params_
    }
    with open(f"models/{disease_name.lower()}_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

# ------------------- HEART DISEASE -------------------
heart_path = "data/heart_cleveland_upload.csv"
if not os.path.exists(heart_path):
    raise FileNotFoundError(f"File not found: {heart_path}")

heart_df = pd.read_csv(heart_path)
if "condition" not in heart_df.columns:
    raise ValueError("Column 'condition' not found in heart dataset.")

X_heart = heart_df.drop("condition", axis=1)
y_heart = heart_df["condition"]
train_save_rf_model(X_heart, y_heart, "Heart")

# ------------------- DIABETES -------------------
diabetes_path = "data/Dataset of Diabetes .csv"
if not os.path.exists(diabetes_path):
    raise FileNotFoundError(f"File not found: {diabetes_path}")

diabetes_df = pd.read_csv(diabetes_path)
diabetes_df["CLASS"] = diabetes_df["CLASS"].astype(str).str.strip().str.upper()
diabetes_df = diabetes_df[diabetes_df["CLASS"].isin(['Y', 'N'])]
diabetes_df["CLASS"] = diabetes_df["CLASS"].map({'Y': 1, 'N': 0})

X_diabetes = diabetes_df.drop(["CLASS", "ID", "No_Pation"], axis=1, errors="ignore")
y_diabetes = diabetes_df["CLASS"]
X_diabetes = pd.get_dummies(X_diabetes, drop_first=True)
train_save_rf_model(X_diabetes, y_diabetes, "Diabetes")

# ------------------- STROKE -------------------
stroke_path = "data/healthcare-dataset-stroke-data.csv"
if not os.path.exists(stroke_path):
    raise FileNotFoundError(f"File not found: {stroke_path}")

stroke_df = pd.read_csv(stroke_path)
if "stroke" not in stroke_df.columns:
    raise ValueError("Column 'stroke' not found in stroke dataset.")
if "id" in stroke_df.columns:
    stroke_df = stroke_df.drop("id", axis=1)

stroke_df = pd.get_dummies(stroke_df, drop_first=True)
X_stroke = stroke_df.drop("stroke", axis=1)
y_stroke = stroke_df["stroke"]

imputer = SimpleImputer(strategy="median")
X_stroke = pd.DataFrame(imputer.fit_transform(X_stroke), columns=X_stroke.columns)
train_save_rf_model(X_stroke, y_stroke, "Stroke")

print("\n All Random Forest models trained, tuned, and saved successfully in 'models/' folder.")
