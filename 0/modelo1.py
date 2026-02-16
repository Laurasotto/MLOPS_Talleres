# Generated from: Untitled1.ipynb
# Converted at: 2026-02-16T23:34:05.017Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell
import os
import joblib
import pandas as pd
from palmerpenguins import load_penguins

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "dt_penguins.joblib")
COLS_PATH = os.path.join(MODEL_DIR, "model_columns.joblib")

def main():
    # 1) Cargar datos (palmerpenguins)
    df = load_penguins()

    # 2) Target y features
    target_col = "species"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3) One-hot
    X = pd.get_dummies(X, drop_first=True)

    # 4) Nulos
    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.mode()[0])

    # 5) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6) Entrenar
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    # 7) Evaluar
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 8) Guardar modelo + columnas
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), COLS_PATH)
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {COLS_PATH}")

if __name__ == "__main__":
    main()
