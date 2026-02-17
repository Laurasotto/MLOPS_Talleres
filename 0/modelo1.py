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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MODEL_DIR = "model"
COLS_PATH = os.path.join(MODEL_DIR, "model_columns.joblib")
ACTIVE_PATH = os.path.join(MODEL_DIR, "active_model.txt")


def main():
    # 1) Cargar datos
    df = load_penguins()

    # 2) Target y features
    target_col = "species"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3) One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # 4) Manejo de nulos
    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.mode()[0])

    # 5) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6) Crear carpeta model/
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 7) Guardar columnas (para que el API alinee)
    joblib.dump(X.columns.tolist(), COLS_PATH)
    print(f"Saved: {COLS_PATH}")

    # 8) Definir modelos
    modelos = {
        "dt": DecisionTreeClassifier(random_state=42, max_depth=5),
        "rf": RandomForestClassifier(random_state=42, n_estimators=300),
        "lr": LogisticRegression(max_iter=2000),
    }

    # 9) Entrenar, evaluar y guardar cada modelo
    results = []
    for name, model in modelos.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(model, model_path)

        results.append((name, acc))
        print(f"Saved: {model_path} | acc={acc:.4f}")

    # 10) Dejar activo el mejor por accuracy
    best_name = sorted(results, key=lambda x: x[1], reverse=True)[0][0]
    with open(ACTIVE_PATH, "w") as f:
        f.write(best_name)

    print(f"Active model: {best_name} (Saved: {ACTIVE_PATH})")


if __name__ == "__main__":
    main()

