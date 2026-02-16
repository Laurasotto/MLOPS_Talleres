# Generated from: Untitled1.ipynb
# Converted at: 2026-02-16T23:34:05.017Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import os

for dirname, _, filenames in os.walk('/sample_data/penguins.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/content/sample_data/penguins.csv')

df

df.drop('Unnamed: 0', axis=1, inplace=True)

df.head(4)

df = pd.read_csv("/content/sample_data/penguins.csv")
print(df.head())

print(df.columns)

df = pd.read_csv("/content/sample_data/penguins.csv")
target_col = "species"

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Cargar datos (cambia la ruta por la tuya real)
df = pd.read_csv("/content/sample_data/penguins.csv")

# 2) Elegir target (columna a predecir) y features
target_col = "species"  # <-- cambia esto
X = df.drop(columns=[target_col])
y = df[target_col]

# 3) Si hay variables categóricas, convertir a numéricas (one-hot)
X = pd.get_dummies(X, drop_first=True)

# 4) Manejo simple de nulos (rápido y efectivo para ejemplo)
X = X.fillna(X.median(numeric_only=True))
y = y.fillna(y.mode()[0])

# 5) Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
)

# 6) Entrenar modelo
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5  # ajusta esto
)
model.fit(X_train, y_train)

# 7) Evaluar
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))