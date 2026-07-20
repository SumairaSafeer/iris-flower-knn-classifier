"""
================================================================
 DecodeLabs — Project 2: Data Classification Using AI
 Goal: Build a basic classification model using the Iris dataset
 Algorithm: K-Nearest Neighbors (KNN)
================================================================

This script follows the exact IPO Framework from the training kit:

  INPUT   -> Load Iris dataset, understand it, scale features
  PROCESS -> Train-test split, apply KNN algorithm, tune K
  OUTPUT  -> Confusion Matrix, F1 Score, Accuracy report

Run with:  python project2_iris_classification.py
================================================================
"""

# ----------------------------------------------------------------
# STEP 1: IMPORT LIBRARIES
# ----------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)

# ----------------------------------------------------------------
# STEP 2: INPUT — LOAD AND UNDERSTAND THE DATASET
# ----------------------------------------------------------------
# "Raw Material: The Iris Benchmark" -> 150 samples, 3 classes, 4 features
iris = load_iris()
X = iris.data                     # Features: sepal length/width, petal length/width
y = iris.target                   # Labels: 0=setosa, 1=versicolor, 2=virginica
feature_names = iris.feature_names
class_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, class_names)

print("=" * 60)
print("STEP 2: DATASET OVERVIEW")
print("=" * 60)
print(f"Samples: {df.shape[0]}, Features: {X.shape[1]}, Classes: {len(class_names)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution (balanced check):")
print(df["species"].value_counts())
print("\nStatistical summary:")
print(df.describe())

# ----------------------------------------------------------------
# STEP 3: PROCESS — TRAIN-TEST SPLIT ("Structural Integrity: The Split")
# ----------------------------------------------------------------
# Shuffle happens automatically inside train_test_split (random_state fixes it
# for reproducibility). 80% train / 20% test, as shown in "The Full Architecture".
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y   # keeps class balance equal in both train & test sets
)

print("\n" + "=" * 60)
print("STEP 3: TRAIN-TEST SPLIT")
print("=" * 60)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# ----------------------------------------------------------------
# STEP 4: THE GATEKEEPER RULE — FEATURE SCALING
# ----------------------------------------------------------------
# KNN is distance-based, so features MUST be on the same scale.
# IMPORTANT: fit the scaler ONLY on training data, then transform both,
# to avoid data leakage from the test set.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("STEP 4: FEATURE SCALING (StandardScaler: mean=0, variance=1)")
print("=" * 60)
print("Sample before scaling:", np.round(X_train[0], 2))
print("Sample after scaling :", np.round(X_train_scaled[0], 2))

# ----------------------------------------------------------------
# STEP 5: TUNING THE ENGINE — CHOOSING THE BEST "K"
# ----------------------------------------------------------------
# Test K values from 1 to 20 and track error rate to find "the elbow"
error_rates = []
k_range = range(1, 21)

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    pred_temp = knn_temp.predict(X_test_scaled)
    error_rates.append(np.mean(pred_temp != y_test))

best_k = k_range[np.argmin(error_rates)]

print("\n" + "=" * 60)
print("STEP 5: CHOOSING OPTIMAL K")
print("=" * 60)
print(f"Best K found: {best_k} (lowest error rate: {min(error_rates):.4f})")

plt.figure(figsize=(8, 5))
plt.plot(k_range, error_rates, marker="o", linestyle="--", color="darkblue")
plt.axvline(best_k, color="orange", linestyle=":", label=f"Optimal K = {best_k}")
plt.title("Tuning the Engine: Choosing K")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("k_tuning_curve.png", dpi=150)
print("Saved plot -> k_tuning_curve.png")

# ----------------------------------------------------------------
# STEP 6: THE WORKFLOW — INSTANTIATE, FIT, PREDICT
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: TRAINING THE FINAL KNN MODEL")
print("=" * 60)

model = KNeighborsClassifier(n_neighbors=best_k)   # INSTANTIATE
model.fit(X_train_scaled, y_train)                 # FIT (memorize the map)
predictions = model.predict(X_test_scaled)          # PREDICT (apply logic)

print(f"Model trained with K={best_k} neighbors.")

# ----------------------------------------------------------------
# STEP 7: OUTPUT VALIDATION — CONFUSION MATRIX & METRICS
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: MODEL EVALUATION")
print("=" * 60)

acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")
cm = confusion_matrix(y_test, predictions)

print(f"Accuracy Score : {acc:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nFull Classification Report:")
print(classification_report(y_test, predictions, target_names=class_names))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (KNN, K={best_k})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved plot -> confusion_matrix.png")

# ----------------------------------------------------------------
# STEP 8: PREDICT ON A NEW, UNSEEN SAMPLE (real-world usage demo)
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8: PREDICTING A NEW FLOWER SAMPLE")
print("=" * 60)

new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # sepal_len, sepal_wid, petal_len, petal_wid
new_sample_scaled = scaler.transform(new_sample)
new_prediction = model.predict(new_sample_scaled)

print(f"New sample: {new_sample[0]}")
print(f"Predicted species: {class_names[new_prediction[0]]}")

print("\n" + "=" * 60)
print("PROJECT 2 COMPLETE — Model trained, tested, and validated.")
print("=" * 60)
