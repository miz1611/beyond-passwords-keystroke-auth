"""
train_model.py
---------------
Trains a Random Forest classifier on keystroke-dynamics features and
reproduces the evaluation artefacts from the paper:

    - Figure 1: Feature Importance Ranking
    - Figure 2: Scatter plot of genuine-user vs imposter clusters
    - Figure 3: Confusion matrix
    - Accuracy + Equal Error Rate (EER)

Usage:
    python train_model.py --data ../data/keystroke_features.csv
    python train_model.py --data ../data/sample_keystroke_data.csv   # demo data
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "avg_dwell_time", "std_dwell_time",
    "avg_flight_time", "std_flight_time", "error_rate",
]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def compute_eer(y_true, y_scores):
    """Equal Error Rate from the ROC curve: point where FAR == FRR."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    return (fpr[eer_index] + fnr[eer_index]) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to features CSV")
    parser.add_argument("--test-size", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(args.data)

    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    eer = compute_eer(y_test, y_scores)
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Equal Error Rate (EER): {eer * 100:.2f}%")

    # --- Figure 1: Feature importance ---
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values()
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color=sns.color_palette("viridis", len(importances)))
    plt.title("Keystroke Dynamics: Feature Importance Analysis")
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Biometric Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"))
    plt.close()

    # --- Figure 2: Scatter of dwell vs flight time by class ---
    plt.figure(figsize=(7, 6))
    for label, colour, name in [(1, "tab:blue", "Genuine user"), (0, "tab:red", "Imposter")]:
        subset = df[df["label"] == label]
        plt.scatter(subset["avg_dwell_time"], subset["avg_flight_time"],
                    label=name, color=colour)
    plt.xlabel("Average Dwell Time (seconds)")
    plt.ylabel("Average Flight Time (seconds)")
    plt.title("Biometric Data Distribution - Keystroke Dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "scatter_clusters.png"))
    plt.close()

    # --- Figure 3: Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Genuine", "Imposter"],
                yticklabels=["Genuine", "Imposter"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Biometric Authentication Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    print(f"\nFigures written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
