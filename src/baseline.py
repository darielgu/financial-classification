"""Stratified DummyClassifier baseline (per-class-prior random)."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


def run_baseline_diagnostic(x_train, y_train, x_test, y_test) -> None:
    """Fit a stratified DummyClassifier on train and report test F1 + accuracy."""
    print("\n" + "=" * 60)
    print("BASELINE DIAGNOSTIC: Stratified Classifier")
    print("=" * 60)
    baseline = DummyClassifier(strategy="stratified", random_state=42)
    baseline.fit(x_train, y_train)
    y_baseline_pred = baseline.predict(x_test)
    baseline_f1 = f1_score(y_test, y_baseline_pred, average="macro", zero_division=0)
    baseline_acc = (y_baseline_pred == y_test).mean()

    print(f"Baseline Accuracy   : {baseline_acc:.4f}")
    print(f"Baseline F1 (macro) : {baseline_f1:.4f}")
    print("=" * 60 + "\n")
