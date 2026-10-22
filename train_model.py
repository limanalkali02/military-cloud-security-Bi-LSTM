"""
train_model.py

Trains the Bi-LSTM intrusion classifier defined in model_engine.py on
realistic synthetic CICIDS2017-style flow sequences (see cicids_features.py),
then:
  1. Calibrates a proper decision threshold on the validation set instead
     of blindly using 0.5 (this is the main lever for cutting false alerts).
  2. Trains a lightweight secondary classifier that names *which* attack
     type a malicious flow matches (DoS Hulk / Slowloris / GoldenEye /
     Heartbleed), so alerts carry real detail instead of a generic flag.
  3. Saves everything needed for inference: model, feature scaler,
     threshold, attack-type classifier, feature schema.

Run:  python3 train_model.py
"""

import json
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
import joblib
import tensorflow as tf

from cicids_features import generate_dataset, ATTACK_CLASSES, FEATURE_NAMES
from model_engine import build_model

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TIME_STEPS = 10
N_PER_CLASS = 700          # 700 x 5 classes = 3500 sequences
SEED = 42


def main():
    print("=" * 70)
    print("STEP 1: Generating realistic synthetic flow-sequence dataset")
    print("=" * 70)
    X, y_bin, y_cls = generate_dataset(n_per_class=N_PER_CLASS, time_steps=TIME_STEPS, seed=SEED)
    print(f"Dataset shape: {X.shape}, positives (attack): {y_bin.sum()}/{len(y_bin)}")

    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    X_train, y_train = X[train_idx], y_bin[train_idx]
    X_val, y_val = X[val_idx], y_bin[val_idx]
    X_test, y_test = X[test_idx], y_bin[test_idx]
    cls_test = y_cls[test_idx]

    print("\n" + "=" * 70)
    print("STEP 2: Min-Max scaling features to [0, 1] (fit on train only)")
    print("=" * 70)
    n_features = X.shape[2]
    scaler = MinMaxScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    def scale(arr):
        shape = arr.shape
        flat = arr.reshape(-1, n_features)
        flat = scaler.transform(flat)
        return flat.reshape(shape).astype(np.float32)

    X_train_s = scale(X_train)
    X_val_s = scale(X_val)
    X_test_s = scale(X_test)

    print("\n" + "=" * 70)
    print("STEP 3: Building and training the Bi-LSTM model")
    print("=" * 70)
    model = build_model(time_steps=TIME_STEPS, num_features=n_features)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=30,
        batch_size=64,
        callbacks=[early_stop],
        verbose=2,
    )

    print("\n" + "=" * 70)
    print("STEP 4: Calibrating decision threshold on the validation set")
    print("=" * 70)
    # Instead of hardcoding 0.5 (a major source of false positives if the
    # model's probability distribution is not well-centered), pick the
    # threshold from the validation ROC curve that maximizes
    # (True Positive Rate - False Positive Rate), a.k.a. Youden's J statistic.
    val_probs = model.predict(X_val_s, verbose=0).ravel()
    fpr, tpr, thresholds = roc_curve(y_val, val_probs)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[best_idx])
    # Guard against degenerate threshold values
    threshold = float(np.clip(threshold, 0.15, 0.9))
    print(f"Calibrated decision threshold: {threshold:.4f}")

    print("\n" + "=" * 70)
    print("STEP 5: Evaluating on held-out test set")
    print("=" * 70)
    test_probs = model.predict(X_test_s, verbose=0).ravel()
    test_preds = (test_probs >= threshold).astype(int)

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    cm = confusion_matrix(y_test, test_preds)

    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print("Confusion Matrix [ [TN FP] [FN TP] ]:")
    print(cm)
    fp_rate = cm[0, 1] / max(cm[0].sum(), 1)
    print(f"False Positive Rate on benign test flows: {fp_rate*100:.2f}%")

    print("\n" + "=" * 70)
    print("STEP 6: Training secondary attack-type classifier (for alert detail)")
    print("=" * 70)
    # Flatten sequences to summary stats (mean+std over time) as features
    # for a lightweight classifier that names the specific attack type.
    def summarize(arr):
        return np.concatenate([arr.mean(axis=1), arr.std(axis=1)], axis=1)

    X_train_summary = summarize(X_train_s)
    X_test_summary = summarize(X_test_s)

    type_clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=SEED)
    type_clf.fit(X_train_summary, y_cls[train_idx])
    type_acc = type_clf.score(X_test_summary, cls_test)
    print(f"Attack-type classifier accuracy (test set): {type_acc*100:.2f}%")

    print("\n" + "=" * 70)
    print("STEP 7: Saving artifacts")
    print("=" * 70)
    model.save(os.path.join(ARTIFACT_DIR, "bilstm_model.keras"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    joblib.dump(type_clf, os.path.join(ARTIFACT_DIR, "attack_type_classifier.pkl"))
    with open(os.path.join(ARTIFACT_DIR, "config.json"), "w") as f:
        json.dump({
            "time_steps": TIME_STEPS,
            "num_features": n_features,
            "threshold": threshold,
            "feature_names": FEATURE_NAMES,
            "attack_classes": ATTACK_CLASSES,
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_false_positive_rate": fp_rate,
        }, f, indent=2)

    print(f"Saved model, scaler, attack-type classifier and config to: {ARTIFACT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
