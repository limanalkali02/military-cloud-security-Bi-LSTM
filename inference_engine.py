"""
inference_engine.py

Loads the trained Bi-LSTM model, feature scaler, calibrated threshold,
and attack-type classifier produced by train_model.py, and exposes a
single function `classify_window()` that the sniffer calls on every
rolling window of flow features.

This is the module that was missing from the original project: previously
nothing loaded model_engine.py's model at all, so the dashboard's "alerts"
were not model output.
"""

import json
import os

import numpy as np
import joblib
import tensorflow as tf

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class IntrusionInferenceEngine:
    def __init__(self, artifact_dir: str = ARTIFACT_DIR):
        config_path = os.path.join(artifact_dir, "config.json")
        model_path = os.path.join(artifact_dir, "bilstm_model.keras")
        scaler_path = os.path.join(artifact_dir, "scaler.pkl")
        type_clf_path = os.path.join(artifact_dir, "attack_type_classifier.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run `python3 train_model.py` first."
            )

        with open(config_path) as f:
            self.config = json.load(f)

        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.type_classifier = joblib.load(type_clf_path)
        self.threshold = self.config["threshold"]
        self.time_steps = self.config["time_steps"]
        self.num_features = self.config["num_features"]

    def classify_window(self, window: np.ndarray):
        """
        window: np.ndarray of shape (time_steps, num_features), raw (unscaled)
        feature values for one flow's rolling window.

        Returns dict:
            is_attack: bool
            confidence: float (model probability of "attack")
            attack_type: str ("BENIGN" if not an attack)
        """
        assert window.shape == (self.time_steps, self.num_features), \
            f"Expected window shape {(self.time_steps, self.num_features)}, got {window.shape}"

        flat = window.reshape(-1, self.num_features)
        scaled = self.scaler.transform(flat).astype(np.float32)
        scaled_window = scaled.reshape(1, self.time_steps, self.num_features)

        prob = float(self.model.predict(scaled_window, verbose=0).ravel()[0])
        is_attack = prob >= self.threshold

        attack_type = "BENIGN"
        if is_attack:
            summary = np.concatenate([scaled.mean(axis=0), scaled.std(axis=0)]).reshape(1, -1)
            attack_type = str(self.type_classifier.predict(summary)[0])

        return {
            "is_attack": is_attack,
            "confidence": prob,
            "attack_type": attack_type,
        }
