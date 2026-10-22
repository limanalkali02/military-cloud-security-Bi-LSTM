"""
model_engine.py

Bi-directional LSTM (Bi-LSTM) based Network Intrusion Detection model.

Designed to work with preprocessed CICIDS2017-style data, where each
sample is a sequence of network packet/flow feature vectors:

    Input shape: (batch_size, time_steps, 78)

    - batch_size : number of sequences per training batch
    - time_steps : number of packets/flows per sequence window
    - 78         : number of engineered features per packet/flow
                   (matches the CICIDS2017 feature set after
                   preprocessing/cleaning)

Output: a single sigmoid unit for binary classification
    - 0 -> Normal traffic
    - 1 -> Attack traffic

Author: (your name here)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
NUM_FEATURES = 78      # Fixed by the CICIDS2017 preprocessed feature set
DEFAULT_TIME_STEPS = 10  # Adjust based on how you windowed/sequenced your data
DROPOUT_RATE = 0.3
LSTM_UNITS_LAYER_1 = 64
LSTM_UNITS_LAYER_2 = 32
LEARNING_RATE = 1e-3


def build_model(time_steps: int = DEFAULT_TIME_STEPS,
                 num_features: int = NUM_FEATURES,
                 dropout_rate: float = DROPOUT_RATE,
                 lstm_units_1: int = LSTM_UNITS_LAYER_1,
                 lstm_units_2: int = LSTM_UNITS_LAYER_2,
                 learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """
    Build and compile a Bi-LSTM binary classifier for intrusion detection.

    Parameters
    ----------
    time_steps : int
        Number of time steps (packets/flows) per input sequence.
    num_features : int
        Number of features per time step (78 for CICIDS2017 preprocessed data).
    dropout_rate : float
        Dropout rate applied after each Bi-LSTM layer.
    lstm_units_1 : int
        Number of units in the first Bi-LSTM layer (per direction).
    lstm_units_2 : int
        Number of units in the second Bi-LSTM layer (per direction).
    learning_rate : float
        Learning rate for the Adam optimizer.

    Returns
    -------
    tf.keras.Model
        A compiled Keras model ready for training.
    """

    # ----------------------------------------------------------------- #
    # Input layer
    # Shape: (time_steps, num_features) -- batch dimension is implicit
    # ----------------------------------------------------------------- #
    inputs = layers.Input(shape=(time_steps, num_features), name="packet_sequence_input")

    # ----------------------------------------------------------------- #
    # First Bi-LSTM layer
    # return_sequences=True because we stack another Bi-LSTM layer after it
    # ----------------------------------------------------------------- #
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_1, return_sequences=True),
        name="bilstm_layer_1"
    )(inputs)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    # ----------------------------------------------------------------- #
    # Second Bi-LSTM layer
    # return_sequences=False -> collapses the time dimension, producing
    # a single feature vector summarizing the whole sequence
    # ----------------------------------------------------------------- #
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_2, return_sequences=False),
        name="bilstm_layer_2"
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    # ----------------------------------------------------------------- #
    # Output layer
    # Single sigmoid unit for binary classification (Normal vs Attack)
    # ----------------------------------------------------------------- #
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Intrusion_Detector")

    # ----------------------------------------------------------------- #
    # Compile
    # Binary crossentropy is standard for sigmoid-based binary classification
    # ----------------------------------------------------------------- #
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


if __name__ == "__main__":
    # Quick sanity check: build the model and print its architecture.
    # Adjust `time_steps` here to match however you windowed your
    # CICIDS2017 sequences during preprocessing.
    model = build_model(time_steps=DEFAULT_TIME_STEPS, num_features=NUM_FEATURES)
    model.summary()
