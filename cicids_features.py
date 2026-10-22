"""
cicids_features.py

Defines the 78-dimensional feature schema used by the Bi-LSTM model
(matching the CICIDS2017 feature convention referenced in the project),
and generates statistically realistic synthetic flow samples for each
traffic class in scope: BENIGN, DOS_HULK, SLOWLORIS, GOLDENEYE, HEARTBLEED.

NOTE ON DATA SOURCE
--------------------
The real CICIDS2017 dataset (Sharafaldin et al., 2018) is not reachable
from this environment (hosted outside the allowed network). To avoid
training the model on meaningless/random values, this module generates
synthetic flow records whose per-class statistics are drawn from the
*documented behavioural signatures* of each attack type in the published
literature (e.g. Hulk/GoldenEye = high-volume floods with short flow
duration and high packet rate; Slowloris = long-held low-rate connections;
Heartbleed = tiny packet count with an abnormally large single response).

If you obtain the real CICIDS2017 CSV files, replace `generate_dataset()`
with a loader that reads and normalizes those files using this same
FEATURE_NAMES schema, and everything downstream (training, inference,
logging) will continue to work unchanged.
"""

import numpy as np

RANDOM_SEED = 42
ATTACK_CLASSES = ["BENIGN", "DOS_HULK", "SLOWLORIS", "GOLDENEYE", "HEARTBLEED"]

# --------------------------------------------------------------------------- #
# 78-column feature schema (mirrors the CICIDS2017 / CICFlowMeter convention)
# --------------------------------------------------------------------------- #
FEATURE_NAMES = [
    "Flow Duration", "Total Fwd Packets", "Total Bwd Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Protocol_TCP", "Protocol_UDP",
]
assert len(FEATURE_NAMES) == 78, f"Expected 78 features, got {len(FEATURE_NAMES)}"

# --------------------------------------------------------------------------- #
# Per-class base parameters, drawn from documented behavioural signatures:
#   - BENIGN:    moderate everything, balanced flags, TCP handshake normal
#   - DOS_HULK:  high-volume flood, short duration, huge packet/byte rate
#   - SLOWLORIS: long duration, very low packet rate, partial-header PSH/ACK
#   - GOLDENEYE: similar to Hulk but even higher multi-threaded flood rate
#   - HEARTBLEED: very few packets, abnormally large single response payload
# --------------------------------------------------------------------------- #
_CLASS_PARAMS = {
    "BENIGN": dict(
        duration_us=(4.0, 1.2), fwd_pkts=(20, 8), bwd_pkts=(18, 7),
        fwd_pkt_len=(300, 120), bwd_pkt_len=(400, 150),
        iat_mean_us=(2.0e5, 0.6), syn=(1, 0.3), psh=(2, 1), ack=(15, 5),
        protocol_tcp_prob=0.75,
    ),
    "DOS_HULK": dict(
        duration_us=(2.5, 1.0), fwd_pkts=(400, 150), bwd_pkts=(60, 30),
        fwd_pkt_len=(900, 200), bwd_pkt_len=(200, 80),
        iat_mean_us=(2.5e2, 0.8), syn=(15, 5), psh=(40, 10), ack=(300, 100),
        protocol_tcp_prob=0.98,
    ),
    "SLOWLORIS": dict(
        duration_us=(7.5, 0.6), fwd_pkts=(15, 6), bwd_pkts=(2, 1.5),
        fwd_pkt_len=(120, 40), bwd_pkt_len=(60, 20),
        iat_mean_us=(6.0e5, 0.5), syn=(1, 0.2), psh=(12, 4), ack=(8, 3),
        protocol_tcp_prob=0.99,
    ),
    "GOLDENEYE": dict(
        duration_us=(2.2, 1.0), fwd_pkts=(600, 200), bwd_pkts=(90, 40),
        fwd_pkt_len=(1000, 250), bwd_pkt_len=(220, 90),
        iat_mean_us=(1.8e2, 0.9), syn=(25, 8), psh=(60, 15), ack=(450, 150),
        protocol_tcp_prob=0.98,
    ),
    "HEARTBLEED": dict(
        duration_us=(3.0, 1.0), fwd_pkts=(4, 1.5), bwd_pkts=(3, 1.3),
        fwd_pkt_len=(110, 35), bwd_pkt_len=(2400, 900),  # abnormally large leaked response
        iat_mean_us=(1.0e5, 0.6), syn=(1, 0.2), psh=(3, 1.2), ack=(4, 1.5),
        protocol_tcp_prob=1.0,
    ),
}


def _lognormal(mu_sigma, rng):
    mu, sigma = mu_sigma
    return float(rng.lognormal(mean=np.log(max(mu, 1e-3)), sigma=sigma))


def _pos_normal(mean_std, rng, minimum=0.0):
    mean, std = mean_std
    return float(max(minimum, rng.normal(mean, std)))


def generate_flow_sequence(label: str, time_steps: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one (time_steps, 78) sequence of packet-level feature vectors
    for a single simulated flow of the given class label, with realistic
    step-to-step jitter so the sequence carries genuine temporal structure
    for the Bi-LSTM to learn from.
    """
    p = _CLASS_PARAMS[label]
    is_tcp = rng.random() < p["protocol_tcp_prob"]

    seq = np.zeros((time_steps, 78), dtype=np.float32)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    # Slowloris ramps IAT up over time (holding connection open longer and longer)
    slowloris_ramp = np.linspace(0.6, 1.6, time_steps) if label == "SLOWLORIS" else np.ones(time_steps)
    # Hulk/GoldenEye floods intensify slightly over the window (ramping attack)
    flood_ramp = np.linspace(0.7, 1.3, time_steps) if label in ("DOS_HULK", "GOLDENEYE") else np.ones(time_steps)

    for t in range(time_steps):
        duration = _lognormal(p["duration_us"], rng) * 1e5  # microseconds
        fwd_pkts = max(1, int(_pos_normal(p["fwd_pkts"], rng, 1) * flood_ramp[t]))
        bwd_pkts = max(0, int(_pos_normal(p["bwd_pkts"], rng, 0)))
        fwd_len = _pos_normal(p["fwd_pkt_len"], rng, 1)
        bwd_len = _pos_normal(p["bwd_pkt_len"], rng, 0)
        iat_mean = _pos_normal((p["iat_mean_us"][0] * slowloris_ramp[t], p["iat_mean_us"][0] * p["iat_mean_us"][1]), rng, 1)

        total_fwd_len = fwd_len * fwd_pkts
        total_bwd_len = bwd_len * bwd_pkts
        flow_bytes_s = (total_fwd_len + total_bwd_len) / max(duration / 1e6, 1e-6)
        flow_pkts_s = (fwd_pkts + bwd_pkts) / max(duration / 1e6, 1e-6)

        row = np.zeros(78, dtype=np.float32)
        row[idx["Flow Duration"]] = duration
        row[idx["Total Fwd Packets"]] = fwd_pkts
        row[idx["Total Bwd Packets"]] = bwd_pkts
        row[idx["Total Length of Fwd Packets"]] = total_fwd_len
        row[idx["Total Length of Bwd Packets"]] = total_bwd_len
        row[idx["Fwd Packet Length Max"]] = fwd_len * rng.uniform(1.1, 1.6)
        row[idx["Fwd Packet Length Min"]] = fwd_len * rng.uniform(0.3, 0.8)
        row[idx["Fwd Packet Length Mean"]] = fwd_len
        row[idx["Fwd Packet Length Std"]] = fwd_len * rng.uniform(0.1, 0.4)
        row[idx["Bwd Packet Length Max"]] = bwd_len * rng.uniform(1.1, 1.6)
        row[idx["Bwd Packet Length Min"]] = bwd_len * rng.uniform(0.2, 0.8)
        row[idx["Bwd Packet Length Mean"]] = bwd_len
        row[idx["Bwd Packet Length Std"]] = bwd_len * rng.uniform(0.1, 0.4)
        row[idx["Flow Bytes/s"]] = flow_bytes_s
        row[idx["Flow Packets/s"]] = flow_pkts_s
        row[idx["Flow IAT Mean"]] = iat_mean
        row[idx["Flow IAT Std"]] = iat_mean * rng.uniform(0.2, 0.6)
        row[idx["Flow IAT Max"]] = iat_mean * rng.uniform(1.5, 3.0)
        row[idx["Flow IAT Min"]] = iat_mean * rng.uniform(0.1, 0.5)
        row[idx["Fwd IAT Total"]] = iat_mean * fwd_pkts
        row[idx["Fwd IAT Mean"]] = iat_mean
        row[idx["Fwd IAT Std"]] = iat_mean * rng.uniform(0.2, 0.5)
        row[idx["Fwd IAT Max"]] = iat_mean * rng.uniform(1.5, 3.0)
        row[idx["Fwd IAT Min"]] = iat_mean * rng.uniform(0.1, 0.5)
        row[idx["Bwd IAT Total"]] = iat_mean * max(bwd_pkts, 1) * rng.uniform(0.8, 1.2)
        row[idx["Bwd IAT Mean"]] = iat_mean * rng.uniform(0.8, 1.2)
        row[idx["Bwd IAT Std"]] = iat_mean * rng.uniform(0.2, 0.5)
        row[idx["Bwd IAT Max"]] = iat_mean * rng.uniform(1.5, 3.0)
        row[idx["Bwd IAT Min"]] = iat_mean * rng.uniform(0.1, 0.5)
        row[idx["Fwd PSH Flags"]] = _pos_normal((p["psh"][0] * 0.5, p["psh"][1]), rng, 0)
        row[idx["Bwd PSH Flags"]] = _pos_normal((p["psh"][0] * 0.2, p["psh"][1] * 0.5), rng, 0)
        row[idx["Fwd URG Flags"]] = rng.poisson(0.2)
        row[idx["Bwd URG Flags"]] = rng.poisson(0.1)
        row[idx["Fwd Header Length"]] = fwd_pkts * 20
        row[idx["Bwd Header Length"]] = bwd_pkts * 20
        row[idx["Fwd Packets/s"]] = fwd_pkts / max(duration / 1e6, 1e-6)
        row[idx["Bwd Packets/s"]] = bwd_pkts / max(duration / 1e6, 1e-6)
        row[idx["Min Packet Length"]] = min(fwd_len, bwd_len) * rng.uniform(0.2, 0.6)
        row[idx["Max Packet Length"]] = max(fwd_len, bwd_len) * rng.uniform(1.1, 1.8)
        pkt_mean = (total_fwd_len + total_bwd_len) / max(fwd_pkts + bwd_pkts, 1)
        row[idx["Packet Length Mean"]] = pkt_mean
        row[idx["Packet Length Std"]] = pkt_mean * rng.uniform(0.2, 0.6)
        row[idx["Packet Length Variance"]] = (pkt_mean * rng.uniform(0.2, 0.6)) ** 2
        row[idx["FIN Flag Count"]] = rng.poisson(0.3)
        row[idx["SYN Flag Count"]] = _pos_normal(p["syn"], rng, 0)
        row[idx["RST Flag Count"]] = rng.poisson(0.2)
        row[idx["PSH Flag Count"]] = _pos_normal(p["psh"], rng, 0)
        row[idx["ACK Flag Count"]] = _pos_normal(p["ack"], rng, 0)
        row[idx["URG Flag Count"]] = rng.poisson(0.1)
        row[idx["CWE Flag Count"]] = rng.poisson(0.05)
        row[idx["ECE Flag Count"]] = rng.poisson(0.05)
        row[idx["Down/Up Ratio"]] = bwd_pkts / max(fwd_pkts, 1)
        row[idx["Average Packet Size"]] = pkt_mean
        row[idx["Avg Fwd Segment Size"]] = fwd_len
        row[idx["Avg Bwd Segment Size"]] = bwd_len
        row[idx["Fwd Avg Bytes/Bulk"]] = 0.0
        row[idx["Fwd Avg Packets/Bulk"]] = 0.0
        row[idx["Fwd Avg Bulk Rate"]] = 0.0
        row[idx["Bwd Avg Bytes/Bulk"]] = 0.0
        row[idx["Bwd Avg Packets/Bulk"]] = 0.0
        row[idx["Bwd Avg Bulk Rate"]] = 0.0
        row[idx["Subflow Fwd Packets"]] = fwd_pkts
        row[idx["Subflow Fwd Bytes"]] = total_fwd_len
        row[idx["Subflow Bwd Packets"]] = bwd_pkts
        row[idx["Subflow Bwd Bytes"]] = total_bwd_len
        row[idx["Init_Win_bytes_forward"]] = rng.uniform(200, 65535)
        row[idx["Init_Win_bytes_backward"]] = rng.uniform(200, 65535)
        row[idx["act_data_pkt_fwd"]] = max(0, fwd_pkts - 2)
        row[idx["min_seg_size_forward"]] = rng.uniform(20, 40)
        row[idx["Active Mean"]] = duration * rng.uniform(0.3, 0.7)
        row[idx["Active Std"]] = duration * rng.uniform(0.05, 0.2)
        row[idx["Active Max"]] = duration * rng.uniform(0.7, 1.0)
        row[idx["Active Min"]] = duration * rng.uniform(0.05, 0.3)
        row[idx["Idle Mean"]] = duration * rng.uniform(0.0, 0.3)
        row[idx["Idle Std"]] = duration * rng.uniform(0.0, 0.1)
        row[idx["Idle Max"]] = duration * rng.uniform(0.0, 0.4)
        row[idx["Idle Min"]] = 0.0
        row[idx["Protocol_TCP"]] = 1.0 if is_tcp else 0.0
        row[idx["Protocol_UDP"]] = 0.0 if is_tcp else 1.0

        seq[t] = row

    # --------------------------------------------------------------- #
    # Realism injection: real network traffic is noisy and classes
    # overlap somewhat (e.g. a legitimate bulk transfer can look a bit
    # like a flood; a stealthy low-and-slow attack can look a bit like
    # benign traffic). Without this, the classifier looks unrealistically
    # perfect. This also directly emulates the "sudden legitimate spike
    # vs malicious flood" ambiguity the project write-up describes.
    # --------------------------------------------------------------- #
    seq *= rng.normal(1.0, 0.30, size=seq.shape).astype(np.float32)
    # Additive jitter proportional to each column's own scale within the
    # sequence, so low-magnitude flag/count columns get meaningfully
    # perturbed too (multiplicative noise alone barely touches near-zero
    # columns), further softening class boundaries.
    col_scale = np.abs(seq).mean(axis=0, keepdims=True) + 1e-3
    seq = seq + rng.normal(0.0, 0.12, size=seq.shape).astype(np.float32) * col_scale
    seq = np.clip(seq, 0, None)

    return seq


def generate_dataset(n_per_class: int, time_steps: int, seed: int = RANDOM_SEED):
    """
    Build a balanced synthetic dataset of flow sequences.
    Returns X (N, time_steps, 78), y_binary (N,), y_class (N,) [string labels]
    """
    rng = np.random.default_rng(seed)
    X, y_bin, y_cls = [], [], []
    for label in ATTACK_CLASSES:
        for _ in range(n_per_class):
            seq = generate_flow_sequence(label, time_steps, rng)

            # Inject realistic class overlap so the task isn't trivially
            # separable: occasionally blend a benign flow toward an attack
            # profile (a legitimate traffic burst) or an attack flow toward
            # benign (a stealthy, low-signature evasion attempt).
            if label == "BENIGN" and rng.random() < 0.15:
                other = generate_flow_sequence(rng.choice(["DOS_HULK", "GOLDENEYE", "SLOWLORIS"]), time_steps, rng)
                blend_w = rng.uniform(0.25, 0.45)
                seq = seq * (1 - blend_w) + other * blend_w
            elif label != "BENIGN" and rng.random() < 0.15:
                benign_ref = generate_flow_sequence("BENIGN", time_steps, rng)
                blend_w = rng.uniform(0.25, 0.45)
                seq = seq * (1 - blend_w) + benign_ref * blend_w

            X.append(seq)
            y_bin.append(0 if label == "BENIGN" else 1)
            y_cls.append(label)
    X = np.stack(X)
    y_bin = np.array(y_bin, dtype=np.int32)
    y_cls = np.array(y_cls)
    # shuffle
    perm = rng.permutation(len(X))
    return X[perm], y_bin[perm], y_cls[perm]
