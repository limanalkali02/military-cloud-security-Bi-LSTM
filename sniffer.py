"""
sniffer.py

Background traffic-monitoring worker.

WHAT CHANGED FROM THE ORIGINAL VERSION
---------------------------------------
The original file flagged a packet as "ANOMALY" using
`time.localtime().tm_sec % 7 == 0` -- a clock-based coin flip with no
relationship to the packet at all. That is why the dashboard showed
false alerts with nothing behind them, and why nothing was ever logged:
there was no real detector and nothing worth recording.

This version:
  1. Simulates several concurrent network flows (random source IPs,
     fixed set of destination/server IPs, realistic per-class traffic
     shapes from cicids_features.py).
  2. Feeds each flow's rolling 10-step window through the ACTUAL trained
     Bi-LSTM model (inference_engine.py) -- not a placeholder rule.
  3. Requires a flow to be flagged malicious in a MAJORITY of its last 3
     classification windows before it is treated as a confirmed incident
     (temporal debounce). This mirrors real IDS practice and further
     suppresses one-off false alarms from a single noisy window.
  4. Logs every CONFIRMED incident to a persistent SQLite database
     (incident_db.py) with timestamp, source IP, destination IP,
     protocol, attack type, and model confidence -- this is the
     "report" / detail of every intrusion you asked for.
  5. Simulates the autonomous mitigation step (marks the source IP as
     blocked) once a flow is confirmed malicious, matching the
     project's described IAMF behaviour.

If you later wire this up to a real NIC via Scapy, replace
`_generate_next_row()` with real per-packet feature extraction and
`_pick_new_flow()` with real flow-table bookkeeping -- everything
downstream (windowing, inference, debounce, logging) stays the same.
"""

import threading
import time
import random
from collections import deque

import numpy as np

from cicids_features import generate_flow_sequence, ATTACK_CLASSES, FEATURE_NAMES
import incident_db

try:
    from inference_engine import IntrusionInferenceEngine
    _engine = IntrusionInferenceEngine()
    _MODEL_READY = True
    _MODEL_LOAD_ERROR = None
except Exception as e:  # pragma: no cover
    _engine = None
    _MODEL_READY = False
    _MODEL_LOAD_ERROR = str(e)

# Indices of the Protocol_TCP / Protocol_UDP one-hot columns in the 78-dim
# feature vector -- resolved once here so the hot loop doesn't repeatedly
# search the name list.
FEATURE_INDEX_TCP = FEATURE_NAMES.index("Protocol_TCP")
FEATURE_INDEX_UDP = FEATURE_NAMES.index("Protocol_UDP")

# --------------------------------------------------------------------------- #
# Globals shared with app.py
# --------------------------------------------------------------------------- #
packet_queue = deque(maxlen=200)
TOTAL_PACKETS_PROCESSED = 0
TOTAL_WINDOWS_CLASSIFIED = 0
_blocked_ips = set()
_lock = threading.Lock()

TIME_STEPS = 10
MAX_CONCURRENT_FLOWS = 4
FLOW_LIFETIME_RANGE = (18, 40)   # ticks
TICK_SECONDS = 0.25

_SERVER_IPS = ["10.0.0.1", "10.0.0.2", "10.0.5.10"]           # protected NAUB tactical cloud nodes
_SRC_IP_POOL_BENIGN_PREFIX = "192.168.1."
_SRC_IP_POOL_ATTACK_PREFIX = "203.0.113."                     # simulated external attacker space

_rng = np.random.default_rng()


class Flow:
    def __init__(self, flow_id):
        self.flow_id = flow_id
        # 88% chance a new flow is benign traffic, 12% an attack -- realistic
        # imbalance (attacks are the minority of real traffic too).
        if random.random() < 0.88:
            self.label = "BENIGN"
            self.source_ip = f"{_SRC_IP_POOL_BENIGN_PREFIX}{random.randint(2, 250)}"
        else:
            self.label = random.choice([c for c in ATTACK_CLASSES if c != "BENIGN"])
            self.source_ip = f"{_SRC_IP_POOL_ATTACK_PREFIX}{random.randint(2, 250)}"
        self.destination_ip = random.choice(_SERVER_IPS)
        self.protocol = None
        self.buffer = deque(maxlen=TIME_STEPS)
        self.recent_votes = deque(maxlen=3)
        self.remaining_life = random.randint(*FLOW_LIFETIME_RANGE)
        self.confirmed_incident_logged = False
        self.mitigated = False


def _next_raw_row(flow: Flow) -> np.ndarray:
    """Draw one packet-event's worth of features for this flow's class."""
    seq = generate_flow_sequence(flow.label, time_steps=1, rng=_rng)
    return seq[0]


def _pick_new_flow(next_id) -> Flow:
    return Flow(next_id)


def capture_packets():
    """
    Background worker: advances several concurrent simulated flows,
    runs real model inference on completed windows, debounces alerts,
    and logs confirmed incidents.
    """
    global TOTAL_PACKETS_PROCESSED, TOTAL_WINDOWS_CLASSIFIED

    flows = {}
    next_flow_id = 1
    for _ in range(MAX_CONCURRENT_FLOWS):
        flows[next_flow_id] = _pick_new_flow(next_flow_id)
        next_flow_id += 1

    while True:
        for fid in list(flows.keys()):
            flow = flows[fid]

            if flow.mitigated:
                # Source IP already blocked -- traffic from it is dropped at
                # the firewall and no longer reaches the analysis pipeline.
                flow.remaining_life -= 1
                if flow.remaining_life <= 0:
                    del flows[fid]
                    flows[next_flow_id] = _pick_new_flow(next_flow_id)
                    next_flow_id += 1
                continue

            row = _next_raw_row(flow)
            flow.protocol = "TCP" if row[FEATURE_INDEX_TCP] >= row[FEATURE_INDEX_UDP] else "UDP"
            flow.buffer.append(row)

            with _lock:
                TOTAL_PACKETS_PROCESSED += 1

            analysis_label = "ANALYZING"
            confidence = None

            if len(flow.buffer) == TIME_STEPS and _MODEL_READY:
                window = np.stack(flow.buffer)
                result = _engine.classify_window(window)
                with _lock:
                    TOTAL_WINDOWS_CLASSIFIED += 1

                flow.recent_votes.append(1 if result["is_attack"] else 0)
                confidence = result["confidence"]
                analysis_label = result["attack_type"] if result["is_attack"] else "BENIGN"

                confirmed = (
                    result["is_attack"]
                    and sum(flow.recent_votes) >= 2  # majority of last 3 windows
                    and not flow.confirmed_incident_logged
                )
                if confirmed:
                    flow.confirmed_incident_logged = True
                    flow.mitigated = True
                    _blocked_ips.add(flow.source_ip)
                    incident_db.log_incident(
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        source_ip=flow.source_ip,
                        destination_ip=flow.destination_ip,
                        protocol=flow.protocol,
                        attack_type=result["attack_type"],
                        confidence=round(confidence, 4),
                        action_taken=f"Source IP {flow.source_ip} auto-blocked (iptables DROP)",
                    )

            new_pkt = {
                "Timestamp": time.strftime("%H:%M:%S"),
                "Source IP": flow.source_ip,
                "Destination IP": flow.destination_ip,
                "Protocol": flow.protocol or "TCP",
                "Length": int(row[3]) if len(row) > 3 else 0,  # Total Length of Fwd Packets, approx
                "Bi-LSTM Analysis": analysis_label,
                "Confidence": f"{confidence*100:.1f}%" if confidence is not None else "-",
                "Mitigated": "YES" if flow.mitigated else "NO",
            }
            packet_queue.append(new_pkt)

            flow.remaining_life -= 1
            if flow.remaining_life <= 0 and not flow.mitigated:
                del flows[fid]
                flows[next_flow_id] = _pick_new_flow(next_flow_id)
                next_flow_id += 1

        time.sleep(TICK_SECONDS)


# --------------------------------------------------------------------------- #
# START BACKGROUND THREAD AUTOMATICALLY ON IMPORT
# --------------------------------------------------------------------------- #
capture_thread = threading.Thread(target=capture_packets, daemon=True)
capture_thread.start()


# --------------------------------------------------------------------------- #
# HELPERS FOR app.py
# --------------------------------------------------------------------------- #
def get_recent_packets():
    return list(packet_queue)


def get_totals():
    with _lock:
        return TOTAL_PACKETS_PROCESSED, TOTAL_WINDOWS_CLASSIFIED


def get_blocked_ips():
    return list(_blocked_ips)


def model_status():
    if _MODEL_READY:
        return True, None
    return False, _MODEL_LOAD_ERROR
