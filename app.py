import streamlit as st
import pandas as pd
import time
import psutil

import sniffer
import incident_db

# 1. Page Configuration
st.set_page_config(
    page_title="Intelligent Tactical Security System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Military/Tactical Theme
st.markdown("""
    <style>
    .reportview-container { background: #0d1117; }
    .metric-card {
        background-color: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #58a6ff; margin: 5px 0; }
    .metric-label { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .banner-nominal {
        background-color: rgba(46, 117, 89, 0.2); border: 1px solid #2ea043; color: #56d364;
        border-radius: 8px; padding: 15px; text-align: center; font-weight: bold; font-size: 1.2rem; margin-bottom: 25px;
    }
    .banner-alert {
        background-color: rgba(248, 81, 73, 0.15); border: 1px solid #f85149; color: #ff7b72;
        border-radius: 8px; padding: 15px; text-align: center; font-weight: bold; font-size: 1.2rem;
        margin-bottom: 25px; animation: pulse 2s infinite;
    }
    .banner-warning {
        background-color: rgba(210, 153, 34, 0.15); border: 1px solid #d29922; color: #e3b341;
        border-radius: 8px; padding: 15px; text-align: center; font-weight: bold; font-size: 1.05rem; margin-bottom: 25px;
    }
    @keyframes pulse { 0% { opacity: 0.8; } 50% { opacity: 1; } 100% { opacity: 0.8; } }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar
st.sidebar.image("https://img.icons8.com/nolan/96/shield.png", width=80)
st.sidebar.title("System Control Panel")
st.sidebar.markdown("---")

monitoring_active = st.sidebar.toggle("Monitoring Active", value=True)
refresh_interval = st.sidebar.slider("UI Refresh Interval (sec)", 0.5, 5.0, 1.0, step=0.5)

st.sidebar.markdown("### 🎛️ Tactical Server Metrics")
cpu_usage = psutil.cpu_percent()
memory_usage = psutil.virtual_memory().percent
st.sidebar.write(f"**Core Processors:** {cpu_usage}%")
st.sidebar.progress(float(cpu_usage) / 100)
st.sidebar.write(f"**Memory Allocation:** {memory_usage}%")
st.sidebar.progress(float(memory_usage) / 100)

st.sidebar.markdown("---")
model_ready, model_error = sniffer.model_status()
if model_ready:
    st.sidebar.success("🧠 Bi-LSTM model: LOADED")
else:
    st.sidebar.error(f"🧠 Bi-LSTM model: NOT LOADED\n\n{model_error}\n\nRun `python3 train_model.py` first.")

st.sidebar.info("📌 **Project:** Bi-LSTM Real-Time Threat Detection\n\n**Environment:** Military Tactical Cloud")

# 4. Header
st.markdown("<h2 style='text-align: left; color: white;'>🛡️ Tactical Cloud Security Console</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #8b949e;'>Real-Time Threat Detection & Automated Intrusion Mitigation System (Bi-LSTM)</p>", unsafe_allow_html=True)

if not model_ready:
    st.markdown(f"""
        <div class="banner-warning">
            ⚠️ Model not loaded — dashboard cannot classify traffic. Run <code>python3 train_model.py</code> once, then restart the app.
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# 5. Live packet feed (from the real sniffer/inference pipeline)
df_packets = pd.DataFrame(sniffer.get_recent_packets())
if df_packets.empty:
    st.info("Warming up — capturing the first packet windows...")
    time.sleep(refresh_interval)
    st.rerun()

recent = df_packets.tail(15)
total_packets, total_windows = sniffer.get_totals()
blocked_ips = sniffer.get_blocked_ips()
total_incidents = incident_db.count_incidents()

# Only a CONFIRMED (logged) incident should trigger the critical banner --
# a single window's raw prediction is shown in the table but is not, by
# itself, treated as a confirmed intrusion (see sniffer.py debounce logic).
has_confirmed_attack = total_incidents > 0
last_incidents = incident_db.fetch_recent_incidents(limit=1)
latest_threat_type = last_incidents[0]["attack_type"] if last_incidents else "None"

# 6. Alert Banner
if has_confirmed_attack and last_incidents:
    st.markdown(f"""
        <div class="banner-alert">
            ⚠️ CONFIRMED INTRUSION: {latest_threat_type.replace('_',' ')} FROM {last_incidents[0]['source_ip']} — SOURCE IP AUTO-BLOCKED
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="banner-nominal">
            ✅ ALL SYSTEMS NOMINAL — MILITARY CLOUD ENVIRONMENT SECURED
        </div>
    """, unsafe_allow_html=True)

# 7. Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Packets Processed</div>
            <div class="metric-value">{total_packets}</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🚨 Confirmed Incidents</div>
            <div class="metric-value" style="color: {'#ff7b72' if total_incidents else '#56d364'};">{total_incidents}</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🧠 Windows Classified</div>
            <div class="metric-value" style="color: #d1f1a5;">{total_windows}</div>
        </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔒 IPs Auto-Blocked</div>
            <div class="metric-value" style="color: #ff7b72;">{len(blocked_ips)}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 8. Live Traffic Section
st.subheader("🛰️ Tactical Traffic Stream (Real-Time, Bi-LSTM Classified)")

def color_rows_by_threat(val):
    if val == "BENIGN":
        return "color: #4dfd9b;"
    elif val == "ANALYZING":
        return "color: #8b949e; font-style: italic;"
    else:
        return "color: #ff5555; font-weight: bold; background-color: rgba(153, 0, 0, 0.15);"

if "Bi-LSTM Analysis" in recent.columns:
    styled_df = recent.style.map(color_rows_by_threat, subset=["Bi-LSTM Analysis"])
    st.dataframe(styled_df, use_container_width=True)
else:
    st.dataframe(recent, use_container_width=True)

st.caption(
    "A single window flagged as an attack type is a raw model prediction, shown here for transparency. "
    "It only becomes a **Confirmed Incident** below once the same flow is flagged in a majority of its "
    "last 3 windows — this debounce is what prevents one noisy reading from triggering a false alarm."
)

st.markdown("<br>", unsafe_allow_html=True)

# 9. Incident Report (persistent, from SQLite)
st.subheader("📋 Incident Report — Confirmed Intrusions")

incidents = incident_db.fetch_recent_incidents(limit=100)
if incidents:
    inc_df = pd.DataFrame(incidents)
    inc_df["confidence"] = (inc_df["confidence"] * 100).round(1).astype(str) + "%"
    inc_df = inc_df.rename(columns={
        "id": "ID", "timestamp": "Timestamp", "source_ip": "Source IP (Attacker)",
        "destination_ip": "Destination IP (Target)", "protocol": "Protocol",
        "attack_type": "Attack Type", "confidence": "Confidence", "action_taken": "Action Taken",
    })
    st.dataframe(inc_df, use_container_width=True)

    csv_bytes = inc_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Incident Report (CSV)",
        data=csv_bytes,
        file_name="incident_report.csv",
        mime="text/csv",
    )

    with st.expander("📊 Incidents by attack type"):
        st.bar_chart(pd.Series(incident_db.count_by_attack_type()))
else:
    st.success("No confirmed intrusions recorded yet. This log is persistent across restarts (security_incidents.db).")

# 10. Auto-Refresh Engine
if monitoring_active:
    time.sleep(refresh_interval)
    st.rerun()
