import io
import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root directory to sys.path so 'src' can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import time
from scapy.all import wrpcap

from src.feature_extraction import extract_all_features, FEATURE_COLUMNS
from src.prediction import ThreatPredictionEngine
from src.live_capture import (
    get_available_interfaces,
    LiveCaptureSession,
    simulate_live_traffic_capture
)

try:
    from src.live_capture import is_pcap_installed
except ImportError:
    def is_pcap_installed() -> bool:
        try:
            from scapy.all import conf
            return bool(getattr(conf, "use_pcap", False))
        except Exception:
            return False
from src.report_generator import generate_executive_pdf_report
from src.feature_dictionary import FEATURE_DICTIONARY

# ---------------------------------------------------------
# Page Setup & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Network Detector — Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber SOC Dark Mode CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f9fafb;
    }
    .verdict-banner-attack {
        background: linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #ef4444;
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }
    .verdict-banner-normal {
        background: linear-gradient(90deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 600;
        border-radius: 8px 8px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_prediction_engine():
    return ThreatPredictionEngine(models_dir="models")


engine = get_prediction_engine()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Network Detector")
    st.caption("Machine Learning Network Intrusion Detection")
    st.link_button("⭐ View on GitHub", "https://github.com/Nitins715/Network-Threat-Detector", use_container_width=True)

    st.markdown("---")
    st.subheader("📁 Input Capture")

    upload_mode = st.radio(
        "Select Input Source",
        ["Upload File (.pcap / .csv)", "🎙️ Live Network Sniff"],
        index=0
    )

    pcap_bytes = None
    csv_df = None
    source_name = None

    if upload_mode == "Upload File (.pcap / .csv)":
        uploaded_file = st.file_uploader(
            "Upload capture or flow dataset",
            type=["pcap", "pcapng", "csv"],
            help="Upload raw PCAP capture file or UNSW-NB15 flow CSV"
        )
        if uploaded_file:
            source_name = uploaded_file.name
            if uploaded_file.name.endswith((".pcap", ".pcapng")):
                pcap_bytes = uploaded_file.getvalue()
            else:
                csv_df = pd.read_csv(uploaded_file)

    else:
        st.subheader("🎙️ Live Network Sniff")
        interfaces = get_available_interfaces()
        # Sort so active adapters with real IP addresses appear first
        sorted_labels = sorted(interfaces.keys(), key=lambda x: ('192.168' not in x and '10.' not in x))

        options = sorted_labels[:10] if sorted_labels else ["Default Interface"]
        selected_iface_label = st.selectbox(
            "Network Adapter",
            options=options,
            help="Select the hardware or virtual network adapter to monitor"
        )
        selected_iface_name = interfaces.get(selected_iface_label, "eth0")

        sniff_engine = st.radio(
            "Sniffing Engine",
            ["Live Traffic Stream (Instant Demo)", "Native Scapy Sniffer (Admin/Npcap)"],
            index=0,
            help="Native Sniffer captures directly from your hardware adapter (requires Admin/Npcap). Live Stream Simulator feeds live pulses directly for safe testing."
        )

        if sniff_engine == "Native Scapy Sniffer (Admin/Npcap)":
            if os.name != 'nt':
                st.info(
                    "☁️ **Streamlit Cloud Environment Notice:**\n\n"
                    "This public demo is hosted on Streamlit's cloud server. A cloud website cannot physically connect to your home Wi-Fi antenna or local Ethernet card over the internet.\n\n"
                    "• 👉 **To test live detection here on the cloud:** Select **'Live Traffic Stream (Instant Demo)'** above or upload a capture file!\n"
                    "• 💻 **To sniff your real physical Wi-Fi:** Run this app locally on your PC (`python -m streamlit run dashboard/app.py`)."
                )
            elif not is_pcap_installed():
                st.warning("⚠️ **Npcap Packet Driver Not Found on Local PC**")
                with st.expander("📥 How to Download & Install Npcap (1-Minute Setup)", expanded=True):
                    st.markdown("""
                    **Why is this needed?** Windows protects your network card and blocks Python from reading raw Wi-Fi/Ethernet packets without Npcap.
                    
                    **Step-by-Step Instructions:**
                    1. **Download:** Click the button below to get the official installer (`npcap-setup.exe`).
                    2. **Run Installer:** During setup, choose these recommended options:
                       - ✅ **Check:** *"Install Npcap in WinPcap API-compatible Mode"* (Required for Scapy).
                       - ⬜ **Uncheck:** *"Restrict Npcap driver's access to Administrators only"* (Allows normal users to sniff without running as Admin).
                    3. **Restart:** Refresh this page or re-run the app.
                    
                    *(Note: On Linux or macOS, raw packet capture works natively without Npcap).*
                    """)
                    st.link_button("🌐 Download Free Npcap Installer", "https://npcap.com/#download", use_container_width=True)
                    st.caption("💡 *Tip: You can also select **'Live Traffic Stream'** above for instant testing without installing anything.*")
            else:
                st.success("✅ **Npcap Active:** Ready to sniff physical network packets.")

        capture_duration = st.slider("Duration (seconds)", 3, 30, 5)

        if st.button("▶️ Start Live Sniff", use_container_width=True, type="primary"):
            progress_bar = st.progress(0.0)
            status_box = st.empty()

            if sniff_engine == "Native Scapy Sniffer (Admin/Npcap)":
                status_box.markdown(f"🔴 **Sniffing live packets on `{selected_iface_name}`...**")
                session = LiveCaptureSession(interface=selected_iface_name)
                started, msg = session.start()
                if not started:
                    st.error(f"Could not bind to `{selected_iface_name}`: {msg}\n\n*Note: On Windows, native raw socket capture requires running as Administrator or installing Npcap (https://npcap.com). Switch to 'Live Traffic Stream' for instant testing.*")
                else:
                    for s in range(capture_duration):
                        time.sleep(1)
                        progress_bar.progress((s + 1) / capture_duration)
                    pkts = session.stop()
                    if pkts:
                        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pcap")
                        os.close(tmp_fd)
                        wrpcap(tmp_path, pkts)
                        with open(tmp_path, "rb") as f:
                            pcap_bytes = f.read()
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                        source_name = f"live_{selected_iface_name}.pcap"
                        st.success(f"Sniffed {len(pkts)} raw packets from {selected_iface_name}!")
                    else:
                        st.warning("No IP packets were observed during the window. Ensure traffic is flowing through the adapter.")
            else:
                status_box.markdown(f"🔴 **Capturing live network telemetry stream...**")
                for s in range(capture_duration):
                    time.sleep(0.4)
                    progress_bar.progress((s + 1) / capture_duration)
                pkts = simulate_live_traffic_capture(duration_sec=capture_duration, attack_ratio=0.45)
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pcap")
                os.close(tmp_fd)
                wrpcap(tmp_path, pkts)
                with open(tmp_path, "rb") as f:
                    pcap_bytes = f.read()
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                source_name = f"live_stream_{selected_iface_name.split()[0]}.pcap"
                st.success(f"Streamed and ingested {len(pkts)} live packets!")

    st.markdown("---")
    with st.expander("⚙️ Model Pipeline Specs", expanded=False):
        st.write("**Supervised Classifiers:**")
        st.write("• Random Forest (50%)")
        st.write("• XGBoost (50%)")
        st.write("**Unsupervised Detector:**")
        st.write("• Isolation Forest (Zero-Day Anomaly)")
        st.write("**Decision Fusion:**")
        st.write("• 50/50 Bayesian Score Combination")
        st.write(f"• Threshold: `{engine.config.get('threshold', 0.5)}`")

# ---------------------------------------------------------
# Top Header & Navigation Tabs
# ---------------------------------------------------------
header_col1, header_col2 = st.columns([3.2, 1.3])
with header_col1:
    st.title("🛡️ Network Threat Detector")
    st.markdown("Automated packet dissection, 32-feature UNSW-NB15 flow extraction, and ensemble threat verdict.")
with header_col2:
    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
    st.link_button("⭐ GitHub Repository", "https://github.com/Nitins715/Network-Threat-Detector", use_container_width=True)

main_tab_threat, main_tab_dictionary = st.tabs([
    "🛡️ SOC Threat Intelligence",
    "📖 UNSW-NB15 Feature Dictionary"
])

# =========================================================
# TAB 1: SOC Threat Intelligence (Live & PCAP Analysis)
# =========================================================
with main_tab_threat:
    if pcap_bytes is not None or csv_df is not None:
        flows_df = None

        with st.spinner("Extracting bidirectional flows & packet telemetry..."):
            if pcap_bytes is not None:
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pcap")
                with os.fdopen(tmp_fd, "wb") as tmp_file:
                    tmp_file.write(pcap_bytes)

                try:
                    flows_df = extract_all_features(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            else:
                flows_df = csv_df

        if flows_df is None or flows_df.empty:
            st.warning("No IP network flows could be reconstructed from the provided input.")
        else:
            with st.spinner("Executing Supervised Ensemble & Isolation Forest Anomaly Analysis..."):
                results = engine.analyze(flows_df)

            verdict = results['verdict']
            threat_intensity = results['threat_intensity']
            risk_level = results['risk_level']
            network_status = results['network_status']
            traffic_behavior = results['traffic_behavior']
            total_flows = results['total_flows']
            attack_count = results['attack_flow_count']
            normal_count = results['normal_flow_count']
            analyst_note = results['analyst_note']
            scores = results['all_flow_scores']
            primary_category = results.get('primary_category', 'Normal')
            category_counts = results.get('category_counts', {})
            playbook = results.get('playbook', {})

            # Enrich flows_df with prediction outputs and explanations
            threshold = engine.config.get('threshold', 0.5)
            flows_df['threat_score'] = scores
            flows_df['prediction'] = ['ATTACK' if s >= threshold else 'NORMAL' for s in scores]
            flows_df['attack_category'] = results.get('flow_categories', ['Normal'] * len(flows_df))
            flows_df['primary_threat_driver'] = engine.explain_all_flows(flows_df)

            # -------------------------------------------------
            # 1. Executive Verdict Banner
            # -------------------------------------------------
            if verdict == "ATTACK":
                st.markdown(f"""
                <div class="verdict-banner-attack">
                    <h1 style="margin: 0; font-size: 2.4rem;">🚨 INTRUSION DETECTED — {verdict}</h1>
                    <p style="margin: 8px 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                        <b>Primary Vector:</b> {primary_category} &nbsp;|&nbsp; <b>Risk Level:</b> {risk_level} &nbsp;|&nbsp; <b>Threat Intensity:</b> {threat_intensity}%
                    </p>
                    <p style="margin: 6px 0 0 0; font-size: 0.95rem; opacity: 0.85;">{analyst_note}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-banner-normal">
                    <h1 style="margin: 0; font-size: 2.4rem;">✅ SYSTEM SECURE — {verdict}</h1>
                    <p style="margin: 8px 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                        <b>Network Status:</b> {network_status} &nbsp;|&nbsp; <b>Risk Level:</b> {risk_level} &nbsp;|&nbsp; <b>Threat Intensity:</b> {threat_intensity}%
                    </p>
                    <p style="margin: 6px 0 0 0; font-size: 0.95rem; opacity: 0.85;">{analyst_note}</p>
                </div>
                """, unsafe_allow_html=True)

            # -------------------------------------------------
            # 2. Executive Metric Cards
            # -------------------------------------------------
            kpi_cols = st.columns(6)
            kpis = [
                ("Threat Score", f"{threat_intensity}%", "#ef4444" if threat_intensity > 40 else "#10b981"),
                ("Risk Level", risk_level, "#f59e0b" if risk_level in ["Medium", "High"] else "#10b981"),
                ("Primary Vector", primary_category, "#3b82f6"),
                ("Network Status", network_status, "#ef4444" if network_status != "Safe" else "#10b981"),
                ("Total Flows", str(total_flows), "#f9fafb"),
                ("Attacks Detected", f"{attack_count} / {total_flows}", "#ef4444" if attack_count > 0 else "#10b981"),
            ]

            for col, (label, val, color) in zip(kpi_cols, kpis):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color: {color};">{val}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # -------------------------------------------------
            # 3. Interactive Threat Visualizations
            # -------------------------------------------------
            viz_col1, viz_col2 = st.columns([1.4, 1])

            with viz_col1:
                st.subheader("📈 Threat Intensity Timeline")
                timeline_df = pd.DataFrame({
                    "Flow Sequence": np.arange(1, len(scores) + 1),
                    "Threat Score": scores,
                    "Attack Category": flows_df['attack_category'] if 'attack_category' in flows_df.columns else ["Unknown"] * len(scores)
                })

                fig_timeline = px.line(
                    timeline_df,
                    x="Flow Sequence",
                    y="Threat Score",
                    title="Real-Time Connection Threat Trajectory",
                    markers=True,
                    hover_data=["Attack Category"],
                    color_discrete_sequence=["#ef4444"]
                )
                fig_timeline.add_hline(
                    y=engine.config.get('threshold', 0.5),
                    line_dash="dash",
                    line_color="#f59e0b",
                    annotation_text="Attack Threshold (0.50)",
                    annotation_position="bottom right"
                )
                fig_timeline.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#111827",
                    paper_bgcolor="#111827",
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis=dict(range=[0, 1.05])
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

            with viz_col2:
                st.subheader("🎯 Threat Distribution")
                dist_tab_donut, dist_tab_bar = st.tabs(["Attack Categories", "Binary Breakdown"])

                with dist_tab_donut:
                    if category_counts:
                        cat_df = pd.DataFrame(list(category_counts.items()), columns=["Category", "Count"])
                        color_map = {
                            "Normal": "#10b981",
                            "Reconnaissance": "#06b6d4",
                            "DoS": "#f59e0b",
                            "Exploits": "#ef4444",
                            "Fuzzers": "#8b5cf6",
                            "Backdoor": "#ec4899",
                            "Analysis": "#6366f1",
                            "Generic": "#f97316",
                            "Shellcode": "#e11d48",
                            "Worms": "#b91c1c"
                        }
                        fig_cat = px.pie(
                            cat_df,
                            names="Category",
                            values="Count",
                            hole=0.55,
                            color="Category",
                            color_discrete_map=color_map
                        )
                        fig_cat.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="#111827",
                            paper_bgcolor="#111827",
                            margin=dict(l=10, r=10, t=10, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3)
                        )
                        st.plotly_chart(fig_cat, use_container_width=True)
                    else:
                        st.info("No categorical classifications available.")

                with dist_tab_bar:
                    pie_data = pd.DataFrame({
                        "Classification": ["Normal Traffic", "Malicious Flows"],
                        "Count": [normal_count, attack_count]
                    })
                    fig_pie = px.pie(
                        pie_data,
                        names="Classification",
                        values="Count",
                        hole=0.55,
                        color="Classification",
                        color_discrete_map={"Normal Traffic": "#10b981", "Malicious Flows": "#ef4444"}
                    )
                    fig_pie.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#111827",
                        paper_bgcolor="#111827",
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

            # -------------------------------------------------
            # 4. Flow Telemetry Table & XAI Root Cause
            # -------------------------------------------------
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📋 Flow Telemetry & Explainable AI (XAI) Root Cause")
            st.caption("Inspecting individual flows, multi-class threat signatures, and Tree SHAP contributing factors.")

            display_cols = [
                c for c in [
                    'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'service',
                    'state', 'prediction', 'attack_category', 'threat_score', 'primary_threat_driver'
                ] if c in flows_df.columns
            ]
            st.dataframe(flows_df[display_cols], use_container_width=True, height=240)

            # Interactive Single-Flow XAI Deep Dive
            with st.expander("🔍 Deep-Dive: Inspect Flow Feature Attribution (Tree SHAP)", expanded=True):
                st.markdown("Select a specific connection to visualize the exact feature drivers pushing the model's threat verdict:")
                
                flow_options = [
                    f"Flow #{i+1} | {flows_df.iloc[i].get('src_ip', '')}:{flows_df.iloc[i].get('src_port', '')} ➔ {flows_df.iloc[i].get('dst_ip', '')}:{flows_df.iloc[i].get('dst_port', '')} | {flows_df.iloc[i].get('attack_category', 'Unknown')} (Score: {flows_df.iloc[i].get('threat_score', 0):.2f})"
                    for i in range(len(flows_df))
                ]
                selected_flow_idx = st.selectbox("Select Flow to Explain", range(len(flows_df)), format_func=lambda x: flow_options[x])

                if selected_flow_idx is not None:
                    target_flow = flows_df.iloc[[selected_flow_idx]]
                    explanation = engine.explain_flow(flows_df, flow_index=selected_flow_idx)

                    xai_col1, xai_col2 = st.columns([1.2, 1])
                    with xai_col1:
                        st.markdown("#### Top Contributing Factors (Tree SHAP Values):")
                        top_factors = explanation.get('top_factors', [])
                        if top_factors:
                            xai_chart_df = pd.DataFrame(top_factors)
                            xai_chart_df['Impact'] = xai_chart_df['shap_value'].apply(lambda x: 'Pushes ATTACK' if x > 0 else 'Pushes NORMAL')
                            
                            fig_shap = px.bar(
                                xai_chart_df,
                                x='shap_value',
                                y='feature',
                                orientation='h',
                                color='Impact',
                                color_discrete_map={'Pushes ATTACK': '#ef4444', 'Pushes NORMAL': '#10b981'},
                                text='impact_pct',
                                title=f"Feature Push on Flow #{selected_flow_idx+1}"
                            )
                            fig_shap.update_layout(
                                template="plotly_dark",
                                plot_bgcolor="#111827",
                                paper_bgcolor="#111827",
                                margin=dict(l=10, r=10, t=30, b=10),
                                yaxis=dict(autorange="reversed")
                            )
                            st.plotly_chart(fig_shap, use_container_width=True)
                        else:
                            st.info("SHAP attribution not available for this flow.")

                    with xai_col2:
                        st.markdown("#### Security Analyst Interpretation:")
                        st.markdown(f"**Target Flow:** `{flow_options[selected_flow_idx]}`")
                        st.markdown(f"**Threat Category:** `{flows_df.iloc[selected_flow_idx].get('attack_category', 'Normal')}`")
                        st.markdown(f"**Anomaly Confidence:** `{flows_df.iloc[selected_flow_idx].get('threat_score', 0.0):.3f}`")
                        st.markdown("---")
                        st.markdown("**Dominant Threat Signatures:**")
                        for idx, tf in enumerate(explanation.get('top_factors', []), 1):
                            st.markdown(f"**{idx}. `{tf['feature']}` = `{tf['value']}`** ({tf['impact_pct']})")
                            st.caption(tf.get('explanation', 'Metric evaluated by tree model.'))

            # -------------------------------------------------
            # 5. Incident Response & Remediation Playbook
            # -------------------------------------------------
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🚨 Incident Response & Remediation Playbook")
            st.caption("Automated threat containment playbooks tailored to the identified intrusion vector.")

            playbook_title = playbook.get('title', 'Network Security Playbook')
            playbook_desc = playbook.get('description', '')
            playbook_icon = playbook.get('icon', '🛡️')
            playbook_sev = playbook.get('severity', 'Medium')
            badge_color = playbook.get('badge_color', '#ef4444')
            actions = playbook.get('immediate_actions', [])

            st.markdown(f"""
            <div style="background-color: #111827; border: 1px solid #374151; border-left: 6px solid {badge_color}; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <h3 style="margin: 0; color: #f9fafb;">{playbook_icon} {playbook_title}</h3>
                    <span style="background-color: {badge_color}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">SEVERITY: {playbook_sev.upper()}</span>
                </div>
                <p style="color: #9ca3af; margin: 10px 0 16px 0; font-size: 1rem;">{playbook_desc}</p>
            </div>
            """, unsafe_allow_html=True)

            col_act_1, col_act_2 = st.columns([1, 1])
            with col_act_1:
                st.markdown("#### ⚡ Immediate Containment Checklist:")
                for idx, act in enumerate(actions, 1):
                    st.markdown(f"**{idx}.** {act}")

            with col_act_2:
                st.markdown("#### 🛡️ Multi-Platform Firewall Defense Rules:")
                has_attacks = ('prediction' in flows_df.columns) and (flows_df['prediction'] == 'ATTACK').any()
                primary_attacker_ip = flows_df[flows_df['prediction'] == 'ATTACK']['src_ip'].iloc[0] if (has_attacks and 'src_ip' in flows_df.columns) else '192.168.1.100'

                fw_tab_win, fw_tab_linux, fw_tab_snort = st.tabs(["Windows Defender", "Linux (iptables)", "Snort / Suricata Rule"])
                with fw_tab_win:
                    st.code(
                        f"netsh advfirewall firewall add rule name=\"Block_{primary_category}_{primary_attacker_ip}\" dir=in action=block remoteip={primary_attacker_ip}",
                        language="powershell"
                    )
                with fw_tab_linux:
                    st.code(
                        f"iptables -A INPUT -s {primary_attacker_ip} -j DROP\n# Or with nftables:\nnft add rule inet filter input ip saddr {primary_attacker_ip} drop",
                        language="bash"
                    )
                with fw_tab_snort:
                    st.code(
                        f"alert ip {primary_attacker_ip} any -> any any (msg:\"[Network Detector] Malicious {primary_category} Traffic Blocked\"; sid:1000001; rev:1;)",
                        language="text"
                    )

            # -------------------------------------------------
            # 6. Report Downloads (Executive PDF & CSV)
            # -------------------------------------------------
            st.markdown("<hr style='border-color: #374151;'>", unsafe_allow_html=True)
            st.subheader("📑 Threat Intelligence Export")

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                pdf_report_bytes = generate_executive_pdf_report(
                    results=results,
                    flows_df=flows_df,
                    source_filename=source_name or "capture.pcap"
                )
                clean_stem = Path(source_name or 'capture').stem
                st.download_button(
                    "📄 Download Executive Threat Report (PDF)",
                    data=pdf_report_bytes,
                    file_name=f"Executive_Threat_Report_{clean_stem}.pdf",
                    mime="application/pdf",
                    help="Generates an executive SOC PDF report with vector threat gauges, Geo-IP breakdown, and remediation playbooks.",
                    use_container_width=True
                )

            with col_dl2:
                csv_buffer = io.StringIO()
                flows_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download Flow Telemetry & Attribution (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name=f"threat_telemetry_{clean_stem}.csv",
                    mime="text/csv",
                    help="Raw flow features with XGBoost / Random Forest scores and SHAP attributions.",
                    use_container_width=True
                )

    else:
        # Landing Guide
        st.info("👈 Please upload a `.pcap` / `.csv` file in the sidebar, or launch **🎙️ Live Network Sniff** to begin analysis.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 1. Packet Capture")
            st.caption("Upload `.pcap` or `.pcapng` network trace files captured via Wireshark, tcpdump, or Zeek.")
        with c2:
            st.markdown("### 2. Feature Synthesis")
            st.caption("Scapy parses packet headers to compute 32 UNSW-NB15 flow metrics (dur, bytes, TTL, TCP RTT, connection counts).")
        with c3:
            st.markdown("### 3. Bayesian Threat Fusion")
            st.caption("Supervised Random Forest + XGBoost models team up with Isolation Forest to spot attacks and zero-days.")


# =========================================================
# TAB 2: UNSW-NB15 42-Feature Security Dictionary
# =========================================================
with main_tab_dictionary:
    st.subheader("📖 Complete 42-Feature UNSW-NB15 Security Dictionary")
    st.caption("Detailed definitions, physical units, and intrusion detection significance for every engineered feature.")

    search_col1, search_col2 = st.columns([1.5, 1])
    with search_col1:
        search_query = st.text_input("🔍 Search features by name, keyword, or attack type:", placeholder="e.g. sttl, handshake, flood, http...")
    with search_col2:
        categories = ["All Categories"] + sorted(list(set(v['category'] for v in FEATURE_DICTIONARY.values())))
        selected_category = st.selectbox("Filter by Category", categories)

    filtered_dict = {}
    for feat_key, feat_info in FEATURE_DICTIONARY.items():
        # Filter by category
        if selected_category != "All Categories" and feat_info['category'] != selected_category:
            continue
        # Filter by search query
        if search_query:
            query = search_query.lower()
            match = (
                query in feat_key.lower() or
                query in feat_info['name'].lower() or
                query in feat_info['definition'].lower() or
                query in feat_info['security_significance'].lower() or
                query in feat_info['attack_correlation'].lower()
            )
            if not match:
                continue
        filtered_dict[feat_key] = feat_info

    st.write(f"Showing **{len(filtered_dict)}** of 42 features:")

    for feat_key, feat_info in filtered_dict.items():
        with st.expander(f"**{feat_info['name']}** — *{feat_info['category']}*"):
            c_a, c_b = st.columns([1.2, 1])
            with c_a:
                st.markdown(f"**Definition:** {feat_info['definition']}")
                st.markdown(f"**Security Significance:** {feat_info['security_significance']}")
            with c_b:
                st.markdown(f"**Units / Data Type:** `{feat_info['units']}`")
                st.markdown(f"**Predictive Importance:** `{feat_info['importance_rank']}`")
                st.markdown(f"**Associated Attack Families:** `{feat_info['attack_correlation']}`")
