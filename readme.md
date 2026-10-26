# 🛡️ Network Detector — Network Intrusion Detection System (NIDS)

> An end-to-end Machine Learning–based Network Threat Intelligence Platform that ingests live network streams, `.pcap` / `.pcapng` packet captures, and flow datasets; extracts 32 flow-level features using Scapy & NFStream; classifies threats with an ensemble of Random Forest and XGBoost classifiers paired with an Isolation Forest anomaly detector; attributes root causes with **Tree SHAP (Explainable AI)**; categorizes threats into **9 attack families**; and deploys automated **firewall incident response playbooks**.

Benchmarked against the **UNSW-NB15** network security dataset.

---

## 🏗️ Architecture & Pipeline

```text
1. Network Ingestion
   ├── 🎙️ Live Network Sniffer (Wi-Fi, Ethernet, Loopback via Scapy AsyncSniffer)
   ├── 📁 File Upload (.pcap, .pcapng, .csv)
   └── 🚀 Demo Presets (benign_traffic.pcap, attack_synflood.pcap)
      │
      ▼
2. Packet Dissection & Flow Aggregation (src/feature_extraction.py)
   ├── Pure-Python Scapy aggregation engine (Windows compatible, zero DLL crashes)
   └── NFStream bidirectional statistics fallback
      │
      ▼
3. 32 Flow Features Extracted
   ├── Baseline: dur, proto, service, state, spkts, dpkts, sbytes, dbytes, sload, dload, etc.
   ├── Packet Level: sttl, dttl, swin, dwin, tcprtt, synack, trans_depth, response_body_len, is_ftp_login
   └── Sliding Window (100 flows): ct_srv_src, ct_state_ttl, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, ct_srv_dst, ct_flw_http_mthd
      │
      ▼
4. Preprocessing & Alignment (src/prediction.py)
   ├── Aligns real-time features to 42 UNSW-NB15 model input space
   └── Handles missing/infinite values & categorical encoding
      │
      ├───► Supervised Ensemble (50% Random Forest + 50% XGBoost)
      └───► Unsupervised Anomaly Detection (Isolation Forest for Zero-Days)
      │
      ▼
5. Bayesian Score Fusion: 0.5 * P(Ensemble) + 0.5 * P(Anomaly)
      │
      ├───► 🔬 Explainable AI (XAI): Tree SHAP feature attribution & top attack drivers
      └───► 🎯 Multi-Class Classifier: Categorizes into 9 specific attack categories
      │
      ▼
6. Interactive SOC Dashboard (dashboard/app.py)
   ├── Real-time Threat Score Timeline & Flow Severity Ratios (Plotly)
   ├── 9-Class Threat Category Distribution Bar Chart
   ├── Flow Inspection Telemetry Table with Primary Attack Drivers
   ├── Top 3 Attack Factors Cards with Percentage Risk Impacts
   └── 🚨 Automated Incident Response & Multi-Platform Firewall Playbooks
```

---

## ⚡ Core Capabilities

### 1. 🎙️ Real-Time Live Capture Mode
* Sniffs live network traffic directly from physical or virtual network adapters (`Wi-Fi`, `Ethernet`, `Software Loopback`).
* Features an asynchronous sniffer engine ([src/live_capture.py](src/live_capture.py)) and a safe live stream generator.
* Flushes captured packets into temporary buffers and feeds them into the flow extractor automatically.

### 2. 🔬 Explainable AI (XAI) with Tree SHAP
* Solves the black-box problem by explaining **why** a flow was flagged as malicious.
* Calculates exact feature attribution using XGBoost's native Tree SHAP.
* Displays the **Top 3 Attack Drivers** with observed raw values and percentage risk impact (e.g., `sttl=254 (+21.4%)`, `ct_src_dport_ltm=44 (+12.6%)`).
* Renders an interactive diverging SHAP bar chart (red for attack push, green for benign push).

### 3. 🎯 Multi-Class Threat Categorization (9 Threat Families)
* In addition to binary detection (`NORMAL` vs `ATTACK`), classifies threats into:
  1. **Reconnaissance** (Port scans & network sweeps)
  2. **DoS / DDoS** (SYN floods, UDP blasts)
  3. **Exploits** (RCE, buffer overflows)
  4. **Fuzzers** (Malformed protocol payloads)
  5. **Backdoors** (C2 beaconing)
  6. **Shellcode** (Injected bytecode)
  7. **Analysis** (Vulnerability crawlers)
  8. **Generic** (Cipher & protocol anomalies)
  9. **Worms** (Lateral spreading)

### 4. 🚨 Automated Incident Response Playbooks
* Generates tailored containment checklists based on the primary attack category.
* Outputs ready-to-run firewall blocking rules:
  * **Windows Defender Firewall** (`netsh advfirewall firewall add rule ...`)
  * **Linux** (`iptables` / `nftables` drop commands)
  * **Snort / Suricata IDS** rule definitions

---

## 🚀 Quickstart

### 1. Run the Dashboard
From the root workspace directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open your browser to:
👉 **`http://localhost:8501`**

### 2. Live Sniffing
1. In the sidebar under **Select Input Source**, pick **🎙️ Live Network Sniff**.
2. Select your network adapter (`Wi-Fi`, `Ethernet`, etc.).
3. Choose your duration and click **▶️ Start Live Sniff**.
4. The dashboard records the live traffic, extracts telemetry, calculates threat scores, and updates the threat charts in real time!

### 3. Training Scripts
* **Isolation Forest (Unsupervised):**
  ```powershell
  .\.venv\Scripts\python.exe src/train_isolation_forest.py --samples 5000
  ```
* **Multi-Class Threat Classifier (9 Categories):**
  ```powershell
  .\.venv\Scripts\python.exe src/train_multiclass.py --sample-size 35000
  ```
* **Generate Synthetic Test PCAPs:**
  ```powershell
  .\.venv\Scripts\python.exe data/generate_pcaps.py
  ```

---

## 📁 Project Structure

```text
Network Intrusion Detection/
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py     # 32-feature extraction engine (Scapy + NFStream)
│   ├── prediction.py             # Ensemble + Isolation Forest + SHAP XAI + Playbooks
│   ├── live_capture.py           # Live adapter sniffing & stream generator
│   ├── train_isolation_forest.py # Standalone Isolation Forest trainer
│   └── train_multiclass.py       # Standalone 9-category classifier trainer
├── dashboard/
│   └── app.py                    # Streamlit SOC Threat Intelligence interface
├── models/
│   ├── random_forest.joblib      # Trained Random Forest pipeline (<25MB)
│   ├── xgboost.joblib            # Trained XGBoost pipeline (<1MB)
│   ├── isolation_forest.joblib   # Trained baseline Isolation Forest
│   ├── iso_num_cols.joblib       # Feature order for anomaly detection
│   ├── multiclass_model.joblib   # Trained 9-category classifier (<1MB)
│   ├── multiclass_labels.joblib  # Multi-class label encoder
│   └── ensemble_config.json      # Pipeline weights and thresholds
├── data/
│   ├── benign_traffic.pcap       # Benign web/DNS test capture
│   ├── attack_synflood.pcap      # SYN flood / port scan test capture
│   └── generate_pcaps.py         # Packet synthesizer script
├── notebooks/                    # Research and exploration notebooks (01-05)
├── requirements.txt              # Standard UTF-8 Python dependencies
└── readme.md                     # Documentation
```
