import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Features expected by the UNSW-NB15 trained models
ALL_42_FEATURES = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts',
    'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload',
    'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit',
    'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
    'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
    'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
    'is_sm_ips_ports'
]

# Features not calculable from raw PCAPs without deep state tracking, filled with 0
TO_FILL_ZERO = [
    'rate', 'dinpkt', 'sjit', 'djit', 'stcpb', 'dtcpb',
    'ackdat', 'ct_dst_ltm', 'ct_ftp_cmd', 'ct_src_ltm'
]

ENSEMBLE_WEIGHT = 0.5
IF_WEIGHT = 0.5
ATTACK_RATIO_THRESHOLD = 0.3


class ThreatPredictionEngine:
    def __init__(self, models_dir: str = "models"):
        p = Path(models_dir)
        if not p.is_absolute() and not p.exists():
            root = Path(__file__).resolve().parent.parent
            if (root / models_dir).exists():
                p = root / models_dir
        self.models_dir = p
        self.rf_model = None
        self.xgb_model = None
        self.config = {}
        self.iso_forest = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads trained pipelines, isolation forest, and ensemble config."""
        rf_path = self.models_dir / "random_forest.joblib"
        xgb_path = self.models_dir / "xgboost.joblib"
        cfg_path = self.models_dir / "ensemble_config.json"
        iso_path = self.models_dir / "isolation_forest.joblib"
        iso_cols_path = self.models_dir / "iso_num_cols.joblib"

        if rf_path.exists():
            self.rf_model = joblib.load(rf_path)
        if xgb_path.exists():
            self.xgb_model = joblib.load(xgb_path)
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                self.config = json.load(f)

        if iso_path.exists() and iso_cols_path.exists():
            self.iso_forest = joblib.load(iso_path)
            self.iso_num_cols = joblib.load(iso_cols_path)
        else:
            self.iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
            self.iso_num_cols = None

        # Multi-class attack classifier (9 threat categories)
        mc_path = self.models_dir / "multiclass_model.joblib"
        mc_labels_path = self.models_dir / "multiclass_labels.joblib"
        if mc_path.exists() and mc_labels_path.exists():
            self.multiclass_model = joblib.load(mc_path)
            self.multiclass_labels = joblib.load(mc_labels_path)
        else:
            self.multiclass_model = None
            self.multiclass_labels = None

    def prepare_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Aligns extracted features with the 42 features required by the pipeline."""
        df = raw_df.copy()

        # Fill absent features with default zeros
        for col in TO_FILL_ZERO:
            if col not in df.columns:
                df[col] = 0.0

        # Ensure all 42 features exist
        for col in ALL_42_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        aligned_df = df[ALL_42_FEATURES].copy()
        aligned_df = aligned_df.fillna(0).replace([np.inf, -np.inf], 0)
        return aligned_df

    # Alias for feature alignment
    _align_features = prepare_features

    def predict_anomaly(self, X_df: pd.DataFrame) -> np.ndarray:
        """Computes anomaly scores normalized between [0, 1] relative to normal baseline."""
        if self.iso_num_cols is not None and hasattr(self.iso_forest, "decision_function"):
            cols = [c for c in self.iso_num_cols if c in X_df.columns]
            X_num = X_df[cols].fillna(0)
            scores = self.iso_forest.decision_function(X_num)
            # Negative scores indicate anomaly, flip so higher = more anomalous
            flipped = -scores
            # Typical decision function range is roughly [-0.2, 0.2]
            # Map with sigmoid or min-max normalization
            prob = 1.0 / (1.0 + np.exp(-10.0 * flipped))
            return prob

        # Fallback if no pre-trained model is saved
        num_cols = X_df.select_dtypes(include=[np.number]).columns
        X_num = X_df[num_cols].fillna(0)
        if len(X_df) > 1:
            scores = self.iso_forest.fit(X_num).decision_function(X_num)
            flipped = -scores
            s_min, s_max = flipped.min(), flipped.max()
            if s_max - s_min == 0:
                return np.zeros(len(X_df))
            return (flipped - s_min) / (s_max - s_min)
        return np.array([0.1])

    def predict_ensemble(self, X_df: pd.DataFrame) -> np.ndarray:
        """Averages predicted probabilities from Random Forest and XGBoost."""
        probs = []
        if self.rf_model is not None:
            rf_prob = self.rf_model.predict_proba(X_df)[:, 1]
            probs.append(rf_prob)
        if self.xgb_model is not None:
            xgb_prob = self.xgb_model.predict_proba(X_df)[:, 1]
            probs.append(xgb_prob)

        if not probs:
            return np.zeros(len(X_df))

        return np.mean(probs, axis=0)

    def analyze(self, raw_df: pd.DataFrame) -> dict:
        """
        Master method: runs ensemble + anomaly detection, applies Bayesian fusion,
        and generates the final SOC verdict.
        """
        if raw_df.empty:
            return {
                'verdict': 'NORMAL',
                'total_flows': 0,
                'risk_level': 'Low',
                'network_status': 'Safe',
                'threat_intensity': 0.0,
                'traffic_behavior': 'Stable',
                'peak_flow_index': 0,
                'analyst_note': 'No network flows detected.',
                'all_flow_scores': [],
                'confidence': 100.0,
                'attack_flow_count': 0,
                'normal_flow_count': 0
            }

        X_df = self.prepare_features(raw_df)
        ensemble_prob = self.predict_ensemble(X_df)
        if_prob = self.predict_anomaly(X_df)

        # Bayesian weighted fusion
        combined_prob = (ENSEMBLE_WEIGHT * ensemble_prob) + (IF_WEIGHT * if_prob)
        threshold = self.config.get("threshold", 0.5)

        predictions = (combined_prob >= threshold).astype(int)
        total_flows = len(predictions)
        attack_count = int(np.sum(predictions == 1))
        attack_ratio = attack_count / total_flows

        verdict = 'ATTACK' if attack_ratio >= ATTACK_RATIO_THRESHOLD else 'NORMAL'
        threat_intensity = round(float(np.mean(combined_prob)) * 100, 1)
        peak_flow_index = int(np.argmax(combined_prob)) + 1
        raw_confidence = abs(float(np.mean(combined_prob)) - float(threshold))
        confidence = round(min(raw_confidence * 250, 100), 1)

        if verdict == 'NORMAL':
            if np.mean(combined_prob) < 0.35:
                risk_level = 'Low'
                network_status = 'Safe'
                traffic_behavior = 'Stable' if np.std(combined_prob) < 0.2 else 'Unstable'
                analyst_note = 'Network flows appear benign. No abnormal behaviors or indicators of compromise.'
            else:
                risk_level = 'Medium'
                network_status = 'Suspicious'
                traffic_behavior = 'Stable' if np.std(combined_prob) < 0.2 else 'Unstable'
                analyst_note = 'Flows are largely normal, but isolated flows showed elevated anomaly scores. Continue monitoring.'
        else:
            risk_level = 'High'
            network_status = 'Danger'
            traffic_behavior = 'Erratic'
            analyst_note = f'High threat intensity detected ({threat_intensity}%). Multi-vector intrusion pattern identified.'

        # Multi-class attack categorization (9 specific threat types)
        flow_categories = self.predict_categories(X_df, combined_prob)
        category_counts = {}
        for cat in flow_categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1

        if verdict == 'ATTACK':
            attack_cat_counts = {k: v for k, v in category_counts.items() if k != 'Normal'}
            primary_category = max(attack_cat_counts, key=attack_cat_counts.get) if attack_cat_counts else 'DoS'
        else:
            primary_category = 'Normal'

        playbook = REMEDIATION_PLAYBOOKS.get(primary_category, REMEDIATION_PLAYBOOKS['Normal'])

        return {
            'verdict': verdict,
            'total_flows': total_flows,
            'attack_flow_count': attack_count,
            'normal_flow_count': total_flows - attack_count,
            'risk_level': risk_level,
            'network_status': network_status,
            'threat_intensity': threat_intensity,
            'traffic_behavior': traffic_behavior,
            'peak_flow_index': peak_flow_index,
            'analyst_note': analyst_note,
            'all_flow_scores': combined_prob.round(4).tolist(),
            'confidence': confidence,
            'primary_category': primary_category,
            'category_breakdown': category_counts,
            'flow_categories': flow_categories,
            'playbook': playbook
        }

    def predict_categories(self, X_df: pd.DataFrame, threat_scores: np.ndarray = None) -> list:
        """
        Classifies network flows into 9 specific attack categories:
        Reconnaissance, DoS, Exploits, Fuzzers, Backdoors, Shellcode, Analysis, Generic, Worms, or Normal.
        """
        if self.multiclass_model is None or self.multiclass_labels is None:
            if threat_scores is not None:
                return ["ATTACK" if s >= 0.5 else "Normal" for s in threat_scores]
            return ["Normal"] * len(X_df)

        probs = self.multiclass_model.predict_proba(X_df)
        classes = list(self.multiclass_labels.classes_)
        attack_classes = [c for c in classes if c != 'Normal']
        attack_indices = [classes.index(c) for c in attack_classes]

        categories = []
        for i in range(len(X_df)):
            score = threat_scores[i] if threat_scores is not None else 0.0
            if score >= 0.5 and attack_indices:
                sub_probs = probs[i, attack_indices]
                best_attack = attack_classes[int(np.argmax(sub_probs))]
                categories.append(best_attack)
            else:
                categories.append("Normal")

        return categories

    def explain_flow(self, X_df: pd.DataFrame, flow_index: int = 0, top_k: int = 5) -> dict:
        """
        Uses Tree SHAP (from XGBoost) to explain exactly why a specific flow was flagged.
        Returns top attack drivers, mitigating factors, and human-readable analyst evidence.
        """
        import xgboost as xgb

        if self.xgb_model is None or X_df.empty or flow_index >= len(X_df):
            return {"error": "Model or flow index unavailable", "top_factors": [], "attack_drivers": []}

        row_df = self._align_features(X_df.iloc[[flow_index]])
        prep = self.xgb_model.named_steps['preprocessor']
        clf = self.xgb_model.named_steps['classifier']

        X_trans = prep.transform(row_df)
        f_names = [str(f) for f in prep.get_feature_names_out()]
        dmat = xgb.DMatrix(X_trans, feature_names=f_names)

        # Native Tree SHAP prediction contributions: last element is bias / base value
        raw_shap = clf.get_booster().predict(dmat, pred_contribs=True)[0]
        base_value = float(raw_shap[-1])
        contributions = raw_shap[:-1]

        pos_sum = max(1e-6, sum(c for c in contributions if c > 0))

        features_info = []
        for i, val in enumerate(contributions):
            fname = f_names[i]
            if fname.startswith("num__"):
                raw_key = fname[5:]
                label = FEATURE_HUMAN_NAMES.get(raw_key, raw_key)
                raw_val = row_df[raw_key].values[0] if raw_key in row_df else 0
                display_val = f"{raw_val:.3g}" if isinstance(raw_val, (int, float)) else str(raw_val)
            elif fname.startswith("cat__"):
                parts = fname[5:].split("_", 1)
                raw_key = parts[0]
                cat_val = parts[1] if len(parts) > 1 else ""
                label = f"{FEATURE_HUMAN_NAMES.get(raw_key, raw_key)}: {cat_val}"
                raw_val = row_df[raw_key].values[0] if raw_key in row_df else ""
                display_val = str(raw_val)
            else:
                raw_key = fname
                label = fname
                display_val = ""

            features_info.append({
                'feature': label,
                'feature_key': raw_key,
                'feature_label': label,
                'value': display_val,
                'raw_value': display_val,
                'shap_value': round(float(val), 4),
                'impact_pct': f"+{round((float(val) / pos_sum * 100), 1)}%" if val > 0 else f"{round((float(val) / pos_sum * 100), 1)}%",
                'explanation': f"{label} pushes threat score by {val:+.3f}"
            })

        # Top positive attack drivers
        attack_drivers = sorted(
            [f for f in features_info if f['shap_value'] > 0],
            key=lambda x: x['shap_value'],
            reverse=True
        )[:top_k]

        # Top negative / mitigating benign factors
        benign_drivers = sorted(
            [f for f in features_info if f['shap_value'] < 0],
            key=lambda x: x['shap_value']
        )[:top_k]

        # Top drivers combined for waterfall chart
        top_waterfall = sorted(
            features_info,
            key=lambda x: abs(x['shap_value']),
            reverse=True
        )[:top_k * 2]

        # Generate SOC Analyst Plain-English Explanation
        if attack_drivers:
            drivers_text = ", ".join([
                f"{d['feature_label']} ({d['feature_key']}={d['raw_value']}, {d['impact_pct']})"
                for d in attack_drivers[:3]
            ])
            summary = f"Flagged as malicious primarily due to: {drivers_text}."
        else:
            summary = "Flow exhibits standard benign telemetry across all network indicators."

        return {
            'flow_index': flow_index,
            'base_value': round(base_value, 4),
            'top_factors': top_waterfall,
            'attack_drivers': attack_drivers,
            'benign_drivers': benign_drivers,
            'waterfall_features': top_waterfall,
            'analyst_summary': summary
        }

    def explain_all_flows(self, X_df: pd.DataFrame) -> list:
        """Computes the primary threat driver for each flow in the dataset."""
        import xgboost as xgb

        if self.xgb_model is None or X_df.empty:
            return []

        prep = self.xgb_model.named_steps['preprocessor']
        clf = self.xgb_model.named_steps['classifier']
        X_aligned = self._align_features(X_df)
        X_trans = prep.transform(X_aligned)
        f_names = [str(f) for f in prep.get_feature_names_out()]
        dmat = xgb.DMatrix(X_trans, feature_names=f_names)

        raw_shap = clf.get_booster().predict(dmat, pred_contribs=True)[:, :-1]

        top_drivers = []
        for i in range(len(X_df)):
            row_contribs = raw_shap[i]
            max_idx = int(np.argmax(row_contribs))
            if row_contribs[max_idx] > 0.05:
                fname = f_names[max_idx]
                clean = fname.replace("num__", "").replace("cat__", "")
                val = row_contribs[max_idx]
                label = FEATURE_HUMAN_NAMES.get(clean.split("_")[0], clean)
                top_drivers.append(f"{label} (+{val:.2f})")
            else:
                top_drivers.append("Benign Profile")

        return top_drivers


FEATURE_HUMAN_NAMES = {
    'sttl': 'Source TTL (Scanner Fingerprint)',
    'dttl': 'Destination TTL',
    'ct_srv_dst': 'Target Service Connection Burst',
    'ct_srv_src': 'Source Service Fan-Out',
    'ct_state_ttl': 'Identical State-TTL Repetition',
    'ct_dst_src_ltm': 'Host-to-Host Connection Frequency',
    'ct_src_dport_ltm': 'Target Port Probing Rate',
    'ct_dst_sport_ltm': 'Source Port Probing Rate',
    'dur': 'Flow Duration',
    'sbytes': 'Source Data Volume (Bytes)',
    'dbytes': 'Destination Data Volume (Bytes)',
    'spkts': 'Source Packet Count',
    'dpkts': 'Destination Packet Count',
    'sload': 'Source Throughput (bps)',
    'dload': 'Destination Throughput (bps)',
    'sloss': 'Source Retransmissions/Resets',
    'dloss': 'Destination Retransmissions/Resets',
    'swin': 'Source TCP Window Advertisement',
    'dwin': 'Destination TCP Window Advertisement',
    'tcprtt': 'TCP Handshake Round Trip Time',
    'synack': 'SYN-ACK Handshake Delay',
    'sinpkt': 'Source Packet Inter-Arrival Time',
    'smean': 'Source Mean Packet Size',
    'dmean': 'Destination Mean Packet Size',
    'trans_depth': 'Pipelined HTTP Transactions',
    'response_body_len': 'HTTP Response Body Size',
    'is_ftp_login': 'FTP Login Status',
    'ct_flw_http_mthd': 'Recent HTTP Requests Frequency',
    'is_sm_ips_ports': 'Spoofed IP/Port Loopback',
    'proto': 'Protocol Type',
    'service': 'Service Protocol',
    'state': 'Connection State'
}


REMEDIATION_PLAYBOOKS = {
    'Reconnaissance': {
        'title': 'Port Scan & Reconnaissance Mitigation Playbook',
        'severity': 'Medium',
        'badge_color': '#f59e0b',
        'icon': '🔍',
        'description': 'Hostile probe or sweep detected targeting open ports, service banners, and network topology.',
        'immediate_actions': [
            'Deploy dynamic rate-limiting or drop rules on the source IP at edge firewalls.',
            'Confirm non-essential listening ports (e.g., 21, 23, 3389) are closed or protected behind VPN/MFA.',
            'Enable TCP SYN cookies on web and application gateways to prevent half-open state starvation.'
        ],
        'firewall_rule': 'netsh advfirewall firewall add rule name="Block_Recon_Scanner" dir=in action=block remoteip={ip}'
    },
    'DoS': {
        'title': 'Denial of Service (DoS/DDoS) Mitigation Playbook',
        'severity': 'Critical',
        'badge_color': '#ef4444',
        'icon': '💥',
        'description': 'High-volume packet flood detected attempting compute or bandwidth exhaustion.',
        'immediate_actions': [
            'Engage edge upstream DDoS scrubbing and border router filtering.',
            'Null-route (blackhole) hostile source ingress prefixes targeting service ports.',
            'Activate connection rate limits and adjust TCP keepalive timeouts to recycle dead sockets.'
        ],
        'firewall_rule': 'netsh advfirewall firewall add rule name="Block_DoS_Flooder" dir=in action=block remoteip={ip}'
    },
    'Exploits': {
        'title': 'Vulnerability Exploit / RCE Mitigation Playbook',
        'severity': 'Critical',
        'badge_color': '#ef4444',
        'icon': '☣️',
        'description': 'Remote code execution, memory corruption, or known CVE exploit payload pattern detected.',
        'immediate_actions': [
            'Isolate the target destination endpoint from the local subnet to prevent lateral movement.',
            'Capture volatile memory (RAM) dump before restarting services for root-cause forensic analysis.',
            'Apply urgent vendor security patches and review Web Application Firewall (WAF) blocking signatures.'
        ],
        'firewall_rule': 'netsh advfirewall firewall add rule name="Isolate_Exploit_Target" dir=in action=block remoteip={ip}'
    },
    'Fuzzers': {
        'title': 'Malformed Payload / Protocol Fuzzer Mitigation Playbook',
        'severity': 'High',
        'badge_color': '#f97316',
        'icon': '⚡',
        'description': 'Malformed, oversized, or irregular packet fields detected attempting to crash protocol handlers.',
        'immediate_actions': [
            'Inspect protocol daemon logs for unhandled buffer exceptions or core dumps.',
            'Enforce strict RFC protocol compliance validation at boundary reverse proxies.',
            'Drop non-standard TCP options and truncated fragmented packet sequences.'
        ],
        'firewall_rule': 'netsh advfirewall firewall add rule name="Block_Fuzzing_Source" dir=in action=block remoteip={ip}'
    },
    'Backdoor': {
        'title': 'Persistent Backdoor & C2 Channel Containment Playbook',
        'severity': 'Critical',
        'badge_color': '#ef4444',
        'icon': '🚪',
        'description': 'Command & Control (C2) beacon or unauthorized persistent backdoor channel detected.',
        'immediate_actions': [
            'Terminate active TCP/UDP sessions to the remote C2 controller IP immediately.',
            'Run full EDR and memory forensic scans on internal endpoints participating in the connection.',
            'Revoke and rotate domain service account credentials, Kerberos tickets, and SSL API keys.'
        ],
        'firewall_rule': 'netsh advfirewall firewall add rule name="Block_C2_Backdoor" dir=in action=block remoteip={ip}'
    },
    'Shellcode': {
        'title': 'Injected Shellcode / Bytecode Mitigation Playbook',
        'severity': 'Critical',
        'badge_color': '#ef4444',
        'icon': '💀',
        'description': 'Direct CPU instruction bytecode / shellcode detected in packet payload stream.',
        'immediate_actions': [
            'Force kill the compromised process listening on the destination port immediately.',
            'Disconnect the endpoint from the network infrastructure and alert incident response tier 3.',
            'Submit payload binary stream to malware sandbox (VirusTotal, Hybrid Analysis) for IOC generation.'
        ],
        'firewall_rule': 'netsh advfirewall firewall add rule name="Block_Shellcode_Source" dir=in action=block remoteip={ip}'
    },
    'Analysis': {
        'title': 'Automated Vulnerability Scanner & Crawler Playbook',
        'severity': 'Low',
        'badge_color': '#3b82f6',
        'icon': '🕵️',
        'description': 'Automated reconnaissance crawler (Nessus, Nikto, DirBuster, or OpenVAS).',
        'immediate_actions': [
            'Verify whether the scan originates from an authorized scheduled security assessment.',
            'If unauthorized, ban source IP on perimeter firewalls and tarpit crawling threads.'
        ],
        'firewall_rule': 'netsh advfirewall firewall add rule name="Block_Scanner_Analysis" dir=in action=block remoteip={ip}'
    },
    'Generic': {
        'title': 'Cryptographic & Protocol Anomaly Playbook',
        'severity': 'Medium',
        'badge_color': '#f59e0b',
        'icon': '🛡️',
        'description': 'Cryptographic collision, cipher anomaly, or generic heuristic deviation detected.',
        'immediate_actions': [
            'Inspect TLS/SSL certificate chains and cipher suite negotiation logs.',
            'Verify integrity of application layer API payloads and HMAC signatures.'
        ],
        'firewall_rule': 'netsh advfirewall firewall add rule name="Block_Generic_Anomaly" dir=in action=block remoteip={ip}'
    },
    'Worms': {
        'title': 'Self-Propagating Network Worm Containment Playbook',
        'severity': 'Critical',
        'badge_color': '#ef4444',
        'icon': '🐛',
        'description': 'Rapid worm propagation pattern attempting cross-host port scans and lateral movement.',
        'immediate_actions': [
            'Sever lateral communications ports (SMB 445, RPC 135, RDP 3389) across local subnets.',
            'Apply immediate network micro-segmentation to quarantine infected broadcast domains.',
            'Issue emergency advisory to patch known lateral exploitation vulnerabilities (e.g. MS17-010).'
        ],
        'firewall_rule': 'netsh advfirewall firewall add rule name="Contain_Worm_Spreader" dir=in action=block remoteip={ip}'
    },
    'Normal': {
        'title': 'Standard Benign Traffic Baseline',
        'severity': 'Info',
        'badge_color': '#10b981',
        'icon': '✅',
        'description': 'Legitimate network communication complying with standard protocol baselines.',
        'immediate_actions': [
            'No remediation needed. Maintain continuous telemetry ingestion.'
        ],
        'firewall_rule': ''
    }
}

