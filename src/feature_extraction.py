import os
import logging
from collections import deque
import numpy as np
import pandas as pd

# Suppress harmless Scapy runtime warnings when Npcap driver isn't installed
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.all import IP, TCP, UDP, Raw, PcapReader

# Try importing NFStreamer; if Npcap/wpcap.dll is missing on Windows, use pure-Scapy fallback
try:
    from nfstream import NFStreamer
    NFSTREAM_AVAILABLE = True
except (ImportError, OSError):
    NFSTREAM_AVAILABLE = False


PROTOCOL_MAP = {
    6: 'tcp',
    17: 'udp',
    1: 'icmp',
    58: 'icmpv6',
    41: 'ipv6',
    132: 'sctp',
    2: 'igmp',
    47: 'gre',
    50: 'esp',
    89: 'ospf',
}

PORT_TO_SERVICE = {
    80: 'http',
    443: 'https',
    8080: 'http',
    21: 'ftp',
    20: 'ftp-data',
    22: 'ssh',
    53: 'dns',
    25: 'smtp',
    110: 'pop3',
    143: 'imap',
    23: 'telnet',
    3306: 'mysql',
    5432: 'postgres',
    6667: 'irc',
    161: 'snmp',
    67: 'dhcp',
    68: 'dhcp',
    123: 'ntp',
    137: 'netbios',
    138: 'netbios',
    139: 'netbios',
    445: 'smb',
    3389: 'rdp',
    5060: 'sip',
}

FEATURE_COLUMNS = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts',
    'sbytes', 'dbytes', 'sttl', 'dttl', 'sload', 'dload',
    'sloss', 'dloss', 'sinpkt', 'swin', 'dwin', 'tcprtt',
    'synack', 'smean', 'dmean', 'trans_depth', 'response_body_len',
    'ct_srv_src', 'ct_state_ttl', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_flw_http_mthd', 'ct_srv_dst',
    'is_sm_ips_ports'
]


def _scapy_aggregate_flows(pcap_path: str) -> pd.DataFrame:
    """
    Pure Python/Scapy flow aggregation fallback.
    Aggregates raw packets into bidirectional flows without requiring external Npcap DLLs.
    """
    flows = {}
    http_methods = (b'GET ', b'POST ', b'PUT ', b'DELETE ', b'HEAD ', b'OPTIONS ', b'PATCH ')

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if not pkt.haslayer(IP):
                continue

            ts = float(pkt.time)
            ip = pkt[IP]
            src_ip, dst_ip = ip.src, ip.dst
            proto = int(ip.proto)
            pkt_len = len(pkt)
            ttl = int(ip.ttl)

            sport, dport = 0, 0
            tcp_win = 0
            tcp_flags = 0
            is_rst = 0
            is_fin = 0
            is_syn = 0
            raw_payload = b''

            if pkt.haslayer(TCP):
                sport = int(pkt[TCP].sport)
                dport = int(pkt[TCP].dport)
                tcp_win = int(pkt[TCP].window)
                tcp_flags = int(pkt[TCP].flags)
                if tcp_flags & 0x04:  # RST
                    is_rst = 1
                if tcp_flags & 0x01:  # FIN
                    is_fin = 1
                if tcp_flags & 0x02:  # SYN
                    is_syn = 1
            elif pkt.haslayer(UDP):
                sport = int(pkt[UDP].sport)
                dport = int(pkt[UDP].dport)

            if pkt.haslayer(Raw):
                raw_payload = bytes(pkt[Raw].load)

            # Bidirectional flow key: canonical unordered pair
            flow_key = tuple(sorted([(src_ip, sport), (dst_ip, dport)])) + (proto,)

            if flow_key not in flows:
                flows[flow_key] = {
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'src_port': sport,
                    'dst_port': dport,
                    'protocol': proto,
                    'start_time': ts,
                    'last_time': ts,
                    'src_packets': 1,
                    'dst_packets': 0,
                    'src_bytes': pkt_len,
                    'dst_bytes': 0,
                    'src_ttl': ttl,
                    'dst_ttl': 0,
                    'src_win': tcp_win,
                    'dst_win': 0,
                    'src_rst': is_rst,
                    'dst_rst': 0,
                    'src_fin': is_fin,
                    'dst_fin': 0,
                    'src_syn': is_syn,
                    'syn_time': ts if (tcp_flags == 0x02) else None,
                    'synack_time': None,
                    'ack_time': None,
                    'src_iat': [],
                    'last_src_ts': ts,
                    'trans_depth': 1 if any(raw_payload.startswith(m) for m in http_methods) else 0,
                    'response_body_len': 0,
                    'is_ftp_login': 1 if (dport == 21 or sport == 21) and b'230' in raw_payload else 0
                }
            else:
                fl = flows[flow_key]
                fl['last_time'] = ts

                if src_ip == fl['src_ip'] and sport == fl['src_port']:
                    # Forward direction
                    fl['src_packets'] += 1
                    fl['src_bytes'] += pkt_len
                    fl['src_rst'] += is_rst
                    fl['src_fin'] += is_fin
                    fl['src_iat'].append(ts - fl['last_src_ts'])
                    fl['last_src_ts'] = ts
                    if fl['src_win'] == 0:
                        fl['src_win'] = tcp_win

                    if tcp_flags == 0x02 and fl['syn_time'] is None:
                        fl['syn_time'] = ts
                    if (tcp_flags & 0x10) and fl['syn_time'] and fl['ack_time'] is None:
                        fl['ack_time'] = ts

                    if any(raw_payload.startswith(m) for m in http_methods):
                        fl['trans_depth'] += 1

                else:
                    # Backward direction
                    fl['dst_packets'] += 1
                    fl['dst_bytes'] += pkt_len
                    fl['dst_rst'] += is_rst
                    fl['dst_fin'] += is_fin
                    if fl['dst_ttl'] == 0:
                        fl['dst_ttl'] = ttl
                    if fl['dst_win'] == 0:
                        fl['dst_win'] = tcp_win

                    if tcp_flags == 0x12 and fl['synack_time'] is None:
                        fl['synack_time'] = ts

                    if raw_payload.startswith(b'HTTP/'):
                        try:
                            headers = raw_payload.split(b'\r\n\r\n', 1)
                            for line in headers[0].split(b'\r\n'):
                                if line.lower().startswith(b'content-length:'):
                                    fl['response_body_len'] += int(line.split(b':')[1].strip())
                        except Exception:
                            pass

                if (dport == 21 or sport == 21) and b'230' in raw_payload:
                    fl['is_ftp_login'] = 1

    records = []
    for fl in flows.values():
        dur = max(0.0, fl['last_time'] - fl['start_time'])
        spkts = fl['src_packets']
        dpkts = fl['dst_packets']
        sbytes = fl['src_bytes']
        dbytes = fl['dst_bytes']

        sload = (sbytes * 8.0) / dur if dur > 0 else 0.0
        dload = (dbytes * 8.0) / dur if dur > 0 else 0.0
        sinpkt = float(np.mean(fl['src_iat'])) if fl['src_iat'] else 0.0
        smean = sbytes / spkts if spkts > 0 else 0.0
        dmean = dbytes / dpkts if dpkts > 0 else 0.0

        # TCP RTT
        t1, t2, t3 = fl['syn_time'], fl['synack_time'], fl['ack_time']
        synack = round(t2 - t1, 6) if (t1 and t2) else 0.0
        tcprtt = round(t3 - t1, 6) if (t1 and t3) else 0.0

        # Protocol mapping
        proto_str = PROTOCOL_MAP.get(fl['protocol'], str(fl['protocol']))

        # Service mapping
        service = PORT_TO_SERVICE.get(fl['dst_port']) or PORT_TO_SERVICE.get(fl['src_port']) or '-'

        # State determination
        if fl['protocol'] == 6:
            if fl['src_rst'] > 0 or fl['dst_rst'] > 0:
                state = 'RST'
            elif fl['src_fin'] > 0 and fl['dst_fin'] > 0:
                state = 'FIN'
            elif fl['src_syn'] > 0:
                state = 'INT'
            else:
                state = 'CON'
        elif fl['protocol'] == 17:
            state = 'CON' if (spkts > 0 and dpkts > 0) else 'REQ'
        elif fl['protocol'] in (1, 58):
            state = 'ECO'
        else:
            state = 'URN'

        is_sm = 1 if (fl['src_ip'] == fl['dst_ip'] or fl['src_port'] == fl['dst_port']) else 0

        records.append({
            'src_ip': fl['src_ip'],
            'dst_ip': fl['dst_ip'],
            'src_port': fl['src_port'],
            'dst_port': fl['dst_port'],
            'protocol': fl['protocol'],
            'dur': dur,
            'proto': proto_str,
            'service': service,
            'state': state,
            'spkts': spkts,
            'dpkts': dpkts,
            'sbytes': sbytes,
            'dbytes': dbytes,
            'sttl': fl['src_ttl'],
            'dttl': fl['dst_ttl'],
            'sload': sload,
            'dload': dload,
            'sloss': fl['src_rst'],
            'dloss': fl['dst_rst'],
            'sinpkt': sinpkt,
            'swin': fl['src_win'],
            'dwin': fl['dst_win'],
            'tcprtt': tcprtt,
            'synack': synack,
            'smean': smean,
            'dmean': dmean,
            'trans_depth': fl['trans_depth'],
            'response_body_len': fl['response_body_len'],
            'is_ftp_login': fl['is_ftp_login'],
            'is_sm_ips_ports': is_sm,
            'start_time': fl['start_time']
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values('start_time').reset_index(drop=True)
    return df


def compute_sliding_window_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Computes connection count features (ct_*) across a rolling window of 100 flows."""
    window = deque(maxlen=100)

    ct_srv_src = []
    ct_srv_dst = []
    ct_dst_src_ltm = []
    ct_src_dport_ltm = []
    ct_dst_sport_ltm = []
    ct_state_ttl = []
    ct_flw_http_mthd = []

    for _, row in df.iterrows():
        src_ip = row['src_ip']
        dst_ip = row['dst_ip']
        src_port = row['src_port']
        dst_port = row['dst_port']
        service = row['service']
        state = row['state']
        sttl = row['sttl']

        ct_srv_src.append(sum(1 for f in window if f['src_ip'] == src_ip and f['service'] == service))
        ct_srv_dst.append(sum(1 for f in window if f['dst_ip'] == dst_ip and f['service'] == service))
        ct_dst_src_ltm.append(sum(1 for f in window if f['src_ip'] == src_ip and f['dst_ip'] == dst_ip))
        ct_src_dport_ltm.append(sum(1 for f in window if f['src_ip'] == src_ip and f['dst_port'] == dst_port))
        ct_dst_sport_ltm.append(sum(1 for f in window if f['dst_ip'] == dst_ip and f['src_port'] == src_port))
        ct_state_ttl.append(sum(1 for f in window if f['state'] == state and f['sttl'] == sttl))
        ct_flw_http_mthd.append(sum(1 for f in window if f.get('had_http', False)))

        had_http = row.get('trans_depth', 0) > 0
        window.append({
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'service': service,
            'state': state,
            'sttl': sttl,
            'had_http': had_http
        })

    df['ct_srv_src'] = ct_srv_src
    df['ct_srv_dst'] = ct_srv_dst
    df['ct_dst_src_ltm'] = ct_dst_src_ltm
    df['ct_src_dport_ltm'] = ct_src_dport_ltm
    df['ct_dst_sport_ltm'] = ct_dst_sport_ltm
    df['ct_state_ttl'] = ct_state_ttl
    df['ct_flw_http_mthd'] = ct_flw_http_mthd

    return df


def extract_all_features(pcap_path: str) -> pd.DataFrame:
    """
    Main extraction pipeline:
    Parses a .pcap/.pcapng file and extracts all 32 UNSW-NB15 flow features,
    attaching flow metadata for visual display.
    """
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    # Flow aggregation (pure Scapy engine ensures zero DLL dependencies on Windows)
    flow_df = _scapy_aggregate_flows(pcap_path)

    if flow_df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ['src_ip', 'dst_ip', 'src_port', 'dst_port'])

    # Compute sliding window connection statistics
    full_df = compute_sliding_window_counts(flow_df)

    # Clean numerical edge cases (NaN / Inf)
    for col in FEATURE_COLUMNS:
        if col in full_df.columns:
            full_df[col] = full_df[col].fillna(0).replace([np.inf, -np.inf], 0)

    return full_df
