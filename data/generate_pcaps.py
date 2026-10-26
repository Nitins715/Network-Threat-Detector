import time
from pathlib import Path
from scapy.all import IP, TCP, UDP, Raw, wrpcap

out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

# 1. Generate Benign Web & DNS Traffic
benign_pkts = []
base_time = time.time()

# Handshake & HTTP GET session
src_ip = "192.168.1.105"
dst_ip = "93.184.216.34"  # example.com
sport = 54321
dport = 80

# SYN -> SYN-ACK -> ACK
syn = IP(src=src_ip, dst=dst_ip, ttl=64) / TCP(sport=sport, dport=dport, flags="S", seq=1000, window=64240)
syn.time = base_time
synack = IP(src=dst_ip, dst=src_ip, ttl=56) / TCP(sport=dport, dport=sport, flags="SA", seq=5000, ack=1001, window=65535)
synack.time = base_time + 0.02
ack = IP(src=src_ip, dst=dst_ip, ttl=64) / TCP(sport=sport, dport=dport, flags="A", seq=1001, ack=5001, window=64240)
ack.time = base_time + 0.04

# HTTP GET request & Response
http_req = IP(src=src_ip, dst=dst_ip, ttl=64) / TCP(sport=sport, dport=dport, flags="PA", seq=1001, ack=5001) / Raw(load=b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n")
http_req.time = base_time + 0.05
http_resp = IP(src=dst_ip, dst=src_ip, ttl=56) / TCP(sport=dport, dport=sport, flags="PA", seq=5001, ack=1050) / Raw(load=b"HTTP/1.1 200 OK\r\nContent-Length: 45\r\n\r\n<html><body>Normal User Browsing</body></html>")
http_resp.time = base_time + 0.09

# DNS query & response
dns_q = IP(src=src_ip, dst="8.8.8.8", ttl=64) / UDP(sport=53322, dport=53) / Raw(load=b"\x12\x34\x01\x00\x00\x01\x00\x00")
dns_q.time = base_time + 0.12
dns_r = IP(src="8.8.8.8", dst=src_ip, ttl=54) / UDP(sport=53, dport=53322) / Raw(load=b"\x12\x34\x81\x80\x00\x01\x00\x01")
dns_r.time = base_time + 0.14

benign_pkts.extend([syn, synack, ack, http_req, http_resp, dns_q, dns_r])
wrpcap(str(out_dir / "benign_traffic.pcap"), benign_pkts)
print("Saved data/benign_traffic.pcap")

# 2. Generate SYN Flood / Reconnaissance Attack (UNSW-NB15 profile: ttl=254, rapid INT states)
attack_pkts = []
attacker_ip = "175.45.176.0"
target_ip = "149.171.126.16"

for i in range(120):
    probe = IP(src=attacker_ip, dst=target_ip, ttl=254) / TCP(sport=30000 + i, dport=80, flags="S", seq=i * 200, window=0)
    probe.time = base_time + 0.0005 * i
    attack_pkts.append(probe)

wrpcap(str(out_dir / "attack_synflood.pcap"), attack_pkts)
print("Saved data/attack_synflood.pcap with UNSW-NB15 attack profile")
