import logging
import os
import random
import time
from pathlib import Path

# Silence harmless Scapy runtime warnings
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from scapy.all import AsyncSniffer, wrpcap, IP, TCP, UDP, Raw, conf


def is_pcap_installed() -> bool:
    """Returns True if Npcap or WinPcap driver is available on the machine."""
    return bool(getattr(conf, "use_pcap", False))


def get_available_interfaces() -> dict:
    """
    Returns a dictionary of available network interfaces with friendly names and IPs.
    Example: {'Wi-Fi (192.168.1.16)': 'Wi-Fi', 'Ethernet': 'Ethernet'}
    """
    interfaces = {}
    try:
        from scapy.arch.windows import get_windows_if_list
        win_ifs = get_windows_if_list()
        for iface in win_ifs:
            name = iface.get('name', '')
            ips = [ip for ip in iface.get('ips', []) if ':' not in ip and not ip.startswith('169.254')]
            desc = iface.get('description', '')
            if name:
                ip_str = f" ({ips[0]})" if ips else ""
                label = f"{name}{ip_str} — {desc[:30]}"
                interfaces[label] = name
    except Exception:
        try:
            from scapy.all import get_if_list
            for iface in get_if_list():
                interfaces[iface] = iface
        except Exception:
            pass

    if not interfaces:
        interfaces["Wi-Fi"] = "Wi-Fi"
        interfaces["Ethernet"] = "Ethernet"

    return interfaces


class LiveCaptureSession:
    """Manages an asynchronous network packet sniffing session using Scapy AsyncSniffer."""
    def __init__(self, interface: str = "Wi-Fi"):
        self.interface = interface
        self.sniffer = None
        self.is_capturing = False
        self.start_time = None
        self.packets = []

    def start(self, filter_bpf: str = "ip"):
        """Starts asynchronous packet capture on the selected network adapter."""
        # Check Windows Npcap and Administrator privileges
        if os.name == 'nt' and not is_pcap_installed():
            try:
                import ctypes
                is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            except Exception:
                is_admin = False

            if not is_admin:
                self.is_capturing = False
                return False, (
                    "Npcap packet capture driver is not installed, and PowerShell is not running as Administrator. "
                    "The Windows kernel blocks physical network card sniffing without Npcap. "
                    "Please install Npcap (https://npcap.com/#download) or launch terminal as Administrator. "
                    "Or switch to 'Live Traffic Stream (Instant Demo)' for immediate testing."
                )

        try:
            kwargs = {'iface': self.interface, 'store': True}
            if is_pcap_installed():
                if filter_bpf:
                    kwargs['filter'] = filter_bpf
            else:
                kwargs['L2socket'] = conf.L3socket

            self.sniffer = AsyncSniffer(**kwargs)
            self.sniffer.start()

            # Short validation to detect immediate thread failure
            time.sleep(0.08)
            if hasattr(self.sniffer, 'thread') and self.sniffer.thread and not self.sniffer.thread.is_alive():
                self.is_capturing = False
                return False, "Sniffer thread exited immediately. Windows blocked raw socket binding."

            self.is_capturing = True
            self.start_time = time.time()
            return True, "Sniffer started successfully."
        except Exception as exc:
            self.is_capturing = False
            return False, str(exc)

    def stop(self) -> list:
        """Stops the active sniffer and returns captured packet list."""
        if self.sniffer is not None and self.is_capturing:
            try:
                self.sniffer.stop()
                self.packets = list(self.sniffer.results) if self.sniffer.results else []
            except Exception:
                self.packets = []
            finally:
                self.is_capturing = False
        return self.packets

    def save_pcap(self, output_path: str) -> bool:
        """Writes captured packets to a .pcap file."""
        if self.packets:
            wrpcap(output_path, self.packets)
            return True
        return False


def simulate_live_traffic_capture(duration_sec: int = 5, attack_ratio: float = 0.3) -> list:
    """
    Fallback traffic generator: synthesizes a live stream of realistic network flows
    when running on Windows without Administrator raw socket permissions or Npcap drivers.
    """
    packets = []
    base_t = time.time()

    # Normal user web requests
    for i in range(25):
        s_ip = "192.168.1.16"
        d_ip = f"142.250.190.{i + 20}"
        sp = 50000 + i
        dp = 443

        p_syn = IP(src=s_ip, dst=d_ip, ttl=64) / TCP(sport=sp, dport=dp, flags="S", seq=100)
        p_syn.time = base_t + (i * 0.1)
        p_ack = IP(src=s_ip, dst=d_ip, ttl=64) / TCP(sport=sp, dport=dp, flags="A", seq=101, ack=201)
        p_ack.time = base_t + (i * 0.1) + 0.02
        packets.extend([p_syn, p_ack])

    # If attack simulation is requested (SYN flood / port scan burst)
    if attack_ratio > 0:
        attacker_ip = "185.220.101.5"
        victim_ip = "192.168.1.16"
        for j in range(40):
            probe = IP(src=attacker_ip, dst=victim_ip, ttl=254) / TCP(sport=40000 + j, dport=80, flags="S", seq=j * 10, window=0)
            probe.time = base_t + (j * 0.005)
            packets.append(probe)

    return packets
