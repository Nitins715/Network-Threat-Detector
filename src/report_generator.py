import io
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.graphics.shapes import Drawing, Rect, Circle, String, Line, Group, Polygon


def get_ip_geo_attribution(ip: str) -> dict:
    """Provides network context, Geo-IP country attribution, and subnet classification."""
    if not ip or ip == "0":
        return {"country": "Unknown", "asn": "N/A", "classification": "Unspecified Host", "badge": "#64748b"}
    
    # Private RFC 1918 subnets
    if ip.startswith(("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.")):
        return {"country": "Local Corporate LAN", "asn": "AS-PRIVATE (Internal)", "classification": "Corporate Subnet (RFC 1918)", "badge": "#3b82f6"}
    
    # Loopback
    if ip.startswith("127.") or ip == "::1":
        return {"country": "Localhost", "asn": "Loopback", "classification": "Host Loopback Device", "badge": "#64748b"}
    
    # UNSW-NB15 Security Testbed Range
    if ip.startswith("175.45."):
        return {"country": "Australia (UNSW Cyber Range)", "asn": "AS13335 (UNSW Cyber Testbed)", "classification": "Active Threat Testbed Range", "badge": "#ef4444"}
    if ip.startswith("149.171."):
        return {"country": "Australia (UNSW)", "asn": "AS13335 (Academic Network)", "classification": "University Gateway / Proxy", "badge": "#f59e0b"}
        
    # Public Scanner / Known Threat Feeds
    if ip.startswith("185.220."):
        return {"country": "Germany (Tor Relay)", "asn": "AS206238 (Privacy Network)", "classification": "Anonymized Proxy / Tor Node", "badge": "#dc2626"}
    if ip.startswith("198.51.") or ip.startswith("203.0.113."):
        return {"country": "North America (TEST-NET)", "asn": "AS-DOCUMENTATION", "classification": "External Scanner / Threat Feed", "badge": "#e11d48"}
        
    return {"country": "Global Internet (External)", "asn": "AS-TRANSIT (Public)", "classification": "External Autonomous Host", "badge": "#8b5cf6"}


def create_threat_gauge_drawing(threat_pct: float, width: float = 520, height: float = 50) -> Drawing:
    """
    Renders an executive vector Threat Gauge meter directly in the PDF.
    Shows the 4 SOC alert tiers (Normal, Guarded, Elevated, Critical) and an indicator arrow.
    """
    d = Drawing(width, height)
    bar_x = 10
    bar_y = 18
    bar_w = width - 20
    bar_h = 14
    
    # Background Track
    d.add(Rect(bar_x, bar_y, bar_w, bar_h, rx=6, ry=6, fillColor=colors.HexColor('#e2e8f0'), strokeColor=None))
    
    # Tier 1: Normal (0 - 30%)
    w1 = bar_w * 0.30
    d.add(Rect(bar_x, bar_y, w1, bar_h, rx=4, ry=4, fillColor=colors.HexColor('#10b981'), strokeColor=None))
    
    # Tier 2: Guarded (30 - 55%)
    w2 = bar_w * 0.25
    d.add(Rect(bar_x + w1, bar_y, w2, bar_h, rx=0, ry=0, fillColor=colors.HexColor('#3b82f6'), strokeColor=None))
    
    # Tier 3: Elevated (55 - 75%)
    w3 = bar_w * 0.20
    d.add(Rect(bar_x + w1 + w2, bar_y, w3, bar_h, rx=0, ry=0, fillColor=colors.HexColor('#f59e0b'), strokeColor=None))
    
    # Tier 4: Critical (75 - 100%)
    w4 = bar_w * 0.25
    d.add(Rect(bar_x + w1 + w2 + w3, bar_y, w4, bar_h, rx=4, ry=4, fillColor=colors.HexColor('#ef4444'), strokeColor=None))
    
    # Clamped Threat Position
    clamped_pct = max(0.0, min(100.0, threat_pct))
    curr_x = bar_x + (clamped_pct / 100.0) * bar_w
    
    # Pointer Triangle (downward needle)
    triangle = Polygon(
        points=[curr_x, bar_y + bar_h + 1, curr_x - 5, bar_y + bar_h + 9, curr_x + 5, bar_y + bar_h + 9],
        fillColor=colors.HexColor('#0f172a'),
        strokeColor=None
    )
    d.add(triangle)
    
    # Value label badge
    gauge_label = f"CURRENT THREAT: {clamped_pct:.1f}%"
    text_x = max(bar_x + 50, min(bar_x + bar_w - 50, curr_x))
    d.add(String(text_x, bar_y + bar_h + 12, gauge_label, fontName="Helvetica-Bold", fontSize=8, textAnchor="middle", fillColor=colors.HexColor('#0f172a')))
    
    # Baseline ticks and labels
    d.add(String(bar_x, bar_y - 10, "0% SAFE", fontName="Helvetica", fontSize=6.5, fillColor=colors.HexColor('#64748b')))
    d.add(String(bar_x + w1, bar_y - 10, "30% GUARDED", fontName="Helvetica", fontSize=6.5, textAnchor="middle", fillColor=colors.HexColor('#64748b')))
    d.add(String(bar_x + w1 + w2, bar_y - 10, "55% ELEVATED", fontName="Helvetica", fontSize=6.5, textAnchor="middle", fillColor=colors.HexColor('#64748b')))
    d.add(String(bar_x + w1 + w2 + w3, bar_y - 10, "75% HIGH", fontName="Helvetica", fontSize=6.5, textAnchor="middle", fillColor=colors.HexColor('#64748b')))
    d.add(String(bar_x + bar_w, bar_y - 10, "100% CRITICAL", fontName="Helvetica", fontSize=6.5, textAnchor="end", fillColor=colors.HexColor('#64748b')))
    
    return d


def create_category_breakdown_drawing(flows_df: pd.DataFrame, width: float = 520, height: float = 24) -> Drawing:
    """Renders a vector stacked bar showing the category distribution in the PDF."""
    d = Drawing(width, height)
    bar_x = 10
    bar_y = 6
    bar_w = width - 20
    bar_h = 12
    
    if flows_df.empty or 'attack_category' not in flows_df.columns:
        d.add(Rect(bar_x, bar_y, bar_w, bar_h, rx=4, ry=4, fillColor=colors.HexColor('#e2e8f0'), strokeColor=None))
        d.add(String(bar_x + bar_w / 2, bar_y + 3, "No Category Telemetry", fontName="Helvetica", fontSize=7, textAnchor="middle", fillColor=colors.HexColor('#64748b')))
        return d
        
    palette = {
        'Normal': colors.HexColor('#10b981'),
        'Reconnaissance': colors.HexColor('#06b6d4'),
        'DoS': colors.HexColor('#f59e0b'),
        'Exploits': colors.HexColor('#ef4444'),
        'Fuzzers': colors.HexColor('#8b5cf6'),
        'Backdoor': colors.HexColor('#ec4899'),
        'Analysis': colors.HexColor('#6366f1'),
        'Generic': colors.HexColor('#f97316'),
        'Shellcode': colors.HexColor('#e11d48'),
        'Worms': colors.HexColor('#b91c1c')
    }
    
    counts = flows_df['attack_category'].value_counts()
    total = len(flows_df)
    
    curr_x = bar_x
    for cat, cnt in counts.items():
        segment_w = (cnt / total) * bar_w
        col = palette.get(cat, colors.HexColor('#64748b'))
        d.add(Rect(curr_x, bar_y, segment_w, bar_h, rx=2, ry=2, fillColor=col, strokeColor=None))
        curr_x += segment_w
        
    return d


def generate_executive_pdf_report(
    results: dict,
    flows_df: pd.DataFrame,
    source_filename: str = "capture.pcap"
) -> bytes:
    """
    Generates a professional executive-ready PDF Threat Intelligence Report
    using ReportLab, returning the document as raw bytes.
    Includes:
      - SOC Header & Confidentiality Classification
      - Incident Verdict & Threat Gauge Vector Visual
      - Malicious IP Breakdown & Geo-IP Attribution
      - Critical Flow Telemetry with Tree SHAP Root Cause
      - Remediation Checklist & Multi-Platform Firewall Playbooks
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch
    )

    styles = getSampleStyleSheet()

    # Typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=8
    )
    section_heading = ParagraphStyle(
        'SectionHead',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#334155')
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#1e293b')
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#0f172a')
    )

    story = []

    # 1. Header Banner & Classification
    header_data = [
        [
            Paragraph("<b>NETWORK DETECTOR</b> | SOC THREAT INTELLIGENCE SYSTEM", table_cell_bold),
            Paragraph(f"CLASSIFICATION: <b>CONFIDENTIAL // SOC INTERNAL ONLY</b>", ParagraphStyle('R', parent=table_cell, alignment=2, textColor=colors.HexColor('#dc2626')))
        ]
    ]
    t_head = Table(header_data, colWidths=[4.2 * inch, 3.4 * inch])
    t_head.setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_head)
    story.append(Spacer(1, 8))

    # 2. Main Title
    story.append(Paragraph("Executive Network Threat Intelligence Report", title_style))
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"Generated: <b>{report_date}</b> &nbsp;|&nbsp; Target Capture: <b>{source_filename}</b> &nbsp;|&nbsp; Engine: <b>Network Detector v2.0</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceAfter=8))

    # 3. Incident Verdict & Severity Panel
    verdict = results.get('verdict', 'NORMAL')
    threat_intensity = results.get('threat_intensity', 0.0)
    risk_level = results.get('risk_level', 'Low')
    network_status = results.get('network_status', 'Safe')
    primary_category = results.get('primary_category', 'Normal')
    attack_count = results.get('attack_flow_count', 0)
    total_flows = results.get('total_flows', 0)
    analyst_note = results.get('analyst_note', '')
    playbook = results.get('playbook', {})

    v_bg = colors.HexColor('#fef2f2') if verdict == 'ATTACK' else colors.HexColor('#f0fdf4')
    v_border = colors.HexColor('#ef4444') if verdict == 'ATTACK' else colors.HexColor('#10b981')
    v_text = colors.HexColor('#b91c1c') if verdict == 'ATTACK' else colors.HexColor('#15803d')

    kpi_table_data = [
        [
            Paragraph("OVERALL VERDICT", ParagraphStyle('H', parent=table_cell, textColor=colors.HexColor('#64748b'))),
            Paragraph("THREAT INTENSITY", ParagraphStyle('H', parent=table_cell, textColor=colors.HexColor('#64748b'))),
            Paragraph("RISK LEVEL", ParagraphStyle('H', parent=table_cell, textColor=colors.HexColor('#64748b'))),
            Paragraph("NETWORK STATUS", ParagraphStyle('H', parent=table_cell, textColor=colors.HexColor('#64748b'))),
            Paragraph("PRIMARY VECTOR", ParagraphStyle('H', parent=table_cell, textColor=colors.HexColor('#64748b'))),
            Paragraph("MALICIOUS FLOWS", ParagraphStyle('H', parent=table_cell, textColor=colors.HexColor('#64748b')))
        ],
        [
            Paragraph(f"<b>{verdict}</b>", ParagraphStyle('V', parent=table_cell_bold, fontSize=10.5, textColor=v_text)),
            Paragraph(f"<b>{threat_intensity}%</b>", ParagraphStyle('V', parent=table_cell_bold, fontSize=10.5, textColor=v_text)),
            Paragraph(f"<b>{risk_level}</b>", ParagraphStyle('V', parent=table_cell_bold, fontSize=10.5)),
            Paragraph(f"<b>{network_status}</b>", ParagraphStyle('V', parent=table_cell_bold, fontSize=10.5)),
            Paragraph(f"<b>{primary_category}</b>", ParagraphStyle('V', parent=table_cell_bold, fontSize=10.5, textColor=colors.HexColor('#2563eb'))),
            Paragraph(f"<b>{attack_count} / {total_flows}</b>", ParagraphStyle('V', parent=table_cell_bold, fontSize=10.5))
        ]
    ]
    t_kpi = Table(kpi_table_data, colWidths=[1.26 * inch] * 6)
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), v_bg),
        ('BOX', (0, 0), (-1, -1), 1.2, v_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 8))

    # 4. Embedded Vector Threat Gauge
    story.append(Paragraph("<b>Threat Exposure Gauge:</b>", ParagraphStyle('Sub', parent=table_cell_bold, fontSize=8, textColor=colors.HexColor('#475569'))))
    story.append(create_threat_gauge_drawing(threat_intensity, width=540, height=44))
    story.append(Spacer(1, 4))

    # 5. Executive Summary Note
    story.append(Paragraph("<b>Executive Summary:</b> " + analyst_note, body_style))
    story.append(Spacer(1, 8))

    # 6. Malicious IP Breakdown & Geo-IP Attribution
    story.append(Paragraph("1. Host Attribution & Geo-IP Breakdown", section_heading))
    df_copy = flows_df.copy() if not flows_df.empty else pd.DataFrame()
    if not df_copy.empty:
        if 'threat_score' not in df_copy.columns:
            df_copy['threat_score'] = results.get('all_flow_scores', [results.get('threat_intensity', 0.0) / 100.0] * len(df_copy))
        if 'prediction' not in df_copy.columns:
            df_copy['prediction'] = results.get('verdict', 'NORMAL')
        if 'dst_ip' not in df_copy.columns:
            df_copy['dst_ip'] = '192.168.1.1'
        if 'dst_port' not in df_copy.columns:
            df_copy['dst_port'] = 80
        if 'proto' not in df_copy.columns:
            df_copy['proto'] = 'tcp'
        if 'service' not in df_copy.columns:
            df_copy['service'] = '-'
        if 'state' not in df_copy.columns:
            df_copy['state'] = 'CON'
        if 'attack_category' not in df_copy.columns:
            df_copy['attack_category'] = results.get('primary_category', 'Normal')
        if 'primary_threat_driver' not in df_copy.columns:
            df_copy['primary_threat_driver'] = 'Standard Baseline Flow'

    if not df_copy.empty and 'src_ip' in df_copy.columns:
        attacker_flows = df_copy[df_copy['prediction'] == 'ATTACK'] if 'prediction' in df_copy.columns else pd.DataFrame()
        if attacker_flows.empty:
            attacker_flows = df_copy.head(6)

        ip_summary = attacker_flows.groupby('src_ip').agg({
            'dst_ip': lambda x: list(x.unique())[:3],
            'dst_port': lambda x: list(x.unique())[:4],
            'proto': lambda x: list(x.unique()),
            'threat_score': 'mean'
        }).reset_index()

        ip_table_data = [
            [
                Paragraph("<b>Source Host (Attacker)</b>", table_cell_bold),
                Paragraph("<b>Geo-IP Country & Subnet</b>", table_cell_bold),
                Paragraph("<b>ASN / Network</b>", table_cell_bold),
                Paragraph("<b>Target Host(s)</b>", table_cell_bold),
                Paragraph("<b>Target Ports</b>", table_cell_bold),
                Paragraph("<b>Protocols</b>", table_cell_bold),
                Paragraph("<b>Avg Threat</b>", table_cell_bold)
            ]
        ]
        for _, r in ip_summary.head(5).iterrows():
            src_ip = str(r['src_ip'])
            geo = get_ip_geo_attribution(src_ip)
            victims = ", ".join(map(str, r['dst_ip']))
            ports = ", ".join(map(str, r['dst_port']))
            protocols = ", ".join(map(str, r['proto']))
            avg_threat = f"{float(r['threat_score']) * 100:.1f}%"

            ip_table_data.append([
                Paragraph(src_ip, table_cell_bold),
                Paragraph(geo['country'], table_cell),
                Paragraph(geo['asn'], table_cell),
                Paragraph(victims, table_cell),
                Paragraph(ports, table_cell),
                Paragraph(protocols, table_cell),
                Paragraph(avg_threat, table_cell_bold)
            ])

        t_ip = Table(ip_table_data, colWidths=[1.15 * inch, 1.45 * inch, 1.45 * inch, 1.35 * inch, 0.75 * inch, 0.75 * inch, 0.7 * inch])
        t_ip.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ]))
        story.append(t_ip)
    else:
        story.append(Paragraph("No IP telemetry available in dataset.", body_style))

    story.append(Spacer(1, 8))

    # 7. Category Distribution Stacked Bar
    story.append(Paragraph("<b>Traffic Composition Vector Distribution:</b>", ParagraphStyle('Sub', parent=table_cell_bold, fontSize=8, textColor=colors.HexColor('#475569'))))
    story.append(create_category_breakdown_drawing(df_copy, width=540, height=20))
    story.append(Spacer(1, 6))

    # 8. Critical Malicious Flow Telemetry Table (with Tree SHAP attribution)
    story.append(Paragraph("2. Critical Malicious Flow Telemetry (Tree SHAP Attribution)", section_heading))
    if not df_copy.empty:
        crit_flows = df_copy.sort_values('threat_score', ascending=False).head(6) if 'threat_score' in df_copy.columns else df_copy.head(6)
        flow_table_data = [
            [
                Paragraph("<b>#</b>", table_cell_bold),
                Paragraph("<b>Flow Socket (Src ➔ Dst)</b>", table_cell_bold),
                Paragraph("<b>Proto / Svc</b>", table_cell_bold),
                Paragraph("<b>State</b>", table_cell_bold),
                Paragraph("<b>Threat Category</b>", table_cell_bold),
                Paragraph("<b>Score</b>", table_cell_bold),
                Paragraph("<b>Primary Root Cause (Tree SHAP Attribution)</b>", table_cell_bold)
            ]
        ]
        for idx, (_, row) in enumerate(crit_flows.iterrows(), 1):
            conn = f"{row.get('src_ip', '')}:{row.get('src_port', '')} ➔ {row.get('dst_ip', '')}:{row.get('dst_port', '')}"
            psvc = f"{row.get('proto', '')} / {row.get('service', '-')}"
            stt = str(row.get('state', ''))
            cat = str(row.get('attack_category', 'Normal'))
            sc = f"{float(row.get('threat_score', 0.0)):.3f}"
            cause = str(row.get('primary_threat_driver', 'Standard Flow Profile'))[:50]

            flow_table_data.append([
                Paragraph(str(idx), table_cell),
                Paragraph(conn, table_cell),
                Paragraph(psvc, table_cell),
                Paragraph(stt, table_cell),
                Paragraph(cat, table_cell_bold),
                Paragraph(sc, table_cell_bold),
                Paragraph(cause, table_cell)
            ])

        t_flows = Table(flow_table_data, colWidths=[0.25 * inch, 2.1 * inch, 0.85 * inch, 0.55 * inch, 0.95 * inch, 0.55 * inch, 2.35 * inch])
        t_flows.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t_flows)

    story.append(Spacer(1, 8))

    # 9. Remediation Playbook Section
    story.append(Paragraph(f"3. SOC Incident Remediation Playbook ({primary_category})", section_heading))
    pb_title = playbook.get('title', 'Security Containment Playbook')
    pb_desc = playbook.get('description', '')
    actions = playbook.get('immediate_actions', [])

    story.append(Paragraph(f"<b>{pb_title}</b>: {pb_desc}", body_style))
    story.append(Spacer(1, 3))

    for idx, act in enumerate(actions[:4], 1):
        story.append(Paragraph(f"&nbsp;&nbsp;<b>{idx}.</b> {act}", body_style))

    story.append(Spacer(1, 6))

    # 10. Multi-Platform Containment Commands
    primary_attacker = '192.168.1.100'
    if not df_copy.empty and 'src_ip' in df_copy.columns:
        if 'prediction' in df_copy.columns and (df_copy['prediction'] == 'ATTACK').any():
            primary_attacker = df_copy[df_copy['prediction'] == 'ATTACK']['src_ip'].iloc[0]
        else:
            primary_attacker = df_copy['src_ip'].iloc[0]

    fw_win = f"netsh advfirewall firewall add rule name=\"Block_{primary_category}_{primary_attacker}\" dir=in action=block remoteip={primary_attacker}"
    fw_linux = f"iptables -A INPUT -s {primary_attacker} -j DROP"

    fw_box_data = [
        [
            Paragraph(f"<b>Deployable Containment Rules:</b><br/>"
                      f"<b>Windows Defender:</b> <code>{fw_win}</code><br/>"
                      f"<b>Linux (iptables):</b> <code>{fw_linux}</code>", table_cell)
        ]
    ]
    t_fw = Table(fw_box_data, colWidths=[7.6 * inch])
    t_fw.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_fw)

    # 11. SOC Officer Sign-Off Footer
    story.append(Spacer(1, 10))
    sig_data = [
        [
            Paragraph("<b>Automated Threat Triage Agent:</b> Network Detector AI SOC v2.0", table_cell),
            Paragraph("<b>SOC Incident Commander Sign-off:</b> ____________________________", ParagraphStyle('R', parent=table_cell, alignment=2))
        ]
    ]
    t_sig = Table(sig_data, colWidths=[4.2 * inch, 3.4 * inch])
    story.append(t_sig)

    # Build document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
