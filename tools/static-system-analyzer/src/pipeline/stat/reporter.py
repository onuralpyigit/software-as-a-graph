"""
Report exporter module.

This module generates markdown/PDF reports with matplotlib charts.
"""

import os
import re
import textwrap
import warnings
from typing import List, Tuple, Dict, Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from pypdf import PdfReader, PdfWriter

from pipeline.aggregator.converter import qos_risk_weight
from .analyzer import AnalysisReport, ComponentAnalysis
from .statistics import (
    extract_cross_cutting_data,
    find_1d_outliers_iqr,
    calculate_outliers,
    compute_topic_bandwidth_stats,
    compute_qos_risk_stats,
    compute_app_balance_stats,
    compute_topic_fanout_stats,
    compute_cross_node_heatmap_stats,
    compute_node_comm_load_stats,
    compute_domain_comm_stats,
    compute_criticality_io_stats,
    compute_lib_dependency_stats,
    compute_node_critical_density_stats,
    compute_domain_diversity_stats,
)
from .metric_ids import (
    NODE_APPLICATION_COUNT,
    NODE_DOMAIN_HIERARCHY_DIVERSITY_COUNT,
    APP_DIRECT_PUBLISH_COUNT,
    APP_DIRECT_SUBSCRIBE_COUNT,
    APP_HIERARCHY_COMPONENT_DISTRIBUTION,
    APP_HIERARCHY_CONFIG_ITEM_DISTRIBUTION,
    APP_HIERARCHY_DOMAIN_TOPIC_VARIETY_COUNT,
    APP_HIERARCHY_CONFIG_ITEM_TOPIC_VARIETY_COUNT,
    LIB_APPLICATION_USAGE_COUNT,
    USES_CYCLE_DISTRIBUTION,
    TOPIC_SIZE_BYTES,
    is_total_metric,
)
from common.logger import log_info
from common.runtime_config import get_runtime_config


def _sanitize_filename(name: str) -> str:
    """Sanitize metric name for use as filename."""
    return name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").lower()


def _markdown_anchor(text: str) -> str:
    """Create a deterministic markdown anchor ID for report headings."""
    normalized = text.lower()
    translation_table = str.maketrans({
        "ı": "i",
        "İ": "i",
        "ğ": "g",
        "Ğ": "g",
        "ü": "u",
        "Ü": "u",
        "ş": "s",
        "Ş": "s",
        "ö": "o",
        "Ö": "o",
        "ç": "c",
        "Ç": "c",
    })
    normalized = normalized.translate(translation_table)
    normalized = re.sub(r"[^a-z0-9\s-]", "", normalized)
    normalized = re.sub(r"\s+", "-", normalized.strip())
    normalized = re.sub(r"-+", "-", normalized)
    return normalized or "section"


def _build_report_anchor(*parts: str) -> str:
    """Build a stable anchor independent from visible heading text."""
    return _markdown_anchor("-".join(str(part) for part in parts if part))


def _localize_report_text(text: str) -> str:
    """Normalize visible report text."""
    localized = str(text)
    replacements = [
        ("Topic ", "Topic"),
        ("System Hierarchy", "System Hierarchy"),
        ("Transport Priority", "Transport Priority"),
        ("Durability", "Durability"),
        ("Reliability", "Reliability"),
        ("lib uses", "lib uses"),
        ("cycle", "cycle"),
    ]
    for source, target in replacements:
        localized = localized.replace(source, target)
    return localized


def _display_metric_label(analysis: ComponentAnalysis) -> str:
    """Return the localized label used in rendered reports."""
    return _localize_report_text(analysis.description or analysis.metric_name)


def _format_name_with_version(name: str, version: str = "") -> str:
    """Format name with version in parentheses if version exists and is not 'NOT_FOUND'."""
    if version and version != "NOT_FOUND":
        return f"{name} ({version})"
    return str(name)


def _safe_tight_layout(fig: plt.Figure, rect: List[float] = None) -> None:
    """Apply tight layout without surfacing benign matplotlib layout warnings."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Tight layout not applied.*",
                category=UserWarning,
            )
            if rect is None:
                fig.tight_layout()
            else:
                fig.tight_layout(rect=rect)
    except Exception:
        # Keep figure generation resilient even when matplotlib cannot optimize layout.
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.97)




# ======================== CROSS-CUTTING (EXTRAS) CHARTS ========================


def _make_top_text_page(lines: List[str], title: str = "") -> plt.Figure:
    """Create a standalone figure containing only the top-N text list."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    if title:
        ax.text(0.5, 0.95, title, fontsize=14, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes)
    txt = "\n".join(lines)
    ax.text(0.05, 0.85, txt, fontsize=10, fontfamily='monospace',
            verticalalignment='top', transform=ax.transAxes)
    return fig


def _chart_topic_size_vs_subscribers(st: Dict[str, Any]) -> Tuple[plt.Figure, List[str]]:
    """Chart 1: Topic Size vs. Subscriber Count (scatter) — bandwidth hotspot detection."""
    sizes = st["sizes"]
    subs = st["subs"]
    labels = st["labels"]
    bandwidth = st["bandwidth"]
    outlier_idx = st["outlier_indices"]
    iqr_upper = st["iqr_upper"]
    iqr = st["iqr"]
    sm = st["summary"]

    top_lines: List[str] = []
    fig, ax = plt.subplots(figsize=(10, 9))
    if sizes:
        ax.scatter(sizes, subs, c='#e74c3c', alpha=0.6, edgecolors='black', s=60)
        if outlier_idx:
            ox = [sizes[i] for i in outlier_idx]
            oy = [subs[i] for i in outlier_idx]
            ax.scatter(ox, oy, facecolors='none', edgecolors='#c0392b',
                       s=200, linewidths=2, zorder=5, label='High Bandwidth Risk')
            ax.legend(fontsize=9, loc='upper right')
            top_lines = [f"Aykırı Değerler — High Bandwidth Risk (Size×Subscriber), IQR üst sınır: {iqr_upper:,.0f}:"]
            for rank, i in enumerate(outlier_idx, 1):
                dev = (bandwidth[i] - iqr_upper) / iqr if iqr > 0 else 0.0
                top_lines.append(f"  {rank:>2}. {labels[i][:45]:<45s}  Size={sizes[i]:>8,}  Subscriber={subs[i]}  Load={bandwidth[i]:,}  Sapma={dev:.1f}×IQR")
    ax.set_xlabel('Topic Size (bytes)', fontsize=11)
    ax.set_ylabel('Subscriber Count', fontsize=11)
    ax.set_title('Topic Size vs Subscriber Count\n(Bandwidth Hotspot Detection)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Interpretive statistics from pre-computed summary
    if sm:
        top_lines.append("")
        top_lines.append("─── Özet İstatistikler ───")
        top_lines.append(f"  Toplam Topic: {sm['total_topics']}")
        top_lines.append(f"  Size — Ort: {sm['size_mean']:,.0f} B, Medyan: {sm['size_median']:,.0f} B, Maks: {sm['size_max']:,.0f} B")
        top_lines.append(f"  Subscriber — Ort: {sm['sub_mean']:.1f}, Medyan: {sm['sub_median']:.0f}, Maks: {sm['sub_max']}")
        if sm['zero_sub_count'] > 0:
            top_lines.append(f"  Subscriber'ı olmayan topic: {sm['zero_sub_count']} ({sm['zero_sub_count']/sm['total_topics']*100:.1f}%)")
        if sm.get('bw_mean', 0) > 0:
            top_lines.append(f"  Bant genişliği yükü — Ort: {sm['bw_mean']:,.0f}, Medyan: {sm['bw_median']:,.0f}")
            top_lines.append(f"  Aykırı değer sayısı: {sm['outlier_count']}")

    return fig, top_lines


def _chart_qos_3d_scatter(st: Dict[str, Any], tick_maps: Dict[str, Dict[str, float]], w2name: Dict[str, Dict[float, str]]) -> Tuple[plt.Figure, List[str]]:
    """Chart: QoS boyut dağılımı — 3 panel, X=konu boyutu, Y=QoS kategorisi."""
    qos_dims = [
        ("durability", "Dayanıklılık (Durability)"),
        ("reliability", "Güvenilirlik (Reliability)"),
        ("transport_priority", "Taşıma Önceliği (Transp. Priority)"),
    ]

    topic_data = st["topic_data"]
    top_outliers = st["top_outliers"]
    risk_upper = st["risk_upper"]
    risk_iqr = st["risk_iqr"]
    sm = st["summary"]

    if not topic_data:
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.text(0.5, 0.5, 'Veri yok', ha='center', va='center', fontsize=14)
        return fig, []

    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    fig.suptitle('Topic Size Distribution by QoS Categories', fontsize=14, fontweight='bold')

    for ax, (dim_key, dim_label) in zip(axes, qos_dims):
        tm = tick_maps[dim_key]
        weight_to_name: Dict[float, str] = {}
        for name, w in tm.items():
            weight_to_name[w] = name

        y_vals = [td[dim_key] for td in topic_data]
        x_vals = [td["size"] for td in topic_data]

        rng = np.random.RandomState(42)
        jitter = rng.uniform(-0.12, 0.12, size=len(y_vals))
        y_jittered = np.array(y_vals) + jitter

        ax.scatter(x_vals, y_jittered, c='#3498db', alpha=0.6,
                   edgecolors='black', linewidths=0.4, s=50, zorder=3)

        for td in top_outliers:
            idx = next((i for i, t in enumerate(topic_data) if t is td), None)
            if idx is not None:
                ax.scatter(x_vals[idx], y_jittered[idx], facecolors='none',
                           edgecolors='#c0392b', s=180, linewidths=2, zorder=5)

        sorted_weights = sorted(weight_to_name.keys())
        ax.set_yticks(sorted_weights)
        ax.set_yticklabels([weight_to_name[w] for w in sorted_weights], fontsize=9)

        ax.set_xlabel('Topic Size (bytes)', fontsize=10)
        ax.set_title(dim_label, fontsize=11)
        ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_handle = Line2D([0], [0], marker='o', color='w', markeredgecolor='#c0392b',
                           markerfacecolor='none', markersize=10, markeredgewidth=2,
                           label='Outliers')
    axes[-1].legend(handles=[legend_handle], fontsize=9, loc='lower right')

    top_lines: List[str] = []
    if top_outliers:
        if risk_iqr > 0:
            top_lines.append(f"Aykırı Değerler — QoS Risk Score (dur_w × rel_w × tp_w) × log₂(size + 1), IQR üst sınır: {risk_upper:,.0f}:")
        else:
            top_lines.append("High Risk Topics — QoS Risk Score (dur_w × rel_w × tp_w) × log₂(size + 1):")
    top_lines.append("")
    for rank, td in enumerate(top_outliers, 1):
        top_lines.append(
            f"  {rank:>2}. {td['name'][:45]:<45s}  Size={td['size']:>8,}  "
            f"Dur={td['dur_name']}  Rel={td['rel_name']}  TP={td['tp_name']}  "
            f"Score={td['risk']:,.0f}"
        )
    top_lines.append("")

    top_lines.append("─── Özet İstatistikler ───")
    top_lines.append(f"  Toplam Topic: {sm['total_topics']}")
    if sm.get("risk_mean") is not None:
        top_lines.append(f"  Risk Score — Ort: {sm['risk_mean']:,.0f}, Medyan: {sm['risk_median']:,.0f}, Std: {sm['risk_std']:,.0f}")
        top_lines.append(f"  Aykırı değer sayısı: {sm['outlier_count']}")
    qos_dist = sm.get("qos_distribution", {})
    for dim_key, dim_label in qos_dims:
        w_counts = qos_dist.get(dim_key, {})
        dist_parts = []
        for w in sorted(w_counts.keys()):
            name = w2name[dim_key].get(w, f"w={w}")
            dist_parts.append(f"{name}={w_counts[w]}")
        top_lines.append(f"  {dim_label}: {', '.join(dist_parts)}")

    return fig, top_lines


def _chart_app_pub_sub_balance(st: Dict[str, Any]) -> Tuple[plt.Figure, List[str]]:
    """Chart 3: App Publish vs. Subscribe Balance (scatter with quadrant lines)."""
    pubs = st["pubs"]
    subs = st["subs"]
    labels = st["labels"]
    io_load = st["io_load"]
    outlier_idx = st["outlier_indices"]
    io_upper = st["io_upper"]
    io_iqr = st["io_iqr"]
    sm = st["summary"]

    top_lines: List[str] = []
    fig, ax = plt.subplots(figsize=(10, 10))
    if pubs:
        ax.scatter(pubs, subs, c='#3498db', alpha=0.6, edgecolors='black', s=60)

        mean_p = sm["pub_mean"]
        mean_s = sm["sub_mean"]
        ax.axvline(mean_p, color='gray', linestyle='--', alpha=0.5, label=f'Avg Publish ({mean_p:.1f})')
        ax.axhline(mean_s, color='gray', linestyle=':', alpha=0.5, label=f'Avg Subscribe ({mean_s:.1f})')

        max_p = max(pubs) if pubs else 1
        max_s = max(subs) if subs else 1
        ax.text(max_p * 0.95, max_s * 0.95, 'High I/O', ha='right', va='top',
                fontsize=9, color='#e74c3c', fontweight='bold', alpha=0.7)
        ax.text(0.02 * max_p, max_s * 0.95, 'Consumer', ha='left', va='top',
                fontsize=9, color='#27ae60', fontweight='bold', alpha=0.7)
        ax.text(max_p * 0.95, 0.02 * max_s, 'Producer', ha='right', va='bottom',
                fontsize=9, color='#e67e22', fontweight='bold', alpha=0.7)
        ax.text(0.02 * max_p, 0.02 * max_s, 'Low Activity', ha='left', va='bottom',
                fontsize=9, color='#95a5a6', fontweight='bold', alpha=0.7)

        if outlier_idx:
            ox = [pubs[i] for i in outlier_idx]
            oy = [subs[i] for i in outlier_idx]
            ax.scatter(ox, oy, facecolors='none', edgecolors='#c0392b',
                       s=200, linewidths=2, zorder=5, label='High I/O Load')
            top_lines = [f"Aykırı Değerler — High I/O Load (Publish+Subscribe), IQR üst sınır: {io_upper:.0f}:"]
            for rank, i in enumerate(outlier_idx, 1):
                dev = (io_load[i] - io_upper) / io_iqr if io_iqr > 0 else 0.0
                top_lines.append(f"  {rank:>2}. {labels[i][:45]:<45s}  Publish={pubs[i]}  Subscribe={subs[i]}  Total={io_load[i]}  Sapma={dev:.1f}×IQR")

        ax.legend(fontsize=9, loc='upper left')

    ax.set_xlabel('Direct Publish Count', fontsize=11)
    ax.set_ylabel('Direct Subscribe Count', fontsize=11)
    ax.set_title('Application Publish-Subscribe Balance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if sm:
        top_lines.append("")
        top_lines.append("─── Özet İstatistikler ───")
        top_lines.append(f"  Toplam Uygulama: {sm['total_apps']}")
        top_lines.append(f"  Publish — Ort: {sm['pub_mean']:.1f}, Medyan: {sm['pub_median']:.0f}, Maks: {sm['pub_max']}")
        top_lines.append(f"  Subscribe — Ort: {sm['sub_mean']:.1f}, Medyan: {sm['sub_median']:.0f}, Maks: {sm['sub_max']}")
        top_lines.append(f"  Kadran dağılımı: High I/O={sm['q_high_io']}, Consumer={sm['q_consumer']}, Producer={sm['q_producer']}, Low={sm['q_low']}")
        if sm['zero_activity'] > 0:
            top_lines.append(f"  Hiç iletişimi olmayan uygulama: {sm['zero_activity']} ({sm['zero_activity']/sm['total_apps']*100:.1f}%)")
        top_lines.append(f"  Aykırı değer sayısı: {sm['outlier_count']}")

    return fig, top_lines


def _chart_topic_fanout(st: Dict[str, Any]) -> Tuple[plt.Figure, List[str]]:
    """Chart 4: Topic Fanout (scatter: publisher count vs subscriber count)."""
    pubs = st["pubs"]
    subs = st["subs"]
    labels = st["labels"]
    fanout = st["fanout"]
    outlier_idx = st["outlier_indices"]
    fanout_upper = st["fanout_upper"]
    fanout_iqr = st["fanout_iqr"]
    sm = st["summary"]

    top_lines: List[str] = []
    fig, ax = plt.subplots(figsize=(10, 9))
    if pubs:
        ax.scatter(pubs, subs, c='#9b59b6', alpha=0.6, edgecolors='black', s=60)
        if outlier_idx:
            ox = [pubs[i] for i in outlier_idx]
            oy = [subs[i] for i in outlier_idx]
            ax.scatter(ox, oy, facecolors='none', edgecolors='#c0392b',
                       s=200, linewidths=2, zorder=5, label='Yüksek Yayılım Riski')
            ax.legend(fontsize=9, loc='upper right')
            top_lines = [f"Aykırı Değerler — High Fanout Risk (Publisher×Subscriber), IQR üst sınır: {fanout_upper:.0f}:"]
            for rank, i in enumerate(outlier_idx, 1):
                dev = (fanout[i] - fanout_upper) / fanout_iqr if fanout_iqr > 0 else 0.0
                top_lines.append(f"  {rank:>2}. {labels[i][:45]:<45s}  Publisher={pubs[i]}  Subscriber={subs[i]}  Fanout={fanout[i]}  Sapma={dev:.1f}×IQR")
    ax.set_xlabel('Publisher Count', fontsize=11)
    ax.set_ylabel('Subscriber Count', fontsize=11)
    ax.set_title('Topic Fanout\n(Publisher and Subscriber Count per Topic)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if sm:
        top_lines.append("")
        top_lines.append("─── Özet İstatistikler ───")
        top_lines.append(f"  Toplam Topic: {sm['total_topics']}")
        top_lines.append(f"  Publisher — Ort: {sm['pub_mean']:.1f}, Medyan: {sm['pub_median']:.0f}, Maks: {sm['pub_max']}")
        top_lines.append(f"  Subscriber — Ort: {sm['sub_mean']:.1f}, Medyan: {sm['sub_median']:.0f}, Maks: {sm['sub_max']}")
        top_lines.append(f"  İletişim kalıpları: 1→N={sm['one_to_many']}, N→1={sm['many_to_one']}, N→M={sm['many_to_many']}")
        if sm['orphan'] > 0:
            top_lines.append(f"  Publisher'ı veya subscriber'ı olmayan topic: {sm['orphan']} ({sm['orphan']/sm['total_topics']*100:.1f}%)")
        if sm.get('fanout_mean', 0) > 0:
            top_lines.append(f"  Yayılım karmaşıklığı (Pub×Sub) — Ort: {sm['fanout_mean']:.1f}, Maks: {sm['fanout_max']}")
        top_lines.append(f"  Aykırı değer sayısı: {sm['outlier_count']}")

    return fig, top_lines


def _chart_cross_node_heatmap(st: Dict[str, Any]) -> Tuple[plt.Figure, List[str]]:
    """Chart 5: Cross-Node Communication Heatmap.
    Cell (i,j) = number of topics where a node-i app publishes and a node-j app subscribes.
    """
    labels = st["labels"]
    matrix = st["matrix"]
    n = len(labels)

    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'Node data not found', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig, []

    fig_size = max(7, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=max(6, 10 - n // 5))
    ax.set_yticklabels(labels, fontsize=max(6, 10 - n // 5))
    ax.set_xlabel('Subscriber Node', fontsize=11)
    ax.set_ylabel('Publisher Node', fontsize=11)
    ax.set_title('Inter-Node Communication Heatmap\n(Topic count: publisher node → subscriber node)',
                 fontsize=13, fontweight='bold')

    for i in range(n):
        for j in range(n):
            val = int(matrix[i][j])
            if val > 0:
                text_color = 'white' if val > matrix.max() * 0.6 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=max(6, 10 - n // 4), color=text_color)

    fig.colorbar(im, ax=ax, label='Shared Topic Count', shrink=0.8)
    _safe_tight_layout(fig)

    top_lines: List[str] = []
    outlier_pairs = st["outlier_pairs"]
    iqr_upper = st["iqr_upper"]
    iqr = st["iqr"]
    if outlier_pairs:
        top_lines = [f"Aykırı Değerler — Node Pairs, IQR üst sınır: {iqr_upper:.0f}:"]
        for rank, (pub_n, sub_n, count, dev) in enumerate(outlier_pairs, 1):
            dev_str = f"  Sapma={dev:.1f}×IQR" if iqr > 0 else ""
            top_lines.append(f"  {rank:>2}. {pub_n[:20]} → {sub_n[:20]}  Topic={count}{dev_str}")

    sm = st["summary"]
    top_lines.append("")
    top_lines.append("─── Özet İstatistikler ───")
    top_lines.append(f"  Node sayısı: {sm['entity_count']}")
    top_lines.append(f"  Aktif iletişim çifti: {sm['nonzero_count']}/{sm['total_cells']} ({sm['active_pct']:.1f}%)" if sm['total_cells'] > 0 else "  Aktif iletişim çifti: 0")
    top_lines.append(f"  Node-içi iletişim: {sm['intra_total']} topic, Node'lar arası: {sm['inter_total']} topic")
    if 'cell_mean' in sm:
        top_lines.append(f"  Hücre değerleri (sıfır hariç) — Ort: {sm['cell_mean']:.1f}, Medyan: {sm['cell_median']:.0f}, Maks: {sm['cell_max']}")
    top_lines.append(f"  Node'lar arası aktif çift: {sm['cross_pairs']}")
    if sm.get('outlier_count', 0) > 0:
        top_lines.append(f"  Aykırı değer sayısı: {sm['outlier_count']}")

    return fig, top_lines


def _chart_node_communication_load(st: Dict[str, Any]) -> Tuple[plt.Figure, List[str]]:
    """Chart 6: Node Communication Load (stacked bar: pub + sub per node)."""
    sorted_labels = st["sorted_labels"]
    sorted_pub = st["sorted_pub"]
    sorted_sub = st["sorted_sub"]

    fig, ax = plt.subplots(figsize=(12, max(5, len(sorted_labels) * 0.4)))
    y_pos = np.arange(len(sorted_labels))

    bars1 = ax.barh(y_pos, sorted_pub, color='#27ae60', alpha=0.8, label='Publish')
    bars2 = ax.barh(y_pos, sorted_sub, left=sorted_pub, color='#e67e22', alpha=0.8, label='Subscribe')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Communication Count', fontsize=11)
    ax.set_title('Node Communication Load\n(Total Publish + Subscribe per Node)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    for i, (p, s) in enumerate(zip(sorted_pub, sorted_sub)):
        total = p + s
        if total > 0:
            ax.text(total + 0.3, i, str(total), va='center', fontsize=8, color='#34495e')

    _safe_tight_layout(fig)

    top_lines: List[str] = []
    outliers = st["outliers"]
    iqr_upper = st["iqr_upper"]
    iqr = st["iqr"]
    if outliers:
        top_lines = [f"Aykırı Değerler — Node Load, IQR üst sınır: {iqr_upper:.0f}:"]
        for rank, (name, p, s, t, dev) in enumerate(outliers, 1):
            dev_str = f"  Sapma={dev:.1f}×IQR" if iqr > 0 else ""
            top_lines.append(f"  {rank:>2}. {name[:30]:<30s}  Publish={p}  Subscribe={s}  Total={t}{dev_str}")

    sm = st["summary"]
    top_lines.append("")
    top_lines.append("─── Özet İstatistikler ───")
    top_lines.append(f"  Node sayısı: {sm['node_count']}")
    if 'pub_total' in sm:
        top_lines.append(f"  Toplam publish: {sm['pub_total']}, Toplam subscribe: {sm['sub_total']}")
        top_lines.append(f"  Node load — Ort: {sm['load_mean']:.1f}, Medyan: {sm['load_median']:.0f}, Std: {sm['load_std']:.1f}")
        if sm.get('cv', 0) > 0:
            top_lines.append(f"  Yük dengesizliği (CV): {sm['cv']:.1f}%")
        if sm.get('zero_load', 0) > 0:
            top_lines.append(f"  İletişimsiz node: {sm['zero_load']} ({sm['zero_load']/sm['node_count']*100:.1f}%)")
    top_lines.append(f"  Aykırı değer sayısı: {sm.get('outlier_count', 0)}")

    return fig, top_lines


def _chart_domain_communication_matrix(st: Dict[str, Any]) -> Tuple[plt.Figure, List[str]]:
    """Chart 7: Domain-to-Domain Communication Heatmap.
    Cell (i,j) = number of topics where a domain-i app publishes and domain-j app subscribes.
    """
    labels = st["labels"]
    matrix = st["matrix"]
    n = len(labels)

    if n < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'Insufficient CSS data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig, []

    fig_size = max(7, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=max(6, 10 - n // 5))
    ax.set_yticklabels(labels, fontsize=max(6, 10 - n // 5))
    ax.set_xlabel('Subscriber CSS', fontsize=11)
    ax.set_ylabel('Publisher CSS', fontsize=11)
    ax.set_title('CSS Communication Matrix\n(Topic count: publisher CSS → subscriber CSS)', fontsize=13, fontweight='bold')

    for i in range(n):
        for j in range(n):
            val = int(matrix[i][j])
            if val > 0:
                text_color = 'white' if val > matrix.max() * 0.6 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=max(6, 10 - n // 4), color=text_color)

    fig.colorbar(im, ax=ax, label='Shared Topic Count', shrink=0.8)
    _safe_tight_layout(fig)

    top_lines: List[str] = []
    outlier_pairs = st["outlier_pairs"]
    iqr_upper = st["iqr_upper"]
    iqr = st["iqr"]
    if outlier_pairs:
        top_lines = [f"Aykırı Değerler — CSS Pairs, IQR üst sınır: {iqr_upper:.0f}:"]
        for rank, (pub_d, sub_d, count, dev) in enumerate(outlier_pairs, 1):
            dev_str = f"  Sapma={dev:.1f}×IQR" if iqr > 0 else ""
            top_lines.append(f"  {rank:>2}. {pub_d} → {sub_d}  Topic={count}{dev_str}")

    sm = st["summary"]
    top_lines.append("")
    top_lines.append("─── Özet İstatistikler ───")
    top_lines.append(f"  CSS sayısı: {sm['entity_count']}")
    top_lines.append(f"  Aktif iletişim çifti: {sm['nonzero_count']}/{sm['total_cells']} ({sm['active_pct']:.1f}%)")
    top_lines.append(f"  CSS-içi iletişim: {sm['intra_total']} topic, CSS'ler arası: {sm['inter_total']} topic")
    top_lines.append(f"  CSS'ler arası aktif çift: {sm['cross_pairs']}")
    if sm.get('outlier_count', 0) > 0:
        top_lines.append(f"  Aykırı değer sayısı: {sm['outlier_count']}")

    return fig, top_lines


def _chart_criticality_io_load(st: Dict[str, Any]) -> Tuple[plt.Figure, List[str]]:
    """Chart 8: Criticality × I/O Load scatter — critical apps vs total communication."""
    crit_pubs = st["crit_pubs"]
    crit_subs = st["crit_subs"]
    crit_labels = st["crit_labels"]
    norm_pubs = st["norm_pubs"]
    norm_subs = st["norm_subs"]

    fig, ax = plt.subplots(figsize=(10, 9))

    if norm_pubs:
        ax.scatter(norm_pubs, norm_subs, c='#3498db', alpha=0.5, edgecolors='black',
                   linewidths=0.4, s=50, label=f'Normal ({len(norm_pubs)})', zorder=3)
    if crit_pubs:
        ax.scatter(crit_pubs, crit_subs, c='#e74c3c', alpha=0.8, edgecolors='black',
                   linewidths=0.6, s=100, marker='D', label=f'Kritik ({len(crit_pubs)})', zorder=4)

    ax.set_xlabel('Direct Publish Count', fontsize=11)
    ax.set_ylabel('Direct Subscribe Count', fontsize=11)
    ax.set_title('Criticality × Communication Load\n(Critical vs Normal Application Publish-Subscribe Distribution)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    top_lines: List[str] = []
    outliers = st["outliers"]
    iqr_upper = st["iqr_upper"]
    if outliers:
        top_lines = [f"Aykırı Değerler — Critical App I/O Load, IQR üst sınır: {iqr_upper:.0f}:"]
        for rank, (name, p, s, t) in enumerate(outliers, 1):
            top_lines.append(f"  {rank:>2}. {name[:40]:<40s}  Publish={p}  Subscribe={s}  Total={t}")

    sm = st["summary"]
    top_lines.append("")
    top_lines.append("─── Özet İstatistikler ───")
    top_lines.append(f"  Toplam uygulama: {sm['total_apps']}")
    top_lines.append(f"  Kritik uygulama: {sm['crit_count']} ({sm['crit_pct']:.1f}%)" if sm['total_apps'] > 0 else "  Kritik uygulama: 0")
    if 'crit_io_mean' in sm:
        top_lines.append(f"  Kritik I/O yükü — Ort: {sm['crit_io_mean']:.1f}, Medyan: {sm['crit_io_median']:.0f}, Maks: {sm['crit_io_max']}")
    if 'norm_io_mean' in sm:
        top_lines.append(f"  Normal I/O yükü — Ort: {sm['norm_io_mean']:.1f}, Medyan: {sm['norm_io_median']:.0f}, Maks: {sm['norm_io_max']}")
    if 'crit_norm_ratio' in sm:
        top_lines.append(f"  Kritik/Normal ortalama yük oranı: {sm['crit_norm_ratio']:.2f}×")
    top_lines.append(f"  Aykırı değer sayısı (kritik): {sm.get('outlier_count', 0)}")

    return fig, top_lines


def _chart_library_dependency_density(st: Dict[str, Any]) -> Tuple[plt.Figure, List[str]]:
    """Chart 9: Library Dependency Density — in-degree (used by) and out-degree (uses) bar chart."""
    labels = st["labels"]
    in_vals = st["in_vals"]
    out_vals = st["out_vals"]

    if not labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'Bağımlılık (uses) verisi bulunamadı', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig, []

    fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.35)))
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, in_vals, color='#27ae60', alpha=0.8, label='Bağımlı olan (in-degree)')
    ax.barh(y_pos, [-v for v in out_vals], color='#e67e22', alpha=0.8, label='Bağımlı olduğu (out-degree)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Bağımlılık Sayısı', fontsize=11)
    ax.set_title('Kütüphane Bağımlılık Yoğunluğu\n(In-degree: bana bağımlı, Out-degree: bağımlı olduğum)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(0, color='black', linewidth=0.5)

    _safe_tight_layout(fig)

    top_lines: List[str] = []
    outliers = st["outliers"]
    iqr_upper = st["iqr_upper"]
    if outliers:
        top_lines = [f"Aykırı Değerler — Yüksek Bağımlılık (in-degree), IQR üst sınır: {iqr_upper:.0f}:"]
        for rank, (name, ind, outd) in enumerate(outliers, 1):
            top_lines.append(f"  {rank:>2}. {name[:40]:<40s}  Bağımlı={ind}  Bağımlılık={outd}")

    sm = st["summary"]
    top_lines.append("")
    top_lines.append("─── Özet İstatistikler ───")
    top_lines.append(f"  Toplam bağımlılık ilişkisi: {sm['total_relations']}")
    top_lines.append(f"  Aktif bileşen: {sm.get('active_count', 0)} (uygulama + kütüphane)")
    top_lines.append(f"  Uygulama: {sm.get('app_count', 0)}, Kütüphane: {sm.get('lib_count', 0)}")
    if 'in_mean' in sm:
        top_lines.append(f"  In-degree — Ort: {sm['in_mean']:.1f}, Maks: {sm['in_max']}")
    if 'out_mean' in sm:
        top_lines.append(f"  Out-degree — Ort: {sm['out_mean']:.1f}, Maks: {sm['out_max']}")
    top_lines.append(f"  Aykırı değer sayısı: {sm.get('outlier_count', 0)}")

    return fig, top_lines


def _chart_node_critical_app_density(st: Dict[str, Any]) -> Tuple[plt.Figure, List[str]]:
    """Chart 10: Node Critical App Density — stacked bar showing critical vs normal apps per node."""
    sorted_labels = st["sorted_labels"]
    sorted_crit = st["sorted_crit"]
    sorted_norm = st["sorted_norm"]

    fig, ax = plt.subplots(figsize=(12, max(5, len(sorted_labels) * 0.4)))
    y_pos = np.arange(len(sorted_labels))

    ax.barh(y_pos, sorted_crit, color='#e74c3c', alpha=0.85, label='Kritik')
    ax.barh(y_pos, sorted_norm, left=sorted_crit, color='#3498db', alpha=0.7, label='Normal')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Application Count', fontsize=11)
    ax.set_title('Critical Application Density per Node\n(Critical vs Normal Application Distribution)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    for i, (c, n) in enumerate(zip(sorted_crit, sorted_norm)):
        total = c + n
        if total > 0:
            pct = c / total * 100
            ax.text(total + 0.3, i, f'{total} (%{pct:.0f} kritik)', va='center', fontsize=8, color='#34495e')

    _safe_tight_layout(fig)

    sm = st["summary"]
    top_lines: List[str] = []
    top_lines.append("─── Özet İstatistikler ───")
    top_lines.append(f"  Node sayısı: {sm['node_count']}")
    top_lines.append(f"  Toplam uygulama: {sm['total_all']} (Kritik: {sm['total_crit']}, Normal: {sm['total_norm']})")
    if sm['total_all'] > 0:
        top_lines.append(f"  Sistem geneli kritiklik oranı: %{sm['system_crit_pct']:.1f}")
    if 'crit_per_node_mean' in sm:
        top_lines.append(f"  Node başına kritik uygulama — Ort: {sm['crit_per_node_mean']:.1f}, Maks: {sm['crit_per_node_max']}")
    if 'max_ratio_node' in sm:
        top_lines.append(f"  En yüksek kritiklik oranı: {sm['max_ratio_node']} (%{sm['max_ratio_pct']:.0f})")
    if sm.get('zero_crit', 0) > 0:
        top_lines.append(f"  Kritik uygulaması olmayan node: {sm['zero_crit']}")

    return fig, top_lines


def _chart_domain_app_topic_diversity(st: Dict[str, Any]) -> Tuple[plt.Figure, List[str]]:
    """Chart 11: Domain App & Topic Diversity — bubble chart (X=app count, Y=topic variety, size=total I/O)."""
    labels = st["labels"]
    app_counts = st["app_counts"]
    topic_counts = st["topic_counts"]
    io_vals = st["io_vals"]

    if len(labels) < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'Insufficient CSS data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig, []

    max_io = max(io_vals) if io_vals else 1
    sizes = [max(50, (v / max_io) * 1500) if max_io > 0 else 50 for v in io_vals]

    fig, ax = plt.subplots(figsize=(10, 9))
    scatter = ax.scatter(app_counts, topic_counts, s=sizes, c=io_vals,
                         cmap='YlOrRd', alpha=0.7, edgecolors='black', linewidths=0.8)

    for i, label in enumerate(labels):
        ax.annotate(label[:20], (app_counts[i], topic_counts[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.85)

    ax.set_xlabel('Application Count', fontsize=11)
    ax.set_ylabel('Topic Variety', fontsize=11)
    ax.set_title('CSS Application and Topic Diversity\n(Bubble size = total I/O load)',
                 fontsize=13, fontweight='bold')
    fig.colorbar(scatter, ax=ax, label='Total I/O Load', shrink=0.8)
    ax.grid(True, alpha=0.3)

    sm = st["summary"]
    ranked = st["ranked"]
    top_lines: List[str] = []
    top_lines.append("─── Özet İstatistikler ───")
    top_lines.append(f"  CSS sayısı: {sm['css_count']}")
    if 'app_mean' in sm:
        top_lines.append(f"  Uygulama/CSS — Ort: {sm['app_mean']:.1f}, Maks: {sm['app_max']}")
    if 'topic_mean' in sm:
        top_lines.append(f"  Topic çeşitliliği/CSS — Ort: {sm['topic_mean']:.1f}, Maks: {sm['topic_max']}")
    if 'io_mean' in sm:
        top_lines.append(f"  I/O yükü/CSS — Ort: {sm['io_mean']:.1f}, Maks: {sm['io_max']}")
    if ranked:
        top_lines.append("")
        top_lines.append("CSS Sıralaması (I/O yüküne göre):")
        for rank, (name, ac, tc, io) in enumerate(ranked, 1):
            top_lines.append(f"  {rank:>2}. {name[:30]:<30s}  Uyg={ac}  Topic={tc}  I/O={io}")

    return fig, top_lines


def _generate_extras_charts(raw_data: Dict[str, Any], images_dir: str = None, *, dds_mask: bool = True) -> List[Tuple[str, str, plt.Figure, List[str]]]:
    """
    Generate all cross-cutting (Extras) charts.
    
    Args:
        raw_data: Raw JSON data from aggregator.
        images_dir: Directory to save PNG images (for markdown). If None, figures are returned only.
    
    Returns:
        List of (chart_id, title, figure, top_lines) tuples.
    """
    if not raw_data:
        return []

    cc = extract_cross_cutting_data(raw_data)

    # Build QoS presentation config (tick_maps / w2name)
    from pipeline.aggregator.converter import _QOS_RISK_WEIGHTS
    w2name: Dict[str, Dict[float, str]] = {}
    if dds_mask:
        for dim, mapping in _QOS_RISK_WEIGHTS.items():
            w2name[dim] = {w: v for v, w in mapping.items()}

    # Pre-compute all statistics
    topic_bandwidth_st = compute_topic_bandwidth_stats(cc)
    qos_risk_st = compute_qos_risk_stats(cc, qos_risk_weight, w2name) if dds_mask else None
    app_balance_st = compute_app_balance_stats(cc)
    topic_fanout_st = compute_topic_fanout_stats(cc)
    cross_node_st = compute_cross_node_heatmap_stats(cc)
    node_comm_st = compute_node_comm_load_stats(cc)
    domain_comm_st = compute_domain_comm_stats(cc)
    criticality_io_st = compute_criticality_io_stats(cc)
    lib_dep_st = compute_lib_dependency_stats(cc)
    node_crit_st = compute_node_critical_density_stats(cc)
    domain_div_st = compute_domain_diversity_stats(cc)

    # Build tick_maps for QoS Y-axis labels
    tick_maps: Dict[str, Dict[float, str]] = {}
    if dds_mask:
        for dim, mapping in _QOS_RISK_WEIGHTS.items():
            tick_maps[dim] = {w: v for v, w in mapping.items()}

    charts = [
        ("size_vs_subscribers", "Topic Size vs Subscriber Count", lambda: _chart_topic_size_vs_subscribers(topic_bandwidth_st)),
        ("qos_3d_scatter", "Topic Size by QoS Categories", lambda: _chart_qos_3d_scatter(qos_risk_st, tick_maps, w2name) if qos_risk_st else (None, [])),
        ("app_pub_sub_balance", "Application Publish-Subscribe Balance", lambda: _chart_app_pub_sub_balance(app_balance_st)),
        ("topic_fanout", "Topic Fanout (Publisher and Subscriber)", lambda: _chart_topic_fanout(topic_fanout_st)),
        ("cross_node_heatmap", "Inter-Node Communication Heatmap", lambda: _chart_cross_node_heatmap(cross_node_st)),
        ("node_comm_load", "Node Communication Load", lambda: _chart_node_communication_load(node_comm_st)),
        ("domain_comm_matrix", "CSS Communication Matrix", lambda: _chart_domain_communication_matrix(domain_comm_st)),
        ("criticality_io", "Criticality × Communication Load", lambda: _chart_criticality_io_load(criticality_io_st)),
        ("lib_dependency", "Library Dependency Density", lambda: _chart_library_dependency_density(lib_dep_st)),
        ("node_critical_density", "Critical Application Density per Node", lambda: _chart_node_critical_app_density(node_crit_st)),
        ("domain_diversity", "CSS Application and Topic Diversity", lambda: _chart_domain_app_topic_diversity(domain_div_st)),
    ]

    results = []
    for chart_id, title, chart_fn in charts:
        try:
            fig, top_lines = chart_fn()
            if fig is not None:
                results.append((chart_id, title, fig, top_lines))
        except Exception:
            pass  # Skip charts that fail due to missing data

    return results


def generate_report(report: AnalysisReport, output_dir: str, output_format: str = "markdown", *, dds_mask: bool = True) -> str:
    """
    Generate report in specified format.
    
    Args:
        report: AnalysisReport object.
        output_dir: Base output directory.
        output_format: 'markdown' or 'pdf'
    
    Returns:
        Path to the created report file.
    """
    if output_format.lower() == "pdf":
        return generate_pdf_report(report, output_dir, dds_mask=dds_mask)
    else:
        return generate_markdown_report(report, output_dir, dds_mask=dds_mask)


def generate_pdf_report(report: AnalysisReport, output_dir: str, *, dds_mask: bool = True) -> str:
    """
    Generate a PDF report with charts, tables and clickable TOC.
    
    Strategy:
    1. Generate content pages (matplotlib) -> temp file, track page numbers
    2. Calculate TOC page count  
    3. Generate TOC pages (matplotlib) -> temp file
    4. Merge: Title + TOC + Content[1:] with pypdf
    5. Add link annotations to TOC pages with pypdf
    """
    from pypdf.generic import ArrayObject, DictionaryObject, FloatObject, NameObject, NumberObject
    
    platform_dir = os.path.join(output_dir, report.platform_name)
    os.makedirs(platform_dir, exist_ok=True)
    
    temp_content_file = os.path.join(platform_dir, "_temp_content.pdf")
    temp_toc_file = os.path.join(platform_dir, "_temp_toc.pdf")
    output_file = os.path.join(platform_dir, "report.pdf")
    
    # ============ STEP 1: Generate content PDF ============
    toc_entries = []  # [(section_title, content_page_idx, [(metric_desc, content_page_idx), ...])]
    
    with PdfPages(temp_content_file) as pdf:
        # Page 0: Title page
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        ax.text(0.5, 0.6, 'İstatistiksel Analiz Raporu', fontsize=28, ha='center', va='center', fontweight='bold')
        ax.text(0.5, 0.45, f'{report.platform_name.upper()}', fontsize=24, ha='center', va='center', color='#3498db')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Summary pages
        summary_start = pdf.get_pagecount()
        _add_summary_to_pdf(pdf, report)
        toc_entries.append(("Sistem Özeti", summary_start, []))
        
        # İstatistikler: Cross-cutting charts
        extras_charts = _generate_extras_charts(report.raw_data, dds_mask=dds_mask)
        extras_start = None
        extras_metrics = []

        if extras_charts:
            extras_start = pdf.get_pagecount()
            for chart_id, chart_title, fig, top_lines in extras_charts:
                metric_page = pdf.get_pagecount()
                extras_metrics.append((chart_title, metric_page))
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                # Add a separate page for top-10 text list
                if top_lines:
                    txt_fig = _make_top_text_page(top_lines, title=chart_title)
                    pdf.savefig(txt_fig, bbox_inches='tight')
                    plt.close(txt_fig)

        # İstatistikler: ComponentAnalysis items (e.g. cycle detection)
        for analysis in report.extras_analysis:
            values = [item.get("value", 0) for item in analysis.ranked_list]
            if not values:
                continue
            if extras_start is None:
                extras_start = pdf.get_pagecount()
            desc = _display_metric_label(analysis)
            metric_start = pdf.get_pagecount()
            extras_metrics.append((desc, metric_start))
            _add_single_analysis_to_pdf(pdf, analysis, "İstatistikler")

        if extras_metrics and extras_start is not None:
            toc_entries.append(("İstatistikler", extras_start, extras_metrics))
    
    # ============ STEP 2: Build TOC items list ============
    toc_items = []  # [(text, content_page_idx, is_section), ...]
    for section_title, section_page, metrics in toc_entries:
        toc_items.append((section_title, section_page, True))
        for metric_desc, metric_page in metrics:
            toc_items.append((metric_desc, metric_page, False))
    
    lines_per_page = 22
    toc_pages_needed = max(1, (len(toc_items) + lines_per_page - 1) // lines_per_page)
    
    # ============ STEP 3: Generate TOC PDF with matplotlib ============
    # Track link positions as ratios (0-1), convert to points after we know actual page size
    toc_link_ratios = []  # [(toc_page_idx, y_ratio_top, y_ratio_bottom, content_page), ...]
    
    with PdfPages(temp_toc_file) as pdf:
        item_idx = 0
        for toc_page_idx in range(toc_pages_needed):
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis('off')
            
            # Header
            if toc_page_idx == 0:
                ax.text(0.5, 0.92, 'İÇİNDEKİLER', fontsize=24, ha='center', va='top', 
                        fontweight='bold', transform=ax.transAxes)
                y_start = 0.80
            else:
                ax.text(0.5, 0.92, 'İÇİNDEKİLER (devam)', fontsize=24, ha='center', va='top', 
                        fontweight='bold', transform=ax.transAxes)
                y_start = 0.85
            
            y_pos = y_start
            line_height = 0.030
            items_on_page = 0
            
            while item_idx < len(toc_items) and items_on_page < lines_per_page:
                text, content_page, is_section = toc_items[item_idx]
                
                if is_section:
                    y_pos -= 0.015  # Extra space before section
                    y_top = y_pos
                    ax.text(0.08, y_pos, f"■ {text}", fontsize=12, ha='left', va='top',
                            fontweight='bold', color='#2c3e50', transform=ax.transAxes)
                    y_pos -= line_height * 1.3
                    y_bottom = y_pos + 0.005
                else:
                    y_top = y_pos
                    display_text = text[:70] + "..." if len(text) > 70 else text
                    ax.text(0.10, y_pos, f"• {display_text}", fontsize=10, ha='left', va='top',
                            color='#0066cc', transform=ax.transAxes)
                    y_pos -= line_height
                    y_bottom = y_pos + 0.005
                
                # Store ratios
                toc_link_ratios.append((toc_page_idx, y_top, y_bottom, content_page))
                
                item_idx += 1
                items_on_page += 1
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    # ============ STEP 4: Merge PDFs ============
    content_reader = PdfReader(temp_content_file)
    toc_reader = PdfReader(temp_toc_file)
    writer = PdfWriter()
    
    # GERÇEK TOC sayfa sayısı - tahmin değil!
    actual_toc_pages = len(toc_reader.pages)
    
    # Title page (content[0])
    writer.add_page(content_reader.pages[0])
    
    # TOC pages
    for page in toc_reader.pages:
        writer.add_page(page)
    
    # Content pages (skip title)
    for i in range(1, len(content_reader.pages)):
        writer.add_page(content_reader.pages[i])
    
    # ============ STEP 5: Add link annotations ============
    # Get actual page dimensions from first TOC page
    first_toc_page = writer.pages[1]
    media_box = first_toc_page.mediabox
    page_width = float(media_box.width)
    page_height = float(media_box.height)
    
    for toc_page_idx, y_ratio_top, y_ratio_bottom, content_page in toc_link_ratios:
        # Calculate final pages
        final_toc_page_idx = 1 + toc_page_idx
        
        # GERÇEK TOC sayfa sayısını kullan!
        if content_page == 0:
            final_target_page = 0
        else:
            final_target_page = actual_toc_pages + content_page
        
        # Ensure target page exists
        if final_target_page >= len(writer.pages):
            continue
        
        # Convert ratios to points (PDF y-axis is from bottom)
        y_top_pt = y_ratio_top * page_height
        y_bottom_pt = y_ratio_bottom * page_height
        
        # Link rectangle
        rect = ArrayObject([
            FloatObject(40),
            FloatObject(y_bottom_pt),
            FloatObject(page_width - 40),
            FloatObject(y_top_pt)
        ])
        
        # Destination: go to top of target page
        target_page_obj = writer.pages[final_target_page]
        dest = ArrayObject([
            target_page_obj.indirect_reference,
            NameObject("/XYZ"),
            NumberObject(0),
            FloatObject(page_height),
            NumberObject(0)  # Keep current zoom
        ])
        
        # Create link annotation
        link = DictionaryObject({
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Link"),
            NameObject("/Rect"): rect,
            NameObject("/Border"): ArrayObject([NumberObject(0), NumberObject(0), NumberObject(0)]),
            NameObject("/Dest"): dest
        })
        
        # Add to page
        toc_page = writer.pages[final_toc_page_idx]
        if "/Annots" not in toc_page:
            toc_page[NameObject("/Annots")] = ArrayObject()
        toc_page["/Annots"].append(writer._add_object(link))
    
    # ============ STEP 6: Add bookmarks (sidebar) ============
    for section_title, content_page, metrics in toc_entries:
        # GERÇEK TOC sayfa sayısını kullan!
        final_page = actual_toc_pages + content_page if content_page > 0 else 0
        parent = writer.add_outline_item(section_title, final_page)
        
        for metric_desc, metric_page in metrics:
            final_metric_page = actual_toc_pages + metric_page
            writer.add_outline_item(metric_desc, final_metric_page, parent=parent)
    
    # Write output
    with open(output_file, "wb") as f:
        writer.write(f)
    
    # Cleanup
    for f in [temp_content_file, temp_toc_file]:
        try:
            os.remove(f)
        except:
            pass
    
    log_info(f"PDF report generated: {output_file}")
    return output_file


def _add_single_analysis_to_pdf(pdf: PdfPages, analysis: ComponentAnalysis, section_title: str):
    """Add chart and outlier pages for a single analysis to PDF."""
    values = [item.get("value", 0) for item in analysis.ranked_list]
    if not values or all(v == 0 for v in values):
        return
    
    desc = _display_metric_label(analysis)
    s = analysis.stats
    is_categorical = analysis.is_categorical
    outliers = calculate_outliers(analysis.ranked_list, analysis.stats, is_categorical=is_categorical) if not is_categorical else []
    show_charts = getattr(analysis, "show_charts", True)
    
    # Total metrics include recursive library breakdowns.
    is_total_metric_analysis = is_total_metric(analysis.metric_name)
    
    if not show_charts:
        _add_table_pages_to_pdf(pdf, analysis, section_title, desc, is_total_metric_analysis)
        return

    if is_categorical:
        # Pie chart and Bar chart
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.suptitle(f'{section_title}\n{desc}', fontsize=14, fontweight='bold')
        
        names = [item.get("name", "?") for item in analysis.ranked_list]
        vals = [item.get("value", 0) for item in analysis.ranked_list]
        total = sum(vals)
        pcts = [(v / total * 100) if total > 0 else 0 for v in vals]
        
        # Pie Chart
        ax = axes[0]
        labels = [f"{n}\n({v})" for n, v in zip(names, vals)]
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        wedges, texts = ax.pie(vals, labels=labels, colors=colors, startangle=90)
        ax.set_title('Dağılım (Pasta Grafik)')
        
        # Bar Chart
        ax = axes[1]
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, vals, color='#3498db', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Sayı')
        ax.set_title('Dağılım (Bar Grafik)')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, val, pct in zip(bars, vals, pcts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{int(val)} (%{pct:.1f})', ha='left', va='center', fontsize=9)
        
        _safe_tight_layout(fig, rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    else:
        # Box plot and histogram
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.suptitle(f'{section_title}\n{desc}', fontsize=14, fontweight='bold')
        
        # Box Plot
        ax = axes[0]
        bp = ax.boxplot(values, vert=False, patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][0].set_alpha(0.7)
        ax.set_xlabel('Değer')
        ax.set_title('Kutu Grafiği')
        ax.grid(True, alpha=0.3)
        stats_text = f'Ort.: {s.mean:.2f}\nMedyan: {s.median:.2f}\nStd: {s.std:.2f}\nQ1: {s.q1:.2f}\nQ3: {s.q3:.2f}'
        ax.annotate(stats_text, xy=(0.98, 0.98), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Histogram
        ax = axes[1]
        n_bins = min(20, max(5, len(values) // 3))
        ax.hist(values, bins=n_bins, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.axvline(s.mean, color='red', linestyle='--', linewidth=2, label=f'Ort.: {s.mean:.2f}')
        ax.axvline(s.median, color='blue', linestyle='-', linewidth=2, label=f'Medyan: {s.median:.2f}')
        ax.set_xlabel('Değer')
        ax.set_ylabel('Frekans')
        ax.set_title('Dağılım Grafiği')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        _safe_tight_layout(fig, rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # Outliers pages (only for numeric metrics)
    if outliers:
        lower_bound = s.outlier_lower
        upper_bound = s.outlier_upper
        _add_outliers_pages_to_pdf(pdf, outliers, section_title, desc, lower_bound, upper_bound)


def _add_summary_to_pdf(pdf: PdfPages, report: AnalysisReport):
    """Add executive summary page to PDF with system overview statistics."""
    
    # ==================== Gather summary statistics ====================
    node_count = 0
    app_count = 0
    lib_count = 0
    topic_count = 0
    
    # Node stats
    node_stats = None
    node_hierarchy_diversity_stats = None
    if report.node_analysis:
        for analysis in report.node_analysis:
            if analysis.ranked_list:
                node_count = len(analysis.ranked_list)
                if analysis.metric_name == NODE_APPLICATION_COUNT:
                    node_stats = analysis.stats
                elif analysis.metric_name == NODE_DOMAIN_HIERARCHY_DIVERSITY_COUNT:
                    node_hierarchy_diversity_stats = analysis.stats
    
    # App stats
    app_direct_pub_stats = None
    app_direct_sub_stats = None
    app_component_distribution = None
    app_config_item_distribution = None
    topic_variety_per_domain = None
    topic_variety_per_config_item = None
    
    for analysis in report.app_analysis:
        if analysis.ranked_list:
            app_count = max(app_count, len(analysis.ranked_list))
            if analysis.metric_name == APP_DIRECT_PUBLISH_COUNT:
                app_direct_pub_stats = analysis.stats
            elif analysis.metric_name == APP_DIRECT_SUBSCRIBE_COUNT:
                app_direct_sub_stats = analysis.stats
            elif analysis.metric_name == APP_HIERARCHY_COMPONENT_DISTRIBUTION:
                app_component_distribution = analysis.ranked_list
            elif analysis.metric_name == APP_HIERARCHY_CONFIG_ITEM_DISTRIBUTION:
                app_config_item_distribution = analysis.ranked_list
            elif analysis.metric_name == APP_HIERARCHY_DOMAIN_TOPIC_VARIETY_COUNT:
                topic_variety_per_domain = analysis.ranked_list
            elif analysis.metric_name == APP_HIERARCHY_CONFIG_ITEM_TOPIC_VARIETY_COUNT:
                topic_variety_per_config_item = analysis.ranked_list
    
    # Lib stats
    lib_usage_stats = None
    for analysis in report.lib_analysis:
        if analysis.ranked_list:
            lib_count = max(lib_count, len(analysis.ranked_list))
            if analysis.metric_name == LIB_APPLICATION_USAGE_COUNT:
                lib_usage_stats = analysis.stats
                break
    
    # Topic stats
    topic_size_stats = None
    for analysis in report.topic_analysis:
        if analysis.ranked_list:
            topic_count = max(topic_count, len(analysis.ranked_list))
            if analysis.metric_name == TOPIC_SIZE_BYTES:
                topic_size_stats = analysis.stats
    
    # ==================== Single Summary Page ====================
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'SİSTEM ÖZETİ', fontsize=24, ha='center', va='top', 
            fontweight='bold', transform=ax.transAxes, color='#2c3e50')
    ax.text(0.5, 0.90, f'{report.platform_name.upper()} Platformu - Genel Bakış', 
            fontsize=14, ha='center', va='top', transform=ax.transAxes, color='#7f8c8d')
    
    # Left column - System Scale
    left_y = 0.80
    
    ax.text(0.08, left_y, 'SİSTEM ÖLÇEĞİ', fontsize=16, ha='left', va='top', 
            fontweight='bold', transform=ax.transAxes, color='#3498db')
    left_y -= 0.06
    
    scale_items = [
        (f"Toplam Node Sayısı:", node_count, '#2980b9'),
        (f"Toplam Uygulama Sayısı:", app_count, '#27ae60'),
        (f"Toplam Kütüphane Sayısı:", lib_count, '#9b59b6'),
        (f"Toplam Topic Sayısı:", topic_count, '#e67e22'),
    ]
    
    for label, value, color in scale_items:
        ax.text(0.08, left_y, label, fontsize=12, ha='left', va='top', 
                transform=ax.transAxes, color='#34495e')
        ax.text(0.35, left_y, f"{value}", fontsize=14, ha='left', va='top', 
                transform=ax.transAxes, color=color, fontweight='bold')
        left_y -= 0.045
    
    left_y -= 0.03
    
    # Communication Load section
    ax.text(0.08, left_y, 'İLETİŞİM YÜKÜ', fontsize=16, ha='left', va='top', 
            fontweight='bold', transform=ax.transAxes, color='#e74c3c')
    left_y -= 0.06
    
    total_pub = int(app_direct_pub_stats.mean * app_count) if app_direct_pub_stats else 0
    total_sub = int(app_direct_sub_stats.mean * app_count) if app_direct_sub_stats else 0
    avg_pub = app_direct_pub_stats.mean if app_direct_pub_stats else 0
    avg_sub = app_direct_sub_stats.mean if app_direct_sub_stats else 0
    
    comm_items = [
        (f"Toplam Pub İlişkisi:", total_pub),
        (f"Toplam Sub İlişkisi:", total_sub),
        (f"Uygulama Baş. Ort. Pub:", f"{avg_pub:.2f}"),
        (f"Uygulama Baş. Ort. Sub:", f"{avg_sub:.2f}"),
    ]
    
    for label, value in comm_items:
        ax.text(0.08, left_y, label, fontsize=12, ha='left', va='top', 
                transform=ax.transAxes, color='#34495e')
        ax.text(0.35, left_y, f"{value}", fontsize=12, ha='left', va='top', 
                transform=ax.transAxes, color='#c0392b', fontweight='bold')
        left_y -= 0.04
    
    # Right column - Node Load & Variability
    right_y = 0.80
    
    # Node load info
    if node_stats and node_stats.count > 0:
        ax.text(0.55, right_y, 'NODE YÜKÜ DAĞILIMI', fontsize=16, ha='left', va='top', 
                fontweight='bold', transform=ax.transAxes, color='#2980b9')
        right_y -= 0.06
        
        node_items = [
            (f"Minimum:", f"{int(node_stats.min_val)} app"),
            (f"Maksimum:", f"{int(node_stats.max_val)} app"),
            (f"Ortalama:", f"{node_stats.mean:.1f} app"),
            (f"Medyan:", f"{node_stats.median:.1f} app"),
        ]
        
        for label, value in node_items:
            ax.text(0.55, right_y, label, fontsize=12, ha='left', va='top', 
                    transform=ax.transAxes, color='#34495e')
            ax.text(0.72, right_y, value, fontsize=12, ha='left', va='top', 
                    transform=ax.transAxes, color='#2980b9', fontweight='bold')
            right_y -= 0.04
        
        right_y -= 0.03
    
    # Topic size stats
    if topic_size_stats and topic_size_stats.count > 0:
        ax.text(0.55, right_y, 'TOPIC BOYUT İSTATİSTİKLERİ', fontsize=16, ha='left', va='top', 
                fontweight='bold', transform=ax.transAxes, color='#16a085')
        right_y -= 0.06
        
        topic_items = [
            (f"Minimum:", f"{int(topic_size_stats.min_val)} B"),
            (f"Maksimum:", f"{int(topic_size_stats.max_val)} B"),
            (f"Ortalama:", f"{topic_size_stats.mean:.0f} B"),
        ]
        
        for label, value in topic_items:
            ax.text(0.55, right_y, label, fontsize=12, ha='left', va='top', 
                    transform=ax.transAxes, color='#34495e')
            ax.text(0.72, right_y, value, fontsize=12, ha='left', va='top', 
                    transform=ax.transAxes, color='#16a085', fontweight='bold')
            right_y -= 0.04
        
        right_y -= 0.03

    hierarchy_summary_lines = []
    if app_component_distribution:
        top_component = max(app_component_distribution, key=lambda item: item.get("value", 0))
        hierarchy_summary_lines.append(f"En yaygın CSC: {top_component.get('name', '?')} ({top_component.get('value', 0)} uyg.)")
    if app_config_item_distribution:
        top_config_item = max(app_config_item_distribution, key=lambda item: item.get("value", 0))
        hierarchy_summary_lines.append(f"En yaygın CSCI: {top_config_item.get('name', '?')} ({top_config_item.get('value', 0)} uyg.)")
    if topic_variety_per_domain:
        top_domain = max(topic_variety_per_domain, key=lambda item: item.get("value", 0))
        hierarchy_summary_lines.append(f"En geniş CSS topic çeşitliliği: {top_domain.get('name', '?')} ({int(top_domain.get('value', 0))} adet topic)")
    if topic_variety_per_config_item:
        top_config_item_topics = max(topic_variety_per_config_item, key=lambda item: item.get("value", 0))
        hierarchy_summary_lines.append(f"En geniş CSCI topic çeşitliliği: {top_config_item_topics.get('name', '?')} ({int(top_config_item_topics.get('value', 0))} adet topic)")

    if hierarchy_summary_lines:
        ax.text(0.55, right_y, 'SİSTEM HİYERARŞİSİ ÖZETİ', fontsize=16, ha='left', va='top',
                fontweight='bold', transform=ax.transAxes, color='#8e44ad')
        right_y -= 0.06
        for line in hierarchy_summary_lines[:4]:
            ax.text(0.55, right_y, f"• {line}", fontsize=10, ha='left', va='top',
                    transform=ax.transAxes, color='#34495e')
            right_y -= 0.032
    
    # Variability indicators (bottom)
    ax.text(0.08, 0.18, 'DEĞİŞKENLİK GÖSTERGELERİ', fontsize=14, ha='left', va='top', 
            fontweight='bold', transform=ax.transAxes, color='#8e44ad')
    
    var_y = 0.13
    if app_direct_pub_stats and app_direct_pub_stats.mean > 0:
        cv_pub = (app_direct_pub_stats.std / app_direct_pub_stats.mean * 100)
        ax.text(0.08, var_y, f"Publish Dağılımı CV: {cv_pub:.1f}%", fontsize=11, ha='left', va='top', 
                transform=ax.transAxes, color='#7f8c8d')
        var_y -= 0.03
    if app_direct_sub_stats and app_direct_sub_stats.mean > 0:
        cv_sub = (app_direct_sub_stats.std / app_direct_sub_stats.mean * 100)
        ax.text(0.08, var_y, f"Subscribe Dağılımı CV: {cv_sub:.1f}%", fontsize=11, ha='left', va='top', 
                transform=ax.transAxes, color='#7f8c8d')
        var_y -= 0.03
    if lib_usage_stats and lib_usage_stats.mean > 0:
        cv_lib = (lib_usage_stats.std / lib_usage_stats.mean * 100)
        ax.text(0.08, var_y, f"Kütüphane Kullanımı CV: {cv_lib:.1f}%", fontsize=11, ha='left', va='top', 
                transform=ax.transAxes, color='#7f8c8d')
        var_y -= 0.03
    if node_hierarchy_diversity_stats and node_hierarchy_diversity_stats.mean > 0:
        ax.text(0.08, var_y, f"Node başına CSS çeşitliliği Ort.: {node_hierarchy_diversity_stats.mean:.2f}", fontsize=11, ha='left', va='top',
                transform=ax.transAxes, color='#7f8c8d')
    
    ax.text(0.55, 0.13, "(CV = Değişkenlik Katsayısı = Std/Ort × 100)", 
            fontsize=9, ha='left', va='top', transform=ax.transAxes, color='#bdc3c7', style='italic')
    ax.text(0.55, 0.10, "Yüksek CV = Dengesiz dağılım", 
            fontsize=9, ha='left', va='top', transform=ax.transAxes, color='#bdc3c7', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def _add_table_pages_to_pdf(pdf: PdfPages, analysis: ComponentAnalysis, section_title: str, desc: str, is_total_metric: bool):
    """Add paginated table pages to PDF showing all items."""
    if analysis.metric_name == USES_CYCLE_DISTRIBUTION:
        _add_cycle_table_pages_to_pdf(pdf, analysis, section_title, desc)
        return

    items = analysis.ranked_list
    total_items = len(items)
    
    if total_items == 0:
        return
    
    show_entities = getattr(analysis, "show_entities", True)
    
    # Items per page - fewer items if we have lib_breakdown or entities since each item takes more lines
    items_per_page = 25 if (is_total_metric or (analysis.is_categorical and show_entities)) else 45
    
    page_num = 0
    for start_idx in range(0, total_items, items_per_page):
        page_num += 1
        end_idx = min(start_idx + items_per_page, total_items)
        page_items = items[start_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        
        total_pages = (total_items + items_per_page - 1) // items_per_page
        title_text = f'{section_title} - {desc}\nTam Liste (Sayfa {page_num}/{total_pages}, Toplam {total_items} adet)'
        ax.text(0.5, 0.98, title_text, fontsize=12, ha='center', va='top', fontweight='bold', transform=ax.transAxes)
        
        # Build table text
        if is_total_metric:
            header = f"{'#':<5} {'İsim':<40} {'Değer':>10}   Kaynak Dağılımı\n"
            separator = "-" * 100 + "\n"
        elif analysis.is_categorical and show_entities:
            header = f"{'#':<5} {'Değer':<25} {'Sayı':>8}   Üyeler\n"
            separator = "-" * 100 + "\n"
        else:
            header = f"{'#':<5} {'İsim':<55} {'Değer':>10}\n"
            separator = "-" * 75 + "\n"
        
        table_text = header + separator
        
        for i, item in enumerate(page_items, start_idx + 1):
            name = str(item.get("name", item.get("id", "?")))
            version = item.get("version", "")
            display_name = _format_name_with_version(name, version)
            value = item.get("value", 0)
            
            if is_total_metric:
                lib_breakdown = item.get("lib_breakdown", [])
                if lib_breakdown:
                    # First line with name and value
                    first_lib = f"{str(lib_breakdown[0][0])}: {lib_breakdown[0][1]}"
                    table_text += f"{i:<5} {display_name[:38]:<40} {value:>10.0f}   {first_lib}\n"
                    # Additional lines for remaining libraries (indented)
                    for lib_name, count in lib_breakdown[1:]:
                        table_text += f"{'':5} {'':40} {'':10}   {lib_name}: {count}\n"
                else:
                    table_text += f"{i:<5} {display_name[:38]:<40} {value:>10.0f}   -\n"
            elif analysis.is_categorical and show_entities:
                entities = item.get("entities", [])
                if entities:
                    # First line with value and count, and first entity
                    first_entity = str(entities[0])[:50] if entities else "-"
                    table_text += f"{i:<5} {display_name[:23]:<25} {value:>8.0f}   {first_entity}\n"
                    # Additional lines for remaining entities (indented)
                    for entity in entities[1:]:
                        table_text += f"{'':5} {'':25} {'':8}   {str(entity)[:50]}\n"
                else:
                    table_text += f"{i:<5} {display_name[:23]:<25} {value:>8.0f}   -\n"
            else:
                table_text += f"{i:<5} {display_name[:53]:<55} {value:>10.0f}\n"
        
        ax.text(0.02, 0.92, table_text, fontsize=8, ha='left', va='top', 
                transform=ax.transAxes, family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def _add_cycle_table_pages_to_pdf(pdf: PdfPages, analysis: ComponentAnalysis, section_title: str, desc: str) -> None:
    """Add cycle analysis pages with wrapped closed chains for PDF output."""
    lines: List[str] = []
    for index_value, item in enumerate(analysis.ranked_list, 1):
        cycle_name = str(item.get("name", "Kapalı yol bulunamadı"))

        wrapped_cycle = textwrap.wrap(
            f"{index_value}. {cycle_name}",
            width=105,
            subsequent_indent="   ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        for wrapped_line in wrapped_cycle:
            lines.append(wrapped_line)
        lines.append("")

    if not lines:
        return

    lines_per_page = 42
    total_pages = (len(lines) + lines_per_page - 1) // lines_per_page

    for page_index, start_line in enumerate(range(0, len(lines), lines_per_page), 1):
        page_lines = lines[start_line:start_line + lines_per_page]
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')

        title_text = f"{section_title} - {desc}\nTam Liste (Sayfa {page_index}/{total_pages}, Toplam {len(analysis.ranked_list)} döngü)"
        ax.text(0.5, 0.98, title_text, fontsize=12, ha='center', va='top', fontweight='bold', transform=ax.transAxes)
        ax.text(
            0.03,
            0.90,
            "\n".join(page_lines),
            fontsize=8.5,
            ha='left',
            va='top',
            transform=ax.transAxes,
            family='monospace',
        )

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def _add_outliers_pages_to_pdf(pdf: PdfPages, outliers: List[Tuple[str, float]], section_title: str, desc: str, lower_bound: float, upper_bound: float):
    """Add paginated outlier pages to PDF showing all outliers."""
    total_outliers = len(outliers)
    
    if total_outliers == 0:
        return
    
    items_per_page = 45
    
    page_num = 0
    for start_idx in range(0, total_outliers, items_per_page):
        page_num += 1
        end_idx = min(start_idx + items_per_page, total_outliers)
        page_items = outliers[start_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        
        total_pages = (total_outliers + items_per_page - 1) // items_per_page
        title_text = f'{section_title} - {desc}\nOutlier\'lar (Sayfa {page_num}/{total_pages}, Toplam {total_outliers} adet)'
        title_text += f'\nAlt limit: {lower_bound:.2f}, Üst limit: {upper_bound:.2f}'
        ax.text(0.5, 0.98, title_text, fontsize=12, ha='center', va='top', fontweight='bold', transform=ax.transAxes)
        
        # Build outlier table text
        header = f"{'#':<5} {'İsim':<50} {'Değer':>12}\n"
        separator = "-" * 70 + "\n"
        
        table_text = header + separator
        for i, (name, value) in enumerate(page_items, start_idx + 1):
            table_text += f"{i:<5} {str(name)[:48]:<50} {value:>12.0f}\n"
        
        ax.text(0.02, 0.88, table_text, fontsize=8, ha='left', va='top', 
                transform=ax.transAxes, family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def generate_markdown_report(report: AnalysisReport, output_dir: str, *, dds_mask: bool = True) -> str:
    """
    Generate a markdown report with tables and charts.
    
    Args:
        report: AnalysisReport object.
        output_dir: Base output directory (e.g., output/stat).
    
    Returns:
        Path to the created markdown file.
    """
    # Create platform-specific directory: output_dir/platform_name/
    platform_dir = os.path.join(output_dir, report.platform_name)
    images_dir = os.path.join(platform_dir, "images")
    os.makedirs(platform_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    output_file = os.path.join(platform_dir, "report.md")
    
    # Pre-generate extras charts so their titles are available for the TOC
    extras_charts = _generate_extras_charts(report.raw_data, images_dir, dds_mask=dds_mask)
    
    lines = []
    lines.append(f"# İstatistiksel Analiz Raporu: {report.platform_name.upper()}")
    lines.append("")

    # TABLE OF CONTENTS
    lines.append("## İçindekiler")
    lines.append("")
    summary_anchor = _build_report_anchor("summary")
    lines.append(f"- [Sistem Özeti](#{summary_anchor})")
    has_extras = extras_charts or report.extras_analysis
    if has_extras:
        extras_anchor = _build_report_anchor("extras")
        lines.append(f"- [İstatistikler](#{extras_anchor})")
        extras_toc_index = 0
        for chart_index, (_, chart_title, _, _) in enumerate(extras_charts, 1):
            extras_toc_index += 1
            anchor = _build_report_anchor("extras", "chart", str(chart_index))
            lines.append(f"  - [{chart_title}](#{anchor})")
        for analysis_index, analysis in enumerate(report.extras_analysis, 1):
            extras_toc_index += 1
            desc = _display_metric_label(analysis)
            anchor = _build_report_anchor("extras", "analysis", str(analysis_index))
            lines.append(f"  - [{desc}](#{anchor})")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # SYSTEM SUMMARY
    lines.append(f"<a id=\"{summary_anchor}\"></a>")
    lines.extend(_generate_markdown_summary(report))
    
    # İSTATİSTİKLER: Cross-cutting charts and analyses
    if has_extras:
        lines.append(f"<a id=\"{_build_report_anchor('extras')}\"></a>")
        lines.append("## İstatistikler")
        lines.append("")
        for chart_index, (chart_id, chart_title, fig, top_lines) in enumerate(extras_charts, 1):
            img_file = f"extras_{chart_id}.png"
            img_path = os.path.join(images_dir, img_file)
            fig.savefig(img_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            lines.append(f"<a id=\"{_build_report_anchor('extras', 'chart', str(chart_index))}\"></a>")
            lines.append(f"### {chart_title}")
            lines.append("")
            lines.append(f"![{chart_title}](images/{img_file})")
            lines.append("")
            if top_lines:
                lines.append("```")
                lines.extend(top_lines)
                lines.append("```")
                lines.append("")
        for analysis_index, analysis in enumerate(report.extras_analysis, 1):
            anchor = _build_report_anchor("extras", "analysis", str(analysis_index))
            lines.extend(_format_analysis_md(analysis, {}, "İstatistikler", anchor))
    
    lines.append("---")
    lines.append("*Rapor Sonu*")
    
    content = "\n".join(lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    log_info(f"Markdown report generated: {output_file}")
    return output_file


def _generate_markdown_summary(report: AnalysisReport) -> List[str]:
    """Generate system summary section for markdown report."""
    lines = []
    lines.append("## Sistem Özeti")
    lines.append("")
    lines.append(f"> **{report.platform_name.upper()}** Platformu Genel Bakış")
    lines.append("")
    
    # Gather summary statistics
    node_count = 0
    app_count = 0
    lib_count = 0
    topic_count = 0
    
    # Node stats
    node_hierarchy_diversity_stats = None
    if report.node_analysis:
        for analysis in report.node_analysis:
            if analysis.ranked_list:
                node_count = len(analysis.ranked_list)
                if analysis.metric_name == NODE_DOMAIN_HIERARCHY_DIVERSITY_COUNT:
                    node_hierarchy_diversity_stats = analysis.stats
    
    # App stats
    app_direct_pub_stats = None
    app_direct_sub_stats = None
    app_component_distribution = None
    app_config_item_distribution = None
    topic_variety_per_domain = None
    topic_variety_per_config_item = None
    
    for analysis in report.app_analysis:
        if analysis.ranked_list:
            app_count = max(app_count, len(analysis.ranked_list))
            if analysis.metric_name == APP_DIRECT_PUBLISH_COUNT:
                app_direct_pub_stats = analysis.stats
            elif analysis.metric_name == APP_DIRECT_SUBSCRIBE_COUNT:
                app_direct_sub_stats = analysis.stats
            elif analysis.metric_name == APP_HIERARCHY_COMPONENT_DISTRIBUTION:
                app_component_distribution = analysis.ranked_list
            elif analysis.metric_name == APP_HIERARCHY_CONFIG_ITEM_DISTRIBUTION:
                app_config_item_distribution = analysis.ranked_list
            elif analysis.metric_name == APP_HIERARCHY_DOMAIN_TOPIC_VARIETY_COUNT:
                topic_variety_per_domain = analysis.ranked_list
            elif analysis.metric_name == APP_HIERARCHY_CONFIG_ITEM_TOPIC_VARIETY_COUNT:
                topic_variety_per_config_item = analysis.ranked_list
    
    # Lib stats
    lib_usage_stats = None
    for analysis in report.lib_analysis:
        if analysis.ranked_list:
            lib_count = max(lib_count, len(analysis.ranked_list))
            if analysis.metric_name == LIB_APPLICATION_USAGE_COUNT:
                lib_usage_stats = analysis.stats
                break
    
    # Topic stats
    for analysis in report.topic_analysis:
        if analysis.ranked_list:
            topic_count = max(topic_count, len(analysis.ranked_list))
    
    # Section 1: System Scale
    lines.append("### Sistem Ölçeği")
    lines.append("")
    lines.append("| Metrik | Değer |")
    lines.append("|--------|-------|")
    lines.append(f"| Toplam Node Sayısı | **{node_count}** |")
    lines.append(f"| Toplam Uygulama Sayısı | **{app_count}** |")
    lines.append(f"| Toplam Kütüphane Sayısı | **{lib_count}** |")
    lines.append(f"| Toplam Topic Sayısı | **{topic_count}** |")
    lines.append("")
    
    # Section 2: Communication Load
    lines.append("### İletişim Yükü")
    lines.append("")
    
    total_pub = int(app_direct_pub_stats.mean * app_count) if app_direct_pub_stats else 0
    total_sub = int(app_direct_sub_stats.mean * app_count) if app_direct_sub_stats else 0
    avg_pub = app_direct_pub_stats.mean if app_direct_pub_stats else 0
    avg_sub = app_direct_sub_stats.mean if app_direct_sub_stats else 0
    
    lines.append("| Metrik | Değer |")
    lines.append("|--------|-------|")
    lines.append(f"| Toplam Publish İlişkisi | **{total_pub}** |")
    lines.append(f"| Toplam Subscribe İlişkisi | **{total_sub}** |")
    lines.append(f"| Uygulama Başına Ort. Publish | **{avg_pub:.2f}** |")
    lines.append(f"| Uygulama Başına Ort. Subscribe | **{avg_sub:.2f}** |")
    lines.append("")

    lines.append("### Sistem Hiyerarşisi Özeti")
    lines.append("")
    lines.append("| Metrik | Değer |")
    lines.append("|--------|-------|")
    if app_component_distribution:
        top_component = max(app_component_distribution, key=lambda item: item.get("value", 0))
        lines.append(f"| En yaygın CSC | **{top_component.get('name', '?')}** ({top_component.get('value', 0)} uyg.) |")
    if app_config_item_distribution:
        top_config_item = max(app_config_item_distribution, key=lambda item: item.get("value", 0))
        lines.append(f"| En yaygın CSCI | **{top_config_item.get('name', '?')}** ({top_config_item.get('value', 0)} uyg.) |")
    if topic_variety_per_domain:
        top_domain = max(topic_variety_per_domain, key=lambda item: item.get("value", 0))
        lines.append(f"| En yüksek CSS topic çeşitliliği | **{top_domain.get('name', '?')}** ({int(top_domain.get('value', 0))} adet topic) |")
    if topic_variety_per_config_item:
        top_config_item_topics = max(topic_variety_per_config_item, key=lambda item: item.get("value", 0))
        lines.append(f"| En yüksek CSCI topic çeşitliliği | **{top_config_item_topics.get('name', '?')}** ({int(top_config_item_topics.get('value', 0))} adet topic) |")
    if node_hierarchy_diversity_stats and node_hierarchy_diversity_stats.mean > 0:
        lines.append(f"| Node başına ortalama CSS çeşitliliği | **{node_hierarchy_diversity_stats.mean:.2f}** |")
    lines.append("")
    
    # Section 3: Variability Indicators
    lines.append("### Değişkenlik Göstergeleri")
    lines.append("")
    lines.append("| Metrik | CV (%) |")
    lines.append("|--------|--------|")
    
    if app_direct_pub_stats and app_direct_pub_stats.mean > 0:
        cv_pub = (app_direct_pub_stats.std / app_direct_pub_stats.mean * 100)
        lines.append(f"| Publish Dağılımı | **{cv_pub:.1f}%** |")
    
    if app_direct_sub_stats and app_direct_sub_stats.mean > 0:
        cv_sub = (app_direct_sub_stats.std / app_direct_sub_stats.mean * 100)
        lines.append(f"| Subscribe Dağılımı | **{cv_sub:.1f}%** |")
    
    if lib_usage_stats and lib_usage_stats.mean > 0:
        cv_lib = (lib_usage_stats.std / lib_usage_stats.mean * 100)
        lines.append(f"| Kütüphane Kullanımı | **{cv_lib:.1f}%** |")
    lines.append("")
    lines.append("*CV (Değişkenlik Katsayısı) = Std/Ort × 100. Yüksek CV değeri dengesiz dağılımı gösterir.*")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    return lines


def _format_analysis_md(analysis: ComponentAnalysis, charts: dict = None, section_title: str = "", anchor_id: str = "") -> list:
    """Format a single analysis as markdown with charts and outliers."""
    lines = []
    
    # Header
    desc = _display_metric_label(analysis)
    if anchor_id:
        lines.append(f"<a id=\"{anchor_id}\"></a>")
    elif section_title:
        lines.append(f"<a id=\"{_markdown_anchor(f'{section_title}-{desc}')}\"></a>")
    lines.append(f"\n### {desc}")
    lines.append("")
    
    # Charts
    if charts:
        if 'piechart' in charts:
            lines.append(f"![Pasta Grafik]({charts['piechart']})")
            lines.append("")
        if 'barchart' in charts:
            lines.append(f"![Bar Grafik]({charts['barchart']})")
            lines.append("")
        if 'boxplot' in charts:
            lines.append(f"![Kutu Grafiği]({charts['boxplot']})")
            lines.append("")
        if 'histogram' in charts:
            lines.append(f"![Dağılım Grafiği]({charts['histogram']})")
            lines.append("")
    
    is_categorical = analysis.is_categorical
    
    # Outliers - only for numeric metrics
    if not is_categorical:
        s = analysis.stats
        outliers = calculate_outliers(analysis.ranked_list, analysis.stats, is_categorical=analysis.is_categorical)
        if outliers:
            lower_bound = s.outlier_lower
            upper_bound = s.outlier_upper
            lines.append("<details>")
            lines.append(f"<summary>Outlier'lar ({len(outliers)} adet) — Alt limit: {lower_bound:.2f}, Üst limit: {upper_bound:.2f}</summary>")
            lines.append("")
            lines.append("| İsim | Değer |")
            lines.append("|------|-------|")
            for name, value in outliers:
                lines.append(f"| {name} | {value:.0f} |")
            lines.append("")
            lines.append("</details>")
            lines.append("")
    
    return lines


def log_report_summary(report: AnalysisReport) -> None:
    """Log a brief summary of the analysis to the logger."""
    log_info("=" * 60)
    log_info("ANALİZ ÖZETİ")
    log_info("=" * 60)
    
    # Node summary
    if report.node_analysis:
        node_stats = report.node_analysis[0].stats
        log_info(f"Nodes: {node_stats.count} | Mean Apps: {node_stats.mean:.2f} | Std: {node_stats.std:.2f}")
    
    # App summary
    for analysis in report.app_analysis:
        s = analysis.stats
        if analysis.is_categorical:
            log_info(f"Apps ({analysis.metric_name}): {s.total_count} items | Mode: {s.mode} ({s.mode_count}, %{s.mode_percentage:.1f})")
        else:
            log_info(f"Apps ({analysis.metric_name}): {s.count} | Mean: {s.mean:.2f} | Std: {s.std:.2f}")
    
    # Lib summary
    for analysis in report.lib_analysis:
        s = analysis.stats
        if analysis.is_categorical:
            log_info(f"Libs ({analysis.metric_name}): {s.total_count} items | Mode: {s.mode} ({s.mode_count}, %{s.mode_percentage:.1f})")
        else:
            log_info(f"Libs ({analysis.metric_name}): {s.count} | Mean: {s.mean:.2f} | Std: {s.std:.2f}")
    
    # Topic summary
    for analysis in report.topic_analysis:
        s = analysis.stats
        if analysis.is_categorical:
            log_info(f"Topics ({analysis.metric_name}): {s.total_count} items | Mode: {s.mode} ({s.mode_count}, %{s.mode_percentage:.1f})")
        else:
            log_info(f"Topics ({analysis.metric_name}): {s.count} | Mean: {s.mean:.2f} | Std: {s.std:.2f}")
    
    log_info("=" * 60)