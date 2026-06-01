"""
Yapısal analiz rapor üreteci.

Üretir:
- Markdown raporu (yapısal metrikler, örüntüler ve anomali skorları)

Not: JSON çıktısı artık aggregated JSON içine enjekte edilir (service.py).
"""

import os
from typing import Dict, List, Any

from common.logger import log_info

from .metrics import AllMetrics
from .patterns import PatternResults
from .scoring import ScoringResults


# ---------------------------------------------------------------------------
# Örüntü açıklamaları (Türkçe)
# ---------------------------------------------------------------------------

PATTERN_DESCRIPTIONS = {
    # Uygulama düzeyi
    "WR": ("Geniş Erişim (Wide Reach)",
           "R(a)↑ ∧ AMP(a)↑",
           "Yüksek erişim ve yüksek amplifikasyon bir arada gözlemlendi."),
    "RS": ("Rol Çarpıklığı (Role Skew)",
           "RA(a)↑ ∨ RA(a)↓",
           "Yayıncı veya abone rolüne belirgin yoğunlaşma."),
    "CS": ("Bağlam Yayılımı (Context Spread)",
           "TC(a)↑",
           "Çok sayıda farklı konu  bağlamıyla etkileşim."),
    "SD": ("Paylaşımlı Bağımlılık Maruziyeti (Shared Dep. Exposure)",
           "LE(a)↑",
           "Çok sayıda paylaşımlı kütüphaneye bağımlılık."),
    # Konu  düzeyi
    "CB": ("İletişim Omurgası (Communication Backbone)",
           "C(t)↑ ∧ I(t)↓",
           "Yüksek kapsam ve düşük dengesizlik — merkezi iletişim rolü."),
    "DC": ("Yönsel Yoğunlaşma (Directional Concentration)",
           "I(t)↑",
           "Yayıncı veya abone yönünde çarpık yoğunlaşma."),
    "PA": ("Çevresel Toplayıcı (Peripheral Aggregator)",
           "LCR(t)↑",
           "Düşük bağlantı çeşitliliğine sahip uygulamaların toplanması."),
    # Düğüm (node) düzeyi
    "IH": ("Etkileşim Sıcak Noktası (Interaction Hotspot)",
           "ND(n)↑ ∧ NID(n)↑",
           "Yoğun düğüm-içi etkileşimle birlikte çok sayıda uygulama."),
    # Kütüphane düzeyi
    "WUL": ("Yaygın Kullanılan Kütüphane (Widely Used Library)",
            "LC(l)↑",
            "Çok sayıda uygulama tarafından kullanılıyor."),
    "CL": ("Yoğunlaşmış Kütüphane (Concentrated Library)",
           "LCon(l)↑",
           "Kullanım belirli çalıştırma düğümlerinde yoğunlaşmış."),
}

METRIC_DESCRIPTIONS = {
    # Uygulama
    "R": "Erişim (Reach)",
    "AMP": "Amplifikasyon (Amplification)",
    "RA": "Rol Asimetrisi (Role Asymmetry)",
    "TC": "Konu Bağlam Çeşitliliği (Topic Context Diversity)",
    "LE": "Kütüphane Maruziyeti (Library Exposure)",
    # Konu 
    "C": "Kapsam (Coverage)",
    "I": "Dengesizlik (Imbalance)",
    "PS": "Fiziksel Yayılım (Physical Spread)",
    "LCR": "Düşük Bağlantı Oranı (Low Connectivity Ratio)",
    # Düğüm (node)
    "ND": "Düğüm Yoğunluğu (Node Density)",
    "NID": "Düğüm Etkileşim Yoğunluğu (Node Interaction Density)",
    # Kütüphane
    "LC": "Kütüphane Kapsamı (Library Coverage)",
    "LCon": "Kütüphane Yoğunlaşması (Library Concentration)",
}


def generate_markdown_report(
    metrics: AllMetrics,
    patterns: PatternResults,
    scores: ScoringResults,
    output_dir: str,
    platform_name: str,
) -> str:
    """
    Yapısal analiz için Markdown raporu üretir.

    Args:
        metrics: Hesaplanmış yapısal metrikler.
        patterns: Tespit edilen örüntüler.
        scores: Birleşik anomali skorları.
        output_dir: Çıktı dizini.
        platform_name: Platform adı.

    Returns:
        Üretilen markdown dosyasının yolu.
    """
    platform_dir = os.path.join(output_dir, platform_name)
    os.makedirs(platform_dir, exist_ok=True)

    md_path = os.path.join(platform_dir, f"{platform_name}_structural.md")
    md_content = _build_markdown(metrics, patterns, scores, platform_name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    log_info(f"Markdown raporu yazıldı: {md_path}")

    return md_path


def log_report_summary(
    metrics: AllMetrics,
    patterns: PatternResults,
    scores: ScoringResults,
) -> None:
    """Yapısal analiz sonuçlarının özetini loglar."""
    log_info("-" * 50)
    log_info("YAPISAL ANALİZ ÖZETİ")
    log_info("-" * 50)
    log_info(f"  Analiz edilen uygulama: {len(metrics.applications)}")
    log_info(f"  Analiz edilen konu:     {len(metrics.topics)}")
    log_info(f"  Analiz edilen düğüm:    {len(metrics.nodes)}")
    log_info(f"  Analiz edilen kütüphane: {len(metrics.libraries)}")

    # Örüntü sayıları
    for level_name, pdict in [
        ("Uygulama", patterns.app_patterns),
        ("Konu", patterns.topic_patterns),
        ("Düğüm", patterns.node_patterns),
        ("Kütüphane", patterns.lib_patterns),
    ]:
        total = sum(len(v) for v in pdict.values())
        details = ", ".join(f"{k}={len(v)}" for k, v in pdict.items() if len(v) > 0)
        log_info(f"  {level_name} örüntüleri: toplam {total} ({details})")

    # En yüksek skorlu bileşenler
    for level_name, score_list in [
        ("Uygulama", scores.applications),
        ("Konu", scores.topics),
        ("Düğüm", scores.nodes),
        ("Kütüphane", scores.libraries),
    ]:
        if score_list:
            top = score_list[0]
            log_info(
                f"  En yüksek {level_name}: {top.name} "
                f"(skor={top.total_score:.4f}, örüntüler={top.matched_patterns})"
            )

    log_info("-" * 50)


# ---------------------------------------------------------------------------
# Markdown oluşturucu
# ---------------------------------------------------------------------------

def _build_markdown(
    metrics: AllMetrics,
    patterns: PatternResults,
    scores: ScoringResults,
    platform_name: str,
) -> str:
    lines = []
    _h = lines.append

    _h(f"# Yapısal Analiz Raporu — {platform_name}\n")
    _h("Bu rapor yapısal metriklerin hesaplanması, göreli çeyreklik yorumu, ")
    _h("yapısal anomali örüntüsü tespiti ve birleşik anomali skorlaması ")
    _h("adımlarını kapsamaktadır.\n")

    # ---- Parametreler ----
    _h("## Parametreler\n")
    _h(f"| Parametre | Değer |")
    _h(f"|-----------|-------|")
    for k, v in scores.parameters.items():
        _h(f"| {k} | {v} |")
    _h("")

    # ================================================================
    # METRİKLER BÖLÜMÜ
    # ================================================================
    _h("---\n")
    _h("## 1. Yapısal Metrikler\n")

    # -- Uygulama Metrikleri --
    _h("### 1.1 Uygulama Düzeyi Metrikleri\n")
    _h("| Uygulama | R | AMP | RA | TC | LE |")
    _h("|----------|---|-----|----|----|-----|")
    for a in sorted(metrics.applications, key=lambda x: x.id):
        _h(f"| {a.name} | {a.reach} | {a.amplification:.4f} | "
           f"{a.role_asymmetry:.4f} | {a.topic_context_diversity} | "
           f"{a.library_exposure} |")
    _h("")

    # Çeyreklikler
    _h("**Çeyreklik Sınırları:**\n")
    _h("| Metrik | Q1 | Q3 | Min | Maks | Dejenere |")
    _h("|--------|----|----|-----|------|----------|")
    for mk in ("R", "AMP", "RA", "TC", "LE"):
        q = patterns.app_quartiles.get(mk)
        if q:
            _h(f"| {mk} | {q.q1:.4f} | {q.q3:.4f} | "
               f"{q.min_val:.4f} | {q.max_val:.4f} | "
               f"{'Evet' if q.degenerate else 'Hayır'} |")
    _h("")

    # -- Konu  Metrikleri --
    _h("### 1.2 Konu  Düzeyi Metrikleri\n")
    _h("| Konu | C | I | PS | LCR |")
    _h("|------|---|---|----|-----|")
    for t in sorted(metrics.topics, key=lambda x: x.id):
        _h(f"| {t.name} | {t.coverage} | {t.imbalance:.4f} | "
           f"{t.physical_spread} | {t.low_connectivity_ratio:.4f} |")
    _h("")

    _h("**Çeyreklik Sınırları:**\n")
    _h("| Metrik | Q1 | Q3 | Min | Maks | Dejenere |")
    _h("|--------|----|----|-----|------|----------|")
    for mk in ("C", "I", "PS", "LCR"):
        q = patterns.topic_quartiles.get(mk)
        if q:
            _h(f"| {mk} | {q.q1:.4f} | {q.q3:.4f} | "
               f"{q.min_val:.4f} | {q.max_val:.4f} | "
               f"{'Evet' if q.degenerate else 'Hayır'} |")
    _h("")

    # -- Düğüm (Node) Metrikleri --
    _h("### 1.3 Düğüm (Node) Düzeyi Metrikleri\n")
    _h("| Düğüm | ND | NID |")
    _h("|-------|----|-----|")
    for n in sorted(metrics.nodes, key=lambda x: x.id):
        _h(f"| {n.name} | {n.density} | {n.interaction_density} |")
    _h("")

    _h("**Çeyreklik Sınırları:**\n")
    _h("| Metrik | Q1 | Q3 | Min | Maks | Dejenere |")
    _h("|--------|----|----|-----|------|----------|")
    for mk in ("ND", "NID"):
        q = patterns.node_quartiles.get(mk)
        if q:
            _h(f"| {mk} | {q.q1:.4f} | {q.q3:.4f} | "
               f"{q.min_val:.4f} | {q.max_val:.4f} | "
               f"{'Evet' if q.degenerate else 'Hayır'} |")
    _h("")

    # -- Kütüphane Metrikleri --
    _h("### 1.4 Kütüphane Düzeyi Metrikleri\n")
    _h("| Kütüphane | LC | LCon |")
    _h("|-----------|----|----- |")
    for l in sorted(metrics.libraries, key=lambda x: x.id):
        _h(f"| {l.name} | {l.coverage} | {l.concentration} |")
    _h("")

    _h("**Çeyreklik Sınırları:**\n")
    _h("| Metrik | Q1 | Q3 | Min | Maks | Dejenere |")
    _h("|--------|----|----|-----|------|----------|")
    for mk in ("LC", "LCon"):
        q = patterns.lib_quartiles.get(mk)
        if q:
            _h(f"| {mk} | {q.q1:.4f} | {q.q3:.4f} | "
               f"{q.min_val:.4f} | {q.max_val:.4f} | "
               f"{'Evet' if q.degenerate else 'Hayır'} |")
    _h("")

    # ================================================================
    # ÖRÜNTÜLER BÖLÜMÜ
    # ================================================================
    _h("---\n")
    _h("## 2. Yapısal Anomali Örüntüleri\n")

    for level_name, pdict in [
        ("Uygulama Düzeyi", patterns.app_patterns),
        ("Konu  Düzeyi", patterns.topic_patterns),
        ("Düğüm (Node) Düzeyi", patterns.node_patterns),
        ("Kütüphane Düzeyi", patterns.lib_patterns),
    ]:
        _h(f"### {level_name} Örüntüleri\n")
        for pname, matches in pdict.items():
            desc_tuple = PATTERN_DESCRIPTIONS.get(pname, (pname, "", ""))
            full_name, formula, explanation = desc_tuple

            _h(f"#### {full_name}\n")
            _h(f"**Koşul:** `{formula}`\n")
            _h(f"{explanation}\n")

            if matches:
                _h(f"**Eşleşen bileşenler ({len(matches)}):**\n")
                for m in matches:
                    _h(f"- {m.name} (`{m.id}`)")
            else:
                _h("*Bu örüntüyle eşleşen bileşen bulunamadı.*")
            _h("")

    # ================================================================
    # SKORLAR BÖLÜMÜ
    # ================================================================
    _h("---\n")
    _h("## 3. Birleşik Anomali Skorları\n")
    _h("Skorlar; örüntü tabanlı aykırılık skoru (OS^P) ile tek boyutlu ")
    _h("aykırılık katkısının (UNI) birleşimiyle hesaplanmaktadır.\n")

    for level_name, score_list in [
        ("Uygulama", scores.applications),
        ("Konu ", scores.topics),
        ("Düğüm (Node)", scores.nodes),
        ("Kütüphane", scores.libraries),
    ]:
        _h(f"### {level_name} Skorları\n")
        if not score_list:
            _h("*Veri bulunamadı.*\n")
            continue

        _h("| Sıra | Bileşen | OS^P | UNI | Skor | Örüntüler |")
        _h("|------|---------|------|-----|------|-----------|")
        for rank, s in enumerate(score_list, 1):
            pats = ", ".join(s.matched_patterns) if s.matched_patterns else "—"
            _h(f"| {rank} | {s.name} | {s.pattern_score:.4f} | "
               f"{s.uni_score:.4f} | {s.total_score:.4f} | {pats} |")
        _h("")

    return "\n".join(lines)
