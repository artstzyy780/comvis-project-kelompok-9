"""Generator for Proposal Defense Presentation Deck (Sidang Proposal UTS).

This module builds a professional 16:9 widescreen PowerPoint presentation (.pptx)
for Computer Vision Project: Nighttime Pedestrian Detection using YOLOv8
and Adaptive Illumination Augmentation.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# ==============================================================================
# DESIGN SYSTEM CONSTANTS
# ==============================================================================
SLIDE_WIDTH_INCHES: float = 13.333
SLIDE_HEIGHT_INCHES: float = 7.5

FONT_FAMILY_TITLE: str = "Segoe UI"
FONT_FAMILY_BODY: str = "Segoe UI"

# Color Palette (Dark Executive Tech Theme)
COLOR_BG_DARK: RGBColor = RGBColor(15, 23, 42)          # #0F172A (Deep Slate)
COLOR_CARD_BG: RGBColor = RGBColor(30, 41, 59)          # #1E293B (Card Slate)
COLOR_CARD_BORDER: RGBColor = RGBColor(51, 65, 85)      # #334155 (Subtle Border)
COLOR_CARD_HIGHLIGHT: RGBColor = RGBColor(39, 53, 76)   # #27354C (Light Card)

COLOR_TEXT_WHITE: RGBColor = RGBColor(248, 250, 252)    # #F8FAFC (Primary Text)
COLOR_TEXT_MUTED: RGBColor = RGBColor(148, 163, 184)    # #94A3B8 (Secondary Text)
COLOR_TEXT_DIM: RGBColor = RGBColor(100, 116, 139)      # #64748B (Dimmed Text)

COLOR_CYAN_ACCENT: RGBColor = RGBColor(56, 189, 248)    # #38BDF8 (Primary Accent)
COLOR_EMERALD_GREEN: RGBColor = RGBColor(52, 211, 153)  # #34D399 (Success/Metric)
COLOR_AMBER_ORANGE: RGBColor = RGBColor(251, 191, 36)   # #FBBF24 (Warning/Urgency)
COLOR_ROSE_RED: RGBColor = RGBColor(251, 113, 133)      # #FB7185 (Problem/Danger)
COLOR_INDIGO_BLUE: RGBColor = RGBColor(129, 140, 248)   # #818CF8 (Secondary Accent)


def apply_slide_background(slide) -> None:
    """Sets a solid dark background for a slide.

    Args:
        slide: The pptx slide object.
    """
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG_DARK


def add_header(
    slide,
    category: str,
    title: str,
    subtitle: Optional[str] = None,
) -> None:
    """Adds a standardized category badge, main title, and optional subtitle.

    Args:
        slide: The pptx slide object.
        category: Section tracking breadcrumb or badge text.
        title: Main punchy slide title.
        subtitle: Optional explanatory subtitle.
    """
    # Category / Breadcrumb
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(11.7), Inches(0.35))
    cat_tf = cat_box.text_frame
    cat_tf.word_wrap = True
    cat_tf.margin_left = cat_tf.margin_top = cat_tf.margin_right = cat_tf.margin_bottom = 0
    cat_p = cat_tf.paragraphs[0]
    cat_p.text = category.upper()
    cat_p.font.name = FONT_FAMILY_BODY
    cat_p.font.size = Pt(10)
    cat_p.font.bold = True
    cat_p.font.color.rgb = COLOR_CYAN_ACCENT

    # Main Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.78), Inches(11.7), Inches(0.65))
    title_tf = title_box.text_frame
    title_tf.word_wrap = True
    title_tf.margin_left = title_tf.margin_top = title_tf.margin_right = title_tf.margin_bottom = 0
    title_p = title_tf.paragraphs[0]
    title_p.text = title
    title_p.font.name = FONT_FAMILY_TITLE
    title_p.font.size = Pt(22)
    title_p.font.bold = True
    title_p.font.color.rgb = COLOR_TEXT_WHITE

    # Subtitle (if present)
    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.42), Inches(11.7), Inches(0.45))
        sub_tf = sub_box.text_frame
        sub_tf.word_wrap = True
        sub_tf.margin_left = sub_tf.margin_top = sub_tf.margin_right = sub_tf.margin_bottom = 0
        sub_p = sub_tf.paragraphs[0]
        sub_p.text = subtitle
        sub_p.font.name = FONT_FAMILY_BODY
        sub_p.font.size = Pt(12)
        sub_p.font.color.rgb = COLOR_TEXT_MUTED


def add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    items: List[str],
    accent_color: RGBColor = COLOR_CYAN_ACCENT,
    tag: Optional[str] = None,
    bg_color: RGBColor = COLOR_CARD_BG,
) -> None:
    """Adds a styled rectangular card container with items.

    Args:
        slide: The pptx slide object.
        left: Left offset in inches.
        top: Top offset in inches.
        width: Card width in inches.
        height: Card height in inches.
        title: Card header title.
        items: List of descriptive string points.
        accent_color: Accent color for the title or tag.
        tag: Optional small badge tag on the card.
        bg_color: Background color of the card shape.
    """
    # Card Shape
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = COLOR_CARD_BORDER
    shape.line.width = Pt(1)

    # Content Text Frame
    tb = slide.shapes.add_textbox(
        Inches(left + 0.25),
        Inches(top + 0.22),
        Inches(width - 0.5),
        Inches(height - 0.44),
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    first_p = tf.paragraphs[0]
    if tag:
        first_p.text = tag.upper()
        first_p.font.name = FONT_FAMILY_BODY
        first_p.font.size = Pt(9)
        first_p.font.bold = True
        first_p.font.color.rgb = accent_color
        first_p.space_after = Pt(3)

        title_p = tf.add_paragraph()
    else:
        title_p = first_p

    title_p.text = title
    title_p.font.name = FONT_FAMILY_TITLE
    title_p.font.size = Pt(14)
    title_p.font.bold = True
    title_p.font.color.rgb = COLOR_TEXT_WHITE
    title_p.space_after = Pt(8)

    for item in items:
        p = tf.add_paragraph()
        p.text = f"•  {item}"
        p.font.name = FONT_FAMILY_BODY
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_after = Pt(5)


def add_metric_chip(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    value: str,
    label: str,
    color: RGBColor = COLOR_EMERALD_GREEN,
) -> None:
    """Adds a small visual metric indicator box.

    Args:
        slide: The pptx slide object.
        left: Left offset in inches.
        top: Top offset in inches.
        width: Chip width in inches.
        height: Chip height in inches.
        value: Large stat number or string.
        label: Descriptive label underneath.
        color: Primary color for the number.
    """
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLOR_CARD_BG
    shape.line.color.rgb = COLOR_CARD_BORDER
    shape.line.width = Pt(1)

    tb = slide.shapes.add_textbox(
        Inches(left + 0.15),
        Inches(top + 0.12),
        Inches(width - 0.3),
        Inches(height - 0.24),
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    val_p = tf.paragraphs[0]
    val_p.text = value
    val_p.font.name = FONT_FAMILY_TITLE
    val_p.font.size = Pt(20)
    val_p.font.bold = True
    val_p.font.color.rgb = color
    val_p.alignment = PP_ALIGN.CENTER

    lbl_p = tf.add_paragraph()
    lbl_p.text = label
    lbl_p.font.name = FONT_FAMILY_BODY
    lbl_p.font.size = Pt(9)
    lbl_p.font.color.rgb = COLOR_TEXT_MUTED
    lbl_p.alignment = PP_ALIGN.CENTER


def add_footer(slide, current: int, total: int) -> None:
    """Adds standard bottom footer with team info and slide number.

    Args:
        slide: The pptx slide object.
        current: Current slide index.
        total: Total slides in deck.
    """
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.3))
    tf = tb.text_frame
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = (
        f"Sidang Proposal UTS Visi Komputer — Kelompok 9 (Deteksi Pedestrian Low-Light)  |  "
        f"Slide {current} of {total}"
    )
    p.font.name = FONT_FAMILY_BODY
    p.font.size = Pt(9)
    p.font.color.rgb = COLOR_TEXT_DIM


# ==============================================================================
# SLIDE BUILDERS
# ==============================================================================

def build_slide_1_cover(prs: Presentation) -> None:
    """Builds Slide 1: Cover / Title Slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_slide_background(slide)

    # Accent Top Badge
    badge = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.8),
        Inches(1.2),
        Inches(4.8),
        Inches(0.4),
    )
    badge.fill.solid()
    badge.fill.fore_color.rgb = RGBColor(14, 116, 144)  # Cyan 700
    badge.line.fill.background()
    b_tf = badge.text_frame
    b_tf.margin_left = b_tf.margin_right = b_tf.margin_top = b_tf.margin_bottom = 0
    bp = b_tf.paragraphs[0]
    bp.text = "SIDANG PROPOSAL UTS — COMPUTER VISION 2026/2027"
    bp.font.name = FONT_FAMILY_BODY
    bp.font.size = Pt(10)
    bp.font.bold = True
    bp.font.color.rgb = COLOR_TEXT_WHITE
    bp.alignment = PP_ALIGN.CENTER

    # Main Project Title
    tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(1.85), Inches(11.7), Inches(1.8))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = "Peningkatan Recall Deteksi Pejalan Kaki\npada Kondisi Minim Cahaya"
    p_title.font.name = FONT_FAMILY_TITLE
    p_title.font.size = Pt(32)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT_WHITE
    p_title.space_after = Pt(10)

    p_sub = tf_title.add_paragraph()
    p_sub.text = (
        "Menggunakan YOLOv8-Nano dan Prapemrosesan Augmentasi Pencahayaan Adaptif (CLAHE & Gamma Correction)"
    )
    p_sub.font.name = FONT_FAMILY_BODY
    p_sub.font.size = Pt(15)
    p_sub.font.color.rgb = COLOR_CYAN_ACCENT

    # Team Box (Card)
    card_team = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.8),
        Inches(4.1),
        Inches(11.7),
        Inches(2.3),
    )
    card_team.fill.solid()
    card_team.fill.fore_color.rgb = COLOR_CARD_BG
    card_team.line.color.rgb = COLOR_CARD_BORDER
    card_team.line.width = Pt(1)

    tb_tcontent = slide.shapes.add_textbox(Inches(1.1), Inches(4.35), Inches(11.1), Inches(1.8))
    tf_tcontent = tb_tcontent.text_frame
    tf_tcontent.word_wrap = True
    tf_tcontent.margin_left = tf_tcontent.margin_top = tf_tcontent.margin_right = tf_tcontent.margin_bottom = 0

    p_grp = tf_tcontent.paragraphs[0]
    p_grp.text = "KELOMPOK 9  |  Mata Kuliah: Visi Komputer (IF0PP006) — Kelas 5PITA"
    p_grp.font.name = FONT_FAMILY_TITLE
    p_grp.font.size = Pt(13)
    p_grp.font.bold = True
    p_grp.font.color.rgb = COLOR_EMERALD_GREEN
    p_grp.space_after = Pt(10)

    # Member 1
    p_m1 = tf_tcontent.add_paragraph()
    p_m1.text = "•  Krisantus Amka Ginting  —  Data & Preprocessing Lead"
    p_m1.font.name = FONT_FAMILY_BODY
    p_m1.font.size = Pt(12)
    p_m1.font.bold = True
    p_m1.font.color.rgb = COLOR_TEXT_WHITE
    p_m1_desc = tf_tcontent.add_paragraph()
    p_m1_desc.text = "    Audit integritas data, eksplorasi class imbalance, pipeline CLAHE kanal-L, dan isolasi data leakage."
    p_m1_desc.font.name = FONT_FAMILY_BODY
    p_m1_desc.font.size = Pt(10)
    p_m1_desc.font.color.rgb = COLOR_TEXT_MUTED
    p_m1_desc.space_after = Pt(6)

    # Member 2
    p_m2 = tf_tcontent.add_paragraph()
    p_m2.text = "•  Linardi Halim  —  Documentation & Reporting Lead"
    p_m2.font.name = FONT_FAMILY_BODY
    p_m2.font.size = Pt(12)
    p_m2.font.bold = True
    p_m2.font.color.rgb = COLOR_TEXT_WHITE
    p_m2_desc = tf_tcontent.add_paragraph()
    p_m2_desc.text = "    Penyusunan proposal dan laporan teknis, naskah presentasi, serta rekap tracker dan logbook."
    p_m2_desc.font.name = FONT_FAMILY_BODY
    p_m2_desc.font.size = Pt(10)
    p_m2_desc.font.color.rgb = COLOR_TEXT_MUTED

    # Member 3
    p_m3 = tf_tcontent.add_paragraph()
    p_m3.text = "•  Kristofer Ryan Giggs Eka Saputra  —  Model & Training Lead"
    p_m3.font.name = FONT_FAMILY_BODY
    p_m3.font.size = Pt(12)
    p_m3.font.bold = True
    p_m3.font.color.rgb = COLOR_TEXT_WHITE
    p_m3_desc = tf_tcontent.add_paragraph()
    p_m3_desc.text = "    Pelatihan model YOLOv8-Nano, eksperimen terkontrol komparatif, evaluasi Recall, dan benchmark FPS."
    p_m3_desc.font.name = FONT_FAMILY_BODY
    p_m3_desc.font.size = Pt(10)
    p_m3_desc.font.color.rgb = COLOR_TEXT_MUTED

    # Member 4
    p_m4 = tf_tcontent.add_paragraph()
    p_m4.text = "•  Leonard Franscois  —  Demo & Deployment Lead"
    p_m4.font.name = FONT_FAMILY_BODY
    p_m4.font.size = Pt(12)
    p_m4.font.bold = True
    p_m4.font.color.rgb = COLOR_TEXT_WHITE
    p_m4_desc = tf_tcontent.add_paragraph()
    p_m4_desc.text = "    Prototype demo inferensi, dokumentasi repository, dan organisational video cadangan."
    p_m4_desc.font.name = FONT_FAMILY_BODY
    p_m4_desc.font.size = Pt(10)
    p_m4_desc.font.color.rgb = COLOR_TEXT_MUTED


def build_slide_2_urgency(prs: Presentation) -> None:
    """Builds Slide 2: Urgensi & Latar Belakang (The 'Why')."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_slide_background(slide)
    add_header(
        slide,
        category="01. LATAR BELAKANG & URGENSI MASALAH",
        title="Mengapa Deteksi Pejalan Kaki Malam Hari Sangat Kritis?",
        subtitle="Keselamatan pejalan kaki di malam hari adalah isu berisiko tinggi (Life-Critical Task) bagi ADAS dan Smart Mobility.",
    )

    # Top Hero Banner / Highlight
    banner = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.8),
        Inches(2.0),
        Inches(11.733),
        Inches(1.0),
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = RGBColor(69, 10, 10)  # Dark Wine Red
    banner.line.color.rgb = COLOR_ROSE_RED
    banner.line.width = Pt(1)

    b_tb = slide.shapes.add_textbox(Inches(1.05), Inches(2.15), Inches(11.2), Inches(0.7))
    b_tf = b_tb.text_frame
    b_tf.word_wrap = True
    b_tf.margin_left = b_tf.margin_top = b_tf.margin_right = b_tf.margin_bottom = 0
    bp1 = b_tf.paragraphs[0]
    bp1.text = "FAKTA KESELAMATAN JALAN: 76% KECELAKAAN FATAL TERJADI SAAT GELAP (SUMBER: NHTSA & IIHS)"
    bp1.font.name = FONT_FAMILY_TITLE
    bp1.font.size = Pt(12)
    bp1.font.bold = True
    bp1.font.color.rgb = COLOR_ROSE_RED
    bp2 = b_tf.add_paragraph()
    bp2.text = (
        "Berdasarkan laporan resmi NHTSA Traffic Safety Facts & IIHS FARS, ~76% fatalitas pejalan kaki "
        "terjadi pada kondisi gelap/malam hari akibat keterbatasan visibilitas pengemudi dan kegagalan sensor optik standar."
    )
    bp2.font.name = FONT_FAMILY_BODY
    bp2.font.size = Pt(11)
    bp2.font.color.rgb = COLOR_TEXT_WHITE

    # 3 Pillar Cards
    card_width = 3.75
    gap = 0.24
    top_pos = 3.25
    height_pos = 3.5

    add_card(
        slide,
        left=0.8,
        top=top_pos,
        width=card_width,
        height=height_pos,
        title="Vulnerable Road Users",
        items=[
            "Pejalan kaki tidak memiliki proteksi fisik layaknya penumpang kendaraan bermotor.",
            "Di malam hari, pakaian gelap dan pencahayaan lampu jalan minim membuat pejalan kaki hampir tak terlihat.",
            "Sistem deteksi dini adalah satu-satunya pertahanan pencegah tabrakan di sistem ADAS modern.",
        ],
        accent_color=COLOR_AMBER_ORANGE,
        tag="Faktor Keselamatan Jiwa",
    )

    add_card(
        slide,
        left=0.8 + card_width + gap,
        top=top_pos,
        width=card_width,
        height=height_pos,
        title="Keterbatasan Kamera Optik",
        items=[
            "Kamera RGB standar mengalami penurunan SNR (Signal-to-Noise Ratio) ekstrem pada intensitas foton rendah.",
            "Solusi sensor aktif seperti LiDAR atau Thermal IR berbiaya sangat tinggi ($$$) untuk kendaraan massal.",
            "Diperlukan solusi berbasis software visi komputer cerdas pada kamera RGB monokuler yang terjangkau.",
        ],
        accent_color=COLOR_CYAN_ACCENT,
        tag="Tantangan Sensorik",
    )

    add_card(
        slide,
        left=0.8 + 2 * (card_width + gap),
        top=top_pos,
        width=card_width,
        height=height_pos,
        title="Tuntutan Real-Time Edge",
        items=[
            "Kendaraan melaju 40-60 km/jam membutuhkan respon deteksi dalam hitungan milidetik (< 50 ms).",
            "Sistem harus mampu berjalan lancar pada perangkat komputasi onboard CPU standar tanpa latensi tinggi.",
            "Akurasi tinggi tidak ada artinya jika komputasi lambat dan tidak responsif di jalan raya.",
        ],
        accent_color=COLOR_EMERALD_GREEN,
        tag="Kebutuhan Operasional",
    )

    add_footer(slide, 2, 11)


def build_slide_3_problems(prs: Presentation) -> None:
    """Builds Slide 3: Rumusan Masalah & Gap Analisis."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_slide_background(slide)
    add_header(
        slide,
        category="02. PERMASALAHAN & ANALISIS KESENJANGAN",
        title="3 Hambatan Utama & Kesenjangan Solusi yang Ada",
        subtitle="Mengapa solusi visi komputer standar saat ini gagal dioperasikan pada kondisi malam hari?",
    )

    # 3 Problem Cards
    col_w = 3.75
    col_gap = 0.24
    top_pos = 1.95
    h_pos = 2.4

    add_card(
        slide,
        left=0.8,
        top=top_pos,
        width=col_w,
        height=h_pos,
        title="1. Kontras Ekstrem & Siluet Gelap",
        items=[
            "Intensitas piksel tubuh pejalan kaki sangat dekat dengan nilai piksel aspal dan latar malam.",
            "Fitur tekstur tepi (edges) dan gradien bentuk tubuh hilang tertelan kegelapan.",
            "Menghasilkan False Negative fatal (objek terlewatkan).",
        ],
        accent_color=COLOR_ROSE_RED,
        tag="Problem Optik",
    )

    add_card(
        slide,
        left=0.8 + col_w + col_gap,
        top=top_pos,
        width=col_w,
        height=h_pos,
        title="2. Derau Sensor & Silau Lampu",
        items=[
            "Kenaikan gain ISO kamera memunculkan grain noise yang mengaburkan kontur objek.",
            "Lampu sorot mobil lawan arah (headlight glare) memicu saturasi piksel over-exposure.",
            "Rentang dinamis menjadi sangat liar dan tak teratur.",
        ],
        accent_color=COLOR_AMBER_ORANGE,
        tag="Problem Lingkungan",
    )

    add_card(
        slide,
        left=0.8 + 2 * (col_w + col_gap),
        top=top_pos,
        width=col_w,
        height=h_pos,
        title="3. Bias Domain Dataset Standar",
        items=[
            "Model detektor populer (YOLO pre-trained COCO) dilatih 90%+ pada foto siang hari.",
            "Fitur bobot neural network bias pada warna dan pencahayaan terang.",
            "Akurasi drop drastis ketika diuji langsung di domain malam.",
        ],
        accent_color=COLOR_INDIGO_BLUE,
        tag="Problem Model",
    )

    # Bottom Comparison Table (Gap Analysis)
    bot_card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.8),
        Inches(4.55),
        Inches(11.733),
        Inches(2.2),
    )
    bot_card.fill.solid()
    bot_card.fill.fore_color.rgb = COLOR_CARD_HIGHLIGHT
    bot_card.line.color.rgb = COLOR_CARD_BORDER

    tb_bot = slide.shapes.add_textbox(Inches(1.05), Inches(4.7), Inches(11.2), Inches(1.9))
    tf_bot = tb_bot.text_frame
    tf_bot.word_wrap = True
    tf_bot.margin_left = tf_bot.margin_top = tf_bot.margin_right = tf_bot.margin_bottom = 0

    p_gtitle = tf_bot.paragraphs[0]
    p_gtitle.text = "ANALISIS KESENJANGAN: MENGAPA PENDEKATAN LAIN BELUM TEPAT GUNA?"
    p_gtitle.font.name = FONT_FAMILY_TITLE
    p_gtitle.font.size = Pt(12)
    p_gtitle.font.bold = True
    p_gtitle.font.color.rgb = COLOR_CYAN_ACCENT
    p_gtitle.space_after = Pt(8)

    p_gap1 = tf_bot.add_paragraph()
    p_gap1.text = (
        "✖ Model Restorasi Deep Generatif (LLNet / EnlightenGAN): Menghasilkan gambar indah tetapi komputasi sangat berat "
        "(< 5 FPS) dan boros VRAM GPU, sehingga mustahil diterapkan di edge device kendaraan."
    )
    p_gap1.font.name = FONT_FAMILY_BODY
    p_gap1.font.size = Pt(11)
    p_gap1.font.color.rgb = COLOR_TEXT_MUTED
    p_gap1.space_after = Pt(5)

    p_gap2 = tf_bot.add_paragraph()
    p_gap2.text = (
        "✖ Filter Histogram Global Konvensional (Global HE): Menaikkan seluruh kecerahan secara membabi-buta, "
        "mengakibatkan silau lampu kendaraan makin over-saturated dan noise langit malam makin parah."
    )
    p_gap2.font.name = FONT_FAMILY_BODY
    p_gap2.font.size = Pt(11)
    p_gap2.font.color.rgb = COLOR_TEXT_MUTED
    p_gap2.space_after = Pt(5)

    p_gap3 = tf_bot.add_paragraph()
    p_gap3.text = (
        "✔ Solusi Kami: Prapemrosesan adaptif lokal berbobot ringan (CLAHE CIE Lab + Gamma Correction) "
        "dipadukan dengan arsitektur YOLOv8-Nano yang ultra-efisien."
    )
    p_gap3.font.name = FONT_FAMILY_BODY
    p_gap3.font.size = Pt(11)
    p_gap3.font.bold = True
    p_gap3.font.color.rgb = COLOR_EMERALD_GREEN

    add_footer(slide, 3, 11)


def build_slide_4_solution(prs: Presentation) -> None:
    """Builds Slide 4: Solusi yang Diusulkan (Pipeline Overview)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_slide_background(slide)
    add_header(
        slide,
        category="03. SOLUSI YANG DIUSULKAN",
        title="Pipeline Cerdas: Adaptive Enhancement + Lightweight YOLOv8",
        subtitle="Kombinasi efisien antara peningkatan visibilitas lokal dan arsitektur deteksi objek generasi terbaru.",
    )

    # 3 Steps Flow
    step_w = 3.75
    step_gap = 0.24
    top_pos = 1.95
    h_pos = 3.6

    add_card(
        slide,
        left=0.8,
        top=top_pos,
        width=step_w,
        height=h_pos,
        title="Tahap 1: Input & Normalisasi",
        items=[
            "Menerima frame video atau citra jalan raya beresolusi 640x640 piksel.",
            "Mengonversi ruang warna citra dari RGB ke CIE L*a*b* untuk memisahkan kanal luminansi (L) dari kanal warna (a, b).",
            "Mencegah pergeseran rona warna (color distortion) saat kontras dinaikkan.",
            "Siap diproses secara frame-by-frame pada stream kamera.",
        ],
        accent_color=COLOR_CYAN_ACCENT,
        tag="Step 1 — Input Stream",
    )

    add_card(
        slide,
        left=0.8 + step_w + step_gap,
        top=top_pos,
        width=step_w,
        height=h_pos,
        title="Tahap 2: Prapemrosesan Adaptif",
        items=[
            "CLAHE (L-channel): Meningkatkan kontras lokal pada tile 8x8 dengan pembatas clip limit = 2.0 untuk meredam amplifikasi noise.",
            "Gamma Correction (γ = 0.6): Meregangkan kurva pencahayaan mid-tone tanpa membuat area lampu sorot menjadi jenuh.",
            "Hasil: Siluet pejalan kaki muncul jelas tanpa menambah beban inferensi deep learning yang berat.",
        ],
        accent_color=COLOR_AMBER_ORANGE,
        tag="Step 2 — Adaptive Preprocessing",
    )

    add_card(
        slide,
        left=0.8 + 2 * (step_w + step_gap),
        top=top_pos,
        width=step_w,
        height=h_pos,
        title="Tahap 3: Deteksi YOLOv8-Nano",
        items=[
            "Arsitektur Ultra-Lightweight: ~3,2 juta parameter, sangat ideal untuk komputasi CPU laptop.",
            "Head Anchor-Free: Langsung memprediksi pusat objek pejalan kaki dengan rasio aspek tubuh yang beragam.",
            "Modul C2f: Menggabungkan fitur spasial resolusi tinggi dengan konteks semantik lalu lintas.",
            "Target Throughput: Berjalan mulus pada ≥ 20–25 FPS.",
        ],
        accent_color=COLOR_EMERALD_GREEN,
        tag="Step 3 — Object Detection",
    )

    # Bottom Metric Chips (Key Strengths)
    add_metric_chip(slide, 0.8, 5.75, 2.75, 1.05, "~3.2 Juta", "Parameter Model (Ringan)", COLOR_CYAN_ACCENT)
    add_metric_chip(slide, 3.8, 5.75, 2.75, 1.05, "≥ 20 FPS", "Kecepatan Inferensi CPU", COLOR_EMERALD_GREEN)
    add_metric_chip(slide, 6.8, 5.75, 2.75, 1.05, "CIE L*a*b*", "Isolasi Luminansi Warna", COLOR_AMBER_ORANGE)
    add_metric_chip(slide, 9.8, 5.75, 2.733, 1.05, "Zero Leakage", "Protokol Validasi Ketat", COLOR_TEXT_WHITE)

    add_footer(slide, 4, 11)


def build_slide_5_metric_philosophy(prs: Presentation) -> None:
    """Builds Slide 5: Filosofi Evaluasi: Mengapa Recall adalah Segalanya."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_slide_background(slide)
    add_header(
        slide,
        category="04. FILOSOFI METRIK & PENILAIAN",
        title="Mengapa Proyek Ini Memprioritaskan Metrik 'Recall'?",
        subtitle="Dalam aplikasi keselamatan ADAS, biaya kegagalan deteksi pejalan kaki memiliki konsekuensi fatal.",
    )

    # Left Card: Asymmetric Cost of Error
    add_card(
        slide,
        left=0.8,
        top=1.95,
        width=5.7,
        height=4.8,
        title="Asimetri Kesalahan: False Negative vs False Positive",
        items=[
            "False Negative (FN) — Fatal Risk:",
            "  • Pejalan kaki ada di depan mobil, tetapi model GAGAL mendeteksi.",
            "  • Kendaraan tidak melakukan pengereman otomatis.",
            "  • Dampak: Tabrakan fatal dan ancaman kehilangan nyawa manusia.",
            "",
            "False Positive (FP) — Low Risk:",
            "  • Bayangan tiang atau semak salah dikira pejalan kaki.",
            "  • Dampak: Sistem memberikan peringatan dini / pengereman defensif.",
            "  • Pengemudi tetap aman tanpa ada bahaya fisik.",
            "",
            "Kesimpulan: Recall pejalan kaki wajib dimaksimalkan mendekati 100%!",
        ],
        accent_color=COLOR_ROSE_RED,
        tag="Prinsip Safety-First",
    )

    # Right Card: The Class Imbalance Trap
    add_card(
        slide,
        left=6.833,
        top=1.95,
        width=5.7,
        height=4.8,
        title="Jebakan Class Imbalance (7.58 : 1)",
        items=[
            "Hasil Audit Dataset Kami:",
            "  • Objek Kendaraan (Vehicle): 6.871 box (88,4%)",
            "  • Objek Pejalan Kaki (Person): 906 box (11,6%)",
            "",
            "Bahaya Metrik Agregat mAP / Akurasi Biasa:",
            "  • Jika model hanya pintar mendeteksi mobil tetapi buta pejalan kaki, nilai mAP keseluruhan masih bisa terlihat 'tinggi' (menipu).",
            "  • Oleh karena itu, dewan penguji dapat melihat ketelitian kami: kami membedah metrik per kelas dan mematok target perbaikan khusus pada Pejalan Kaki.",
            "",
            "Target Kami: Peningkatan Recall Pejalan Kaki > 10% dibanding baseline.",
        ],
        accent_color=COLOR_AMBER_ORANGE,
        tag="Validasi Metrik Akademik",
    )

    add_footer(slide, 5, 11)


def build_slide_6_data_readiness(prs: Presentation) -> None:
    """Builds Slide 6: Kesiapan Data & Bukti Eksplorasi Awal."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_slide_background(slide)
    add_header(
        slide,
        category="05. KESIAPAN DATA & PROGRESS AWAL",
        title="Dataset Terverifikasi Penuh & Bebas Data Leakage",
        subtitle="Kelompok 9 telah menuntaskan audit komprehensif pada dataset Roboflow Low-Light v2 (Checkpoint CP03).",
    )

    # Left: Audit Table & Details
    add_card(
        slide,
        left=0.8,
        top=1.95,
        width=6.0,
        height=4.8,
        title="Distribusi Data Hasil Audit (src/audit_dataset.py)",
        items=[
            "Total Volume: 2.400 citra dengan 7.777 bounding box anotasi.",
            "Train Split (2.100 citra): 6.018 Vehicle | 798 Person (Ratio 7.54:1)",
            "Valid Split (200 citra): 581 Vehicle | 69 Person (Ratio 8.42:1)",
            "Test Split (100 citra): 272 Vehicle | 39 Person (Ratio 6.97:1)",
            "",
            "Integritas Anotasi 100% Lolos Uji:",
            "  • Seluruh gambar memiliki pasangan file label .txt yang valid.",
            "  • Tidak ditemukan anotasi korup, koordinat NaN, atau out-of-bound.",
            "",
            "Protokol Pencegahan Data Leakage:",
            "  • Split folder terisolasi ketat di tingkat file system.",
            "  • Augmentasi & CLAHE HANYA diaplikasikan pada data training.",
            "  • Data uji (Test set) dijaga murni dan tidak pernah terlihat model.",
        ],
        accent_color=COLOR_CYAN_ACCENT,
        tag="Dataset Roboflow Low-Light v2 (CC BY 4.0)",
    )

    # Right: Image Container
    img_card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(7.1),
        Inches(1.95),
        Inches(5.433),
        Inches(4.8),
    )
    img_card.fill.solid()
    img_card.fill.fore_color.rgb = COLOR_CARD_BG
    img_card.line.color.rgb = COLOR_CARD_BORDER

    # Insert Image if exists
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(project_root, "results", "audit", "sample_annotations.png")
    if os.path.exists(img_path):
        slide.shapes.add_picture(
            img_path,
            Inches(7.25),
            Inches(2.1),
            width=Inches(5.133),
            height=Inches(3.85),
        )

        cap_tb = slide.shapes.add_textbox(Inches(7.25), Inches(6.05), Inches(5.133), Inches(0.6))
        cap_tf = cap_tb.text_frame
        cap_tf.word_wrap = True
        cap_tf.margin_left = cap_tf.margin_top = cap_tf.margin_right = cap_tf.margin_bottom = 0
        cap_p = cap_tf.paragraphs[0]
        cap_p.text = "Gambar 1: Sampel Visual Audit Bounding Box (Person: Hijau, Vehicle: Biru)"
        cap_p.font.name = FONT_FAMILY_BODY
        cap_p.font.size = Pt(9)
        cap_p.font.color.rgb = COLOR_EMERALD_GREEN
        cap_p.alignment = PP_ALIGN.CENTER

    add_footer(slide, 6, 11)


def build_slide_7_experiments(prs: Presentation) -> None:
    """Builds Slide 7: Desain Eksperimen Terkontrol (Controlled Methodology)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_slide_background(slide)
    add_header(
        slide,
        category="06. METODOLOGI EKSPERIMEN TERKONTROL",
        title="4 Skenario Pengujian Komparatif yang Ketat",
        subtitle="Riset dirancang terstruktur secara ilmiah untuk membuktikan kontribusi nyata masing-masing teknik.",
    )

    # 4 Cards Grid (2x2)
    card_w = 5.7
    card_h = 2.25
    x1 = 0.8
    x2 = 6.833
    y1 = 1.95
    y2 = 4.45

    add_card(
        slide,
        left=x1,
        top=y1,
        width=card_w,
        height=card_h,
        title="B0: Baseline Zero-Shot (YOLOv8n COCO)",
        items=[
            "Tujuan: Mengukur performa awal model standar tanpa domain adaptation.",
            "Prapemrosesan: Citra asli apa adanya (raw low-light RGB).",
            "Ekspektasi: Membuktikan hipotesis kegagalan deteksi pejalan kaki siluet gelap.",
        ],
        accent_color=COLOR_TEXT_MUTED,
        tag="Eksperimen Pembanding Dasar",
    )

    add_card(
        slide,
        left=x2,
        top=y1,
        width=card_w,
        height=card_h,
        title="Exp-A: Fine-Tuning Domain Adaptif",
        items=[
            "Tujuan: Mengukur perbaikan hasil transfer learning langsung pada data malam.",
            "Prapemrosesan: Augmentasi geometri standar (tanpa manipulasi luminansi khusus).",
            "Ekspektasi: Peningkatan mAP umum, namun Recall pejalan kaki gelap masih terbatas.",
        ],
        accent_color=COLOR_CYAN_ACCENT,
        tag="Eksperimen Fine-Tuned Standard",
    )

    add_card(
        slide,
        left=x1,
        top=y2,
        width=card_w,
        height=card_h,
        title="Exp-B: Fine-Tuned + L-Channel CLAHE",
        items=[
            "Tujuan: Menguji dampak penajaman kontras lokal pada kanal luminansi CIE L*a*b*.",
            "Konfigurasi: Clip limit = 2.0, grid tile = 8x8 (mencegah over-amplifikasi noise).",
            "Target Utama: Lonjakan Recall Pejalan Kaki > 10% dibanding Exp-A.",
        ],
        accent_color=COLOR_EMERALD_GREEN,
        tag="Eksperimen Inti Usulan (Proposed)",
    )

    add_card(
        slide,
        left=x2,
        top=y2,
        width=card_w,
        height=card_h,
        title="Exp-C: Fine-Tuned + Gamma Correction",
        items=[
            "Tujuan: Menguji dampak penyesuaian non-linear kurva intensitas cahaya (γ = 0.6).",
            "Prapemrosesan: Pengangkatan kecerahan mid-tone tanpa saturasi berlebih.",
            "Analisis: Membandingkan komparasi trade-off ketajaman visual CLAHE vs Gamma.",
        ],
        accent_color=COLOR_AMBER_ORANGE,
        tag="Eksperimen Komparasi Pencerahan",
    )

    add_footer(slide, 7, 11)


def build_slide_8_deployment(prs: Presentation) -> None:
    """Builds Slide 8: Rencana Deployment & Deliverables."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_slide_background(slide)
    add_header(
        slide,
        category="07. IMPLEMENTASI & LUARAN SISTEM",
        title="Rencana Deployment Nyata & Target Deliverables",
        subtitle="Bukan sekadar model di atas kertas, proyek ini menghasilkan prototype interaktif yang dapat didemonstrasikan langsung.",
    )

    col_w = 3.75
    col_gap = 0.24
    top_pos = 1.95
    h_pos = 3.6

    add_card(
        slide,
        left=0.8,
        top=top_pos,
        width=col_w,
        height=h_pos,
        title="Lingkungan Komputasi",
        items=[
            "Training Hardware: NVIDIA Tesla T4 GPU (Google Colab 16GB VRAM) — menjamin training cepat dan stabil (50 epochs).",
            "Inference Target: CPU Laptop Standar (Intel Core i5/i7) — membuktikan efisiensi tanpa butuh GPU mahal.",
            "PyTorch & Ultralytics Framework yang terstandarisasi industri.",
        ],
        accent_color=COLOR_CYAN_ACCENT,
        tag="Infrastruktur & Alat",
    )

    add_card(
        slide,
        left=0.8 + col_w + col_gap,
        top=top_pos,
        width=col_w,
        height=h_pos,
        title="Prototype Aplikasi Interaktif",
        items=[
            "Pipeline Inferensi OpenCV / Streamlit:",
            "  • Menerima input video berkendara malam atau static image.",
            "  • Visualisasi live split-screen: Sebelum enhancement vs Sesudah CLAHE.",
            "  • Bounding box deteksi real-time dengan confidence score dan indikator FPS live.",
        ],
        accent_color=COLOR_EMERALD_GREEN,
        tag="Antarmuka Demo Pengguna",
    )

    add_card(
        slide,
        left=0.8 + 2 * (col_w + col_gap),
        top=top_pos,
        width=col_w,
        height=h_pos,
        title="Luaran Akademik & Kode",
        items=[
            "Dokumen Teknis Lengkap: Format 2-Kolom IEEE Technical Report standar publikasi.",
            "Repositori GitHub Terstruktur: Skrip training modular, log audit, dan config reproducible.",
            "Rencana Ekspansi UAS: Eksplorasi kuantisasi model INT8 untuk deployment edge mikrokontroler.",
        ],
        accent_color=COLOR_INDIGO_BLUE,
        tag="Deliverables Akhir",
    )

    # Bottom Metric Chips
    add_metric_chip(slide, 0.8, 5.75, 3.75, 1.05, "NVIDIA T4", "GPU Cloud Training Ready", COLOR_CYAN_ACCENT)
    add_metric_chip(slide, 4.79, 5.75, 3.75, 1.05, "Interactive UI", "Streamlit / OpenCV Demo", COLOR_EMERALD_GREEN)
    add_metric_chip(slide, 8.78, 5.75, 3.75, 1.05, "IEEE Report", "Standardized Documentation", COLOR_INDIGO_BLUE)

    add_footer(slide, 8, 11)


def build_slide_9_execution_plan(prs: Presentation) -> None:
    """Builds Slide 9: Pembagian Peran & Roadmap Eksekusi."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_slide_background(slide)
    add_header(
        slide,
        category="08. EKSEKUSI & ROADMAP PROYEK",
        title="Pembagian Peran yang Jelas & Jadwal Terukur",
        subtitle="Kedua anggota memiliki akuntabilitas tugas yang tegas dan saling melengkapi.",
    )

    # 2 Member Cards
    card_w = 5.7
    card_h = 2.4
    add_card(
        slide,
        left=0.8,
        top=1.95,
        width=card_w,
        height=card_h,
        title="Krisantus Amka Ginting  &  Linardi Halim",
        items=[
            "Data Strategy, Preprocessing & Documentation",
            "Tanggung Jawab Utama:",
            "  • Audit kualitas anotasi dataset dan penyelesaian class imbalance.",
            "  • Implementasi modul pencerahan citra adaptif (CLAHE kanal-L & Gamma).",
            "  • Penyusunan proposal, laporan teknis, dan rekap tracker/logbook.",
        ],
        accent_color=COLOR_CYAN_ACCENT,
        tag="Peran Tim: Data & Preprocessing Lead",
    )

    add_card(
        slide,
        left=6.833,
        top=1.95,
        width=card_w,
        height=card_h,
        title="Kristofer Ryan Giggs Eka Saputra  &  Leonard Franscois",
        items=[
            "Model Architecture, Training & Deployment",
            "Tanggung Jawab Utama:",
            "  • Setup environment training YOLOv8-Nano pada GPU Google Colab.",
            "  • Eksekusi 4 eksperimen komparatif terkontrol (B0, Exp-A, Exp-B, Exp-C).",
            "  • Benchmarking FPS, prototype demo inferensi, dan video cadangan.",
        ],
        accent_color=COLOR_EMERALD_GREEN,
        tag="Peran Tim: Model & Deployment Lead",
    )

    # Roadmap Timeline Banner
    r_card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.8),
        Inches(4.55),
        Inches(11.733),
        Inches(2.2),
    )
    r_card.fill.solid()
    r_card.fill.fore_color.rgb = COLOR_CARD_HIGHLIGHT
    r_card.line.color.rgb = COLOR_CARD_BORDER

    tb_r = slide.shapes.add_textbox(Inches(1.05), Inches(4.7), Inches(11.2), Inches(1.9))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = 0

    rp_title = tf_r.paragraphs[0]
    rp_title.text = "ROADMAP EKSEKUSI MENUJU SIDANG AKHIR (UAS)"
    rp_title.font.name = FONT_FAMILY_TITLE
    rp_title.font.size = Pt(12)
    rp_title.font.bold = True
    rp_title.font.color.rgb = COLOR_AMBER_ORANGE
    rp_title.space_after = Pt(8)

    phases = [
        ("Fase 1 (UTS — SELESAI)", "Audit dataset 2.400 citra, formulasi proposal IEEE, setup repositori, dan verifikasi pipeline."),
        ("Fase 2 (Minggu 8–11)", "Training 4 varian eksperimen pada Google Colab T4, tuning hyperparameter, dan evaluasi Recall."),
        ("Fase 3 (Minggu 12–14)", "Analisis kegagalan (failure cases), perbandingan error slices, integrasi prototype demo Streamlit."),
        ("Fase 4 (UAS — Final)", "Penyusunan laporan akhir komprehensif, video demonstrasi sistem real-time, dan presentasi akhir."),
    ]

    for p_label, p_desc in phases:
        p = tf_r.add_paragraph()
        p.text = f"•  {p_label}:  {p_desc}"
        p.font.name = FONT_FAMILY_BODY
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_WHITE if "SELESAI" in p_label else COLOR_TEXT_MUTED
        p.space_after = Pt(4)

    add_footer(slide, 9, 11)


def build_slide_10_feasibility(prs: Presentation) -> None:
    """Builds Slide 10: Mengapa Proyek Ini Sangat Layak Diterima? (Why Approve)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_slide_background(slide)
    add_header(
        slide,
        category="09. KELAYAKAN & NILAI STRATEGIS",
        title="4 Alasan Utama Proyek Ini Sangat Layak Diterima",
        subtitle="Ringkasan argumentasi kelayakan akademik, teknis, dan kepraktisan untuk dewan penguji.",
    )

    card_w = 5.7
    card_h = 2.25
    x1 = 0.8
    x2 = 6.833
    y1 = 1.95
    y2 = 4.45

    add_card(
        slide,
        left=x1,
        top=y1,
        width=card_w,
        height=card_h,
        title="1. Masalah Nyata & Kritis (High Urgency)",
        items=[
            "Menjawab tantangan riil keselamatan sistem ADAS malam hari.",
            "Berorientasi langsung pada mitigasi kecelakaan fatal pejalan kaki.",
            "Relevansi topik Visi Komputer tingkat lanjut yang berdampak sosial tinggi.",
        ],
        accent_color=COLOR_ROSE_RED,
        tag="Pilar 1 — Relevansi & Urgensi",
    )

    add_card(
        slide,
        left=x2,
        top=y1,
        width=card_w,
        height=card_h,
        title="2. Fondasi Metodologis Kokoh (Sound Methodology)",
        items=[
            "Dataset 2.400 citra telah diaudit 100% dan bebas dari kebocoran data.",
            "Desain eksperimen komparatif 4 tahap yang objektif dan reproducible.",
            "Fokus evaluasi pada Recall per kelas yang jujur terhadap class imbalance.",
        ],
        accent_color=COLOR_CYAN_ACCENT,
        tag="Pilar 2 — Keabsahan Ilmiah",
    )

    add_card(
        slide,
        left=x1,
        top=y2,
        width=card_w,
        height=card_h,
        title="3. Realistis & Bebas Over-Engineering",
        items=[
            "Model ringan (~3,2M parameter) memungkinkan eksekusi cepat di hardware laptop umum.",
            "Tidak memaksakan arsitektur deep generative yang lambat dan berat.",
            "Infrastruktur pelatihan (Colab T4) dan dataset sudah tersedia penuh.",
        ],
        accent_color=COLOR_EMERALD_GREEN,
        tag="Pilar 3 — Kelayakan Eksekusi",
    )

    add_card(
        slide,
        left=x2,
        top=y2,
        width=card_w,
        height=card_h,
        title="4. Luaran Terukur & Siap Diuji (Clear Deliverables)",
        items=[
            "Target peningkatan Recall > 10% dan target kecepatan ≥ 20–25 FPS.",
            "Menghasilkan prototype aplikasi interaktif yang siap didemokan langsung.",
            "Kesesuaian penuh dengan standar IEEE Technical Report dan rubrik penilaian.",
        ],
        accent_color=COLOR_AMBER_ORANGE,
        tag="Pilar 4 — Ketercapaian Hasil",
    )

    add_footer(slide, 10, 11)


def build_slide_11_qa(prs: Presentation) -> None:
    """Builds Slide 11: Penutup & Tanya Jawab (Q&A)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_slide_background(slide)

    # Center Box
    center_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(1.5),
        Inches(1.5),
        Inches(10.333),
        Inches(4.5),
    )
    center_box.fill.solid()
    center_box.fill.fore_color.rgb = COLOR_CARD_BG
    center_box.line.color.rgb = COLOR_CYAN_ACCENT
    center_box.line.width = Pt(2)

    tb = slide.shapes.add_textbox(Inches(1.8), Inches(1.9), Inches(9.733), Inches(3.7))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p1 = tf.paragraphs[0]
    p1.text = "TERIMA KASIH"
    p1.font.name = FONT_FAMILY_TITLE
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TEXT_WHITE
    p1.alignment = PP_ALIGN.CENTER
    p1.space_after = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Kami Siap Menerima Masukan, Saran, & Pertanyaan dari Dewan Penguji"
    p2.font.name = FONT_FAMILY_BODY
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_CYAN_ACCENT
    p2.alignment = PP_ALIGN.CENTER
    p2.space_after = Pt(24)

    p3 = tf.add_paragraph()
    p3.text = "Peningkatan Recall Deteksi Pejalan Kaki pada Kondisi Minim Cahaya\nMenggunakan YOLOv8 dan Augmentasi Pencahayaan Adaptif"
    p3.font.name = FONT_FAMILY_BODY
    p3.font.size = Pt(13)
    p3.font.color.rgb = COLOR_TEXT_MUTED
    p3.alignment = PP_ALIGN.CENTER
    p3.space_after = Pt(20)

    p4 = tf.add_paragraph()
    p4.text = (
        "Kelompok 9 (Kelas 5PITA) — Krisantus Amka Ginting, Linardi Halim, "
        "Kristofer Ryan Giggs Eka Saputra, Leonard Franscois\n"
        "Visi Komputer (IF0PP006) — Semester Ganjil 2026/2027"
    )
    p4.font.name = FONT_FAMILY_BODY
    p4.font.size = Pt(12)
    p4.font.bold = True
    p4.font.color.rgb = COLOR_EMERALD_GREEN
    p4.alignment = PP_ALIGN.CENTER

    add_footer(slide, 11, 11)


# ==============================================================================
# MAIN GENERATOR FUNCTION
# ==============================================================================

def generate_presentation(output_paths: List[str]) -> None:
    """Generates the complete presentation and saves it to specified paths.

    Args:
        output_paths: List of absolute file paths to save the presentation to.
    """
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_WIDTH_INCHES)
    prs.slide_height = Inches(SLIDE_HEIGHT_INCHES)

    print("Building slides...")
    build_slide_1_cover(prs)
    build_slide_2_urgency(prs)
    build_slide_3_problems(prs)
    build_slide_4_solution(prs)
    build_slide_5_metric_philosophy(prs)
    build_slide_6_data_readiness(prs)
    build_slide_7_experiments(prs)
    build_slide_8_deployment(prs)
    build_slide_9_execution_plan(prs)
    build_slide_10_feasibility(prs)
    build_slide_11_qa(prs)

    for path in output_paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            prs.save(path)
            print(f"Presentation successfully saved to: {path}")
        except PermissionError:
            alt_path = path.replace(".pptx", "_v02.pptx")
            prs.save(alt_path)
            print(f"Warning: File {path} is currently locked (open in PowerPoint). Saved copy to: {alt_path}")


if __name__ == "__main__":
    # Path relatif terhadap root proyek agar skrip dapat dijalankan dari mesin mana pun.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    targets = [
        os.path.join(base_dir, "K09_Sidang_Proposal_DeteksiPedestrianLowLight.pptx"),
        os.path.join(base_dir, "docs", "proposal", "K09_Sidang_Proposal_DeteksiPedestrianLowLight.pptx"),
    ]
    generate_presentation(targets)
