"""Interactive Demonstration Dashboard for Low-Light Pedestrian Detection (CP12).

This Streamlit application provides real-time inference and side-by-side model
comparison for the Computer Vision Final Project (IF0PP006) - Kelompok 09:
- Krisantus Amka Ginting (412024025) - Data & Preprocessing Lead
- Linardi Halim (212024005) - Documentation & Reporting Lead
- Kristofer Ryan Giggs Eka Saputra (412024025) - Model & Training Lead
- Leonard Franscois (412024035) - Demo & Deployment Lead

Key Features:
- Model Selection (Exp-C Gamma Proposed, Exp-A Standard, Exp-B CLAHE, B0 Baseline)
- Side-by-side comparative inspection on difficult low-light scenes
- Real-time latency (ms) and throughput (FPS) measurement on CPU
- Interactive confidence and IoU threshold tuning
- Embedded experimental results (CP10) and failure case analysis (CP11)
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Constants & Configuration
# ----------------------------------------------------------------------------
APP_TITLE: str = "Deteksi Pejalan Kaki Minim Cahaya (Low-Light Pedestrian Detection)"
PROJECT_ROOT: Path = Path(__file__).resolve().parent
TEST_IMAGES_DIR: Path = PROJECT_ROOT / "dataset_dedup" / "test" / "images"
UNIFIED_RESULTS_PATH: Path = PROJECT_ROOT / "results" / "experiments" / "unified_results.md"
FAILURE_CASES_PATH: Path = PROJECT_ROOT / "results" / "experiments" / "failure_cases.md"

# Model Registry
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Exp_D_Robust": {
        "name": "Exp-D: Robust Fine-Tuned (Anti-Forgetting & Multi-Scale)",
        "path": PROJECT_ROOT / "hasil_final_model" / "hasil_final_model" / "Exp_D_Robust_best.pt"
        if (PROJECT_ROOT / "hasil_final_model" / "hasil_final_model" / "Exp_D_Robust_best.pt").exists()
        else PROJECT_ROOT / "Exp_D_Robust_best.pt",
        "description": "Model transfer learning terkontrol (freeze=10, lr0=0.001, scale=0.5). Sangat handal mendeteksi pejalan kaki dekat/jauh di malam dan siang hari.",
        "is_zero_shot": False,
        "badge": "🛡️ Robust Transfer Model",
    },
    "Exp_A_Standard": {
        "name": "Exp-A: Standard Low-Light Fine-Tuned (Champion / Best Model)",
        "path": PROJECT_ROOT / "hasil_final_model" / "hasil_final_model" / "Exp_A_Standard_best.pt",
        "description": "Model terbaik dengan performa tertinggi pada test set bersih (Recall 81.01%, Precision 62.75%, mAP 83.93%).",
        "is_zero_shot": False,
        "badge": "🏆 Champion / Best Model",
    },
    "Exp_C_Gamma": {
        "name": "Exp-C: Gamma Correction (Proposed Method)",
        "path": PROJECT_ROOT / "hasil_final_model" / "hasil_final_model" / "Exp_C_Gamma_best.pt",
        "description": "Model usulan augmentasi pencahayaan non-linear (gamma=0.6). Recall 65.82% pada test set bersih.",
        "is_zero_shot": False,
        "badge": "🔬 Proposed Method",
    },
    "Exp_B_CLAHE": {
        "name": "Exp-B: L-channel CLAHE (Proposed Method)",
        "path": PROJECT_ROOT / "hasil_final_model" / "hasil_final_model" / "Exp_B_CLAHE_best.pt",
        "description": "Model usulan kontras adaptif CLAHE pada kanal-L CIE L*a*b*. Mengalami amplifikasi noise di malam hari.",
        "is_zero_shot": False,
        "badge": "🔬 Proposed Method",
    },
    "B0_ZeroShot": {
        "name": "B0: Zero-Shot Baseline (COCO Pretrained)",
        "path": PROJECT_ROOT / "yolov8n.pt",
        "description": "YOLOv8n bawaan MS COCO tanpa adaptasi domain malam hari (Recall anjlok ke 15.19%).",
        "is_zero_shot": True,
        "badge": "📉 Baseline COCO",
    },
}

COCO_TO_TARGET_MAPPING: Dict[int, int] = {
    0: 1,  # person -> person
    2: 0,  # car -> Vehicle
    3: 0,  # motorcycle -> Vehicle
    5: 0,  # bus -> Vehicle
    7: 0,  # truck -> Vehicle
}

TARGET_CLASSES: Dict[int, str] = {
    0: "Vehicle",
    1: "person",
}

# Color mapping in RGB format
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 180, 240),   # Vehicle: Cyan/Light Blue
    1: (240, 40, 40),   # person: Bright Red
}

SAMPLE_PRESETS: List[Tuple[str, str]] = [
    ("Pejalan Kaki Siluet Gelap (Scene 14969583)", "14969583-55cd5dd2_augmented_1_jpg.rf.47834ba812a385f88105c1a8e60ef8f9.jpg"),
    ("Pejalan Kaki Menyeberang Jalan (Scene 1a5d0011)", "1a5d0011-c34c50cc_augmented_0_jpg.rf.46301686bd9a0f1df9bac672db570f74.jpg"),
    ("Pejalan Kaki Jarak Menengah (Scene 08eff217)", "08eff217-6061163f_augmented_1_jpg.rf.9435b8f3effb43e7eaadb8ab2adb4f90.jpg"),
    ("Kepadatan Kendaraan Malam (Scene 1f08ba8a)", "1f08ba8a-1e15976e_augmented_0_jpg.rf.ce5e0d948ba77840d1db22a0084e62a1.jpg"),
    ("Siluet Minim Cahaya Ekstrem (Scene 231bdb08)", "231bdb08-bb0a4a36_augmented_0_jpg.rf.473b90888a76e7c43ae1dfcbc63acc03.jpg"),
]


# ----------------------------------------------------------------------------
# Model Loader Cache
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat bobot model YOLOv8...")
def load_yolo_model(model_key: str) -> Optional[YOLO]:
    """Load and cache a YOLO model instance by registry key.

    Args:
        model_key: Identifier key in MODEL_REGISTRY.

    Returns:
        Loaded Ultralytics YOLO instance, or None if weights not found.
    """
    spec = MODEL_REGISTRY.get(model_key)
    if not spec:
        return None
    weights_path: Path = spec["path"]
    if not weights_path.exists():
        st.error(f"Berkas bobot model tidak ditemukan: {weights_path}")
        return None
    return YOLO(str(weights_path))


# ----------------------------------------------------------------------------
# Inference & Annotation Engine
# ----------------------------------------------------------------------------
def run_inference(
    model: YOLO,
    image_rgb: np.ndarray,
    is_zero_shot: bool,
    conf_threshold: float,
    iou_threshold: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute forward inference and render bounding boxes onto an image.

    Args:
        model: Ultralytics YOLO model.
        image_rgb: Source image in RGB format.
        is_zero_shot: True if model uses stock COCO taxonomy.
        conf_threshold: Minimum confidence score.
        iou_threshold: IoU NMS threshold.

    Returns:
        Annotated RGB image and a dictionary of inference diagnostics.
    """
    start_time: float = time.perf_counter()
    results = model.predict(
        source=image_rgb,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        device="cpu",
        verbose=False,
    )
    latency_ms: float = (time.perf_counter() - start_time) * 1000.0
    fps: float = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    annotated: np.ndarray = image_rgb.copy()
    person_count: int = 0
    vehicle_count: int = 0
    detections_list: List[Dict[str, Any]] = []

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy_arr = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)

            for xyxy, conf, raw_cls in zip(xyxy_arr, conf_arr, cls_arr):
                target_cls: Optional[int] = None
                if is_zero_shot:
                    target_cls = COCO_TO_TARGET_MAPPING.get(raw_cls)
                else:
                    if raw_cls in TARGET_CLASSES:
                        target_cls = raw_cls

                if target_cls is None:
                    continue

                if target_cls == 1:
                    person_count += 1
                elif target_cls == 0:
                    vehicle_count += 1

                x1, y1, x2, y2 = map(int, xyxy)
                color = CLASS_COLORS[target_cls]
                label_text = f"{TARGET_CLASSES[target_cls]} {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                label_y1 = max(0, y1 - text_h - 4)
                cv2.rectangle(
                    annotated,
                    (x1, label_y1),
                    (x1 + text_w + 6, label_y1 + text_h + 4),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label_text,
                    (x1 + 3, label_y1 + text_h + 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

                detections_list.append(
                    {
                        "class": TARGET_CLASSES[target_cls],
                        "confidence": float(conf),
                        "box": [x1, y1, x2, y2],
                    }
                )

    diagnostics: Dict[str, Any] = {
        "latency_ms": latency_ms,
        "fps": fps,
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "total_detections": person_count + vehicle_count,
        "detections": detections_list,
    }
    return annotated, diagnostics


# ----------------------------------------------------------------------------
# View Renderers (Static Images & Real-time Streams)
# ----------------------------------------------------------------------------
def render_static_detection(
    image_rgb: np.ndarray,
    primary_model: YOLO,
    primary_model_key: str,
    enable_comparison: bool,
    secondary_model: Optional[YOLO],
    secondary_model_key: Optional[str],
    conf_threshold: float,
    iou_threshold: float,
) -> None:
    """Render detection results on a static image with optional side-by-side view.

    Args:
        image_rgb: RGB image array.
        primary_model: Loaded primary YOLO model.
        primary_model_key: Registry key of primary model.
        enable_comparison: Flag indicating side-by-side comparison mode.
        secondary_model: Loaded secondary model if comparison enabled.
        secondary_model_key: Registry key of secondary model.
        conf_threshold: Confidence threshold.
        iou_threshold: IoU NMS threshold.
    """
    spec_p = MODEL_REGISTRY[primary_model_key]

    if enable_comparison and secondary_model_key and secondary_model:
        spec_s = MODEL_REGISTRY[secondary_model_key]
        col_left, col_right = st.columns(2)

        # Primary Model Column
        with col_left:
            st.subheader(f"🔹 {spec_p['name']}")
            st.caption(f"{spec_p['description']}")
            annotated_p, diag_p = run_inference(
                primary_model,
                image_rgb,
                spec_p["is_zero_shot"],
                conf_threshold,
                iou_threshold,
            )
            st.image(annotated_p, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🚶 Pedestrian", diag_p["person_count"])
            c2.metric("🚗 Vehicle", diag_p["vehicle_count"])
            c3.metric("⏱️ Latensi", f"{diag_p['latency_ms']:.1f} ms")
            c4.metric("⚡ FPS", f"{diag_p['fps']:.1f}")

        # Secondary Model Column
        with col_right:
            st.subheader(f"🔸 {spec_s['name']}")
            st.caption(f"{spec_s['description']}")
            annotated_s, diag_s = run_inference(
                secondary_model,
                image_rgb,
                spec_s["is_zero_shot"],
                conf_threshold,
                iou_threshold,
            )
            st.image(annotated_s, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🚶 Pedestrian", diag_s["person_count"])
            c2.metric("🚗 Vehicle", diag_s["vehicle_count"])
            c3.metric("⏱️ Latensi", f"{diag_s['latency_ms']:.1f} ms")
            c4.metric("⚡ FPS", f"{diag_s['fps']:.1f}")

        # Summary Comparison Card
        st.markdown("---")
        p_gain = diag_p["person_count"] - diag_s["person_count"]
        if p_gain > 0:
            st.success(
                f"🎯 **Analisis:** Model utama **{spec_p['name']}** mendeteksi **+{p_gain} pejalan kaki lebih banyak** dibanding {spec_s['name']} pada kondisi cahaya ini!"
            )
        elif p_gain == 0:
            st.info(
                f"ℹ️ Kedua model mendeteksi jumlah pejalan kaki yang sama ({diag_p['person_count']} pejalan kaki)."
            )
        else:
            st.warning(
                f"⚠️ Model pembanding mendeteksi {-p_gain} pejalan kaki lebih banyak pada ambang batas saat ini."
            )
    else:
        # Single Model Display
        st.subheader(f"🎯 {spec_p['name']}")
        st.caption(spec_p["description"])
        annotated_img, diag = run_inference(
            primary_model,
            image_rgb,
            spec_p["is_zero_shot"],
            conf_threshold,
            iou_threshold,
        )
        st.image(annotated_img, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚶 Pejalan Kaki Terdeteksi", diag["person_count"])
        c2.metric("🚗 Kendaraan Terdeteksi", diag["vehicle_count"])
        c3.metric("⏱️ Latensi CPU", f"{diag['latency_ms']:.1f} ms")
        c4.metric("⚡ FPS Inferensi", f"{diag['fps']:.1f}")


def process_video_feed(
    video_source: Any,
    primary_model: YOLO,
    primary_model_key: str,
    enable_comparison: bool,
    secondary_model: Optional[YOLO],
    secondary_model_key: Optional[str],
    conf_threshold: float,
    iou_threshold: float,
    frame_skip: int = 1,
    is_live_camera: bool = False,
) -> None:
    """Stream and infer video frames in real-time with UI metrics updates.

    Args:
        video_source: Video file path or camera device index integer.
        primary_model: Loaded primary YOLO model.
        primary_model_key: Registry key of primary model.
        enable_comparison: Flag indicating side-by-side comparison mode.
        secondary_model: Loaded secondary model if comparison enabled.
        secondary_model_key: Registry key of secondary model.
        conf_threshold: Confidence threshold.
        iou_threshold: IoU NMS threshold.
        frame_skip: Number of frames to step (1 = process every frame).
        is_live_camera: True if capturing from live hardware webcam.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error(
            "❌ Tidak dapat membuka sumber video/kamera. Pastikan perangkat terhubung "
            "dan tidak sedang digunakan oleh aplikasi lain (seperti Zoom/Teams)."
        )
        return

    stop_stream = st.button("⏹️ Hentikan Pemutaran / Siaran", key="stop_stream_btn")

    spec_p = MODEL_REGISTRY[primary_model_key]
    spec_s = MODEL_REGISTRY.get(secondary_model_key) if (enable_comparison and secondary_model_key) else None

    if enable_comparison and spec_s and secondary_model:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader(f"🔹 {spec_p['name']}")
            frame_ph_p = st.empty()
            metric_ph_p = st.empty()
        with col_right:
            st.subheader(f"🔸 {spec_s['name']}")
            frame_ph_s = st.empty()
            metric_ph_s = st.empty()
    else:
        st.subheader(f"🎯 {spec_p['name']}")
        frame_ph_p = st.empty()
        metric_ph_p = st.empty()

    frame_counter: int = 0
    try:
        while cap.isOpened() and not stop_stream:
            ret, frame_bgr = cap.read()
            if not ret:
                if not is_live_camera:
                    st.info("ℹ️ Pemutaran video telah selesai.")
                break

            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            annotated_p, diag_p = run_inference(
                primary_model,
                frame_rgb,
                spec_p["is_zero_shot"],
                conf_threshold,
                iou_threshold,
            )
            frame_ph_p.image(annotated_p, use_container_width=True)
            metric_ph_p.markdown(
                f"**🚶 Pedestrian:** `{diag_p['person_count']}` | "
                f"**🚗 Kendaraan:** `{diag_p['vehicle_count']}` | "
                f"**⏱️ Latensi:** `{diag_p['latency_ms']:.1f} ms` | "
                f"**⚡ FPS:** `{diag_p['fps']:.1f}`"
            )

            if enable_comparison and spec_s and secondary_model:
                annotated_s, diag_s = run_inference(
                    secondary_model,
                    frame_rgb,
                    spec_s["is_zero_shot"],
                    conf_threshold,
                    iou_threshold,
                )
                frame_ph_s.image(annotated_s, use_container_width=True)
                metric_ph_s.markdown(
                    f"**🚶 Pedestrian:** `{diag_s['person_count']}` | "
                    f"**🚗 Kendaraan:** `{diag_s['vehicle_count']}` | "
                    f"**⏱️ Latensi:** `{diag_s['latency_ms']:.1f} ms` | "
                    f"**⚡ FPS:** `{diag_s['fps']:.1f}`"
                )

            time.sleep(0.01)
    finally:
        cap.release()


# ----------------------------------------------------------------------------
# Application UI
# ----------------------------------------------------------------------------
def main() -> None:
    """Streamlit application layout and event routing."""
    st.set_page_config(
        page_title="Deteksi Pedestrian Low-Light | K09",
        page_icon="🚶‍♂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main-header { font-size: 2.1rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
        .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='main-header'>🚶‍♂️ {APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Visi Komputer (IF0PP006) — Kelompok 09 | Evaluasi Inferensi Model Terkendali & Sidik Jari Kinerja</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------------
    # Sidebar: Model & Parameter Settings
    # ------------------------------------------------------------------------
    st.sidebar.header("⚙️ Konfigurasi Model & Inferensi")

    model_options = list(MODEL_REGISTRY.keys())
    primary_model_key = st.sidebar.selectbox(
        "Pilih Model Utama:",
        options=model_options,
        index=0,  # Exp_C_Gamma as default
        format_func=lambda k: MODEL_REGISTRY[k]["name"],
    )

    enable_comparison = st.sidebar.checkbox(
        "Bandingkan 2 Model (Side-by-Side)",
        value=True,
        help="Menampilkan komparasi deteksi berdampingan pada citra yang sama.",
    )

    secondary_model_key: Optional[str] = None
    if enable_comparison:
        remaining_options = [k for k in model_options if k != primary_model_key]
        default_secondary_idx = remaining_options.index("B0_ZeroShot") if "B0_ZeroShot" in remaining_options else 0
        secondary_model_key = st.sidebar.selectbox(
            "Pilih Model Pembanding:",
            options=remaining_options,
            index=default_secondary_idx,
            format_func=lambda k: MODEL_REGISTRY[k]["name"],
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Ambang Batas Deteksi")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold (Min Keyakinan)",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Ambang batas probabilitas deteksi (Operating Point recall-first = 0.25).",
    )
    iou_threshold = st.sidebar.slider(
        "NMS IoU Threshold (Supresi Overlap)",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Ambang batas tumpang-tindih bounding box untuk Non-Maximum Suppression.",
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Kunci Kategori Warna Bounding Box:**\n"
        "- 🔴 **Merah**: Pejalan Kaki (`person`)\n"
        "- 🔵 **Biru/Cyan**: Kendaraan (`Vehicle`)"
    )

    # ------------------------------------------------------------------------
    # Main Tabs
    # ------------------------------------------------------------------------
    tab_demo, tab_results, tab_failures, tab_about = st.tabs(
        ["🖼️ Live Demo Inferensi", "📊 Tabel Hasil Terpadu (CP10)", "🔍 Analisis Kasus Gagal (CP11)", "👥 Tim & Kontrak (CP00)"]
    )

    # ------------------------------------------------------------------------
    # TAB 1: Live Demo
    # ------------------------------------------------------------------------
    with tab_demo:
        input_source = st.radio(
            "Sumber Masukan Uji (Input Source):",
            options=[
                "Pilih Sampel Test Set (Citra Malam)",
                "Unggah Citra Sendiri (JPG/PNG)",
                "Snapshot Kamera Laptop (Foto Langsung)",
                "Live Kamera Laptop (Webcam Real-Time)",
                "Unggah Berkas Video (MP4/AVI/MOV)",
            ],
            horizontal=True,
        )

        image_rgb: Optional[np.ndarray] = None

        if input_source == "Pilih Sampel Test Set (Citra Malam)":
            selected_preset_label = st.selectbox(
                "Pilih Kasus Uji Malam Hari:",
                options=[label for label, _ in SAMPLE_PRESETS],
            )
            selected_filename = dict(SAMPLE_PRESETS)[selected_preset_label]
            preset_path = TEST_IMAGES_DIR / selected_filename

            if preset_path.exists():
                image_pil = Image.open(preset_path).convert("RGB")
                image_rgb = np.array(image_pil)
            else:
                st.warning(f"File sampel belum tersedia di `{preset_path}`. Silakan unggah citra.")

        elif input_source == "Unggah Citra Sendiri (JPG/PNG)":
            uploaded_file = st.file_uploader(
                "Unggah foto jalanan malam hari:", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                image_pil = Image.open(uploaded_file).convert("RGB")
                image_rgb = np.array(image_pil)

        elif input_source == "Snapshot Kamera Laptop (Foto Langsung)":
            st.info("📸 Arahkan kamera laptop ke arah subjek dan klik tombol di bawah untuk mengambil foto.")
            camera_pic = st.camera_input("Ambil foto langsung dari kamera laptop:")
            if camera_pic is not None:
                image_pil = Image.open(camera_pic).convert("RGB")
                image_rgb = np.array(image_pil)

        elif input_source == "Live Kamera Laptop (Webcam Real-Time)":
            st.info("📹 Deteksi langsung frame-by-frame dari kamera webcam laptop secara real-time.")
            cam_col1, cam_col2 = st.columns([1, 2])
            with cam_col1:
                cam_idx = st.number_input(
                    "Device Index Kamera:",
                    min_value=0,
                    max_value=5,
                    value=0,
                    step=1,
                    help="Index 0 adalah kamera utama/bawaan laptop.",
                )
                start_webcam = st.toggle("🔴 Aktifkan Kamera Laptop", value=False)
            with cam_col2:
                st.caption(
                    "Tips: Pastikan tidak ada aplikasi lain yang sedang mengunci kamera (Zoom, Teams, dsb). "
                    "Gunakan slider di bilah samping untuk mengubah Confidence & NMS IoU secara dinamis."
                )

            if start_webcam:
                primary_model = load_yolo_model(primary_model_key)
                secondary_model = (
                    load_yolo_model(secondary_model_key)
                    if (enable_comparison and secondary_model_key)
                    else None
                )
                if primary_model:
                    process_video_feed(
                        video_source=int(cam_idx),
                        primary_model=primary_model,
                        primary_model_key=primary_model_key,
                        enable_comparison=enable_comparison,
                        secondary_model=secondary_model,
                        secondary_model_key=secondary_model_key,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        frame_skip=1,
                        is_live_camera=True,
                    )

        elif input_source == "Unggah Berkas Video (MP4/AVI/MOV)":
            st.info("🎥 Unggah rekaman video lalu aktifkan pemutaran untuk inferensi deteksi per-frame.")
            uploaded_video = st.file_uploader(
                "Unggah berkas rekaman video (MP4, AVI, MOV, MKV):",
                type=["mp4", "avi", "mov", "mkv"],
            )
            if uploaded_video is not None:
                suffix = Path(uploaded_video.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                    tfile.write(uploaded_video.read())
                    temp_video_path = tfile.name

                v_col1, v_col2 = st.columns([1, 2])
                with v_col1:
                    start_video = st.toggle("▶️ Mulai Putar & Deteksi Video", value=False)
                with v_col2:
                    frame_skip_val = st.slider(
                        "Lewati Frame (Frame Skip untuk percepat CPU)",
                        min_value=1,
                        max_value=5,
                        value=1,
                        help="Nilai 2 berarti memproses setiap 2 frame sekali untuk menghemat CPU.",
                    )

                if start_video:
                    primary_model = load_yolo_model(primary_model_key)
                    secondary_model = (
                        load_yolo_model(secondary_model_key)
                        if (enable_comparison and secondary_model_key)
                        else None
                    )
                    if primary_model:
                        process_video_feed(
                            video_source=temp_video_path,
                            primary_model=primary_model,
                            primary_model_key=primary_model_key,
                            enable_comparison=enable_comparison,
                            secondary_model=secondary_model,
                            secondary_model_key=secondary_model_key,
                            conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold,
                            frame_skip=frame_skip_val,
                            is_live_camera=False,
                        )

                if not start_video:
                    try:
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                    except OSError:
                        pass

        # Static Image Rendering (Presets, Uploaded Image, or Snapshot Webcam)
        if image_rgb is not None:
            st.markdown("---")
            primary_model = load_yolo_model(primary_model_key)
            secondary_model = (
                load_yolo_model(secondary_model_key)
                if (enable_comparison and secondary_model_key)
                else None
            )

            if primary_model:
                render_static_detection(
                    image_rgb=image_rgb,
                    primary_model=primary_model,
                    primary_model_key=primary_model_key,
                    enable_comparison=enable_comparison,
                    secondary_model=secondary_model,
                    secondary_model_key=secondary_model_key,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                )

    # ------------------------------------------------------------------------
    # TAB 2: Unified Benchmark Results (CP10)
    # ------------------------------------------------------------------------
    with tab_results:
        st.subheader("📊 Laporan Evaluasi Terpadu (Single Split, Per-Class Metrics)")
        st.write(
            "Hasil resmi pengujian komparatif seluruh model pada **split uji bebas-kebocoran (`dataset_dedup/test`)** "
            "yang memuat **373 citra** dan **79 kotak ground-truth pejalan kaki**."
        )

        if UNIFIED_RESULTS_PATH.exists():
            with open(UNIFIED_RESULTS_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content)
        else:
            st.warning("Berkas `results/experiments/unified_results.md` belum dibuat.")

    # ------------------------------------------------------------------------
    # TAB 3: Failure Cases Analysis (CP11)
    # ------------------------------------------------------------------------
    with tab_failures:
        st.subheader("🔍 Analisis Kasus Gagal Deteksi Pejalan Kaki")
        st.write(
            "Identifikasi sistematis kasus di mana model melewatkan pejalan kaki (*False Negatives*) "
            "atau memicu alarm palsu (*False Positives*) berdasarkan 5 kategori heuristik akademis."
        )

        grid_image_path = PROJECT_ROOT / "results" / "experiments" / "failure_cases.png"
        if grid_image_path.exists():
            st.image(str(grid_image_path), caption="Visual Evidence: Galeri Kasus Gagal Model", use_container_width=True)

        if FAILURE_CASES_PATH.exists():
            with open(FAILURE_CASES_PATH, "r", encoding="utf-8") as f:
                fail_content = f.read()
            st.markdown(fail_content)
        else:
            st.warning("Berkas `results/experiments/failure_cases.md` belum tersedia.")

    # ------------------------------------------------------------------------
    # TAB 4: About & Contributing (CP00)
    # ------------------------------------------------------------------------
    with tab_about:
        st.subheader("👥 Anggota Kelompok 09 & Kontrak Kerja")
        st.markdown(
            """
            | Nama Lengkap | NIM | Peran Utama | Fokus Tanggung Jawab |
            |:---|:---|:---|:---|
            | **Krisantus Amka Ginting** | 412024025 | Data & Preprocessing Lead | Audit kebocoran scene Roboflow, pipeline CLAHE/Gamma, rebuild dataset dedup |
            | **Linardi Halim** | 212024005 | Documentation & Reporting Lead | Naskah laporan teknis, sinkronisasi tracker CP00–CP13, slide deck sidang |
            | **Kristofer Ryan Giggs Eka Saputra** | 412024025 | Model & Training Lead | Pelatihan eksperimen A/B/C di Colab GPU T4, evaluasi metrik terpadu Wilson CI |
            | **Leonard Franscois** | 412024035 | Demo & Deployment Lead | Aplikasi web demo interaktif Streamlit, benchmark throughput & latensi CPU |
            """
        )
        st.markdown("---")
        st.markdown(
            "**Repositori GitHub:** [`artstzyy780/comvis-project-kelompok-9`](https://github.com/artstzyy780/comvis-project-kelompok-9)\n\n"
            "**Prinsip Saintifik Utama:** *Anti-Leakage Data Split (Scene-level Perceptual Hashing), Reproducibility (Fixed Seed 42), Recall-First Operating Point.*"
        )


if __name__ == "__main__":
    main()
