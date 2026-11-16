"""Flask web application for low-light pedestrian detection (Kelompok 09).

This application provides interactive inference and side-by-side model comparison
for the Computer Vision Final Project (IF0PP006).

Migrated from Streamlit to Flask for a cleaner, more professional interface.
"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from PIL import Image
from ultralytics import YOLO

from src.adaptive_postprocessing import (
    AdaptiveIlluminationProfile,
    compute_adaptive_illumination_profile,
)

# ----------------------------------------------------------------------------
# Constants & Configuration
# ----------------------------------------------------------------------------
APP_TITLE = "Deteksi Pejalan Kaki pada Kondisi Minim Cahaya"
PROJECT_ROOT = Path(__file__).resolve().parent
TEST_IMAGES_DIR = PROJECT_ROOT / "dataset_dedup" / "test" / "images"
UNIFIED_RESULTS_PATH = PROJECT_ROOT / "results" / "experiments" / "unified_results.md"
FAILURE_CASES_PATH = PROJECT_ROOT / "results" / "experiments" / "failure_cases.md"
FAILURE_GRID_PATH = PROJECT_ROOT / "results" / "experiments" / "failure_cases.png"

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Exp_D_Robust": {
        "name": "Exp-D: Robust Fine-Tuned (Anti-Forgetting & Multi-Scale)",
        "path": PROJECT_ROOT / "hasil_final_model" / "hasil_final_model" / "Exp_D_Robust_best.pt"
        if (PROJECT_ROOT / "hasil_final_model" / "hasil_final_model" / "Exp_D_Robust_best.pt").exists()
        else PROJECT_ROOT / "Exp_D_Robust_best.pt",
        "description": "Model transfer learning terkontrol (freeze=10, lr0=0.001, scale=0.5) untuk deteksi pejalan kaki jarak dekat dan jauh pada kondisi malam dan siang.",
        "is_zero_shot": False,
        "badge": "Robust Transfer Model",
    },
    "Exp_A_Standard": {
        "name": "Exp-A: Standard Low-Light Fine-Tuned (Best on Clean Test Set)",
        "path": PROJECT_ROOT / "hasil_final_model" / "hasil_final_model" / "Exp_A_Standard_best.pt",
        "description": "Model fine-tuning standar dengan performa tertinggi pada test set bersih (Recall 81.01%, Precision 62.75%, mAP 83.93%).",
        "is_zero_shot": False,
        "badge": "Best Performing Model",
    },
    "Exp_C_Gamma": {
        "name": "Exp-C: Gamma Correction (Proposed Method)",
        "path": PROJECT_ROOT / "hasil_final_model" / "hasil_final_model" / "Exp_C_Gamma_best.pt",
        "description": "Model dengan augmentasi pencahayaan non-linear (gamma=0.6). Recall 65.82% pada test set bersih.",
        "is_zero_shot": False,
        "badge": "Proposed Method",
    },
    "Exp_B_CLAHE": {
        "name": "Exp-B: L-Channel CLAHE (Proposed Method)",
        "path": PROJECT_ROOT / "hasil_final_model" / "hasil_final_model" / "Exp_B_CLAHE_best.pt",
        "description": "Model dengan kontras adaptif CLAHE pada kanal-L CIE L*a*b*. Sensitif terhadap amplifikasi noise pada kondisi malam.",
        "is_zero_shot": False,
        "badge": "Proposed Method",
    },
    "B0_ZeroShot": {
        "name": "B0: Zero-Shot Baseline (COCO Pretrained)",
        "path": PROJECT_ROOT / "yolov8n.pt",
        "description": "YOLOv8n pretrained MS COCO tanpa adaptasi domain malam hari (Recall rendah pada kondisi low-light).",
        "is_zero_shot": True,
        "badge": "Baseline",
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

# BGR colors for OpenCV drawing
CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (240, 180, 0),   # Vehicle: cyan/blue-ish
    1: (40, 40, 240),   # person: red
}

SAMPLE_PRESETS: List[Tuple[str, str]] = [
    ("Pejalan kaki siluet gelap (Scene 14969583)", "14969583-55cd5dd2_augmented_1_jpg.rf.47834ba812a385f88105c1a8e60ef8f9.jpg"),
    ("Pejalan kaki menyeberang jalan (Scene 1a5d0011)", "1a5d0011-c34c50cc_augmented_0_jpg.rf.46301686bd9a0f1df9bac672db570f74.jpg"),
    ("Pejalan kaki jarak menengah (Scene 08eff217)", "08eff217-6061163f_augmented_1_jpg.rf.9435b8f3effb43e7eaadb8ab2adb4f90.jpg"),
    ("Kepadatan kendaraan malam (Scene 1f08ba8a)", "1f08ba8a-1e15976e_augmented_0_jpg.rf.ce5e0d948ba77840d1db22a0084e62a1.jpg"),
    ("Siluet minim cahaya ekstrem (Scene 231bdb08)", "231bdb08-bb0a4a36_augmented_0_jpg.rf.473b90888a76e7c43ae1dfcbc63acc03.jpg"),
]

# ----------------------------------------------------------------------------
# App & Model Cache
# ----------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB

_model_cache: Dict[str, YOLO] = {}
_model_lock = threading.Lock()


def get_model(model_key: str) -> Optional[YOLO]:
    """Load and cache a YOLO model by registry key."""
    spec = MODEL_REGISTRY.get(model_key)
    if not spec:
        return None
    with _model_lock:
        if model_key in _model_cache:
            return _model_cache[model_key]
        weights_path: Path = spec["path"]
        if not weights_path.exists():
            return None
        model = YOLO(str(weights_path))
        _model_cache[model_key] = model
        return model


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------

def run_inference(
    model: YOLO,
    image_bgr: np.ndarray,
    is_zero_shot: bool,
    conf_threshold: float,
    iou_threshold: float,
    adaptive_threshold: bool = False,
    enable_tracking: bool = False,
    imgsz: int = 640,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run inference (or tracking) and draw annotations on a BGR image."""
    illum_profile: Optional[AdaptiveIlluminationProfile] = None
    effective_conf: float = conf_threshold

    if adaptive_threshold:
        illum_profile = compute_adaptive_illumination_profile(
            image_bgr, base_conf=conf_threshold, is_bgr=True
        )
        effective_conf = illum_profile.effective_conf_threshold

    start_time = time.perf_counter()
    tracking_active = enable_tracking
    if enable_tracking:
        try:
            results = model.track(
                source=image_bgr,
                persist=True,
                tracker="bytetrack.yaml",
                conf=effective_conf,
                iou=iou_threshold,
                imgsz=imgsz,
                device="cpu",
                verbose=False,
            )
        except Exception:
            tracking_active = False
            results = model.predict(
                source=image_bgr,
                conf=effective_conf,
                iou=iou_threshold,
                imgsz=imgsz,
                device="cpu",
                verbose=False,
            )
    else:
        results = model.predict(
            source=image_bgr,
            conf=effective_conf,
            iou=iou_threshold,
            imgsz=imgsz,
            device="cpu",
            verbose=False,
        )
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    annotated = image_bgr.copy()
    person_count = 0
    vehicle_count = 0
    detections_list: List[Dict[str, Any]] = []

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy_arr = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                boxes.id.cpu().numpy().astype(int)
                if (tracking_active and boxes.id is not None)
                else None
            )
            for idx, (xyxy, conf, raw_cls) in enumerate(zip(xyxy_arr, conf_arr, cls_arr)):
                target_cls: Optional[int] = None
                if is_zero_shot:
                    target_cls = COCO_TO_TARGET_MAPPING.get(raw_cls)
                elif raw_cls in TARGET_CLASSES:
                    target_cls = raw_cls
                if target_cls is None:
                    continue

                if target_cls == 1:
                    person_count += 1
                else:
                    vehicle_count += 1

                x1, y1, x2, y2 = map(int, xyxy)
                color = CLASS_COLORS_BGR[target_cls]
                tid_prefix = ""
                track_id_val = None
                if track_ids is not None and idx < len(track_ids):
                    track_id_val = int(track_ids[idx])
                    tid_prefix = f"ID:{track_id_val} "
                label_text = f"{tid_prefix}{TARGET_CLASSES[target_cls]} {conf:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                label_y1 = max(0, y1 - text_h - 6)
                cv2.rectangle(
                    annotated,
                    (x1, label_y1),
                    (x1 + text_w + 6, label_y1 + text_h + 6),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label_text,
                    (x1 + 3, label_y1 + text_h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                detections_list.append(
                    {
                        "class": TARGET_CLASSES[target_cls],
                        "confidence": round(float(conf), 4),
                        "box": [x1, y1, x2, y2],
                        "track_id": track_id_val,
                    }
                )

    diagnostics: Dict[str, Any] = {
        "latency_ms": round(latency_ms, 2),
        "fps": round(fps, 2),
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "total_detections": person_count + vehicle_count,
        "detections": detections_list,
        "effective_conf": round(effective_conf, 2),
        "tracking_active": tracking_active,
        "illumination": None,
    }
    if illum_profile is not None:
        diagnostics["illumination"] = {
            "luminance": illum_profile.luminance,
            "glare_ratio": illum_profile.glare_ratio,
            "effective_conf_threshold": illum_profile.effective_conf_threshold,
            "mode_label": illum_profile.mode_label,
            "is_glare_detected": illum_profile.is_glare_detected,
            "is_low_light_boost_active": illum_profile.is_low_light_boost_active,
        }
    return annotated, diagnostics


def encode_image_b64(image_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def read_image_to_bgr(file_storage) -> Optional[np.ndarray]:
    try:
        data = file_storage.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def read_markdown(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.route("/")
def index():
    available_presets = []
    for label, filename in SAMPLE_PRESETS:
        available_presets.append({
            "label": label,
            "filename": filename,
            "exists": (TEST_IMAGES_DIR / filename).exists(),
        })
    return render_template(
        "index.html",
        app_title=APP_TITLE,
        models={k: {"name": v["name"], "description": v["description"], "badge": v["badge"], "available": v["path"].exists()}
                for k, v in MODEL_REGISTRY.items()},
        model_order=list(MODEL_REGISTRY.keys()),
        presets=available_presets,
    )


@app.route("/results")
def results_page():
    content = read_markdown(UNIFIED_RESULTS_PATH)
    return render_template("results.html", app_title=APP_TITLE, content=content,
                           path=str(UNIFIED_RESULTS_PATH))


@app.route("/failures")
def failures_page():
    content = read_markdown(FAILURE_CASES_PATH)
    has_grid = FAILURE_GRID_PATH.exists()
    return render_template("failures.html", app_title=APP_TITLE, content=content,
                           has_grid=has_grid, path=str(FAILURE_CASES_PATH))


@app.route("/about")
def about_page():
    return render_template("about.html", app_title=APP_TITLE)


@app.route("/static/failure_grid")
def failure_grid():
    if not FAILURE_GRID_PATH.exists():
        return "Not found", 404
    return send_from_directory(FAILURE_GRID_PATH.parent, FAILURE_GRID_PATH.name)


@app.route("/api/preset_image/<path:filename>")
def preset_image(filename: str):
    # serve preset as base64/json for client preview convenience
    safe_name = Path(filename).name
    img_path = TEST_IMAGES_DIR / safe_name
    if not img_path.exists():
        return jsonify({"error": "File tidak ditemukan."}), 404
    img = cv2.imread(str(img_path))
    if img is None:
        return jsonify({"error": "Gagal membaca citra."}), 400
    return jsonify({"image": encode_image_b64(img)})


@app.route("/api/detect", methods=["POST"])
def api_detect():
    try:
        primary_key = request.form.get("primary_model", "Exp_D_Robust")
        compare = request.form.get("compare", "false").lower() == "true"
        secondary_key = request.form.get("secondary_model")
        conf = float(request.form.get("conf", 0.25))
        iou = float(request.form.get("iou", 0.45))
        adaptive = request.form.get("adaptive", "false").lower() == "true"
        imgsz = int(request.form.get("imgsz", 640))
        source_type = request.form.get("source_type", "upload")

        image_bgr = None
        if source_type == "preset":
            fname = Path(request.form.get("preset_filename", "")).name
            path = TEST_IMAGES_DIR / fname
            if path.exists():
                image_bgr = cv2.imread(str(path))
        else:
            file = request.files.get("image")
            if file:
                image_bgr = read_image_to_bgr(file)

        if image_bgr is None:
            return jsonify({"error": "Citra masukan tidak valid atau tidak ditemukan."}), 400

        primary_model = get_model(primary_key)
        if primary_model is None:
            return jsonify({"error": f"Model utama '{primary_key}' tidak tersedia."}), 400
        primary_spec = MODEL_REGISTRY[primary_key]

        annotated_p, diag_p = run_inference(
            primary_model, image_bgr, primary_spec["is_zero_shot"],
            conf, iou, adaptive_threshold=adaptive, enable_tracking=False, imgsz=imgsz,
        )
        response: Dict[str, Any] = {
            "primary": {
                "model_key": primary_key,
                "model_name": primary_spec["name"],
                "image": encode_image_b64(annotated_p),
                "diagnostics": diag_p,
            }
        }

        if compare and secondary_key and secondary_key != primary_key:
            secondary_model = get_model(secondary_key)
            if secondary_model is not None:
                secondary_spec = MODEL_REGISTRY[secondary_key]
                annotated_s, diag_s = run_inference(
                    secondary_model, image_bgr, secondary_spec["is_zero_shot"],
                    conf, iou, adaptive_threshold=adaptive, enable_tracking=False, imgsz=imgsz,
                )
                response["secondary"] = {
                    "model_key": secondary_key,
                    "model_name": secondary_spec["name"],
                    "image": encode_image_b64(annotated_s),
                    "diagnostics": diag_s,
                }

        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": f"Terjadi kesalahan saat inferensi: {exc}"}), 500


def mjpeg_generator(
    source: Any,
    primary_key: str,
    secondary_key: Optional[str],
    conf: float,
    iou: float,
    adaptive: bool,
    tracking: bool,
    imgsz: int,
    compare: bool,
):
    """Yield MJPEG frames for webcam or video file streaming."""
    # Try to open with best backend on Windows
    cap = None
    if isinstance(source, int):
        for backend in ([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]):
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                break
            cap.release()
            cap = None
        if cap is None:
            return
    else:
        cap = cv2.VideoCapture(source)

    if not cap or not cap.isOpened():
        return

    primary_model = get_model(primary_key)
    if primary_model is None:
        cap.release()
        return
    primary_spec = MODEL_REGISTRY[primary_key]
    secondary_model = get_model(secondary_key) if (compare and secondary_key) else None
    secondary_spec = MODEL_REGISTRY.get(secondary_key) if secondary_key else None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if not isinstance(source, int):
                    break  # end of video file
                time.sleep(0.01)
                continue

            annotated_p, _ = run_inference(
                primary_model, frame, primary_spec["is_zero_shot"],
                conf, iou, adaptive_threshold=adaptive, enable_tracking=tracking, imgsz=imgsz,
            )
            if secondary_model and secondary_spec:
                annotated_s, _ = run_inference(
                    secondary_model, frame, secondary_spec["is_zero_shot"],
                    conf, iou, adaptive_threshold=adaptive, enable_tracking=tracking, imgsz=imgsz,
                )
                # resize to same height and concatenate side by side
                h = max(annotated_p.shape[0], annotated_s.shape[0])
                def resize_h(img, h):
                    if img.shape[0] == h:
                        return img
                    w = int(img.shape[1] * h / img.shape[0])
                    return cv2.resize(img, (w, h))
                annotated_p = resize_h(annotated_p, h)
                annotated_s = resize_h(annotated_s, h)
                output = np.hstack([annotated_p, annotated_s])
            else:
                output = annotated_p

            ok, buf = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            frame_bytes = buf.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        cap.release()


@app.route("/api/video_feed")
def video_feed():
    source_type = request.args.get("source", "camera")
    cam_index = int(request.args.get("cam_index", 0))
    primary_key = request.args.get("primary_model", "Exp_D_Robust")
    secondary_key = request.args.get("secondary_model")
    compare = request.args.get("compare", "false").lower() == "true"
    conf = float(request.args.get("conf", 0.25))
    iou = float(request.args.get("iou", 0.45))
    adaptive = request.args.get("adaptive", "false").lower() == "true"
    tracking = request.args.get("tracking", "false").lower() == "true"
    imgsz = int(request.args.get("imgsz", 416))

    if source_type == "camera":
        source = cam_index
    else:
        return "Unsupported source for streaming endpoint.", 400

    return Response(
        mjpeg_generator(source, primary_key, secondary_key, conf, iou, adaptive, tracking, imgsz, compare),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
