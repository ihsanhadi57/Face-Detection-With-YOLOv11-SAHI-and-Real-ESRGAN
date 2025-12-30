# -*- coding: utf-8 -*-
import streamlit as st

# Panggil st.set_page_config() SATU KALI di paling atas
st.set_page_config(layout="wide", page_title="Inference")

from PIL import Image
import numpy as np
import cv2
import sys
import os
from pathlib import Path
import time
import torch
import pyiqa
from torchvision.transforms.functional import to_tensor
import pandas as pd
import zipfile
from io import BytesIO

# Tambahkan parent directory ke Python path agar bisa import dari utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import dari skrip Anda
from sahi.predict import get_sliced_prediction
from sahi.prediction import PredictionResult
from utils.yolo_wrapper import YOLOv11PoseDetectionModel
from utils.visualization import draw_detections, save_face_crops
from utils.enhancer import FaceEnhancer

# ============================================
# KONSTANTA: IMAGE SIZE SESUAI TRAINING MODEL
# ============================================
MODEL_IMAGE_SIZE = 1024  # Model dilatih dengan imgsz=1024, batch=16

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px; 
    }
    .stTabs [data-baseweb="tab"] { 
        padding: 10px 20px; 
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 5px 5px 0 0;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .stTabs [aria-selected="true"] { 
        background-color: #1f77b4 !important;
        color: white !important;
        border-color: #1f77b4 !important;
    }
    
    /* Caption styling */
    .caption { 
        font-size: 0.85em; 
        color: #666; 
        text-align: center; 
    }
    
    /* Image container */
    .image-container { 
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 5px; 
        padding: 5px; 
        margin-bottom: 10px; 
    }
    
    /* Section header */
    .section-header { 
        font-size: 1.2rem; 
        font-weight: 600; 
        margin-top: 2rem; 
        margin-bottom: 1rem;
        padding: 0.75rem;
        background: linear-gradient(90deg, rgba(31, 119, 180, 0.15) 0%, rgba(31, 119, 180, 0.05) 100%);
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        color: inherit;
    }
    
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .section-header {
            background: linear-gradient(90deg, rgba(31, 119, 180, 0.25) 0%, rgba(31, 119, 180, 0.1) 100%);
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI-FUNGSI LOGIKA ---

@st.cache_resource
def load_yolo_model(model_path, confidence, device, image_size):
    """Load YOLO model dengan image_size yang benar"""
    if not Path(model_path).exists():
        st.error(f"Path model YOLO tidak ditemukan: {model_path}")
        return None
    try:
        model = YOLOv11PoseDetectionModel(
            model_path=model_path, 
            confidence_threshold=confidence, 
            device=device, 
            image_size=image_size,
            load_at_init=True
        )
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        return None

@st.cache_resource
def load_enhancer_model(model_name):
    try:
        return FaceEnhancer(model_name=model_name)
    except Exception as e:
        st.error(f"Gagal memuat model Enhancer: {e}")
        return None

@st.cache_resource
def load_iqa_metric(metric_name, device):
    try:
        return pyiqa.create_metric(metric_name, device=device)
    except Exception as e:
        return None

def get_available_metrics(device):
    """Deteksi metrik IQA yang tersedia"""
    available = {}
    
    metrics_to_try = {
        'niqe': {'name': 'NIQE', 'inverse': True, 'description': 'Natural Image Quality Evaluator'},
        'brisque': {'name': 'BRISQUE', 'inverse': True, 'description': 'Blind/Referenceless Image Spatial Quality Evaluator'},
    }
    
    for key, info in metrics_to_try.items():
        try:
            metric = load_iqa_metric(key, device)
            if metric is not None:
                available[key] = {
                    'metric': metric,
                    'name': info['name'],
                    'inverse': info['inverse'],
                    'description': info['description']
                }
        except Exception as e:
            pass
    
    return available

def calculate_iqa_scores(image_np, device, available_metrics):
    """Menghitung skor IQA berdasarkan metrik yang tersedia"""
    scores = {}
    
    if image_np is None or image_np.size == 0:
        return scores
    
    if image_np.shape[0] < 10 or image_np.shape[1] < 10:
        return scores
    
    try:
        if image_np.max() > 1.0:
            image_np = image_np.astype(np.float32) / 255.0
        
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
        
        img_tensor = to_tensor(image_np).unsqueeze(0).to(device)
        
        for metric_key, metric_info in available_metrics.items():
            try:
                score = metric_info['metric'](img_tensor).item()
                scores[metric_key] = score
            except Exception as e:
                continue
        
    except Exception as e:
        pass
    
    return scores

def calculate_face_crop_quality(sahi_results, image_np, device, available_metrics):
    """Analisis kualitas per face crop"""
    face_quality_scores = []
    
    for idx, pred in enumerate(sahi_results.object_prediction_list):
        bbox = pred.bbox
        x1, y1, x2, y2 = map(int, [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
        
        h, w = image_np.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        face_crop = image_np[y1:y2, x1:x2]
        
        if face_crop.size > 0 and face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
            face_scores = calculate_iqa_scores(face_crop, device, available_metrics)
            
            face_info = {
                'face_id': idx + 1,
                'bbox': (x1, y1, x2, y2),
                'width': x2 - x1,
                'height': y2 - y1,
                'confidence': pred.score.value if hasattr(pred.score, 'value') else 0.0,
                **face_scores
            }
            
            face_quality_scores.append(face_info)
    
    return face_quality_scores

def display_iqa_metrics(scores, available_metrics=None, original_scores=None):
    """Menampilkan skor IQA dalam format horizontal dengan indikator warna"""
    
    if not scores or not available_metrics:
        st.info("Tidak ada metrik yang tersedia")
        return
    
    metric_keys = [k for k in scores.keys() if k in available_metrics]
    num_metrics = len(metric_keys)
    
    if num_metrics == 0:
        st.info("Tidak ada skor yang berhasil dihitung")
        return
    
    metric_order = ['niqe', 'brisque']
    
    sorted_keys = []
    for key in metric_order:
        if key in metric_keys:
            sorted_keys.append(key)
    
    metric_texts = []
    for metric_key in sorted_keys:
        metric_info = available_metrics[metric_key]
        current_value = scores[metric_key]
        
        indicator = ""
        if original_scores and metric_key in original_scores:
            original_value = original_scores[metric_key]
            delta = current_value - original_value
            
            if metric_info['inverse']:
                if delta < -0.5:
                    indicator = "🟢"
                elif delta < 0:
                    indicator = "🟡"
                elif delta > 0.5:
                    indicator = "🔴"
                else:
                    indicator = "🟠"
            else:
                if delta > 0.05:
                    indicator = "🟢"
                elif delta > 0:
                    indicator = "🟡"
                elif delta < -0.05:
                    indicator = "🔴"
                else:
                    indicator = "🟠"
        
        metric_texts.append(f"{indicator} **{metric_info['name']}:** {current_value:.3f}")
    
    st.write(" | ".join(metric_texts))

def display_face_quality_analysis(face_quality_scores, available_metrics, crops_dir):
    """Menampilkan analisis kualitas per face crop"""
    
    if not face_quality_scores:
        st.warning("Tidak ada data kualitas face crop untuk ditampilkan")
        return
    
    st.markdown('<div class="section-header">📊 Analisis Kualitas Detail per Wajah</div>', unsafe_allow_html=True)
    
    crop_files = sorted([f for f in os.listdir(crops_dir) if f.endswith(('.jpg', '.png'))])
    
    for i in range(0, len(face_quality_scores), 3):
        cols = st.columns(3)
        
        for j in range(3):
            if i + j < len(face_quality_scores):
                face_info = face_quality_scores[i + j]
                face_id = face_info['face_id']
                
                with cols[j]:
                    if i + j < len(crop_files):
                        crop_path = os.path.join(crops_dir, crop_files[i + j])
                        if os.path.exists(crop_path):
                            img = Image.open(crop_path)
                            st.image(img, use_container_width=True)
                    
                    st.caption(f"**Face #{face_id}** | {face_info['width']}x{face_info['height']}px | Conf: {face_info['confidence']:.2f}")
                    
                    st.write("**Quality Scores:**")
                    
                    metric_order = ['niqe', 'brisque']
                    
                    has_scores = False
                    for metric_key in metric_order:
                        if metric_key in available_metrics and metric_key in face_info:
                            has_scores = True
                            metric_info = available_metrics[metric_key]
                            
                            if metric_info['inverse']:
                                quality_indicator = "🟢" if face_info[metric_key] < 5.0 else "🟡" if face_info[metric_key] < 10.0 else "🔴"
                            else:
                                quality_indicator = "🟢" if face_info[metric_key] > 0.5 else "🟡" if face_info[metric_key] > 0.3 else "🔴"
                            
                            st.text(f"{quality_indicator} {metric_info['name']}: {face_info[metric_key]:.3f}")
                    
                    if not has_scores:
                        st.info("⚠️ Tidak ada skor yang berhasil dihitung untuk wajah ini")

def perform_enhancement(enhancer_model, image_np):
    enhanced_image, success = enhancer_model.enhance_image(image_np)
    return enhanced_image if success else None

def perform_sahi_detection(detection_model, image_np, slice_h, slice_w, overlap_r, match_threshold, temp_dir="temp_streamlit"):
    """
    ✅ FIXED: Perform SAHI detection with IOS metric (optimal from grid search)
    """
    os.makedirs(temp_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_dir, "temp_sahi_input.jpg")
    cv2.imwrite(temp_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    sahi_result = get_sliced_prediction(
        temp_image_path, 
        detection_model, 
        slice_height=slice_h, 
        slice_width=slice_w, 
        overlap_height_ratio=overlap_r, 
        overlap_width_ratio=overlap_r,
        postprocess_match_metric='IOS',             
        postprocess_match_threshold=match_threshold 
    )
    
    sahi_result.object_prediction_list = detection_model.attach_keypoints_to_predictions(sahi_result.object_prediction_list)
    return sahi_result

def perform_standard_detection(detection_model, image_np, temp_dir="temp_streamlit"):
    """
    Performs detection without SAHI and returns a SAHI-compatible result object.
    
    ✅ FIXED VERSION with comprehensive debugging
    """
    import os
    import cv2
    from sahi.prediction import PredictionResult
    
    os.makedirs(temp_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_dir, "temp_sahi_input.jpg")
    cv2.imwrite(temp_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    print(f"\n{'='*60}")
    print(f"[DEBUG perform_standard_detection]")
    print(f"{'='*60}")
    
    # Step 1: Perform inference
    print(f"\n[Step 1] perform_inference")
    detection_model.perform_inference(image_np)
    
    # Step 2: Create object prediction list
    print(f"\n[Step 2] _create_object_prediction_list_from_original_predictions")
    detection_model._create_object_prediction_list_from_original_predictions(
        shift_amount_list=[[0, 0]],
        full_shape_list=[[image_np.shape[0], image_np.shape[1]]]
    )
    
    # Step 3: Extract predictions (CRITICAL - ini yang bikin masalah!)
    print(f"\n[Step 3] Extract predictions from wrapper")
    
    # Check available attributes
    available_attrs = [attr for attr in dir(detection_model) if 'prediction' in attr.lower()]
    print(f"  Available prediction attributes: {available_attrs}")
    
    # Try different attribute names
    object_prediction_list = []
    
    # Try 1: _object_prediction_list_per_image (private attribute)
    if hasattr(detection_model, '_object_prediction_list_per_image'):
        print(f"  ✓ Found: _object_prediction_list_per_image")
        temp_list = detection_model._object_prediction_list_per_image
        print(f"    Type: {type(temp_list)}")
        print(f"    Length: {len(temp_list) if isinstance(temp_list, list) else 'N/A'}")
        
        if isinstance(temp_list, list) and len(temp_list) > 0:
            object_prediction_list = temp_list[0]
            print(f"    Extracted list[0]: {len(object_prediction_list)} items")
    
    # Try 2: object_prediction_list_per_image (public attribute) 
    elif hasattr(detection_model, 'object_prediction_list_per_image'):
        print(f"  ✓ Found: object_prediction_list_per_image")
        temp_list = detection_model.object_prediction_list_per_image
        print(f"    Type: {type(temp_list)}")
        print(f"    Length: {len(temp_list) if isinstance(temp_list, list) else 'N/A'}")
        
        if isinstance(temp_list, list) and len(temp_list) > 0:
            object_prediction_list = temp_list[0]
            print(f"    Extracted list[0]: {len(object_prediction_list)} items")
    
    # Try 3: object_prediction_list (flat list)
    elif hasattr(detection_model, 'object_prediction_list'):
        print(f"  ✓ Found: object_prediction_list")
        temp_list = detection_model.object_prediction_list
        print(f"    Type: {type(temp_list)}")
        
        # ❌ CRITICAL: Jangan ambil [0] jika ini sudah flat list!
        if isinstance(temp_list, list):
            if len(temp_list) > 0 and isinstance(temp_list[0], list):
                # Nested list [[pred1, pred2, ...]]
                object_prediction_list = temp_list[0]
                print(f"    Nested list detected, extracted [0]: {len(object_prediction_list)} items")
            else:
                # Flat list [pred1, pred2, ...]
                object_prediction_list = temp_list
                print(f"    Flat list detected: {len(object_prediction_list)} items")
    
    else:
        print(f"  ❌ No prediction list found!")
    
    # Validate
    if not isinstance(object_prediction_list, list):
        print(f"  ⚠️ WARNING: object_prediction_list is not a list! Type: {type(object_prediction_list)}")
        object_prediction_list = [object_prediction_list] if object_prediction_list else []
    
    print(f"\n  ✅ Final extracted list: {len(object_prediction_list)} predictions")
    
    # Step 4: Attach keypoints
    print(f"\n[Step 4] attach_keypoints_to_predictions")
    print(f"  Input: {len(object_prediction_list)} predictions")
    
    object_prediction_list = detection_model.attach_keypoints_to_predictions(
        object_prediction_list
    )
    
    print(f"  Output: {len(object_prediction_list)} predictions")
    
    # Step 5: Create PredictionResult
    print(f"\n[Step 5] Create PredictionResult")
    print(f"  Input: {len(object_prediction_list)} predictions")
    
    result = PredictionResult(
        image=image_np,
        object_prediction_list=object_prediction_list,
        durations_in_seconds={'prediction': 0.0}
    )
    
    print(f"  Result type: {type(result.object_prediction_list)}")
    print(f"  Result length: {len(result.object_prediction_list)}")
    
    print(f"\n{'='*60}")
    print(f"[END DEBUG] Final: {len(result.object_prediction_list)} predictions")
    print(f"{'='*60}\n")
    
    return result

def process_single_image(image_np, image_name, yolo_model, enhancer_model, enable_sahi, enable_enhancer, 
                         slice_size, overlap_ratio, sahi_match_threshold, 
                         device, available_metrics, temp_dir, batch_mode=False):
    """Fungsi untuk memproses satu gambar (IOS metric fixed)"""
    
    results = {
        'image_name': image_name,
        'num_detections': 0,
        'detect_time': 0,
        'enhance_time': 0,
        'original_scores': {},
        'enhanced_scores': {},
        'face_quality_scores': [],
        'viz_paths': {},
        'crop_paths': []
    }
    
    original_scores = calculate_iqa_scores(image_np, device, available_metrics)
    results['original_scores'] = original_scores
    
    detection_input_image = image_np
    enhanced_scores = None
    
    if enable_enhancer and enhancer_model:
        start_time = time.time()
        enhanced_image = perform_enhancement(enhancer_model, image_np)
        results['enhance_time'] = time.time() - start_time
        
        if enhanced_image is not None:
            detection_input_image = enhanced_image
            enhanced_scores = calculate_iqa_scores(enhanced_image, device, available_metrics)
            results['enhanced_scores'] = enhanced_scores
    
    start_time = time.time()
    
    image_temp_dir = os.path.join(temp_dir, f"image_{image_name.replace('.', '_')}")
    os.makedirs(image_temp_dir, exist_ok=True)
    crops_dir = os.path.join(image_temp_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    if enable_sahi:
        # ✅ IOS metric is fixed (optimal from grid search)
        sahi_results = perform_sahi_detection(yolo_model, detection_input_image, slice_size, slice_size, 
                                              overlap_ratio, sahi_match_threshold, image_temp_dir)
    else:
        sahi_results = perform_standard_detection(yolo_model, detection_input_image, image_temp_dir)
    
    results['detect_time'] = time.time() - start_time
    results['num_detections'] = len(sahi_results.object_prediction_list)
    
    viz_input_path = os.path.join(image_temp_dir, "temp_sahi_input.jpg")
    
    viz_output_clean_path = os.path.join(image_temp_dir, f"{image_name}_detection.jpg")
    draw_detections(viz_input_path, sahi_results, viz_output_clean_path, show_confidence=False, 
                   show_keypoints=False, box_color=(0, 0, 255))
    results['viz_paths']['clean'] = viz_output_clean_path
    
    viz_output_detail_path = os.path.join(image_temp_dir, f"{image_name}_detail.jpg")
    draw_detections(viz_input_path, sahi_results, viz_output_detail_path, show_confidence=True, 
                   show_keypoints=True, box_color=(0, 255, 0), text_color=(0, 0, 0))
    results['viz_paths']['detail'] = viz_output_detail_path
    
    if results['num_detections'] > 0:
        crop_paths = save_face_crops(viz_input_path, sahi_results, crops_dir, prefix=f"{image_name}_face")
        results['crop_paths'] = crop_paths
        
        face_quality_scores = calculate_face_crop_quality(sahi_results, detection_input_image, device, available_metrics)
        results['face_quality_scores'] = face_quality_scores
    
    return results

# --- ANTARMUKA STREAMLIT ---

st.title("Aplikasi Deteksi Wajah: SmallFace-SuperDetect")


# Inisialisasi available_metrics di session_state
if 'available_metrics' not in st.session_state:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    with st.spinner("Mendeteksi metrik IQA yang tersedia..."):
        st.session_state.available_metrics = get_available_metrics(device)

with st.sidebar:
    st.header("🖼️ Input Gambar")
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])
    uploaded_files = [uploaded_file] if uploaded_file else []
    
    st.header("⚙️ Konfigurasi Deteksi")
    
    st.subheader("Pengaturan Model")
    
    model_path_input = "models/yolo11s-pose-default/yolo11s_pose/weights/best.pt"
    
    st.info("Model: **YOLO11s-Pose-Default**")
    
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    st.subheader("Pengaturan SAHI")
    enable_sahi = st.checkbox("Aktifkan Sahi", value=False)
    
    # ✅ FIXED: Use grid search optimal parameters (fixed values)
    slice_size = 640
    overlap_ratio = 0.25
    sahi_match_threshold = 0.5  # ✅ Fixed optimal value from grid search
    
    st.subheader("Pengaturan Enhance")
    enable_enhancer = st.checkbox("Aktifkan Enhance", value=False)
    enhancer_model_name = st.selectbox("Pilih Model Enhance (2x/4x)", ('RealESRGAN_x2plus', 'RealESRGAN_x4plus'), index=0)

# --- AREA UTAMA ---
if uploaded_files:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    uploaded_file = uploaded_files[0]
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    image_name = uploaded_file.name
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image_pil, use_container_width=True)

    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        process_button = st.button("🚀 Mulai Proses Deteksi", use_container_width=True, type="primary")

    if process_button:
        with st.spinner("Memuat model..."):
            yolo_model = load_yolo_model(model_path_input, conf_threshold, device, MODEL_IMAGE_SIZE)
            enhancer_model = load_enhancer_model(enhancer_model_name) if enable_enhancer else None
        
        if yolo_model is None: 
            st.stop()

        with st.spinner("Memproses gambar..."):
            
            results = process_single_image(
                image_np=image_np,
                image_name=image_name,
                yolo_model=yolo_model,
                enhancer_model=enhancer_model,
                enable_sahi=enable_sahi,
                enable_enhancer=enable_enhancer,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
                sahi_match_threshold=sahi_match_threshold,  # ✅ IOS metric is fixed
                device=device,
                available_metrics=st.session_state.available_metrics,
                temp_dir="temp_streamlit"
            )

        st.success(f"✅ Deteksi selesai! Ditemukan **{results['num_detections']}** wajah dalam **{results['detect_time']:.2f}** detik.")

        with col1:
            st.write("**Analisis Kualitas Gambar Asli:**")
            display_iqa_metrics(results['original_scores'], st.session_state.available_metrics)

        if enable_enhancer and results['enhanced_scores']:
            temp_image_path = os.path.join("temp_streamlit", f"image_{image_name.replace('.', '_')}", "temp_sahi_input.jpg")
            if os.path.exists(temp_image_path):
                 enhanced_image_display = Image.open(temp_image_path)
                 with col2:
                    st.subheader("Hasil Enhancement")
                    st.image(enhanced_image_display, use_container_width=True)
                    st.write("**Kualitas Setelah Enhancement:**")
                    display_iqa_metrics(results['enhanced_scores'], st.session_state.available_metrics, results['original_scores'])

        with col3:
            st.subheader("Hasil Deteksi")
            detail_path = results['viz_paths'].get('detail')
            if detail_path and os.path.exists(detail_path):
                st.image(Image.open(detail_path), use_container_width=True)
                with open(detail_path, "rb") as file:
                    st.download_button("📥 Download Hasil", file, os.path.basename(detail_path), "image/jpeg", use_container_width=True)
            else:
                st.warning("Gagal menampilkan hasil deteksi.")

        st.markdown("---")
        st.markdown('<div class="section-header">📋 Detail Hasil Deteksi</div>', unsafe_allow_html=True)
        
        tab_detail, tab_crops, tab_quality = st.tabs([
            "🎯 Deteksi Detail", 
            f"✂️ Cropped Face ({results['num_detections']})",
            "📊 Analisis Kualitas"
        ])

        with tab_detail:
            detail_path = results['viz_paths'].get('detail')
            if detail_path and os.path.exists(detail_path):
                st.image(Image.open(detail_path), use_container_width=True)
            else:
                st.warning("Tidak ada visualisasi detail.")
        
        with tab_crops:
            if results['num_detections'] > 0 and results['crop_paths']:
                st.info(f"✅ Berhasil menyimpan **{len(results['crop_paths'])}** wajah.")
                for i in range(0, len(results['crop_paths']), 4):
                    crop_cols = st.columns(4)
                    for j, c in enumerate(crop_cols):
                        if i + j < len(results['crop_paths']):
                            img_path = results['crop_paths'][i+j]
                            if os.path.exists(img_path):
                                img = Image.open(img_path)
                                c.image(img, use_container_width=True)
                                c.caption(f"**Face #{i+j+1}** | {img.size[0]}x{img.size[1]}px")
            else:
                st.warning("⚠️ Tidak ada wajah terdeteksi.")
        
        with tab_quality:
            if results['num_detections'] > 0 and results['face_quality_scores']:
                crops_dir = os.path.join("temp_streamlit", f"image_{image_name.replace('.', '_')}", "crops")
                display_face_quality_analysis(results['face_quality_scores'], st.session_state.available_metrics, crops_dir)
            else:
                st.info("Tidak ada wajah untuk dianalisis.")
else:
    st.info("⬆️ Silakan unggah gambar di panel samping untuk memulai.")
    with st.expander("ℹ️ Cara Penggunaan", expanded=True):
        st.markdown(f"""      
        ### Langkah-langkah:
        1. **Unggah Gambar** - Pilih gambar di sidebar
        2. **Atur Konfigurasi** - Aktifkan SAHI dan/atau Enhance jika diperlukan
        3. **Mulai Proses** - Klik tombol "🚀 Mulai Proses Deteksi"
        4. **Lihat Hasil** - Hasil akan ditampilkan di area utama
        
        ### Tentang Fitur:
        - **SAHI**: Menggunakan konfigurasi optimal dari grid search (IOS metric, threshold 0.5)
        - **Enhance**: Meningkatkan kualitas gambar sebelum deteksi
        
        ### Tentang Metrik IQA:
        - **NIQE & BRISQUE**: Lower is better (nilai rendah = kualitas baik)
        """)