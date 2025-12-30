import streamlit as st
from PIL import Image
import time
import os
from pathlib import Path

# --- Konfigurasi ---
# Daftar gambar hasil evaluasi yang akan ditampilkan.
# Pengguna dapat mengubah path dan menambahkan judul untuk setiap gambar di sini.
EVALUATION_RESULTS = [
    {
        "title": "Evaluasi Baseline",
        "path": "eval_results/s-default/standard/dual_baseline_visualization.png"
    },
    {
        "title": "Evaluasi Dengan Sahi ",
        "path": "eval_results/s-default/yolo+sahi/dual_sahi_uniform_visualization.png"
    },
    {
        "title": "Evaluasi Dengan Enhance 2x",
        "path": "eval_results/s-default/enhance+yolo/dual_full-enhance_baseline_visualization.png"
    },
    {
        "title": "Evaluasi Full Pipeline Enhance 2x",
        "path": "eval_results/s-default/full pipeline/dual_full-enhance_sahi_uniform_visualization.png"
    }
    # Tambahkan dictionary lain di sini untuk gambar tambahan
]

# Path ke dataset WiderFace untuk menampilkan sampel
WIDERFACE_INPUT_DIR = "data/dataset/widerface/WIDER_train/images/0--Parade"
NUM_SAMPLE_IMAGES = 4  # Jumlah gambar sampel yang akan ditampilkan

st.title("Evaluasi WiderFace")

st.markdown("""

""")

# --- Tampilkan Input Dataset (Simulasi) ---
st.subheader("Input Dataset ")
st.info(f"Dataset yang digunakan untuk evaluasi: **{WIDERFACE_INPUT_DIR}**")

st.write("**Beberapa Contoh Gambar dari Dataset:**")
try:
    # Dapatkan beberapa path gambar dari direktori sampel
    sample_image_files = [f for f in os.listdir(WIDERFACE_INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:NUM_SAMPLE_IMAGES]
    
    if sample_image_files:
        cols = st.columns(NUM_SAMPLE_IMAGES)
        for i, img_file in enumerate(sample_image_files):
            with cols[i]:
                img_path = os.path.join(WIDERFACE_INPUT_DIR, img_file)
                img = Image.open(img_path)
                st.image(img, caption=img_file, width='stretch')
    else:
        st.warning("Tidak ada gambar sampel yang ditemukan di direktori yang ditentukan.")

except FileNotFoundError:
    st.error(f"Direktori sampel tidak ditemukan: {WIDERFACE_INPUT_DIR}. Pastikan path dataset benar.")

# --- Tombol Mulai Evaluasi ---
if st.button("Mulai Evaluasi pada Dataset WiderFace "):
    with st.spinner("Sedang melakukan evaluasi... mohon tunggu..."):
        # Simulasi proses evaluasi dengan jeda waktu
        time.sleep(3)
    
    st.success("Evaluasi selesai!")
    
    # Tampilkan gambar hasil dari path yang telah ditentukan
    st.header("Hasil Evaluasi ")
    for result in EVALUATION_RESULTS:
        st.subheader(result["title"])
        
        result_image_path = Path(os.path.dirname(__file__)).parent.parent / result["path"]
        
        if result_image_path.exists():
            try:
                img = Image.open(result_image_path)
                st.image(img, width='stretch')
            except Exception as e:
                st.error(f"Gagal memuat gambar hasil '{result['path']}': {e}")
        else:
            st.error(f"Path gambar hasil tidak ditemukan: {result['path']}. Mohon periksa path yang di-hardcode di dalam script.")
        st.markdown("---")


st.caption("")
