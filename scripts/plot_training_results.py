import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re

# Data dari file hasil training.txt
training_data = """
Tabel 5.1. Hasil Komparatif Eksperimen Pelatihan Fase 1 (Pada Best Epoch)
Kode	Arsitektur & Konfigurasi	Input Size	Batch	Best Epoch 	mAP @0.5 (Val)	mAP @0.5:0.95	Precision	Recall
(a)	yolo11n-pose-default	        1024	    16	        97	    0.7228	        0.4109	        0.8628	    0.6450
(b)	yolo11n-pose-custom	            768	        8	        93	    0.7319	        0.4188	        0.8630	    0.6596
(c)	yolo11n-pose-custom-repeat	    960	        4	        99	    0.7334	        0.4238	        0.8730	    0.6533
(d)	yolo11s-pose-default	        1024	    16	        98	    0.7537	        0.4356	        0.8658	    0.6848
(e)	yolo11s-pose-custom	            960	        4           92	    0.7272	        0.4181	        0.8793	    0.6408
(f)	yolo11s-pose-custom-repeat	    960	        4	        96	    0.7307	        0.4202	        0.8690	    0.6488
(g)	yolo11m-pose-default	        960	        8	        100	    0.7571	        0.4368	        0.8692	    0.6908
(h)	yolo11m-pose-custom-repeat	    768	        4	        55	    0.7206	        0.4124	        0.8704	    0.6382
(i)	yolo11m-pose-custom	            768	        4	        97	    0.72517	        0.4176	        0.8620	    0.6446
(j)	yolo11l-pose-default	        960	        8	        99	    0.7612	        0.4419	        0.8696	    0.6896
"""

# Parsing manual yang lebih robust
lines = training_data.strip().split('\n')
parsed_data = []
# Skip the first 2 header lines
for line in lines[2:]:
    # Split by 2 or more spaces
    parts = re.split(r'\s{2,}', line.strip())
    
    # Further split the first part to separate code and architecture
    first_part_split = re.split(r'\s+', parts[0])
    kode = first_part_split[0]
    arsitektur = first_part_split[1]
    
    # The rest of the parts are the numeric values
    numeric_values = parts[1:]
    
    row = {
        'Kode': kode,
        'Arsitektur': arsitektur,
        'Input Size': numeric_values[0],
        'Batch': numeric_values[1],
        'Best Epoch': numeric_values[2],
        'mAP_50_Val': numeric_values[3],
        'mAP_50_95': numeric_values[4],
        'Precision': numeric_values[5],
        'Recall': numeric_values[6]
    }
    parsed_data.append(row)

# Buat DataFrame dari data yang sudah di-parse
df = pd.DataFrame(parsed_data)

# Ekstrak arsitektur dasar (n, s, m, l)
df['Arch_Base'] = df['Arsitektur'].apply(lambda x: re.search(r'yolo11(n|s|m|l)', x).group(1) if re.search(r'yolo11(n|s|m|l)', x) else 'unknown')

# Pastikan tipe data numerik
for col in ['Input Size', 'mAP_50_Val', 'mAP_50_95']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Hapus baris yang gagal di-parse
df.dropna(subset=['Input Size', 'mAP_50_95', 'Arch_Base'], inplace=True)

# --- Membuat Plot ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 9))

# Buat scatter plot dengan seaborn
sns.scatterplot(
    data=df,
    x='Input Size',
    y='mAP_50_95',
    hue='Arch_Base',
    size='mAP_50_Val',  # Ukuran titik merepresentasikan mAP@0.5
    sizes=(150, 600),   # Range ukuran titik
    style='Arsitektur',  # Bentuk marker berbeda untuk tiap arsitektur
    palette='viridis',
    ax=ax,
    edgecolor='black',
    alpha=0.8
)

# Beri label pada setiap titik
for i, row in df.iterrows():
    ax.text(row['Input Size'] + 8, row['mAP_50_95'], f"mAP: {row['mAP_50_95']:.4f}", fontsize=9, ha='left', va='center')

# Atur judul dan label
ax.set_title('Pengaruh Input Size terhadap Performa Model (mAP@0.5:0.95)', fontsize=18, pad=20, weight='bold')
ax.set_xlabel('Input Size (imgsz)', fontsize=14)
ax.set_ylabel('mAP @0.5:0.95', fontsize=14)
ax.legend(title='Arsitektur', loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# Atur grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Simpan plot ke file
output_path = 'training_results_by_input_size.png'
plt.savefig(output_path)

print(f"Plot berhasil dibuat dan disimpan di: {output_path}")

# Tampilkan plot jika dijalankan secara interaktif
# plt.show()
