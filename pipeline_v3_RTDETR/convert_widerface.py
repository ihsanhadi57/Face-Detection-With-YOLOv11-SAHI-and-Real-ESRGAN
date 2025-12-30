import os
from pathlib import Path
from PIL import Image

def convert_widerface_to_yolo(anno_file, images_root, out_root, split='train'):
    """
    Output:
      <out_root>/labels/<split>/<relpath>.txt
    - Mirror struktur folder dari images_root
    - Skip bbox invalid==1 atau ukuran tidak masuk akal
    - Parser robust: cari baris jpg, lalu angka; kalau format kacau → skip blok tsb.
    """
    def is_jpg_line(s: str) -> bool:
        return s.lower().endswith(".jpg")

    labels_root = Path(out_root) / 'labels' / split
    labels_root.mkdir(parents=True, exist_ok=True)

    with open(anno_file, 'r') as f:
        # buang baris kosong/whitespace
        lines = [l.strip() for l in f if l.strip() != '']

    i = 0
    n_imgs = 0
    n_boxes = 0
    n_skipped_invalid = 0
    n_skipped_blocks = 0

    L = len(lines)
    print(f"Loaded {L} non-empty lines from {anno_file}")

    while i < L:
        # --- cari baris jpg berikutnya ---
        while i < L and not is_jpg_line(lines[i]):
            i += 1
        if i >= L:
            break

        rel_img_path = lines[i]; i += 1

        # --- ambil jumlah bbox (num) ---
        if i >= L:
            break
        try:
            num = int(lines[i]); i += 1
        except ValueError:
            # format kacau di blok ini → lewati sampai ketemu jpg berikutnya
            n_skipped_blocks += 1
            continue

        abs_img_path = Path(images_root) / rel_img_path
        # siapkan label path yang mirror folder
        label_path = Path(out_root) / 'labels' / split / rel_img_path
        label_path = label_path.with_suffix('.txt')
        label_path.parent.mkdir(parents=True, exist_ok=True)

        # kalau gambar tak ada → loncati num baris bbox
        if not abs_img_path.exists():
            i += num
            n_skipped_blocks += 1
            continue

        try:
            w_img, h_img = Image.open(abs_img_path).convert('RGB').size
        except Exception:
            # tak bisa buka gambar → lewati blok
            i += num
            n_skipped_blocks += 1
            continue

        yolo_lines = []
        # pastikan tidak melewati batas
        upto = min(i + num, L)
        for _ in range(num):
            if i >= L:
                break
            parts = lines[i].split(); i += 1
            if len(parts) < 10:
                # baris bbox tidak lengkap
                n_skipped_invalid += 1
                continue

            x, y, w, h = map(float, parts[:4])
            invalid = int(parts[7])  # kolom ke-8

            if invalid == 1 or w <= 0 or h <= 0:
                n_skipped_invalid += 1
                continue

            # clip ke dalam gambar
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(x + w, w_img)
            y2 = min(y + h, h_img)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 1 or h < 1:
                n_skipped_invalid += 1
                continue

            cx = (x1 + w / 2) / w_img
            cy = (y1 + h / 2) / h_img
            nw = w / w_img
            nh = h / h_img

            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)

            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            n_boxes += 1

        # tulis label (boleh kosong)
        with open(label_path, 'w') as f:
            f.writelines(yolo_lines)

        n_imgs += 1

    print(f"[{split}] images: {n_imgs}, boxes kept: {n_boxes}, "
          f"skipped_invalid/odd: {n_skipped_invalid}, skipped_blocks: {n_skipped_blocks}")
