import os
import shutil
from pathlib import Path
from tqdm import tqdm

class WiderFaceClassifier:
    def __init__(self, base_path="data/dataset/widerface"):
        self.base_path = Path(base_path)
        self.val_images_path = self.base_path / "WIDER_val" / "images"
        self.label_file = self.base_path / "wider_face_split" / "wider_face_val_bbx_gt.txt"
        
        # Output directories
        self.output_base = self.base_path / "classified_val"
        self.categories = {
            'small_clear': self.output_base / "small_clear",
            'small_degraded': self.output_base / "small_degraded", 
            'medium_large': self.output_base / "medium_large"
        }
        
        # Thresholds untuk klasifikasi ukuran (dalam pixels)
        self.small_threshold = 50  # bbox width/height < 50px = small
        self.medium_threshold = 150  # bbox < 150px = medium, >= 150 = large
        
    def create_output_dirs(self):
        """Buat direktori output untuk setiap kategori"""
        for category_path in self.categories.values():
            category_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {category_path}")
    
    def parse_label_file(self):
        """Parse file label.txt WIDER FACE format (dengan atau tanpa landmarks)"""
        print("\n📖 Parsing label file...")
        
        with open(self.label_file, 'r') as f:
            lines = f.readlines()
        
        annotations = {}
        i = 0
        
        print("Detecting label format...")
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Baris yang berisi path gambar (misal: 0--Parade/0_Parade_marchingband_1_849.jpg)
            if line.endswith('.jpg') or line.endswith('.png'):
                img_path = line
                i += 1
                
                if i >= len(lines):
                    break
                
                # Baris berikutnya: bisa jumlah face atau langsung koordinat
                next_line = lines[i].strip()
                parts = next_line.split()
                
                faces = []
                
                # Cek apakah baris berisi hanya 1 angka (num_faces) atau banyak angka (koordinat)
                try:
                    if len(parts) == 1:
                        # Format standar: ada jumlah face terpisah
                        num_faces = int(parts[0])
                        i += 1
                        
                        for j in range(num_faces):
                            if i >= len(lines):
                                break
                            
                            parts = lines[i].strip().split()
                            face_info = self._parse_face_line(parts)
                            if face_info:
                                faces.append(face_info)
                            i += 1
                    else:
                        # Format alternatif: langsung koordinat tanpa num_faces
                        # Baca sampai ketemu line berikutnya yang .jpg atau habis
                        while i < len(lines):
                            curr_line = lines[i].strip()
                            
                            # Stop jika ketemu image path baru atau line kosong
                            if curr_line.endswith('.jpg') or curr_line.endswith('.png'):
                                break
                            
                            if not curr_line:
                                i += 1
                                break
                            
                            parts = curr_line.split()
                            face_info = self._parse_face_line(parts)
                            if face_info:
                                faces.append(face_info)
                            i += 1
                
                except ValueError as e:
                    print(f"⚠️  Error parsing at line {i}: {e}")
                    i += 1
                    continue
                
                if faces:  # Hanya simpan jika ada faces
                    annotations[img_path] = faces
            else:
                i += 1
        
        print(f"✓ Parsed {len(annotations)} images")
        return annotations
    
    def _parse_face_line(self, parts):
        """Parse satu baris face annotation dengan berbagai format"""
        try:
            if len(parts) < 4:
                return None
            
            # Ambil bounding box (4 nilai pertama)
            face_info = {
                'x': int(float(parts[0])),
                'y': int(float(parts[1])),
                'w': int(float(parts[2])),
                'h': int(float(parts[3]))
            }
            
            # Default values untuk atribut
            face_info['blur'] = 0
            face_info['expression'] = 0
            face_info['illumination'] = 0
            face_info['invalid'] = 0
            face_info['occlusion'] = 0
            face_info['pose'] = 0
            
            # Cek apakah ada atribut tambahan setelah bbox
            # Format bisa: x y w h [landmarks...] blur expression illumination invalid occlusion pose
            # Atau: x y w h blur expression illumination invalid occlusion pose
            
            if len(parts) >= 10:
                # Cari 6 nilai integer di akhir (atribut WIDER FACE)
                # Biasanya di posisi akhir atau setelah landmarks
                try:
                    # Coba ambil dari akhir (skip landmarks jika ada)
                    if len(parts) >= 15:  # Ada landmarks (10 nilai float) + atribut
                        face_info['blur'] = int(float(parts[-6]))
                        face_info['expression'] = int(float(parts[-5]))
                        face_info['illumination'] = int(float(parts[-4]))
                        face_info['invalid'] = int(float(parts[-3]))
                        face_info['occlusion'] = int(float(parts[-2]))
                        face_info['pose'] = int(float(parts[-1]))
                    else:  # Hanya bbox + atribut
                        face_info['blur'] = int(float(parts[4]))
                        face_info['expression'] = int(float(parts[5]))
                        face_info['illumination'] = int(float(parts[6]))
                        face_info['invalid'] = int(float(parts[7]))
                        face_info['occlusion'] = int(float(parts[8]))
                        face_info['pose'] = int(float(parts[9]))
                except (ValueError, IndexError):
                    pass  # Gunakan default values
            
            return face_info
            
        except (ValueError, IndexError) as e:
            print(f"⚠️  Failed to parse face line: {parts[:5]}... - {e}")
            return None
    
    def classify_face(self, face_info):
        """
        Klasifikasi face berdasarkan Opsi B:
        1. Small-Clear: wajah kecil dalam kondisi ideal
        2. Small-Degraded: wajah kecil + blur/occlusion/poor lighting
        3. Medium-Large: wajah ukuran sedang-besar
        """
        w = face_info['w']
        h = face_info['h']
        size = max(w, h)  # Gunakan dimensi terbesar
        
        # Cek apakah face invalid
        if face_info['invalid'] == 1:
            return None  # Skip invalid faces
        
        # Cek ukuran
        is_small = size < self.small_threshold
        
        if is_small:
            # Cek kondisi degraded
            is_degraded = (
                face_info['blur'] >= 1 or  # Ada blur
                face_info['occlusion'] >= 1 or  # Ada occlusion
                face_info['illumination'] == 1 or  # Pencahayaan ekstrem
                face_info['pose'] == 1  # Pose atypical
            )
            
            if is_degraded:
                return 'small_degraded'
            else:
                return 'small_clear'
        else:
            return 'medium_large'
    
    def classify_image(self, faces):
        """
        Klasifikasi gambar berdasarkan mayoritas face atau prioritas
        Prioritas: small_degraded > small_clear > medium_large
        """
        if not faces:
            return 'medium_large'  # Default jika tidak ada face
        
        classifications = []
        for face in faces:
            cat = self.classify_face(face)
            if cat:
                classifications.append(cat)
        
        if not classifications:
            return 'medium_large'
        
        # Gunakan prioritas: jika ada small_degraded, kategorikan sebagai small_degraded
        if 'small_degraded' in classifications:
            return 'small_degraded'
        elif 'small_clear' in classifications:
            return 'small_clear'
        else:
            return 'medium_large'
    
    def copy_images(self, annotations):
        """Copy images ke folder kategori masing-masing"""
        print("\n📂 Copying images to categories...")
        
        stats = {cat: 0 for cat in self.categories.keys()}
        
        for img_path, faces in tqdm(annotations.items(), desc="Processing"):
            # Tentukan kategori gambar
            category = self.classify_image(faces)
            
            # Source image path
            src_img = self.val_images_path / img_path
            
            if not src_img.exists():
                print(f"⚠️  Image not found: {src_img}")
                continue
            
            # Destination path (flatten structure: 0--Parade/img.jpg -> 0--Parade_img.jpg)
            flat_name = img_path.replace('/', '_')
            dst_img = self.categories[category] / flat_name
            
            # Copy image
            shutil.copy2(src_img, dst_img)
            stats[category] += 1
        
        return stats
    
    def run(self):
        """Jalankan proses klasifikasi lengkap"""
        print("="*60)
        print("🎯 WIDER FACE Classification - Opsi B")
        print("="*60)
        print(f"Base path: {self.base_path}")
        print(f"Small threshold: {self.small_threshold}px")
        print(f"Medium threshold: {self.medium_threshold}px")
        
        # 1. Buat direktori output
        self.create_output_dirs()
        
        # 2. Parse label file
        annotations = self.parse_label_file()
        
        # 3. Copy images berdasarkan klasifikasi
        stats = self.copy_images(annotations)
        
        # 4. Tampilkan statistik
        print("\n" + "="*60)
        print("📊 CLASSIFICATION RESULTS")
        print("="*60)
        total = sum(stats.values())
        for category, count in stats.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{category:20s}: {count:5d} images ({percentage:5.2f}%)")
        print(f"{'TOTAL':20s}: {total:5d} images")
        print("="*60)
        
        # 5. Simpan statistik ke file
        stats_file = self.output_base / "classification_stats.txt"
        with open(stats_file, 'w') as f:
            f.write("WIDER FACE Classification Statistics - Opsi B\n")
            f.write("="*60 + "\n\n")
            f.write(f"Small threshold: {self.small_threshold}px\n")
            f.write(f"Medium threshold: {self.medium_threshold}px\n\n")
            f.write("Categories:\n")
            f.write("1. Small-Clear: Wajah kecil dalam kondisi ideal\n")
            f.write("2. Small-Degraded: Wajah kecil + blur/occlusion/poor lighting\n")
            f.write("3. Medium-Large: Wajah ukuran sedang-besar\n\n")
            f.write("Results:\n")
            f.write("-"*60 + "\n")
            for category, count in stats.items():
                percentage = (count / total * 100) if total > 0 else 0
                f.write(f"{category:20s}: {count:5d} images ({percentage:5.2f}%)\n")
            f.write(f"{'TOTAL':20s}: {total:5d} images\n")
        
        print(f"\n✓ Statistics saved to: {stats_file}")
        print(f"✓ Classified images saved to: {self.output_base}")
        print("\n🎉 Classification completed!")


if __name__ == "__main__":
    # Inisialisasi classifier
    classifier = WiderFaceClassifier(base_path="data/dataset/widerface")
    
    # Jalankan klasifikasi
    classifier.run()