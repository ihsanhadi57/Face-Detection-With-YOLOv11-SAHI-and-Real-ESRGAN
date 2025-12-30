import os
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

class WiderFaceClassifierFaceLevel:
    """
    Face-Level Classifier untuk WIDER Face
    
    Berbeda dengan image-level classifier, classifier ini:
    1. Menyimpan SEMUA face dalam satu gambar
    2. Mengklasifikasi SETIAP face secara individual
    3. Membuat "ignore list" untuk setiap difficulty level
    4. Compatible dengan official WIDER Face evaluation
    """
    
    def __init__(self, base_path="data/dataset/widerface"):
        self.base_path = Path(base_path)
        self.label_file = self.base_path / "wider_face_split" / "wider_face_val_bbx_gt.txt"
        
        # Output files
        self.output_dir = self.base_path / "face_level_annotations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thresholds untuk klasifikasi
        self.small_threshold = 50  # < 50px = small
        self.medium_threshold = 150  # < 150px = medium, >= 150 = large
        
        print("="*70)
        print("🎯 WIDER FACE Face-Level Classifier (Official Compatible)")
        print("="*70)
        print(f"Base path: {self.base_path}")
        print(f"Small threshold: {self.small_threshold}px")
        print(f"Output: {self.output_dir}")
        print("="*70)
    
    def parse_label_file(self):
        """Parse file label WIDER FACE"""
        print("\n📖 Parsing label file...")
        
        with open(self.label_file, 'r') as f:
            lines = f.readlines()
        
        annotations = {}
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Image path line
            if line.endswith('.jpg') or line.endswith('.png'):
                img_path = line
                i += 1
                
                if i >= len(lines):
                    break
                
                next_line = lines[i].strip()
                parts = next_line.split()
                faces = []
                
                try:
                    if len(parts) == 1:
                        # Format with num_faces
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
                        # Format without num_faces (direct coordinates)
                        while i < len(lines):
                            curr_line = lines[i].strip()
                            
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
                    print(f"⚠️  Error at line {i}: {e}")
                    i += 1
                    continue
                
                if faces:
                    annotations[img_path] = faces
            else:
                i += 1
        
        print(f"✓ Parsed {len(annotations)} images with faces")
        return annotations
    
    def _parse_face_line(self, parts):
        """Parse satu baris face annotation"""
        try:
            if len(parts) < 4:
                return None
            
            face_info = {
                'bbox': [
                    int(float(parts[0])),  # x
                    int(float(parts[1])),  # y
                    int(float(parts[2])),  # w
                    int(float(parts[3]))   # h
                ],
                'blur': 0,
                'expression': 0,
                'illumination': 0,
                'invalid': 0,
                'occlusion': 0,
                'pose': 0
            }
            
            # Parse attributes if available
            if len(parts) >= 10:
                try:
                    # With or without landmarks
                    if len(parts) >= 15:  # Has landmarks
                        face_info['blur'] = int(float(parts[-6]))
                        face_info['expression'] = int(float(parts[-5]))
                        face_info['illumination'] = int(float(parts[-4]))
                        face_info['invalid'] = int(float(parts[-3]))
                        face_info['occlusion'] = int(float(parts[-2]))
                        face_info['pose'] = int(float(parts[-1]))
                    else:  # No landmarks
                        face_info['blur'] = int(float(parts[4]))
                        face_info['expression'] = int(float(parts[5]))
                        face_info['illumination'] = int(float(parts[6]))
                        face_info['invalid'] = int(float(parts[7]))
                        face_info['occlusion'] = int(float(parts[8]))
                        face_info['pose'] = int(float(parts[9]))
                except (ValueError, IndexError):
                    pass
            
            return face_info
            
        except (ValueError, IndexError) as e:
            return None
    
    def classify_face(self, face_info):
        """
        Klasifikasi satu face ke dalam category
        
        Returns:
            - 'medium_large': Face besar (untuk easy evaluation)
            - 'small_clear': Face kecil tapi jelas (untuk medium evaluation)
            - 'small_degraded': Face kecil dan degraded (untuk hard evaluation)
            - None: Invalid face (skip)
        """
        # Skip invalid faces
        if face_info['invalid'] == 1:
            return None
        
        bbox = face_info['bbox']
        w, h = bbox[2], bbox[3]
        
        # Skip invalid bbox
        if w <= 0 or h <= 0:
            return None
        
        size = max(w, h)
        
        # Klasifikasi berdasarkan size
        is_small = size < self.small_threshold
        
        if is_small:
            # Check degradation factors
            is_degraded = (
                face_info['blur'] >= 1 or
                face_info['occlusion'] >= 1 or
                face_info['illumination'] == 1 or
                face_info['pose'] == 1
            )
            
            return 'small_degraded' if is_degraded else 'small_clear'
        else:
            return 'medium_large'
    
    def create_face_level_annotations(self, raw_annotations):
        """
        Buat annotations dengan face-level classification
        
        Output format:
        {
            "image_path": {
                "all_faces": [
                    {"bbox": [x, y, w, h], "category": "medium_large", "attributes": {...}},
                    ...
                ],
                "easy_indices": [0, 1],      # Indices face untuk easy eval
                "medium_indices": [0, 1, 2], # Indices face untuk medium eval
                "hard_indices": [0, 1, 2, 3] # Indices face untuk hard eval
            }
        }
        """
        print("\n🔍 Creating face-level annotations...")
        
        face_level_data = {}
        stats = defaultdict(int)
        difficulty_stats = defaultdict(lambda: defaultdict(int))
        
        for img_path, faces in tqdm(raw_annotations.items(), desc="Classifying"):
            all_faces = []
            easy_indices = []
            medium_indices = []
            hard_indices = []
            
            for face_idx, face_info in enumerate(faces):
                category = self.classify_face(face_info)
                
                if category is None:
                    continue
                
                # Store face with category
                face_data = {
                    'bbox': face_info['bbox'],
                    'category': category,
                    'attributes': {
                        'blur': face_info['blur'],
                        'expression': face_info['expression'],
                        'illumination': face_info['illumination'],
                        'occlusion': face_info['occlusion'],
                        'pose': face_info['pose']
                    }
                }
                
                current_idx = len(all_faces)
                all_faces.append(face_data)
                
                # Determine which difficulty sets this face belongs to
                # EASY: Only large faces (medium_large)
                if category == 'medium_large':
                    easy_indices.append(current_idx)
                    difficulty_stats['easy'][category] += 1
                
                # MEDIUM: Large + small clear (medium_large + small_clear)
                if category in ['medium_large', 'small_clear']:
                    medium_indices.append(current_idx)
                    difficulty_stats['medium'][category] += 1
                
                # HARD: All faces
                hard_indices.append(current_idx)
                difficulty_stats['hard'][category] += 1
                
                stats[category] += 1
            
            # Only save images with valid faces
            if all_faces:
                face_level_data[img_path] = {
                    'all_faces': all_faces,
                    'easy_indices': easy_indices,
                    'medium_indices': medium_indices,
                    'hard_indices': hard_indices
                }
        
        print(f"✓ Processed {len(face_level_data)} images")
        return face_level_data, stats, difficulty_stats
    
    def save_annotations(self, face_level_data):
        """Save annotations to JSON file"""
        output_file = self.output_dir / "face_level_gt.json"
        
        print(f"\n💾 Saving annotations to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(face_level_data, f, indent=2)
        
        print(f"✓ Saved to: {output_file}")
        return output_file
    
    def print_statistics(self, stats, difficulty_stats, face_level_data):
        """Print detailed statistics"""
        print("\n" + "="*70)
        print("📊 FACE-LEVEL CLASSIFICATION STATISTICS")
        print("="*70)
        
        # Overall face category distribution
        print("\n1. Overall Face Categories:")
        print("-"*70)
        total_faces = sum(stats.values())
        for category, count in sorted(stats.items()):
            percentage = (count / total_faces * 100) if total_faces > 0 else 0
            print(f"  {category:20s}: {count:6d} faces ({percentage:5.1f}%)")
        print(f"  {'TOTAL':20s}: {total_faces:6d} faces")
        
        # Difficulty-level distribution
        print("\n2. Face Distribution per Difficulty Level:")
        print("-"*70)
        for difficulty in ['easy', 'medium', 'hard']:
            total_diff = sum(difficulty_stats[difficulty].values())
            print(f"\n  {difficulty.upper()} set: {total_diff} faces to evaluate")
            for category, count in sorted(difficulty_stats[difficulty].items()):
                percentage = (count / total_diff * 100) if total_diff > 0 else 0
                print(f"    - {category:20s}: {count:6d} ({percentage:5.1f}%)")
        
        # Image-level statistics
        print("\n3. Image Distribution:")
        print("-"*70)
        images_with_easy = sum(1 for v in face_level_data.values() if len(v['easy_indices']) > 0)
        images_with_medium = sum(1 for v in face_level_data.values() if len(v['medium_indices']) > 0)
        images_with_hard = sum(1 for v in face_level_data.values() if len(v['hard_indices']) > 0)
        total_images = len(face_level_data)
        
        print(f"  Total images: {total_images}")
        print(f"  Images in EASY set:   {images_with_easy:5d} ({images_with_easy/total_images*100:5.1f}%)")
        print(f"  Images in MEDIUM set: {images_with_medium:5d} ({images_with_medium/total_images*100:5.1f}%)")
        print(f"  Images in HARD set:   {images_with_hard:5d} ({images_with_hard/total_images*100:5.1f}%)")
        
        # Mixed categories
        print("\n4. Mixed Category Images (images with multiple face categories):")
        print("-"*70)
        mixed_images = 0
        for img_path, data in face_level_data.items():
            categories = set(face['category'] for face in data['all_faces'])
            if len(categories) > 1:
                mixed_images += 1
        
        print(f"  Images with mixed face categories: {mixed_images} ({mixed_images/total_images*100:.1f}%)")
        print(f"  Images with single face category:  {total_images - mixed_images} ({(total_images-mixed_images)/total_images*100:.1f}%)")
        
        print("="*70)
    
    def save_statistics_file(self, stats, difficulty_stats, face_level_data):
        """Save statistics to text file"""
        stats_file = self.output_dir / "statistics.txt"
        
        with open(stats_file, 'w') as f:
            f.write("WIDER FACE Face-Level Classification Statistics\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Small threshold: {self.small_threshold}px\n")
            f.write(f"  Medium threshold: {self.medium_threshold}px\n\n")
            
            f.write("Categories Definition:\n")
            f.write("  - medium_large: Face >= 50px (for EASY evaluation)\n")
            f.write("  - small_clear: Face < 50px, no degradation (for MEDIUM evaluation)\n")
            f.write("  - small_degraded: Face < 50px with blur/occlusion (for HARD evaluation)\n\n")
            
            f.write("Difficulty Sets:\n")
            f.write("  - EASY: Only evaluate medium_large faces\n")
            f.write("  - MEDIUM: Evaluate medium_large + small_clear faces\n")
            f.write("  - HARD: Evaluate all faces (including small_degraded)\n\n")
            
            total_faces = sum(stats.values())
            f.write(f"Total faces: {total_faces}\n\n")
            
            for category, count in sorted(stats.items()):
                percentage = (count / total_faces * 100) if total_faces > 0 else 0
                f.write(f"{category:20s}: {count:6d} ({percentage:5.1f}%)\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"✓ Statistics saved to: {stats_file}")
    
    def run(self):
        """Run complete face-level classification"""
        # 1. Parse labels
        raw_annotations = self.parse_label_file()
        
        # 2. Create face-level annotations
        face_level_data, stats, difficulty_stats = self.create_face_level_annotations(raw_annotations)
        
        # 3. Save annotations
        output_file = self.save_annotations(face_level_data)
        
        # 4. Print statistics
        self.print_statistics(stats, difficulty_stats, face_level_data)
        
        # 5. Save statistics
        self.save_statistics_file(stats, difficulty_stats, face_level_data)
        
        print("\n🎉 Face-level classification completed!")
        print(f"📄 Output file: {output_file}")
        print(f"📊 Statistics file: {self.output_dir / 'statistics.txt'}")
        
        return face_level_data


if __name__ == "__main__":
    # Run classifier
    classifier = WiderFaceClassifierFaceLevel(base_path="data/dataset/widerface")
    face_level_data = classifier.run()
    
    # Print example
    print("\n" + "="*70)
    print("📝 EXAMPLE OUTPUT (first image):")
    print("="*70)
    
    first_img = next(iter(face_level_data.keys()))
    first_data = face_level_data[first_img]
    
    print(f"\nImage: {first_img}")
    print(f"Total faces: {len(first_data['all_faces'])}")
    print(f"Easy set (medium_large only): {len(first_data['easy_indices'])} faces")
    print(f"Medium set (medium_large + small_clear): {len(first_data['medium_indices'])} faces")
    print(f"Hard set (all faces): {len(first_data['hard_indices'])} faces")
    
    print("\nFace details:")
    for i, face in enumerate(first_data['all_faces']):
        in_easy = "✓" if i in first_data['easy_indices'] else "✗"
        in_medium = "✓" if i in first_data['medium_indices'] else "✗"
        in_hard = "✓" if i in first_data['hard_indices'] else "✗"
        
        bbox = face['bbox']
        size = max(bbox[2], bbox[3])
        
        print(f"  Face {i}: {face['category']:20s} | Size: {size:3d}px | "
              f"Easy:{in_easy} Med:{in_medium} Hard:{in_hard}")