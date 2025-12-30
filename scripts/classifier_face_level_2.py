import os
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

class WiderFaceSubCategoryClassifier:
    """
    Sub-Category Classifier untuk WIDER Face - 6 Kategori Detail
    
    Kategori:
    1. Large Clear: ≥150px, no degradation
    2. Medium Clear: 50-150px, no degradation
    3. Small Clear: <50px, no degradation
    4. Large Degraded: ≥150px, blur/occlusion/pose
    5. Medium Degraded: 50-150px, blur/occlusion/pose
    6. Small Degraded: <50px, blur/occlusion/pose
    """
    
    def __init__(self, base_path="data/dataset/widerface"):
        self.base_path = Path(base_path)
        self.label_file = self.base_path / "wider_face_split" / "wider_face_val_bbx_gt.txt"
        
        # Output files
        self.output_dir = self.base_path / "subcategory_annotations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thresholds
        self.small_threshold = 50    # <50px = small
        self.large_threshold = 150   # ≥150px = large, between = medium
        
        print("="*80)
        print("🎯 WIDER FACE Sub-Category Classifier (6 Categories)")
        print("="*80)
        print(f"Base path: {self.base_path}")
        print(f"Small threshold: <{self.small_threshold}px")
        print(f"Medium range: {self.small_threshold}-{self.large_threshold}px")
        print(f"Large threshold: ≥{self.large_threshold}px")
        print(f"Output: {self.output_dir}")
        print("="*80)
    
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
                        # Format without num_faces
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
        Klasifikasi face ke dalam 6 sub-kategori
        
        Returns: category name atau None jika invalid
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
        
        # Tentukan size category
        if size < self.small_threshold:
            size_cat = 'small'
        elif size < self.large_threshold:
            size_cat = 'medium'
        else:
            size_cat = 'large'
        
        # Tentukan degradation
        is_degraded = (
            face_info['blur'] >= 1 or
            face_info['occlusion'] >= 1 or
            face_info['illumination'] == 1 or
            face_info['pose'] == 1
        )
        
        degradation_cat = 'degraded' if is_degraded else 'clear'
        
        # Combine: size_degradation
        category = f"{size_cat}_{degradation_cat}"
        
        return category
    
    def create_subcategory_annotations(self, raw_annotations):
        """
        Buat annotations dengan 6 sub-kategori
        
        Output format:
        {
            "image_path": {
                "all_faces": [
                    {"bbox": [x, y, w, h], "category": "large_clear", "size": 200, "attributes": {...}},
                    ...
                ],
                "large_clear": [0, 2],      # Indices
                "large_degraded": [1],
                "medium_clear": [3, 4],
                "medium_degraded": [5],
                "small_clear": [6],
                "small_degraded": [7, 8]
            }
        }
        """
        print("\n🔍 Creating sub-category annotations...")
        
        subcategory_data = {}
        stats = defaultdict(int)
        
        # All possible categories
        categories = [
            'large_clear', 'large_degraded',
            'medium_clear', 'medium_degraded',
            'small_clear', 'small_degraded'
        ]
        
        for img_path, faces in tqdm(raw_annotations.items(), desc="Classifying"):
            all_faces = []
            category_indices = {cat: [] for cat in categories}
            
            for face_idx, face_info in enumerate(faces):
                category = self.classify_face(face_info)
                
                if category is None:
                    continue
                
                # Calculate face size
                bbox = face_info['bbox']
                face_size = max(bbox[2], bbox[3])
                
                # Store face with category
                face_data = {
                    'bbox': face_info['bbox'],
                    'category': category,
                    'size': face_size,
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
                
                # Add to category indices
                category_indices[category].append(current_idx)
                stats[category] += 1
            
            # Only save images with valid faces
            if all_faces:
                subcategory_data[img_path] = {
                    'all_faces': all_faces,
                    **category_indices  # Unpack all category indices
                }
        
        print(f"✓ Processed {len(subcategory_data)} images")
        return subcategory_data, stats
    
    def save_annotations(self, subcategory_data):
        """Save annotations to JSON file"""
        output_file = self.output_dir / "subcategory_gt.json"
        
        print(f"\n💾 Saving annotations to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(subcategory_data, f, indent=2)
        
        print(f"✓ Saved to: {output_file}")
        return output_file
    
    def print_statistics(self, stats, subcategory_data):
        """Print detailed statistics"""
        print("\n" + "="*80)
        print("📊 SUB-CATEGORY CLASSIFICATION STATISTICS")
        print("="*80)
        
        # Overall distribution
        print("\n1. Face Distribution by Sub-Category:")
        print("-"*80)
        total_faces = sum(stats.values())
        
        # Group by size
        for size in ['large', 'medium', 'small']:
            clear_cat = f'{size}_clear'
            degraded_cat = f'{size}_degraded'
            
            clear_count = stats.get(clear_cat, 0)
            degraded_count = stats.get(degraded_cat, 0)
            size_total = clear_count + degraded_count
            
            print(f"\n  {size.upper()} ({size_total} faces):")
            print(f"    - {clear_cat:20s}: {clear_count:6d} ({clear_count/total_faces*100:5.1f}%)")
            print(f"    - {degraded_cat:20s}: {degraded_count:6d} ({degraded_count/total_faces*100:5.1f}%)")
        
        print(f"\n  {'TOTAL':20s}: {total_faces:6d} faces")
        
        # Image-level statistics
        print("\n2. Image Distribution:")
        print("-"*80)
        
        category_image_counts = defaultdict(int)
        for img_path, data in subcategory_data.items():
            for category in ['large_clear', 'large_degraded', 'medium_clear', 
                           'medium_degraded', 'small_clear', 'small_degraded']:
                if len(data[category]) > 0:
                    category_image_counts[category] += 1
        
        total_images = len(subcategory_data)
        print(f"  Total images: {total_images}")
        print()
        for category in ['large_clear', 'large_degraded', 'medium_clear',
                        'medium_degraded', 'small_clear', 'small_degraded']:
            count = category_image_counts[category]
            print(f"  Images with {category:20s}: {count:5d} ({count/total_images*100:5.1f}%)")
        
        # Size distribution
        print("\n3. Size Statistics:")
        print("-"*80)
        all_sizes = []
        for data in subcategory_data.values():
            for face in data['all_faces']:
                all_sizes.append(face['size'])
        
        import numpy as np
        print(f"  Min size:  {np.min(all_sizes):.1f}px")
        print(f"  Max size:  {np.max(all_sizes):.1f}px")
        print(f"  Mean size: {np.mean(all_sizes):.1f}px")
        print(f"  Median:    {np.median(all_sizes):.1f}px")
        
        print("="*80)
    
    def save_statistics_file(self, stats, subcategory_data):
        """Save statistics to text file"""
        stats_file = self.output_dir / "statistics.txt"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("WIDER FACE Sub-Category Classification Statistics\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Small: <{self.small_threshold}px\n")
            f.write(f"  Medium: {self.small_threshold}-{self.large_threshold}px\n")
            f.write(f"  Large: >={self.large_threshold}px\n\n")
            
            f.write("Categories:\n")
            f.write("  1. Large Clear: >=150px, no degradation\n")
            f.write("  2. Large Degraded: >=150px, blur/occlusion/pose\n")
            f.write("  3. Medium Clear: 50-150px, no degradation\n")
            f.write("  4. Medium Degraded: 50-150px, blur/occlusion/pose\n")
            f.write("  5. Small Clear: <50px, no degradation\n")
            f.write("  6. Small Degraded: <50px, blur/occlusion/pose\n\n")
            
            total_faces = sum(stats.values())
            f.write(f"Total faces: {total_faces}\n\n")
            
            f.write("Distribution:\n")
            for category in ['large_clear', 'large_degraded', 'medium_clear',
                           'medium_degraded', 'small_clear', 'small_degraded']:
                count = stats.get(category, 0)
                percentage = (count / total_faces * 100) if total_faces > 0 else 0
                f.write(f"  {category:20s}: {count:6d} ({percentage:5.1f}%)\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Statistics saved to: {stats_file}")
    
    def run(self):
        """Run complete sub-category classification"""
        # 1. Parse labels
        raw_annotations = self.parse_label_file()
        
        # 2. Create sub-category annotations
        subcategory_data, stats = self.create_subcategory_annotations(raw_annotations)
        
        # 3. Save annotations
        output_file = self.save_annotations(subcategory_data)
        
        # 4. Print statistics
        self.print_statistics(stats, subcategory_data)
        
        # 5. Save statistics
        self.save_statistics_file(stats, subcategory_data)
        
        print("\n🎉 Sub-category classification completed!")
        print(f"📄 Output file: {output_file}")
        print(f"📊 Statistics file: {self.output_dir / 'statistics.txt'}")
        
        return subcategory_data


if __name__ == "__main__":
    # Run classifier
    classifier = WiderFaceSubCategoryClassifier(base_path="data/dataset/widerface")
    subcategory_data = classifier.run()
    
    # Print example
    print("\n" + "="*80)
    print("📝 EXAMPLE OUTPUT (first image):")
    print("="*80)
    
    first_img = next(iter(subcategory_data.keys()))
    first_data = subcategory_data[first_img]
    
    print(f"\nImage: {first_img}")
    print(f"Total faces: {len(first_data['all_faces'])}")
    print(f"\nFaces per category:")
    for category in ['large_clear', 'large_degraded', 'medium_clear',
                    'medium_degraded', 'small_clear', 'small_degraded']:
        count = len(first_data[category])
        if count > 0:
            print(f"  {category:20s}: {count} faces")
    
    print("\nFace details:")
    for i, face in enumerate(first_data['all_faces'][:5]):  # Show first 5
        print(f"  Face {i}: {face['category']:20s} | Size: {face['size']:3d}px | "
              f"Bbox: {face['bbox']}")