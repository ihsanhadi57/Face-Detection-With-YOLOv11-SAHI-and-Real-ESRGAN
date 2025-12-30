import json
import argparse
from pathlib import Path

def find_image_category(image_path_str: str):
    """
    Finds and displays the sub-category classification for a given image
    from the pre-computed subcategory_gt.json file.
    """
    json_file = Path("data/dataset/widerface/subcategory_annotations/subcategory_gt.json")
    
    if not json_file.exists():
        print(f"Error: Annotation file not found at {json_file}")
        print("Please run the main 'classifier_face_level_2.py' script first to generate it.")
        return

    print(f"📖 Loading annotations from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    print("✓ Annotations loaded.")

    # The keys in the JSON file are in the format '0--Parade/0_Parade_marchingband_1_849.jpg'
    # We will accept the full path or just the relative part.
    # The image_path_str might be a full path, so we normalize it to match the key format.
    image_key = image_path_str.replace("\\\\", "/").replace("\\", "/")
    
    # Find the matching key in the JSON data
    found_data = None
    if image_key in data:
        found_data = data[image_key]
    else:
        # If the direct key doesn't match, try to find a key that ends with the provided path
        for key in data.keys():
            if key.endswith(image_key):
                found_data = data[key]
                image_key = key
                break

    if not found_data:
        print(f"\n❌ Image '{image_path_str}' not found in the annotation file.")
        return

    print("\n" + "="*80)
    print(f"🖼️  Classification for Image: {image_key}")
    print("="*80)

    all_faces = found_data.get('all_faces', [])
    if not all_faces:
        print("No valid faces were found for this image.")
        return

    print(f"Total Faces: {len(all_faces)}\n")

    # Print summary
    print("Summary by Category:")
    categories = [
        'large_clear', 'large_degraded',
        'medium_clear', 'medium_degraded',
        'small_clear', 'small_degraded'
    ]
    for category in categories:
        count = len(found_data.get(category, []))
        if count > 0:
            print(f"  - {category:20s}: {count} faces")

    # Print details for each face
    print("\nDetailed Face Info:")
    for i, face in enumerate(all_faces):
        print(f"  - Face #{i+1}:")
        print(f"    - Category: {face['category']}")
        print(f"    - Size:     {face['size']}px")
        print(f"    - BBox:     {face['bbox']}")
        # print(f"    - Attributes: {face['attributes']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check the pre-computed WIDER Face sub-category for a specific image.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="The path of the image to check. \n" 
             "Example: '0--Parade/0_Parade_marchingband_1_849.jpg'"
    )

    args = parser.parse_args()
    find_image_category(args.image_path)
