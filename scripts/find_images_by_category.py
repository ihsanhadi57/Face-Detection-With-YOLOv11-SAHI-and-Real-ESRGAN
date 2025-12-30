import json
import argparse
from pathlib import Path

def find_images_by_category(category_name: str, limit: int):
    """
    Finds and lists all images that contain at least one face of the specified sub-category.
    """
    json_file = Path("data/dataset/widerface/subcategory_annotations/subcategory_gt.json")
    
    if not json_file.exists():
        print(f"Error: Annotation file not found at {json_file}")
        print("Please run the main 'classifier_face_level_2.py' script first to generate it.")
        return

    # Validate category name
    valid_categories = [
        'large_clear', 'large_degraded',
        'medium_clear', 'medium_degraded',
        'small_clear', 'small_degraded'
    ]
    if category_name not in valid_categories:
        print(f"Error: Invalid category name '{category_name}'.")
        print("Please choose one of the following:")
        for cat in valid_categories:
            print(f"  - {cat}")
        return

    print(f"📖 Loading annotations from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    print("✓ Annotations loaded.")

    print("\n" + "="*80)
    print(f"🖼️  Searching for images containing faces of category: '{category_name}'")
    print("="*80)

    matching_images = []
    for image_path, image_data in data.items():
        # The image_data dictionary contains keys for each category,
        # holding a list of indices of faces in that category.
        if image_data.get(category_name): # Check if the list is not empty
            matching_images.append(image_path)

    if not matching_images:
        print(f"No images found with faces in the '{category_name}' category.")
        return
    
    print(f"Found {len(matching_images)} images. Showing up to {limit}:")
    for i, image_path in enumerate(matching_images[:limit]):
        print(f"  {i+1:4d}: {image_path}")

    if len(matching_images) > limit:
        print(f"\n(And {len(matching_images) - limit} more...)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find all images in the WIDER Face dataset that contain faces of a specific sub-category.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "category",
        type=str,
        help="The category to search for. Must be one of: \n" 
             "  - large_clear\n"
             "  - large_degraded\n"
             "  - medium_clear\n"
             "  - medium_degraded\n"
             "  - small_clear\n"
             "  - small_degraded"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="The maximum number of image paths to display. Default is 50."
    )

    args = parser.parse_args()
    find_images_by_category(args.category, args.limit)
