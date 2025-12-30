import os
from tqdm import tqdm

def fix_prediction_files(directory):
    """
    Iterates through all .txt files in a directory, reads them,
    counts the number of bounding box lines, and inserts the count
    as the second line if it's missing.
    """
    print(f"Starting to fix files in {directory}...")
    all_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    
    if not all_files:
        print("No .txt files found to fix.")
        return

    for filename in tqdm(all_files, desc="Fixing files"):
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            if len(lines) < 1:
                continue

            # Check if the second line is already a single integer (the count)
            if len(lines) > 1 and lines[1].strip().isdigit():
                # Already in correct format, check if count is right
                actual_count = len(lines) - 2
                if int(lines[1].strip()) == actual_count:
                    continue # Count is correct, skip
            
            # If format is wrong, fix it
            image_name = lines[0]
            box_lines = lines[1:]
            
            # Filter out any empty lines
            box_lines = [line for line in box_lines if line.strip()]
            
            count = len(box_lines)
            
            new_content = [image_name]
            new_content.append(str(count) + '\n')
            new_content.extend(box_lines)
            
            with open(filepath, 'w') as f:
                f.writelines(new_content)

        except Exception as e:
            print(f"\nError processing file {filename}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    prediction_dir = os.path.join("data", "eval_results", "retinaface")
    fix_prediction_files(prediction_dir)
    print("Finished fixing all prediction files.")
