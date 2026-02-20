import glob
import os
import re

def extract_image_number(filename: str) -> int:
    """
    Extracts the image number from the filename.
    Returns -1 if no number is found.
    Assumption: The relevant number is the first sequence of digits.
    """
    # Remove extension first to avoid confusion with potential numbers there
    name_no_ext = os.path.splitext(filename)[0]
    match = re.search(r'(\d+)', name_no_ext)
    if match:
        return int(match.group(1))
    return -1

def is_ground_truth(file_path: str, target_keyword: str) -> bool:
    """
    Determines if an image is a Ground Truth Positive based on its path.
    
    Since Train and Test datasets are now separated:
    - Any image in '/img/{keyword}_test/' is a POSITIVE.
    - Any image in '/bgimg/' is a NEGATIVE.
    - Images in '/img/{keyword}_train/' should NOT appear in the test list.
    """
    normalized_path = file_path.replace('\\', '/')
    
    # Check for the specific test directory
    test_dir_marker = f"/img/{target_keyword}_test/"
    
    # If the path contains the test directory, it's a ground truth positive.
    if test_dir_marker in "/" + normalized_path:
        return True
        
    return False

def calculate_ap(rank_filepath: str, keyword: str) -> tuple[float, int]:
    """
    Calculates Average Precision (AP) for a rank file.
    Args:
        rank_filepath: Path to the ranking file.
        keyword: The target query keyword (e.g., 'apple').
    """
    with open(rank_filepath, 'r') as f:
        # Filter out empty lines
        lines = [line.strip() for line in f if line.strip()]

    # First, calculate total positives in this specific list (the denominator)
    # This is critical because Top-N training removes some positives from the test set.
    # e.g., if we train on Top-50, only 150 positives remain in the list.
    total_positives_in_list = 0
    gt_list = [] # Store boolean IsRelevant status for each rank
    
    for line in lines:
        parts = line.split()
        if not parts: continue
        image_path = parts[0]
        
        is_relevant = is_ground_truth(image_path, keyword)
        gt_list.append(is_relevant)
        
        if is_relevant:
            total_positives_in_list += 1
            
    if total_positives_in_list == 0:
        return 0.0, 0

    # Calculate AP based on this count
    cumulative_precision = 0.0
    correct_so_far = 0
    
    for rank, is_rel in enumerate(gt_list, start=1):
        if is_rel:
            correct_so_far += 1
            precision_at_k = correct_so_far / rank
            cumulative_precision += precision_at_k
            
    ap = cumulative_precision / total_positives_in_list
    return ap, total_positives_in_list

def main():
    rank_files = sorted(glob.glob("ass2_rank_*.txt"))
    
    print(f"{'File':<50} | {'AP':<8} | {'Rel. Count'}")
    print("-" * 75)
    
    for filepath in rank_files:
        # Expected filename: ass2_rank_{type}_{keyword}_top{N}.txt
        parts = filepath.split('_')
        
        if len(parts) < 4:
            print(f"{filepath:<50} | ERROR    | Invalid Filename")
            continue
            
        target_keyword = parts[3]
        
        ap, count = calculate_ap(filepath, target_keyword)
        print(f"{filepath:<50} | {ap:.4f}   | {count}")

if __name__ == "__main__":
    main()
