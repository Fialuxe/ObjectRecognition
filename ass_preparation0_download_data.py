import os
import glob
from icrawler.builtin import BingImageCrawler
from PIL import Image

# Function to download images (Bing/Google)
def download_images(keyword, max_num, root_dir):
    print(f"Downloading {max_num} images for '{keyword}' from Bing...")
    # Use BingImageCrawler as Google often fails
    crawler = BingImageCrawler(storage={"root_dir": root_dir}, downloader_threads=4)
    crawler.crawl(keyword=keyword, max_num=max_num)

# Function to verify integrity and remove invalid files
def clean_directory(directory):
    if not os.path.exists(directory):
        print(f"Skipping {directory} (does not exist)")
        return 0

    print(f"Cleaning {directory}...")
    files = glob.glob(os.path.join(directory, "*"))
    files.sort() # Ensure consistent order
    
    count = 0
    removed = 0
    valid_files = []
    
    for f in files:
        try:
            # Remove small files (likely thumbnails or errors)
            if os.path.getsize(f) < 5000: # 5KB
                os.remove(f)
                removed += 1
                continue
            
            try:
                with Image.open(f) as img:
                    img.verify() # Verify integrity
                    
                with Image.open(f) as img:
                    # Check format vs extension behavior
                    real_format = img.format
                    if real_format:
                        # Define allowed extensions for each format
                        format_map = {
                            'JPEG': ['.jpg', '.jpeg'],
                            'PNG':  ['.png'],
                            'GIF':  ['.gif'],
                            'WEBP': ['.webp']
                        }
                        
                        file_ext = os.path.splitext(f)[1].lower()
                        expected_exts = format_map.get(real_format.upper(), [])
                        
                        if expected_exts and file_ext not in expected_exts:
                            print(f"Removing {os.path.basename(f)}: Format {real_format} but extension {file_ext}")
                            os.remove(f)
                            removed += 1
                            continue

                    # Convert to RGB if necessary (e.g. RGBA, P, L)
                    if img.mode != 'RGB':
                        rgb_img = img.convert('RGB')
                        rgb_img.save(f)
                        # print(f"Converted {os.path.basename(f)} from {img.mode} to RGB")
            except Exception:
                # If cannot open, remove
                os.remove(f)
                removed += 1
                continue
                
            valid_files.append(f)
            count += 1
            
        except OSError:
            try:
                os.remove(f)
                removed += 1
            except:
                pass

    print(f"  Valid files count: {count} (Removed: {removed})")
    
    # Renumber files to be sequential (000001.jpg, ...)
    # This helps fill gaps so the crawler downloads new images correctly
    print("  Renumbering files...")
    for i, old_path in enumerate(valid_files):
        ext = os.path.splitext(old_path)[1]
        new_name = f"{i+1:06d}{ext}"
        new_path = os.path.join(directory, new_name)
        
        if old_path != new_path:
            # Handle potential collision if new_path already exists (shouldn't happen if sorted, but safe)
            if os.path.exists(new_path) and new_path not in valid_files[i:]:
                 # Temporary rename if needed? With specific loop it should be OK if we process safely
                 pass 
            os.rename(old_path, new_path)
            
    return count

def download_and_ensure(keyword, min_count, root_dir):
    current_target = min_count
    attempts = 0
    max_attempts = 5
    
    while attempts < max_attempts:
        attempts += 1
        print(f"\n[{keyword}] Attempt {attempts}/{max_attempts}: Target download count {current_target}")
        
        download_images(keyword, current_target, root_dir)
        valid_count = clean_directory(root_dir)
        
def download_and_ensure(keywords, min_count, root_dir, max_attempts=None):
    # Ensure keywords is a list for consistent handling
    if isinstance(keywords, str):
        keywords = [keywords]

    current_target = min_count
    attempts = 0
    
    while True:
        attempts += 1
        # Cycle through keywords
        current_keyword = keywords[(attempts - 1) % len(keywords)]
        
        if max_attempts:
             print(f"\n[{current_keyword}] Attempt {attempts}/{max_attempts}: Target download count {current_target}")
        else:
             print(f"\n[{current_keyword}] Attempt {attempts} (Infinite): Target download count {current_target}")
        
        download_images(current_keyword, current_target, root_dir)
        valid_count = clean_directory(root_dir)
        
        if valid_count >= min_count:
            print(f"[{current_keyword}] Successfully reached {valid_count} images (>= {min_count}).")
            return
        
        if max_attempts and attempts >= max_attempts:
             print(f"[{current_keyword}] WARNING: Failed to reach {min_count} images after {max_attempts} attempts. Final count: {valid_count}")
             return

        print(f"[{current_keyword}] Only {valid_count} valid images found. Need {min_count}. Retrying with next keyword...")
        # Increase target to account for failure rate
        shortfall = min_count - valid_count
        # Aim a bit higher to cover future failures
        current_target += int(shortfall * 1.5) + 10 


def main():
    # Assignment 1: 2-Class Classification
    # Pair 1: Chihuahua vs Muffin
    #download_and_ensure("chihuahua", 200, "img/chihuahua", max_attempts=None)
    #download_and_ensure("muffin", 200, "img/muffin", max_attempts=None)
    
    # Pair 2: Poodle vs Fried Chicken
    #download_and_ensure("poodle", 200, "img/poodle", max_attempts=None)
    #download_and_ensure("fried chicken (food)", 200, "img/fried_chicken", max_attempts=None)

    # Pair 3: Bus vs Truck (Easy Pair)
    #download_and_ensure("bus", 200, "img/bus", max_attempts=None)
    download_and_ensure("truck", 200, "img/truck", max_attempts=None)

    # Assignment 2: Re-ranking
    # Training & Test Data (Single Folder Split)
    # We download MORE images (400) to ensure we have enough for:
    # As we cannot acquire 400 photos only query "apple", 
    # this program will donwload another 200 photos for "apple fruit" in directory "apple2", and merge them into "apple" directory manually.
    # download_and_ensure("apple", 400, "img/apple", max_attempts=5) 
    # download_and_ensure("apple fruit", 200, "img/apple2", max_attempts=5)
    #download_and_ensure(["kiwi", "kiwi fruit", "green kiwi"], 400, "img/kiwi", max_attempts=None)

    # Final cleanup pass for bgimg (if used separately, not downloading here)
    if os.path.exists("bgimg"):
        clean_directory("bgimg")

if __name__ == "__main__":
    main()
