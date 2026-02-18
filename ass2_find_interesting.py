import os
import glob
import shutil
import time
# Try to import BingImageCrawler
try:
    from icrawler.builtin import BingImageCrawler
except ImportError:
    print("icrawler not found. Please install it with 'pip install icrawler'")
    exit(1)

# Try to import clean_directory from the preparation script
# Since the file is in the same directory, we can import it.
import sys
sys.path.append(os.getcwd())
try:
    from ass_preparation0_download_data import clean_directory
except ImportError:
    # If import fails (e.g. running from different dir), we define a simple cleaner or copy it.
    # For now assume it works as they are in same dir.
    print("Warning: Could not import clean_directory from ass_preparation0_download_data.py")
    def clean_directory(d):
        return 0

def download_images_noisy(keyword, max_num, root_dir):
    print(f"Downloading {max_num} noisy images for '{keyword}' from Bing...")
    crawler = BingImageCrawler(storage={"root_dir": root_dir}, downloader_threads=4)
    crawler.crawl(keyword=keyword, max_num=max_num)

def process_category(category, noise_keyword, root_dir):
    print(f"\nProcessing {category}...")
    
    # 1. Truncate to top 200
    if os.path.exists(root_dir):
        files = sorted(glob.glob(os.path.join(root_dir, "*")))
        print(f"  Found {len(files)} existing images in {root_dir}")
        
        count_kept = 0
        count_removed = 0
        
        for f in files:
            basename = os.path.basename(f)
            # Try to parse number, assuming 6-digit format from previous script
            try:
                name_pure = os.path.splitext(basename)[0]
                num = int(name_pure)
                
                if num > 200:
                    os.remove(f)
                    count_removed += 1
                else:
                    count_kept += 1
            except ValueError:
                # If not a number, maybe keep it? Or delete it?
                # User's request implies clear structure 000001..000200.
                # I'll output a warning but keep it if it's less than 200 alphabetically?
                pass
                
        print(f"  Kept {count_kept} images. Removed {count_removed} images.")
    else:
        print(f"  Directory {root_dir} does not exist!")
        # Create it? No, ass_preparation0 should have created it.
        # But if missing, create it to put noise in.
        os.makedirs(root_dir, exist_ok=True)

    # 2. Download noise
    # We need 100 images (201-300).
    temp_dir = os.path.join(root_dir, "temp_noise")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Download 300 to be safe (increased from 150)
    download_images_noisy(noise_keyword, 300, temp_dir)
    
    # Clean temp (this checks for valid images and renumbers them 000001...)
    print(f"  Cleaning temp directory {temp_dir}...")
    clean_directory(temp_dir)
    
    # 3. Move and Rename
    noise_files = sorted(glob.glob(os.path.join(temp_dir, "*")))
    
    target_start = 201
    moved_count = 0
    
    for f in noise_files:
        if moved_count >= 100:
            break
            
        # New name
        target_num = target_start + moved_count
        ext = os.path.splitext(f)[1]
        new_name = f"{target_num:06d}{ext}"
        dest_path = os.path.join(root_dir, new_name)
        
        # Check if destination exists (it shouldn't if we truncated)
        if os.path.exists(dest_path):
            os.remove(dest_path)
            
        shutil.move(f, dest_path)
        moved_count += 1
        
    print(f"  Added {moved_count} noise images to {root_dir} (range {target_start}-{target_start+moved_count-1})")
    
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def main():
    # Use 'Hard Negatives' (Distractors) to confuse the model and lower accuracy
    # For Apple (Red/Round): Tomato, Red Ball
    # For Kiwi (Brown/Furry/Green): Potato, Stone
    
    root_apple = "img/apple"
    root_kiwi = "img/kiwi"
    
    # Apple -> Noise: Tomato
    process_category("apple", "tomato", root_apple) 
    
    # Kiwi -> Noise: Potato 
    process_category("kiwi", "potato", root_kiwi)

if __name__ == "__main__":
    main()
