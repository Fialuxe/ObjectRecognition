# Function to download images (Bing/Google)
def download_images(keyword, max_num, root_dir):
    print(f"Downloading {max_num} images for '{keyword}' from Bing...")
    # Use BingImageCrawler as Google often fails
    crawler = BingImageCrawler(storage={"root_dir": root_dir}, downloader_threads=4)
    crawler.crawl(keyword=keyword, max_num=max_num)

# Function to clean the downloaded data
def clean_directory(directory):
    if not os.path.exists(directory):
        print(f"Skipping {directory} (does not exist)")
        return

    print(f"Cleaning {directory}...")
    files = glob.glob(os.path.join(directory, "*"))
    count = 0
    removed = 0
    
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
            except Exception as e:
                # If cannot open, remove
                os.remove(f)
                removed += 1
                continue
                
            count += 1
            
        except OSError as e:
            try:
                os.remove(f)
                removed += 1
            except:
                pass

    print(f"  Final count: {count} videos/images (Removed: {removed})")


def main():
    # Assignment 1: 2-Class Classification
    # Pair 1: Chihuahua vs Muffin
    download_images("chihuahua", 200, "img/chihuahua")
    download_images("muffin", 200, "img/muffin")
    
    # Pair 2: Poodle vs Fried Chicken
    download_images("poodle", 200, "img/poodle")
    download_images("fried chicken (food)", 200, "img/fried_chicken")

    # Pair 3: Bus vs Truck (Easy Pair)
    download_images("bus", 200, "img/bus")
    download_images("truck", 200, "img/truck")

    # Assignment 2: Re-ranking
    # Training & Test Data (Single Folder Split)
    # We download MORE images (400) to ensure we have enough for:
    # - Train Top-N (25/50)
    # - Test Rest (350+) + Background Noise Injection
    download_images("apple", 400, "img/apple") 
    download_images("kiwi", 400, "img/kiwi")

    # Clean the downloaded data
    dirs = [
        "img/chihuahua", 
        "img/muffin", 
        "img/poodle", 
        "img/fried_chicken", 
        "img/bus",
        "img/truck",
        "img/truck",
        "img/apple", 
        "img/kiwi",
        "bgimg"
    ]
    for d in dirs:
        clean_directory(d)

if __name__ == "__main__":
    main()
