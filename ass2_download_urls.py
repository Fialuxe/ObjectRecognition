import os
import requests
import shutil
from concurrent.futures import ThreadPoolExecutor

def download_image(url, save_path):
    try:
        # User supplied Flickr URLs, sometimes headers are needed or just robust handling
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

def process_list(list_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(list_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
        
    print(f"Processing {list_file}: {len(urls)} URLs pointing to {output_dir}")
    
    # We want sequential filenames 000001.jpg
    
    def task(index, url):
        # Determine extension (default to jpg if not in url)
        ext = os.path.splitext(url)[1]
        if not ext: ext = '.jpg'
        # Basic query param cleanup
        if '?' in ext: ext = ext.split('?')[0]
        
        filename = f"{index+1:06d}{ext}"
        save_path = os.path.join(output_dir, filename)
        
        if download_image(url, save_path):
            if (index + 1) % 10 == 0:
                print(f"  Downloaded {index+1}/{len(urls)}")
            return True
        return False

    # Parallel download
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i, url in enumerate(urls):
            futures.append(executor.submit(task, i, url))
            
        # Wait for all
        results = [f.result() for f in futures]
        
    print(f"Filtered/Downloaded {sum(results)} images to {output_dir}")

def main():
    # Define pairs
    pairs = [
        ("ass2_imagelist_apple.txt", "img/apple_test"),
        ("ass2_imagelist_kiwi.txt", "img/kiwi_test")
    ]
    
    for list_file, out_dir in pairs:
        if os.path.exists(list_file):
            process_list(list_file, out_dir)
        else:
            print(f"List file not found: {list_file}")

if __name__ == "__main__":
    main()
