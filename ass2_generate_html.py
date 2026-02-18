import glob
import os

def generate_evaluation_html(txt_file, is_original=False):
    base_name = os.path.splitext(os.path.basename(txt_file))[0]
    # Distinguish output filenames
    if is_original:
        out_file = base_name.replace('ass2_rank_original', 'ass2_eval_original') + ".html"
        title_prefix = "Original Rank"
    else:
        out_file = base_name.replace('ass2_rank_reranked', 'ass2_eval_reranked') + ".html"
        title_prefix = "Re-ranked Result"
        
    print(f"Processing {txt_file} -> {out_file}")
    
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        
    # Skip header
    if len(lines) > 0 and "score" in lines[0]:
        lines = lines[1:]
        
    # Extract data
    images = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 2:
            path = parts[0]
            # Handle potential relative paths or different separators
            path = path.replace("\\", "/") 
            score = parts[1]
            images.append({
                "rank": i + 1,
                "path": path,
                "score": score,
                "name": os.path.basename(path)
            })

    # Generate HTML with embedded JS for counting
    html = []
    html.append(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title_prefix}</title>")
    html.append("""
    <style>
        body { font-family: sans-serif; background: #f5f5f5; padding: 20px; }
        .header { position: sticky; top: 0; background: white; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); z-index: 100; display: flex; justify-content: space-between; align-items: center; }
        .stats { font-size: 18px; font-weight: bold; }
        .container { display: flex; flex-wrap: wrap; margin-top: 20px; }
        .box { width: 200px; margin: 10px; padding: 10px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: all 0.2s; border: 2px solid transparent; position: relative; cursor: pointer; }
        .box.correct { border-color: #4CAF50; background: #E8F5E9; }
        .box:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .rank { font-weight: bold; font-size: 14px; color: #333; margin-bottom: 5px; }
        img { width: 100%; height: 150px; object-fit: cover; border-radius: 4px; }
        .info { font-size: 12px; color: #666; margin-top: 5px; word-break: break-all; }
        .score { color: #888; font-size: 11px; }
        .check-overlay { position: absolute; top: 10px; right: 10px; background: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
        .checkbox { cursor: pointer; width: 16px; height: 16px; }
    </style>
    <script>
        function toggleCard(element) {
            const checkbox = element.querySelector('.checkbox');
            checkbox.checked = !checkbox.checked;
            updateStats();
            updateStyle(element, checkbox.checked);
        }

        function toggleCheckbox(event, element) {
            event.stopPropagation();
            const card = element.closest('.box');
            updateStats();
            updateStyle(card, element.checked);
        }

        function updateStyle(card, isChecked) {
            if (isChecked) {
                card.classList.add('correct');
            } else {
                card.classList.remove('correct');
            }
        }

        function updateStats() {
            const checkboxes = document.querySelectorAll('.checkbox');
            let count100 = 0;
            let countTotal = 0;
            
            checkboxes.forEach(cb => {
                if (cb.checked) {
                    const rank = parseInt(cb.dataset.rank);
                    if (rank <= 100) count100++;
                    countTotal++;
                }
            });
            
            document.getElementById('stat-top100').textContent = count100;
            document.getElementById('stat-total').textContent = countTotal;
        }
    </script>
    """)
    html.append("</head><body>")
    
    html.append("<div class='header'>")
    html.append(f"<div><h1>{title_prefix}: {base_name}</h1></div>")
    html.append("<div class='stats'>Precision@100: <span id='stat-top100' style='color:#4CAF50'>0</span> / 100 <span style='font-size:0.8em; color:#888'>(Total Selected: <span id='stat-total'>0</span>)</span></div>")
    html.append("</div>")
    
    html.append("<div class='container'>")
    
    for img in images:
        html.append(f"<div class='box' onclick='toggleCard(this)'>")
        html.append(f"<div class='check-overlay'><input type='checkbox' class='checkbox' data-rank='{img['rank']}' onclick='toggleCheckbox(event, this)'></div>")
        html.append(f"<div class='rank'>#{img['rank']}</div>")
        html.append(f"<img src='{img['path']}' loading='lazy' alt='{img['name']}'>")
        html.append(f"<div class='info'>{img['name']}</div>")
        html.append(f"<div class='score'>Score: {img['score']}</div>")
        html.append("</div>")
            
    html.append("</div>")
    html.append("</body></html>")
    
    with open(out_file, 'w') as f:
        f.write("\n".join(html))
    print(f"Generated {out_file}")

def main():
    # Process Re-ranked results
    files_result = glob.glob("ass2_rank_reranked_*.txt")
    for f in files_result:
        generate_evaluation_html(f, is_original=False)
        
    # Process Original rankings
    files_original = glob.glob("ass2_rank_original_*.txt")
    for f in files_original:
        generate_evaluation_html(f, is_original=True)

    if not files_result and not files_original:
        print("No ass2_rank_*.txt files found.")

if __name__ == "__main__":
    main()
