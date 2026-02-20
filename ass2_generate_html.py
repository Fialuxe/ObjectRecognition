import os

def generate_html(rank_file, output_html, keyword):
    with open(rank_file, 'r') as f:
        lines = f.readlines()
    
    # Extract ranking data
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            path = parts[0]
            score = float(parts[1])
            data.append({'path': path, 'score': score})

    # Generate HTML content with JS for manual review
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ranking Result: {os.path.basename(rank_file)}</title>
        <style>
            body {{ font-family: sans-serif; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .item {{ margin: 5px; border: 1px solid #ccc; padding: 5px; width: 160px; text-align: center; }}
            img {{ max-width: 150px; max-height: 150px; cursor: pointer; }}
            .score {{ font-size: 12px; color: #555; }}
            .rank {{ font-weight: bold; color: #007bff; }}
            .controls {{ position: sticky; top: 0; background: #fff; padding: 10px; border-bottom: 2px solid #000; z-index: 100; }}
            .checked {{ background-color: #e0ffe0; border-color: #009900; }}
            #export-area {{ margin-top: 20px; }}
        </style>
        <script>
            function toggleCheck(id) {{
                const checkbox = document.getElementById('cb-' + id);
                checkbox.checked = !checkbox.checked;
                updateStyle(id);
            }}

            function updateStyle(id) {{
                const div = document.getElementById('div-' + id);
                const checkbox = document.getElementById('cb-' + id);
                if (checkbox.checked) {{
                    div.classList.add('checked');
                }} else {{
                    div.classList.remove('checked');
                }}
            }}

            function checkAll(n) {{
                for (let i = 1; i <= n; i++) {{
                    const checkbox = document.getElementById('cb-' + i);
                    if (checkbox) {{
                        checkbox.checked = true;
                        updateStyle(i);
                    }}
                }}
            }}

            function exportMistakes() {{
                // Logic: Find unchecked items within top 100 (or specified N)
                const N = 100;
                let mistakes = [];
                let mistakeHtml = "<h2>Mistakes (Unchecked Top " + N + ")</h2><p>These are the images you did NOT verify as correct.</p><div style='display:flex; flex-wrap:wrap;'>";
                
                for (let i = 1; i <= Math.min(N, {len(data)}); i++) {{
                    const checkbox = document.getElementById('cb-' + i);
                    const imgPath = document.getElementById('img-' + i).src;
                    
                    if (checkbox && !checkbox.checked) {{
                        mistakes.push(imgPath);
                        mistakeHtml += "<div style='margin:5px; border:1px solid red; padding:5px; text-align:center;'><img src='" + imgPath + "' style='max-width:100px;'><br>Rank: " + i + "</div>";
                    }}
                }}
                mistakeHtml += "</div>";
                
                const win = window.open("", "Mistakes", "width=800,height=600");
                win.document.write("<html><body style='font-family:sans-serif'>" + mistakeHtml + "</body></html>");
            }}
        </script>
    </head>
    <body>
        <div class="controls">
            <h2>{os.path.basename(rank_file)} ({keyword})</h2>
            <button onclick="checkAll(100)">Check Top 100 (Assume Correct)</button>
            <button onclick="exportMistakes()">Export Unchecked (Mistakes)</button>
            <p>Usage: 1. Click "Check Top 100". 2. Manually uncheck the MISTAKES (non-target images). 3. Click "Export Unchecked".</p>
        </div>
        <div class="container">
    """
    
    for i, item in enumerate(data, 1):
        html_content += f"""
            <div class="item" id="div-{i}">
                <div class="rank">Rank: {i}</div>
                <img id="img-{i}" src="{item['path']}" onclick="toggleCheck({i})">
                <br>
                <input type="checkbox" id="cb-{i}" onclick="updateStyle({i})"> Correct
                <div class="score">{item['score']:.4f}</div>
                <div class="fname">{os.path.basename(item['path'])}</div>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    with open(output_html, 'w') as f:
        f.write(html_content)
    print(f"Generated {output_html}")
    
def main():
    import glob
    files = glob.glob("ass2_rank_*.txt")
    
    for f in files:
        keyword = "apple" if "apple" in f else "kiwi"
        out_name = f.replace("ass2_rank_", "ass2_eval_").replace(".txt", ".html")
        print(f"Processing {f} -> {out_name}")
        generate_html(f, out_name, keyword)

if __name__ == "__main__":
    main()
