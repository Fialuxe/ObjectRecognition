import os
import shutil

# 設定：ここにあるフォルダを順番に結合します
source_folders = ['apple', 'apple2']
target_folder = 'new_apple'

# 保存先フォルダを作成（すでにある場合はそのまま使う）
os.makedirs(target_folder, exist_ok=True)

counter = 1

print("処理を開始します...")

# 指定したフォルダ順に処理
for folder in source_folders:
    # ファイル名順に並べ替え
    files = sorted(os.listdir(folder))
    
    for filename in files:
        # 画像ファイル(.jpg, .jpeg)だけを対象にする（大文字小文字無視）
        if filename.lower().endswith(('.jpg', '.jpeg')):
            
            # 元のファイルの場所
            old_path = os.path.join(folder, filename)
            
            # 新しい名前 (6桁の連番.jpg)
            new_filename = f"{counter:06d}.jpg"
            new_path = os.path.join(target_folder, new_filename)
            
            # コピーを実行
            shutil.copy2(old_path, new_path)
            
            # 確認用の表示
            print(f"[{folder}] {filename} -> {new_filename}")
            
            # 番号を進める
            counter += 1

print("-" * 30)
print(f"完了！ 合計 {counter - 1} 枚の画像を {target_folder} に保存しました。")
