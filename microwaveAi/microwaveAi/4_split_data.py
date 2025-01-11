import os
import random
import shutil

data_dir = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/2_expand_data"
output_dir = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/3_split_data"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 出力ディレクトリの作成（存在する場合は削除してから作成）
for folder in ["train", "val", "test"]:
    target_dir = os.path.join(output_dir, folder)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # ディレクトリごと削除
    os.makedirs(target_dir, exist_ok=True)

# backgroundとmicrowaveのファイルをそれぞれリストに格納
background_files = []
microwave_files = []

for filename in os.listdir(data_dir):
    if filename.startswith("background_adjusted"):
        background_files.append(filename)
    elif filename.startswith("microwave_adjusted"):
        microwave_files.append(filename)

# 各クラスのファイルをシャッフル
random.shuffle(background_files)
random.shuffle(microwave_files)

# 各クラスごとにtrain/val/testに分割
for class_files, class_name in [(background_files, "background"), (microwave_files, "microwave")]:
    num_files = len(class_files)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)

    for i, filename in enumerate(class_files):
        src_path = os.path.join(data_dir, filename)
        if i < num_train:
            dst_path = os.path.join(output_dir, "train", filename)
        elif i < num_train + num_val:
            dst_path = os.path.join(output_dir, "val", filename)
        else:
            dst_path = os.path.join(output_dir, "test", filename)
        shutil.copy(src_path, dst_path)
    print(f"{class_name} files split complete.")

print("Data split complete.")