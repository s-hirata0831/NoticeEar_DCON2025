import librosa
import numpy as np
import os
import re

def extract_mfcc(file_path, n_mfcc=40):
    """
    音声ファイルからMFCCを抽出する関数
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfccs.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

split_data_dir = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/3_split_data"
output_dir = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/4_features"

os.makedirs(output_dir, exist_ok=True)

for split in ["train", "val", "test"]:
    mfcc_features = []
    labels = []
    filenames = []
    max_len = 0

    data_dir = os.path.join(split_data_dir, split)
    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(data_dir, filename)
            mfccs = extract_mfcc(file_path)

            if mfccs is not None:
                if filename.startswith("background_adjusted"):
                    label = 0  # backgroundは0
                elif filename.startswith("microwave_adjusted"):
                    label = 1  # microwaveは1
                else:
                    print(f"Warning: Unknown filename format: {filename}")
                    continue # 次のファイルへ

                labels.append(label)
                mfcc_features.append(mfccs)
                filenames.append(filename)
                max_len = max(max_len, mfccs.shape[0])

    # パディング
    padded_mfcc_features = []
    for mfcc in mfcc_features:
        padding_len = max_len - mfcc.shape[0]
        padded_mfcc = np.pad(mfcc, ((0, padding_len), (0, 0)), 'constant')
        padded_mfcc_features.append(padded_mfcc)

    mfcc_features = np.array(padded_mfcc_features)
    labels = np.array(labels)
    filenames = np.array(filenames)

    np.save(os.path.join(output_dir, f"mfcc_features_{split}.npy"), mfcc_features)
    np.save(os.path.join(output_dir, f"labels_{split}.npy"), labels)
    np.save(os.path.join(output_dir, f"filenames_{split}.npy"), filenames)
    print(f"{split} MFCC extraction complete. Shape: {mfcc_features.shape}")

print("All MFCC extraction complete.")