import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold

# === 設定 ===
DATASET_DIR = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/1_dataset_and_prep/dataset"
OUTPUT_DIR = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/1_dataset_and_prep/processed_data"
SAMPLE_RATE = 16000
MFCC_FEATURES = 13
MAX_FRAMES = 160

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        n_fft = min(2048, 2**int(np.floor(np.log2(len(audio)))))
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_FEATURES, n_fft=n_fft).T

        scaler = StandardScaler()
        mfcc_normalized = scaler.fit_transform(mfcc)
        
        if mfcc_normalized.shape[0] < MAX_FRAMES:
            pad_width = MAX_FRAMES - mfcc_normalized.shape[0]
            mfcc_padded = np.pad(mfcc_normalized, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc_padded = mfcc_normalized[:MAX_FRAMES, :]
        
        return mfcc_padded
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_dataset(dataset_dir):
    data, labels, label_map = [], [], {}
    for idx, category in enumerate(sorted(os.listdir(dataset_dir))):
        category_path = os.path.join(dataset_dir, category)
        if os.path.isdir(category_path):
            print(f"Processing category: {category}")
            label_map[category] = idx
            for fold in os.listdir(category_path):
                fold_path = os.path.join(category_path, fold)
                if os.path.isdir(fold_path):
                    for file in os.listdir(fold_path):
                        if file.endswith(".wav"):
                            file_path = os.path.join(fold_path, file)
                            mfcc_features = process_audio(file_path)
                            if mfcc_features is not None:
                                data.append(mfcc_features)
                                labels.append(idx)
    return np.array(data), np.array(labels), label_map

def save_data(data, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    encoder = OneHotEncoder(sparse_output=False)
    labels_one_hot = encoder.fit_transform(labels.reshape(-1, 1))

    np.save(os.path.join(output_dir, "data.npy"), data)
    np.save(os.path.join(output_dir, "labels.npy"), labels_one_hot)

    print("Data and labels saved successfully.")
    print(f"Data shape: {data.shape}, Labels shape: {labels_one_hot.shape}")

def main():
    data, labels, label_map = process_dataset(DATASET_DIR)
    save_data(data, labels, OUTPUT_DIR)
    print(f"Label map: {label_map}")

if __name__ == "__main__":
    main()
