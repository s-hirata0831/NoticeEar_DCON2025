import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

"""
前処理用プログラムv2.0
【変更点】
UrbanSound8Kの10-fold-cross-validation用の処理を追加。
前処理後のプログラムを各フォルダごとに分割して保存するように変更。
OneHotEncoder でラベルをOne-hotエンコーディングした形に変換。
"""

# === 設定 ===
DATASET_DIR = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/1_dataset_and_prep/dataset"  # データセットのルートディレクトリ
OUTPUT_DIR = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/1_dataset_and_prep/processed_data"  # 前処理済みデータの保存先
SAMPLE_RATE = 16000  # サンプリングレート
MFCC_FEATURES = 13  # MFCCの次元数
MAX_FRAMES = 160  # 最大フレーム数（約1.6秒を想定）

os.makedirs(OUTPUT_DIR, exist_ok=True)  # 出力ディレクトリの作成

def process_audio(file_path):
    """
    音声ファイルを読み込み、前処理を行い、特徴量を返す。
    """
    try:
        # 音声ファイルの読み込み
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # 特徴量抽出（MFCC）
        n_fft = min(2048, 2**int(np.floor(np.log2(len(audio)))))
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_FEATURES, n_fft=n_fft)
        mfcc = mfcc.T  # 転置してフレーム数 x 特徴量次元の形状に
        
        # 特徴量の正規化
        scaler = StandardScaler()
        mfcc_normalized = scaler.fit_transform(mfcc)
        
        # フレーム数を統一（パディングまたはカット）
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
    """
    データセット内の全ての音声ファイルを処理し、結果を収集する。
    """
    data = []
    labels = []
    label_map = {}  # カテゴリ名を数値にマッピングする辞書

    # カテゴリごとに処理
    for idx, category in enumerate(sorted(os.listdir(dataset_dir))):  # ソートして順序を固定
        category_path = os.path.join(dataset_dir, category)
        if os.path.isdir(category_path):
            print(f"Processing category: {category}")
            
            # カテゴリを数値ラベルにマッピング
            if category not in label_map:
                label_map[category] = idx  # 新しいカテゴリを追加
            
            # foldごとに処理
            for fold in os.listdir(category_path):
                fold_path = os.path.join(category_path, fold)
                if os.path.isdir(fold_path):
                    for file in os.listdir(fold_path):
                        if file.endswith(".wav"):
                            file_path = os.path.join(fold_path, file)
                            
                            # 音声データの処理
                            mfcc_features = process_audio(file_path)
                            if mfcc_features is not None:
                                data.append(mfcc_features)
                                labels.append(label_map[category])  # 数値ラベルを追加
                                
                                # デバッグログを出力
                                print(f"Added file: {file_path}, Label: {label_map[category]}")

    return np.array(data), np.array(labels), label_map

def save_folds(data, labels, label_map, output_dir):
    """
    データを10-foldクロスバリデーション用に分割して保存する。
    """
    encoder = OneHotEncoder(sparse_output=False)  # One-hotエンコーディング
    labels_one_hot = encoder.fit_transform(labels.reshape(-1, 1))  # reshape(-1, 1) で2次元に変換
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-foldクロスバリデーション
    fold_idx = 0

    for train_idx, test_idx in kf.split(data):
        # トレーニングデータとテストデータに分割
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = labels_one_hot[train_idx], labels_one_hot[test_idx]

        # 各fold用の保存先ディレクトリ
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        # データを保存
        np.save(os.path.join(fold_dir, "X_train.npy"), X_train)
        np.save(os.path.join(fold_dir, "X_test.npy"), X_test)
        np.save(os.path.join(fold_dir, "y_train.npy"), y_train)
        np.save(os.path.join(fold_dir, "y_test.npy"), y_test)
        
        print(f"Saved fold {fold_idx + 1} data to {fold_dir}")
        fold_idx += 1

def main():
    """
    データセット全体に対して前処理を実行し、クロスバリデーション用に保存。
    """
    data, labels, label_map = process_dataset(DATASET_DIR)
    save_folds(data, labels, label_map, OUTPUT_DIR)

if __name__ == "__main__":
    main()
