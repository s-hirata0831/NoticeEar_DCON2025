import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        
        # 無音部分の除去
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        
        # 短い音声データに対して適切なn_fftを設定
        n_fft = min(2048, 2**int(np.floor(np.log2(len(audio_trimmed)))))
        
        # 特徴量抽出（MFCC）
        mfcc = librosa.feature.mfcc(y=audio_trimmed, sr=sr, n_mfcc=MFCC_FEATURES, n_fft=n_fft)
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


def process_dataset(dataset_dir, output_dir):
    """
    データセット内の全ての音声ファイルを処理し、結果を保存する。
    """
    data = []
    labels = []
    
    # カテゴリごとに処理
    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        if os.path.isdir(category_path):
            print(f"Processing category: {category}")
            
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
                                labels.append(category)
    
    # 保存
    np.save(os.path.join(output_dir, "data.npy"), np.array(data))
    np.save(os.path.join(output_dir, "labels.npy"), np.array(labels))
    print(f"Saved processed data and labels to {output_dir}")

def main():
    """
    データセット全体に対して前処理を実行。
    """
    process_dataset(DATASET_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()
