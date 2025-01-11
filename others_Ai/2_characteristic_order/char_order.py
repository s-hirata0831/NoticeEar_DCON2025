import librosa
import numpy as np
import pandas as pd

def extract_features(audio, sr=22050, n_mfcc=13):
    """
    音声データからMFCC特徴量を抽出
    """
    if len(audio) < 2048:
        n_fft = 512  # 短い信号に対して小さなn_fftを設定
    else:
        n_fft = 2048
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def process_audio_files(csv_path='/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/1_prep/preprocessed_data.csv', output_path='/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/2_characteristic_order/audio_features.csv'):
    """
    前処理された音声データから特徴量を抽出し、保存する
    """
    df = pd.read_csv(csv_path)
    features = []
    labels = []

    for index, row in df.iterrows():
        # 括弧、改行、及び省略記号を除去してからリスト化
        audio_str = row['audio'].replace('[', '').replace(']', '').replace('\n', ' ').replace('...', '').strip()
        try:
            audio = np.array(list(map(float, audio_str.split())))
        except ValueError:
            print(f"Skipping row {index} due to conversion error.")
            continue
        label = row['label']
        mfcc_features = extract_features(audio)
        features.append(mfcc_features)
        labels.append(label)

    features_df = pd.DataFrame(features)
    features_df['label'] = labels

    features_df.to_csv(output_path, index=False)

# 実行
process_audio_files()
