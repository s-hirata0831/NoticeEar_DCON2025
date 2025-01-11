import librosa
import numpy as np
import pandas as pd
import os

def resample_audio(file_path, target_sr=22050):
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def pad_or_trim(audio, length=22050):
    if len(audio) > length:
        return audio[:length]
    else:
        return np.pad(audio, (0, max(0, length - len(audio))), 'constant')

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def preprocess_and_save(microwave_files, background_files, save_path='preprocessed_data.csv'):
    processed_audios = []
    labels = []

    # 終了音の前処理
    for file in microwave_files:
        audio, _ = resample_audio(file)
        audio = pad_or_trim(audio)
        audio = normalize_audio(audio)
        processed_audios.append(audio)
        labels.append(1)  # 終了音のラベルは1

    # 背景音の前処理
    for file in background_files:
        audio, _ = resample_audio(file)
        audio = pad_or_trim(audio)
        audio = normalize_audio(audio)
        processed_audios.append(audio)
        labels.append(0)  # 背景音のラベルは0

    # データを保存
    data = {'audio': processed_audios, 'label': labels}
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)

# 使用例
microwave_files = [f'/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/0_microwave/microwave_{i}.wav' for i in range(1, 19)]  # 終了音ファイルのリスト
background_files = [f'/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/1_background/background_{i}.wav' for i in range(1, 19)]  # 背景音ファイルのリスト

preprocess_and_save(microwave_files, background_files)
