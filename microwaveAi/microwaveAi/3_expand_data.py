import librosa
import soundfile as sf
import numpy as np
import os
import random

def time_stretch_and_resize(y, sr, rate, target_length):
    """
    タイムストレッチを行い、指定した長さにリサイズする関数
    """
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    target_samples = int(target_length * sr)

    if len(y_stretched) > target_samples:
        y_resized = y_stretched[:target_samples]  # 切り取り
    elif len(y_stretched) < target_samples:
        padding_len = target_samples - len(y_stretched)
        y_resized = np.pad(y_stretched, (0, padding_len), 'constant')  # ゼロパディング
    else:
        y_resized = y_stretched
    return y_resized

def augment_audio(input_path, output_path, num_augmentations=10, target_length=float(4.0), target_sr=16000):
    """
    音声データを拡張する関数 (タイムストレッチ後にリサイズを追加)
    """
    try:
        y, sr = librosa.load(input_path, sr=target_sr)
        for i in range(num_augmentations):
            y_augmented = y.copy()

            # ピッチシフト
            n_steps = random.uniform(-2, 2)
            y_augmented = librosa.effects.pitch_shift(y_augmented, sr=sr, n_steps=n_steps)

            # タイムストレッチ
            rate = random.uniform(0.9, 1.1)
            y_augmented = time_stretch_and_resize(y_augmented, sr, rate, target_length) # ここでリサイズ

            # ノイズ付加
            noise_level = random.uniform(0.001, 0.005)
            noise = np.random.normal(0, noise_level, len(y_augmented))
            y_augmented = y_augmented + noise

            # 音量調整
            gain = random.uniform(0.8, 1.2)
            y_augmented = y_augmented * gain
            
            output_filename = os.path.splitext(os.path.basename(input_path))[0] + f"_aug_{i}.wav"
            output_filepath = os.path.join(output_folder, output_filename)
            sf.write(output_filepath, y_augmented, target_sr, format='WAV')

    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# 入力フォルダと出力フォルダを設定
input_folder = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/1_prep_background"
output_folder = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/2_expand_data"

os.makedirs(output_folder, exist_ok=True)

# 拡張を行う
num_total_files = 10000
num_original_files = 40 # microwave_adjusted_1.wavからmicrowave_adjus　ted_14.wavまでなので14個
num_augmentations_per_file = (num_total_files // num_original_files) + 1

for i in range(1, num_original_files + 1):
    input_filename = f"background_adjusted_{i}.wav" # ファイル名からパスを生成するように修正
    input_filepath = os.path.join(input_folder, input_filename)
    if os.path.exists(input_filepath):
        augment_audio(input_filepath, output_folder, num_augmentations_per_file, target_length=4.0) # target_lengthを渡す
    else:
        print(f"{input_filepath} not found")

print("Augmentation complete.")