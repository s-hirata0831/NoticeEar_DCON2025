import librosa
import soundfile as sf
import os
import numpy as np

def adjust_audio_length(input_folder, output_folder, target_duration=float(4.0)):
    """
    指定されたフォルダ内のWAVファイルをすべて指定の長さに調整します。

    Args:
        input_folder (str): 入力WAVファイルが格納されているフォルダパス。
        output_folder (str): 出力WAVファイルを保存するフォルダパス。
        target_duration (float): 目標とする音声の長さ（秒）。
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(1, 17):  # microwave_1.wav から microwave_16.wav まで
        input_file = os.path.join(input_folder, f"/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/0_prep_microwave/microwave_{i}.wav")
        output_file = os.path.join(output_folder, f"/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/0_prep_microwave/microwave_{i}_adjusted.wav")

        try:
            y, sr = librosa.load(input_file, sr=None) # サンプリングレートを保持

            current_duration = librosa.get_duration(y=y, sr=sr)

            if current_duration != target_duration:
                # 目標時間に合わせて伸縮
                y_adjusted = librosa.resample(y, orig_sr=sr, target_sr=int(sr * (current_duration / target_duration)))
                
                # 長すぎる場合は切り取り、短すぎる場合は無音を追加
                if len(y_adjusted) > int(target_duration * sr):
                    y_adjusted = y_adjusted[:int(target_duration * sr)]
                elif len(y_adjusted) < int(target_duration * sr):
                    padding_length = int(target_duration * sr) - len(y_adjusted)
                    y_adjusted = np.pad(y_adjusted, (0, padding_length), mode='constant')

                sf.write(output_file, y_adjusted, sr)
                print(f"{input_file} を {output_file} に調整しました。")
            else:
                print(f"{input_file} は既に {target_duration} 秒です。")

        except FileNotFoundError:
            print(f"{input_file} が見つかりませんでした。")
        except Exception as e:
            print(f"{input_file} の処理中にエラーが発生しました: {e}")

# 使用例
input_folder = "input_audio"  # 入力WAVファイルがあるフォルダ
output_folder = "output_audio" # 出力WAVファイルを保存するフォルダ
adjust_audio_length(input_folder, output_folder)