import librosa
import soundfile as sf
import os
import numpy as np

def split_audio(input_folder, output_folder, split_duration=4.0):
    """
    指定されたフォルダ内のWAVファイルを指定の長さ（split_duration）ごとに分割します。
    最後の分割ファイルがsplit_duration未満の場合は削除します。

    Args:
        input_folder (str): 入力WAVファイルが格納されているフォルダパス。
        output_folder (str): 出力WAVファイルを保存するフォルダパス。
        split_duration (float): 分割する音声の長さ（秒）。
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_file = os.path.join(input_folder, filename)
            try:
                y, sr = librosa.load(input_file, sr=None)
                total_duration = librosa.get_duration(y=y, sr=sr)
                num_splits = int(np.ceil(total_duration / split_duration))

                for i in range(num_splits):
                    start_time = i * split_duration
                    end_time = min((i + 1) * split_duration, total_duration)

                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)

                    y_split = y[start_sample:end_sample]
                    split_duration_actual = librosa.get_duration(y=y_split, sr=sr) # 分割後の実際の時間

                    output_filename = os.path.splitext(filename)[0] + f"_split_{i+1}.wav"
                    output_file = os.path.join(output_folder, output_filename)

                    if split_duration_actual >= split_duration: # 分割後の時間が規定値以上の場合のみ保存
                        sf.write(output_file, y_split, sr)
                        print(f"{input_file} を {output_filename} に分割しました。")
                    else:
                        print(f"{output_filename} は{split_duration}秒未満のため、保存しませんでした。実際の長さ: {split_duration_actual}秒")
            except FileNotFoundError:
                print(f"{input_file} が見つかりませんでした。")
            except Exception as e:
                print(f"{input_file} の処理中にエラーが発生しました: {e}")

# 使用例
input_folder = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/1_prep_background"  # 入力WAVファイルがあるフォルダ
output_folder = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/1_prep_background" # 出力WAVファイルを保存するフォルダ
split_audio(input_folder, output_folder)