import librosa
import soundfile as sf
import os

def preprocess_audio(input_path, output_path, target_sr=16000):
    """
    音声ファイルの前処理を行う関数
    """
    try:
        # 音声ファイルの読み込み
        y, sr = librosa.load(input_path, sr=None, mono=True)  # mono=Trueでモノラル化

        # リサンプリング
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # 正規化 (音量を-1から1の範囲に調整)
        y = librosa.util.normalize(y)

        # ファイルに保存
        sf.write(output_path, y, target_sr, format='WAV')

    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# 例: フォルダ内のすべての.wavファイルを処理
input_folder = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/1_background"  # 入力音声ファイルのフォルダ
output_folder = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/1_prep_background"  # 前処理後の音声ファイルのフォルダ

os.makedirs(output_folder, exist_ok=True) # 出力フォルダを作成

for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)
        preprocess_audio(input_file, output_file)

print("Preprocessing complete.")