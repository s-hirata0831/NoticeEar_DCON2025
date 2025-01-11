import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import tensorflow as tf
import time
import os

# モデルのロード
model_path = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwave_model.h5"
model = tf.keras.models.load_model(model_path)

# 特徴量抽出関数 (変更なし)
def extract_mfcc(y, sr, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T

# 音声入力の設定 (変更なし)
fs = 16000
seconds = float(4.0)
max_len = 126

try:
    while True:
        print("Recording...")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        print("Recording complete.")

        temp_filename = "temp_audio.wav"
        sf.write(temp_filename, myrecording, fs)

        y, sr = librosa.load(temp_filename, sr=fs)
        mfccs = extract_mfcc(y, sr)

        if mfccs is not None:
            # パディング
            pad_width = max_len - mfccs.shape[0]
            if pad_width > 0:
                mfccs = np.pad(mfccs, pad_width=((0, pad_width), (0, 0)), mode='constant')
            elif pad_width < 0:
              mfccs = mfccs[:max_len, :] #切り取り

            mfccs = mfccs[np.newaxis, ..., np.newaxis]  # バッチ次元とチャンネル次元を追加

            # 推論
            predictions = model.predict(mfccs)

            # 各クラスの確率を表示
            print("Predictions:", predictions) # 追加

            # 推論結果の表示を改善
            predicted_class = np.argmax(predictions)
            class_names = ["background", "microwave"]  # クラス名を設定
            print(f"Predicted class: {class_names[predicted_class]} (Probability: {predictions[0][predicted_class]:.4f})") #変更

        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped.")
finally:
    try:
        os.remove(temp_filename)
    except FileNotFoundError:
        pass