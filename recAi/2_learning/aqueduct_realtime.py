import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import queue
import os

# === 設定 ===
OUTPUT_MODEL_PATH = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/2_learning/final_model.h5"  # 保存済みのモデル
NUM_CLASSES = 6  # 分類クラス数
LABELS = ["horn", "microwave", "animals(dog)", "guns", "siren", "back_ground"]  # クラスラベルを指定
MAX_FRAMES = 160  # 最大フレーム数
MFCC_FEATURES = 13  # MFCC特徴量の数
SAMPLE_RATE = 16000  # マイクのサンプリングレート（Hz）
BUFFER_DURATION = 2  # 推論対象の録音時間（秒）

# 音声データを格納するキュー
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """
    マイクからの音声をリアルタイムで取得するためのコールバック関数。
    """
    if status:
        print(f"Error: {status}")
    audio_queue.put(indata.copy())

def preprocess_audio_stream(audio_buffer, sample_rate, max_frames=MAX_FRAMES, mfcc_features=MFCC_FEATURES):
    """
    マイクから取得した音声データを前処理する。
    """
    # 音声データを1次元に変換
    y = np.squeeze(audio_buffer)
    
    # MFCC特徴量の抽出
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=mfcc_features)
    
    # フレーム数を固定長にパディングまたは切り取り
    if mfcc.shape[1] < max_frames:
        padding = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode="constant")
    else:
        mfcc = mfcc[:, :max_frames]
    
    # モデル入力に対応する次元に変換
    mfcc = np.expand_dims(mfcc, axis=-1)  # チャンネル次元を追加
    return mfcc

def predict_stream(model, audio_buffer):
    """
    マイク入力データを分類モデルで推論する。
    """
    # 音声データの前処理
    processed_audio = preprocess_audio_stream(audio_buffer, SAMPLE_RATE)
    
    # バッチ次元の追加
    processed_audio = np.expand_dims(processed_audio, axis=0)
    
    # 推論
    predictions = model.predict(processed_audio)
    
    # 推論結果の取得
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]
    
    return LABELS[predicted_class], confidence

if __name__ == "__main__":
    # モデルの読み込み
    model = tf.keras.models.load_model(OUTPUT_MODEL_PATH)
    
    # マイクからリアルタイム音声を取得
    with sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=1, blocksize=int(SAMPLE_RATE * BUFFER_DURATION)):
        print("Listening...")
        try:
            while True:
                # 音声データを取得
                if not audio_queue.empty():
                    audio_buffer = audio_queue.get()
                    
                    # 推論を実行
                    predicted_label, confidence_score = predict_stream(model, audio_buffer)
                    print(f"Predicted Class: {predicted_label}, Confidence Score: {confidence_score:.4f}")
        except KeyboardInterrupt:
            print("\nStopped.")
