import numpy as np
import tensorflow as tf
import librosa
import os

# === 設定 ===
OUTPUT_MODEL_PATH = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/2_learning/final_model_v2.h5"  # 保存済みのモデル
NUM_CLASSES = 6  # 分類クラス数
LABELS = ["horn", "microwave", "animals(dog)", "guns", "siren", "back_ground"]  # クラスラベルを指定
MAX_FRAMES = 160  # 最大フレーム数
MFCC_FEATURES = 13  # MFCC特徴量の数

def preprocess_audio(audio_path, max_frames=MAX_FRAMES, mfcc_features=MFCC_FEATURES):
    """
    音声データの前処理を行い、モデルに入力可能な形式に変換する。
    """
    # 音声ファイルの読み込み
    y, sr = librosa.load(audio_path, sr=None)
    
    # MFCC特徴量の抽出
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_features)
    
    # フレーム数を固定長にパディングまたは切り取り
    if mfcc.shape[1] < max_frames:
        padding = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode="constant")
    else:
        mfcc = mfcc[:, :max_frames]
    
    # モデル入力に対応する次元に変換
    mfcc = np.expand_dims(mfcc, axis=-1)  # チャンネル次元を追加
    return mfcc

def predict_audio(model_path, audio_path):
    """
    音声ファイルを分類モデルで推論する。
    """
    # モデルの読み込み
    model = tf.keras.models.load_model(model_path)
    
    # 音声ファイルの前処理
    processed_audio = preprocess_audio(audio_path)
    
    # バッチ次元の追加
    processed_audio = np.expand_dims(processed_audio, axis=0)
    
    # 推論
    predictions = model.predict(processed_audio)
    
    # 推論結果の取得
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]
    
    return LABELS[predicted_class], confidence

if __name__ == "__main__":
    # 推論対象の音声ファイル
    #audio_file_path = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/2_learning/back_geound.mp3"
    audio_file_path = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/1_dataset_and_prep/dataset/4_guns/fold1/guns_1_5.wav"
    
    if not os.path.exists(audio_file_path):
        print(f"Error: File not found: {audio_file_path}")
    else:
        # 推論の実行
        predicted_label, confidence_score = predict_audio(OUTPUT_MODEL_PATH, audio_file_path)
        print(f"Predicted Class: {predicted_label}")
        print(f"Confidence Score: {confidence_score:.4f}")
