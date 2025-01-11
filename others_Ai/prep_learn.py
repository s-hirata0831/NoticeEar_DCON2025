import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd  # リアルタイム録音用
import soundfile as sf # wavファイルの保存用

def extract_mel_spectrogram(file_path):
    """音声ファイルからメルスペクトログラムを抽出する関数"""
    try:
        y, sr = librosa.load(file_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max) # dBスケールに変換
        return log_mel_spectrogram
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

microwave_mel_spectrograms = []
for i in range(1, 19):
    file_path = f"/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/0_prep_microwave/microwave_{i}.wav"
    mel_spectrogram = extract_mel_spectrogram(file_path)
    if mel_spectrogram is not None:
        microwave_mel_spectrograms.append(mel_spectrogram)

background_mel_spectrograms = []
for i in range(1, 19):
    file_path = f"/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/1_prep_background/background_{i}.wav"
    mel_spectrogram = extract_mel_spectrogram(file_path)
    if mel_spectrogram is not None:
        background_mel_spectrograms.append(mel_spectrogram)

# 可変長に対応するため、リストに格納
print("microwave_mel_spectrograms length:", len(microwave_mel_spectrograms))
print("example shape:", microwave_mel_spectrograms[0].shape)
print("background_mel_spectrograms length:", len(background_mel_spectrograms))
print("example shape:", background_mel_spectrograms[0].shape)

# ここから追記

# メルスペクトログラムをNumPy配列に変換し、チャンネル次元を追加
def pad_mel_spectrogram(mel_spectrograms, max_len=None):
    """メルスペクトログラムをパディングする関数。max_lenを指定可能"""
    if max_len is None:
        max_len = max(x.shape[1] for x in mel_spectrograms)  # 最大の幅を取得
    padded_mel_spectrograms = []
    for mel_spectrogram in mel_spectrograms:
        pad_width = max_len - mel_spectrogram.shape[1]
        padded_mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        padded_mel_spectrograms.append(padded_mel_spectrogram)
    return np.array(padded_mel_spectrograms)

# 両方のデータセットを合わせた最大の長さを計算
max_len = max(max(x.shape[1] for x in microwave_mel_spectrograms), max(x.shape[1] for x in background_mel_spectrograms))

microwave_mel_spectrograms_np = pad_mel_spectrogram(microwave_mel_spectrograms, max_len)
background_mel_spectrograms_np = pad_mel_spectrogram(background_mel_spectrograms, max_len)

# チャンネル次元を追加
microwave_mel_spectrograms_np = np.expand_dims(microwave_mel_spectrograms_np, axis=-1)
background_mel_spectrograms_np = np.expand_dims(background_mel_spectrograms_np, axis=-1)

print("microwave_mel_spectrograms_np shape:", microwave_mel_spectrograms_np.shape)
print("background_mel_spectrograms_np shape:", background_mel_spectrograms_np.shape)

# データとラベルを作成
X = np.concatenate([microwave_mel_spectrograms_np, background_mel_spectrograms_np])
y = np.concatenate([np.zeros(len(microwave_mel_spectrograms_np)), np.ones(len(background_mel_spectrograms_np))])

# データの分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# データ形状の確認
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

# CNNモデルの作成 (前回と同様)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cnn_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

input_shape = X_train.shape[1:]  # 入力形状をX_trainから取得
model = create_cnn_model(input_shape)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# モデルの学習
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# モデルの評価
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# 学習曲線のプロット
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save("/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/microwave_detection_model.h5")

# モデルのロード (推論時に使用)
loaded_model = keras.models.load_model("/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/microwave_detection_model.h5")

def predict_from_audio(audio, sr):
    """音声データから予測を行う関数"""
    try:
        # メルスペクトログラム抽出
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # パディング (学習時と同じ最大長を使用)
        max_len = X_train.shape[2] # 学習データの幅を取得
        pad_width = max_len - log_mel_spectrogram.shape[1]
        padded_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')

        # チャンネル次元を追加
        input_data = np.expand_dims(padded_mel_spectrogram, axis=0) # バッチ次元を追加
        input_data = np.expand_dims(input_data, axis=-1)

        # 予測
        prediction = loaded_model.predict(input_data)
        return prediction[0][0] # 確率値を返す

    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def record_and_predict(duration=3, sample_rate=44100):
    """録音して予測を行う関数"""
    print(f"{duration}秒間録音します...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # 録音終了まで待機
    print("録音終了")

    prediction = predict_from_audio(recording.flatten(), sample_rate)

    if prediction is not None:
        print(f"電子レンジの確率: {prediction:.4f}")
        if prediction > 0.5:
            print("電子レンジの音が検出されました！")
        else:
            print("電子レンジの音は検出されませんでした。")
    return recording, sample_rate

def continuous_prediction(duration=3, sample_rate=44100):
    """連続的に録音と予測を行う関数"""
    while True:
        print(f"{duration}秒間録音します...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()  # 録音終了まで待機
        print("録音終了")

        prediction = predict_from_audio(recording.flatten(), sample_rate)

        if prediction is not None:
            print(f"電子レンジの確率: {prediction:.4f}")
            if prediction > 0.5:
                print("電子レンジの音が検出されました！")
            else:
                print("電子レンジの音は検出されませんでした。")
        # wavファイルに保存する場合
        sf.write('output.wav', recording, sample_rate)
        
# 録音して予測 (1回のみ)
# recording, sample_rate = record_and_predict()

# 連続的に録音と予測
continuous_prediction()