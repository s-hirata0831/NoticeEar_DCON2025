import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# データ拡張関数
def augment_audio(audio, sr):
    """音声データを拡張する関数"""
    augmented_audio = audio.copy()

    # ピッチシフト
    n_steps = np.random.randint(-2, 3)  # -2から2半音シフト
    augmented_audio = librosa.effects.pitch_shift(augmented_audio, sr=sr, n_steps=n_steps)

    # 時間伸縮
    rate = np.random.uniform(0.8, 1.2)  # 0.8倍から1.2倍に伸縮
    augmented_audio = librosa.effects.time_stretch(augmented_audio, rate=rate)

    # ノイズ付加
    noise_level = np.random.uniform(0.005, 0.01) # ノイズレベルを調整
    noise = np.random.normal(0, noise_level * np.max(np.abs(augmented_audio)), len(augmented_audio))
    augmented_audio += noise

    return augmented_audio

def extract_mel_spectrogram(file_path, augment=False):
    """音声ファイルからメルスペクトログラムを抽出する関数。データ拡張オプション付き"""
    try:
        y, sr = librosa.load(file_path, sr=None)
        if augment:
            y = augment_audio(y, sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return log_mel_spectrogram
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# データ読み込みと拡張
microwave_mel_spectrograms = []
background_mel_spectrograms = []
num_augmentations = 2  # 各データに対する拡張数

for i in range(1, 19):
    microwave_file = f"/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/0_prep_microwave/microwave_{i}.wav"
    background_file = f"/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/1_prep_background/background_{i}.wav"

    microwave_mel = extract_mel_spectrogram(microwave_file)
    background_mel = extract_mel_spectrogram(background_file)
    if microwave_mel is not None and background_mel is not None:
        microwave_mel_spectrograms.append(microwave_mel)
        background_mel_spectrograms.append(background_mel)

        # データ拡張
        for _ in range(num_augmentations):
            aug_microwave_mel = extract_mel_spectrogram(microwave_file, augment=True)
            aug_background_mel = extract_mel_spectrogram(background_file, augment=True)
            if aug_microwave_mel is not None and aug_background_mel is not None:
                microwave_mel_spectrograms.append(aug_microwave_mel)
                background_mel_spectrograms.append(aug_background_mel)

# パディング
def pad_mel_spectrogram(mel_spectrograms, max_len=None):
    if max_len is None:
        max_len = max(x.shape[1] for x in mel_spectrograms)
    padded_mel_spectrograms = []
    for mel_spectrogram in mel_spectrograms:
        pad_width = max_len - mel_spectrogram.shape[1]
        padded_mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        padded_mel_spectrograms.append(padded_mel_spectrogram)
    return np.array(padded_mel_spectrograms)

max_len = max(max(x.shape[1] for x in microwave_mel_spectrograms), max(x.shape[1] for x in background_mel_spectrograms))

microwave_mel_spectrograms_np = pad_mel_spectrogram(microwave_mel_spectrograms, max_len)
background_mel_spectrograms_np = pad_mel_spectrogram(background_mel_spectrograms, max_len)

microwave_mel_spectrograms_np = np.expand_dims(microwave_mel_spectrograms_np, axis=-1)
background_mel_spectrograms_np = np.expand_dims(background_mel_spectrograms_np, axis=-1)

# データとラベル作成、シャッフル
X = np.concatenate([microwave_mel_spectrograms_np, background_mel_spectrograms_np])
y = np.concatenate([np.zeros(len(microwave_mel_spectrograms_np)), np.ones(len(background_mel_spectrograms_np))])
X, y = shuffle(X, y, random_state=42)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# CNNモデル作成 (L2正則化を追加)
def create_cnn_model(input_shape, l2_reg=0.001):  # L2正則化パラメータを追加
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

input_shape = X_train.shape[1:]
model = create_cnn_model(input_shape)

# コンパイル (AdamWオプティマイザに変更)
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# モデル学習 (EarlyStoppingを追加)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

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
#loaded_model = keras.models.load_model("/Users/hiratasoma/Documents/NoticeEar_DCON2025/精度の高いモデル.h5")

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

#44100
def continuous_prediction(duration=3, sample_rate=16000):
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