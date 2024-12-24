import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# データのロード
data_path = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/1_dataset_and_prep/processed_data/data.npy"
labels_path = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/1_dataset_and_prep/processed_data/labels.npy"
X = np.load(data_path)
y = np.load(labels_path)

# Xとyの形状を確認
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# データ正規化
X = X / np.max(X)

# 実際のユニークなクラス数を確認
num_classes = len(np.unique(y))
print("Number of unique classes:", num_classes)

# データ分割（学習用、検証用、テスト用）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 入力層の形状確認
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# num_classesの確認
print("Number of classes:", num_classes)

# モデル構築（CNN + LSTM）
input_layer = tf.keras.layers.Input(shape=(X.shape[1], X.shape[2]))

# CNN部分
cnn = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu')(input_layer)
cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)
cnn = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(cnn)
cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)
cnn = tf.keras.layers.Flatten()(cnn)

# LSTM部分
lstm = tf.keras.layers.LSTM(128, return_sequences=False)(tf.keras.layers.Reshape((-1, 1))(cnn))

# 全結合層
dense = tf.keras.layers.Dense(128, activation='relu')(lstm)
dropout = tf.keras.layers.Dropout(0.3)(dense)

# 出力層（num_classes に対応）
output_layer = tf.keras.layers.Dense(3, activation='softmax')(dropout)  # 3クラス分類に変更

# モデルコンパイル
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # categorical_crossentropyに変更

# 早期終了のコールバック
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# モデル学習
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping])

# モデル評価
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 混同行列と分類レポート
y_pred = model.predict(X_test)
# 多クラス分類の場合、one-hotエンコードから最大値を取得
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred))

conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 精度と損失のプロット
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# モデル保存（TensorFlow Lite変換準備）
model.save("/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/2_learning/cnn_lstm_model.h5")
