import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# データディレクトリ
data_dir = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/4_features"

# データのロード
X_train = np.load(os.path.join(data_dir, "mfcc_features_train.npy"), allow_pickle=True)
y_train = np.load(os.path.join(data_dir, "labels_train.npy"), allow_pickle=True) #astype(int)は不要
X_val = np.load(os.path.join(data_dir, "mfcc_features_val.npy"), allow_pickle=True) #astype(int)は不要
y_val = np.load(os.path.join(data_dir, "labels_val.npy"), allow_pickle=True) #astype(int)は不要
X_test = np.load(os.path.join(data_dir, "mfcc_features_test.npy"), allow_pickle=True) #astype(int)は不要
y_test = np.load(os.path.join(data_dir, "labels_test.npy"), allow_pickle=True) #astype(int)は不要

# num_classesを正しく設定 (二値分類なので2)
num_classes = 2

# ラベルをone-hotベクトルに変換 (num_classes を指定)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# 入力データの形状を調整 (チャンネル次元を追加)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"Train data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of classes: {num_classes}")

# モデルの構築 (CNN)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax') #出力層はsoftmax
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy', #多クラス分類なのでcategorical_crossentropy
              metrics=['accuracy'])

model.summary()

# EarlyStoppingのコールバックを設定
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)

# 学習
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# モデルの保存
model.save("microwave_model.h5")

# 評価
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# 学習過程の可視化
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()