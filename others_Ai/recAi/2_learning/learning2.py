import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import statistics

# 設定
PROCESSED_DATA_DIR = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/1_dataset_and_prep/processed_data"  # 前処理済みデータのディレクトリ
EPOCHS = 50  # エポック数
BATCH_SIZE = 32  # バッチサイズ
NUM_CLASSES = 6 # 分類数
OUTPUT_MODEL_PATH = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/2_learning/final_model.h5" # 出力モデルのパス

def train_and_evaluate(fold_dir):
    """指定されたfoldで学習と評価を行う"""
    X_train = np.load(os.path.join(fold_dir, "X_train.npy"))
    X_test = np.load(os.path.join(fold_dir, "X_test.npy"))
    y_train = np.load(os.path.join(fold_dir, "y_train.npy"))
    y_test = np.load(os.path.join(fold_dir, "y_test.npy"))

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    input_shape = X_train.shape[1:]

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    # LSTMレイヤーをDenseレイヤーに変更
    model.add(keras.layers.Dense(64, activation='relu')) # 中間層を追加
    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

    return accuracy, precision, recall, f1, model

def main():
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    models = [] # 各foldのモデルを保存

    for i in range(1, 11):
        fold_dir = os.path.join(PROCESSED_DATA_DIR, f"fold_{i}")
        print(f"Training and evaluating fold {i}...")
        accuracy, precision, recall, f1, model = train_and_evaluate(fold_dir)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        models.append(model)

    # 全foldの平均値を計算
    mean_accuracy = statistics.mean(accuracies)
    mean_precision = statistics.mean(precisions)
    mean_recall = statistics.mean(recalls)
    mean_f1 = statistics.mean(f1_scores)

    print("=== Cross-validation results ===")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean F1-score: {mean_f1:.4f}")

    # 最も性能の良いモデルを保存 (ここではF1スコアが最大のものを選定)
    best_model_index = f1_scores.index(max(f1_scores))
    best_model = models[best_model_index]
    best_model.save(OUTPUT_MODEL_PATH)
    print(f"Best model saved to {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()