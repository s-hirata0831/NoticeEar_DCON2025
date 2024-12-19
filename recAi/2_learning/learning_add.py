from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf

# データのロード
data_path = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/1_dataset_and_prep/processed_data/data.npy"
labels_path = "/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/1_dataset_and_prep/processed_data/labels.npy"
X = np.load(data_path)
y = np.load(labels_path)  # 既にOne-hotエンコーディング済み

# データ正規化
X = X / np.max(X)

# 結果の確認
print(f"元のラベル: {y[:5]}")  # One-hotエンコード済みラベルの確認
print(f"データの形状: {X.shape}, ラベルの形状: {y.shape}")

