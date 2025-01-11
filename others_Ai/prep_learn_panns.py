import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pretrainedmodels
import os
import sounddevice as sd

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# オーディオデータ拡張（torchaudioを使用）
def augment_audio(waveform, sample_rate):
    augmented_waveform = waveform.clone()

    # ランダムなノイズ付加
    noise_factor = np.random.uniform(0, 0.05)
    noise = torch.randn_like(augmented_waveform) * noise_factor * torch.max(torch.abs(augmented_waveform))
    augmented_waveform += noise

    # ピッチシフト
    n_steps = np.random.randint(-1, 2)
    augmented_waveform = torchaudio.transforms.PitchShift(sample_rate, n_steps)(augmented_waveform)

    # タイムストレッチ
    rate = np.random.uniform(0.8, 1.2)
    augmented_waveform = torchaudio.transforms.SpeedPerturbation(sample_rate, rate)(augmented_waveform)

    return augmented_waveform

# データセットクラス
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            if waveform.shape[0] > 1: #ステレオ音源をモノラルに変換
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform.squeeze() # (1, samples) -> (samples)

            # リサンプリング
            if sample_rate != 32000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=32000)
                waveform = resampler(waveform)

            if self.augment:
                waveform = augment_audio(waveform, sample_rate)

            # waveformがNoneでないか確認
            if waveform is None:
                print(f"Warning: waveform is None after augmentation for {file_path}. Returning a zero tensor.")
                waveform = torch.zeros(1)  # 長さ1のゼロテンソルを返す
            return waveform, label

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # ファイルが存在しない場合は警告を表示してスキップ
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
            return None, None

# データ準備
microwave_files = [f"/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/0_prep_microwave/microwave_{i}.wav" for i in range(1, 19)]
background_files = [f"/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/dataset/1_prep_background/background_{i}.wav" for i in range(1, 19)]

microwave_labels = [0] * len(microwave_files)
background_labels = [1] * len(background_files)

all_files = microwave_files + background_files
all_labels = microwave_labels + background_labels

# データセットとデータローダー
dataset = AudioDataset(all_files, all_labels, augment=True)
train_files, test_files, train_labels, test_labels = train_test_split(all_files, all_labels, test_size=0.2, random_state=42)
train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=0.25, random_state=42)

train_dataset = AudioDataset(train_files, train_labels, augment=True)
val_dataset = AudioDataset(val_files, val_labels, augment=False)
test_dataset = AudioDataset(test_files, test_labels, augment=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=True)

# モデル定義 (PANNs Cnn14)
class PANNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet') # Cnn14の代わりにresnet18を使用(計算資源の都合上)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) # チャンネル次元を追加
        x = x.expand(-1, 3, -1) #モノラル音源を3チャンネルに拡張
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)

# モデル、損失関数、最適化関数
model = PANNModel(num_classes=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for waveforms, labels in train_loader:
        if waveforms is None or labels is None:
            print("Skipping batch due to None data.")
            continue
        waveforms = waveforms.to(device)
        labels = labels.float().to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 検証
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for waveforms, labels in val_loader:
            if waveforms is None or labels is None:
                print("Skipping batch due to None data in validation.")
                continue
            waveforms = waveforms.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# 評価
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for waveforms, labels in test_loader:
        if waveforms is None or labels is None:
            print("Skipping batch due to None data in test.")
            continue
        waveforms = waveforms.to(device)
        labels = labels.cpu().numpy()
        outputs = model(waveforms).cpu().numpy()
        preds = (outputs > 0.5).astype(int)
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.flatten())

precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# モデル保存
torch.save(model.state_dict(), "/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/microwave_detection_model_pann.pth")

# 推論部分
def predict_from_audio(audio, sr, model_path="/Users/hiratasoma/Documents/NoticeEar_DCON2025/microwaveAi/microwave_detection_model_pann.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
      waveform = torch.from_numpy(audio).float()
      if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
      if waveform.shape[0] > 1: #ステレオ音源をモノラルに変換
        waveform = torch.mean(waveform, dim=0, keepdim=True)
      waveform = waveform.squeeze()
      if sr != 32000:
          resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)
          waveform = resampler(waveform)
      waveform = waveform.to(device)
      model = PANNModel(num_classes=1).to(device)
      model.load_state_dict(torch.load(model_path))
      model.eval()
      with torch.no_grad():
        output = model(waveform.unsqueeze(0))
      return output.item()
    except Exception as e:
      print(f"Prediction error: {e}")
      return None

# 連続予測を実行する関数
def continuous_prediction(duration=3, sample_rate=16000, max_iterations=10):
    """連続的に録音と予測を行う関数"""
    for _ in range(max_iterations):
        record_and_predict(duration, sample_rate)

def record_and_predict(duration=3, sample_rate=16000):
    """録音して予測を行う関数"""
    print(f"{duration}秒間録音します...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # 録音終了まで待機
    print("録音終了")

    prediction = predict_from_audio(recording.flatten(), sample_rate)

    if prediction is not None:
        print(f"電子レンジの確率: {prediction:.4f}")
        if prediction > 0.7:
            print("電子レンジの音が検出されました！")
        else:
            print("電子レンジの音は検出されませんでした。")

    # wavファイルに保存する場合
    sf.write('output.wav', recording, sample_rate)

import sounddevice as sd
# 連続予測を実行 (3秒ごとに最大10回)
continuous_prediction(duration=3, sample_rate=16000, max_iterations=10)