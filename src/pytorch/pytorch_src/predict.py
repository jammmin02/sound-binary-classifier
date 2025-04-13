import os
import uuid
import librosa
import numpy as np
import subprocess
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 설정
sr = 22050
n_mfcc = 13
hop_length = 512
max_len = 130
segment_duration = 5.0
model_path = "models/cnn_lstm_pytorch.pth"
test_folder = "test_audio_batch"
save_dir = "test_visuals"

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.lstm = nn.LSTM(input_size=64 * 3, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def convert_to_wav(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.wav':
        return file_path
    temp_wav = f"temp_{uuid.uuid4().hex[:8]}.wav"
    command = ['ffmpeg', '-y', '-i', file_path, '-ac', '1', '-ar', str(sr), temp_wav]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_wav

def extract_features(y_audio):
    try:
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
        features = np.vstack([mfcc, zcr])  # (14, N)

        if np.isnan(features).any():
            print("⚠️ NaN detected in features.")
            return None

        if features.shape[1] < max_len:
            features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
        else:
            features = features[:, :max_len]

        return features[np.newaxis, np.newaxis, :, :]  # (1, 1, 130, 14)
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None

def predict_file(model, file_path):
    y_full, _ = librosa.load(file_path, sr=sr)
    duration = librosa.get_duration(y=y_full, sr=sr)
    segments = int(np.ceil(duration / segment_duration))

    results = []
    for i in range(segments):
        start = int(i * segment_duration * sr)
        end = int(min((i + 1) * segment_duration * sr, len(y_full)))
        segment = y_full[start:end]

        x = extract_features(segment)
        if x is None:
            print(f"[{file_path} - seg{i+1}] Skipped due to NaN or error.")
            continue

        try:
            x_tensor = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                prob = model(x_tensor).item()
            label = "quiet" if prob < 0.5 else "loud"
            print(f"[{file_path} - seg{i+1}] prob: {prob:.4f} → {label}")
            results.append((f"seg{i+1}", prob, label))
        except Exception as e:
            print(f"[{file_path} - seg{i+1}] ❌ Model error: {e}")
            continue

    return results

def plot_results(results, true_labels, pred_labels):
    os.makedirs(save_dir, exist_ok=True)
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n✅ Accuracy: {acc * 100:.2f}% ({sum(np.array(true_labels)==np.array(pred_labels))}/{len(true_labels)})")
    pred_counts = Counter(pred_labels)
    print(f"🔊 loud: {pred_counts['loud']} | 🤫 quiet: {pred_counts['quiet']}")

    results.sort(key=lambda x: x[1], reverse=True)
    names = [x[0] for x in results]
    probs = [x[1] for x in results]
    labels = [x[2] for x in results]
    colors = ['skyblue' if label == 'quiet' else 'tomato' for label in labels]

    plt.figure(figsize=(14, max(5, len(names) * 0.4)))
    sns.set(style="whitegrid")
    bars = plt.barh(names, probs, color=colors, edgecolor='black')
    plt.xlabel("Prediction Probability (loud)")
    plt.title("All Segment Predictions")
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    for bar, prob, label in zip(bars, probs, labels):
        plt.text(prob + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{prob:.2f} ({label})", va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_segments_result.png"))
    plt.show()

    cm = confusion_matrix(true_labels, pred_labels, labels=["quiet", "loud"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["quiet", "loud"])
    plt.figure(figsize=(5, 4))
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":
    print("✅ Loading model...")
    model = CNNLSTM()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    if not os.path.exists(test_folder):
        print(f"❗ Test folder '{test_folder}' does not exist.")
        exit()

    results_all = []
    true_labels = []
    pred_labels = []

    for fname in os.listdir(test_folder):
        if not fname.lower().endswith(('.wav', '.mp4', '.m4a', '.mp3')):
            continue

        fpath = os.path.join(test_folder, fname)
        if not fpath.endswith('.wav'):
            fpath = convert_to_wav(fpath)

        true_label = "loud" if "loud" in fname.lower() else "quiet"
        results = predict_file(model, fpath)

        for r in results:
            segname = f"{fname} - {r[0]}"
            results_all.append((segname, r[1], r[2]))
            true_labels.append(true_label)
            pred_labels.append(r[2])

    if results_all:
        plot_results(results_all, true_labels, pred_labels)
    else:
        print("❗ No predictions made.")
