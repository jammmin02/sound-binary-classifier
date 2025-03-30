import os
import subprocess
import uuid
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow.keras.models import load_model

# 🔧 Config
sr = 22050
n_mfcc = 13
hop_length = 512
max_len = 130
segment_duration = 5.0
model_path = "cnn_lstm_model.h5"

# 📥 Load model
model = load_model(model_path)

# 🎧 Convert to WAV if needed
def convert_to_wav(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.wav':
        return file_path
    temp_wav = f"temp_{uuid.uuid4().hex[:8]}.wav"
    command = ['ffmpeg', '-y', '-i', file_path, '-ac', '1', '-ar', str(sr), temp_wav]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_wav

# 🎛️ Feature extraction
def preprocess_segment(y_audio):
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])
    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]
    return features.T[np.newaxis, ..., np.newaxis]  # (1, 130, 14, 1)

# 🔍 Predict per audio segment
def predict_file(file_path):
    y_full, _ = librosa.load(file_path, sr=sr)
    duration = librosa.get_duration(y=y_full, sr=sr)
    segments = int(np.ceil(duration / segment_duration))
    results = []
    for i in range(segments):
        start = int(i * segment_duration * sr)
        end = int(min((i + 1) * segment_duration * sr, len(y_full)))
        segment = y_full[start:end]
        if len(segment) < sr:
            continue
        x = preprocess_segment(segment)
        pred = model.predict(x, verbose=0)[0][0]
        label = "quiet" if pred < 0.5 else "loud"
        results.append((f"seg{i+1}", pred, label))
    return results

# 📊 Plot prediction results
def plot_results(results, true_labels, pred_labels, save_dir="test_visuals"):
    if not results:
        print("❗ No prediction results found.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # 1. Accuracy summary
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n✅ Overall Accuracy: {acc * 100:.2f}% ({sum(np.array(true_labels)==np.array(pred_labels))}/{len(true_labels)})")

    # 2. Prediction Summary
    pred_counts = Counter(pred_labels)
    print(f"🔊 Predicted loud: {pred_counts['loud']} | 🤫 Predicted quiet: {pred_counts['quiet']}")

    # 3. Overall bar chart
    results.sort(key=lambda x: x[1], reverse=True)
    names = [x[0] for x in results]
    probs = [x[1] for x in results]
    labels = [x[2] for x in results]
    colors = ['skyblue' if label == 'quiet' else 'tomato' for label in labels]

    plt.figure(figsize=(14, max(5, len(names)*0.4)))
    sns.set(style="whitegrid")
    bars = plt.barh(names, probs, color=colors, edgecolor='black')
    plt.xlabel("Prediction Probability (loud)", fontsize=13)
    plt.title("All Segment Predictions", fontsize=16)
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    for bar, prob, label in zip(bars, probs, labels):
        plt.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{prob:.2f} ({label})", va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_segments_result.png"))
    plt.show()

    # 4. Confusion Matrix
    label_order = ['quiet', 'loud']
    cm = confusion_matrix(true_labels, pred_labels, labels=label_order)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_order)
    plt.figure(figsize=(5, 4))
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (True vs Predicted)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()

# 🚀 Run predictions
if __name__ == "__main__":
    test_folder = "test_audio_batch"
    results_all = []
    true_labels = []
    pred_labels = []

    if not os.path.exists(test_folder):
        print(f"❗ Folder '{test_folder}' does not exist.")
    else:
        for fname in os.listdir(test_folder):
            if not fname.lower().endswith(('.wav', '.mp4', '.m4a', '.mp3')):
                continue
            fpath = os.path.join(test_folder, fname)
            if not fname.endswith('.wav'):
                fpath = convert_to_wav(fpath)

            # ✅ Label from filename
            true_label = "loud" if "loud" in fname.lower() else "quiet"

            results = predict_file(fpath)

            for r in results:
                segment_name = f"{fname} - {r[0]}"
                results_all.append((segment_name, r[1], r[2]))
                true_labels.append(true_label)
                pred_labels.append(r[2])

        # 📊 Show all results
        plot_results(results_all, true_labels, pred_labels)
