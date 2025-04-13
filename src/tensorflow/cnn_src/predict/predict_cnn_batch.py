import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import subprocess
import uuid
import matplotlib.pyplot as plt

# 설정
sr = 22050
n_mfcc = 13
hop_length = 512
max_len = 130
segment_duration = 5.0
model_path = "cnn_model.h5"
input_folder = "test_audio_batch"

model = load_model(model_path)

def convert_to_wav(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.wav':
        return file_path
    temp_wav = f"temp_{uuid.uuid4().hex[:8]}.wav"
    command = ['ffmpeg', '-y', '-i', file_path, '-ac', '1', '-ar', str(sr), temp_wav]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_wav
    except subprocess.CalledProcessError:
        print(f"❗ 변환 실패: {file_path}")
        return None

def preprocess_audio(y_audio):
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_len]
    features = features[..., np.newaxis]
    return np.expand_dims(features, axis=0)

def predict_segments(file_path, original_name):
    results = []
    y_full, _ = librosa.load(file_path, sr=sr)
    total_duration = librosa.get_duration(y=y_full, sr=sr)
    num_segments = int(np.ceil(total_duration / segment_duration))

    for i in range(num_segments):
        offset = int(i * segment_duration * sr)
        end = int(min((i + 1) * segment_duration * sr, len(y_full)))
        y_segment = y_full[offset:end]

        if len(y_segment) < sr:  # 너무 짧은 건 skip
            continue

        x = preprocess_audio(y_segment)
        pred = model.predict(x, verbose=0)[0][0]
        label = "quiet" if pred < 0.5 else "loud"
        segment_name = f"{original_name}_seg{i+1}"
        results.append((segment_name, pred, label))

    return results

def predict_batch(folder):
    all_results = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.wav', '.mp3', '.mp4', '.m4a')):
            continue

        path = os.path.join(folder, fname)
        ext = os.path.splitext(fname)[1].lower()
        temp_file = None
        if ext != '.wav':
            temp_file = convert_to_wav(path)
            if temp_file is None:
                continue
            path = temp_file

        results = predict_segments(path, os.path.splitext(fname)[0])
        all_results.extend(results)

        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

    return all_results

def plot_results(results):
    if not results:
        print("❗ 예측 결과가 없습니다.")
        return

    results.sort(key=lambda x: x[1], reverse=True)
    names = [x[0] for x in results]
    preds = [x[1] for x in results]
    labels = [x[2] for x in results]
    colors = ['skyblue' if l == 'quiet' else 'tomato' for l in labels]

    plt.figure(figsize=(14, max(6, len(names) * 0.4)))
    bars = plt.barh(names, preds, color=colors)
    plt.xlim(0, 1)
    plt.xlabel("예측 확률 (loud)", fontsize=12)
    plt.title("📊 CNN 예측 결과 시각화 (파랑: quiet / 빨강: loud)", fontsize=14)
    plt.gca().invert_yaxis()

    for bar, prob, label in zip(bars, preds, labels):
        plt.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{prob:.2f} ({label})", va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig("prediction_segments_visualization.png")
    plt.show()
    print("✅ 시각화 저장 완료: prediction_segments_visualization.png")

# 실행
if __name__ == "__main__":
    if not os.path.exists(input_folder):
        print(f"❗ '{input_folder}' 폴더가 존재하지 않아요!")
    else:
        results = predict_batch(input_folder)
        for f, p, l in results:
            print(f"🎧 {f} → {p:.4f} → {l}")
        plot_results(results)
