import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import uuid

# 설정
base_dir = 'data'  # quiet / loud 폴더가 있어야 함
output_dir = 'outputs/cnn_lstm'
visual_dir = 'visuals'
sr = 22050
n_mfcc = 13
hop_length = 512
max_len = 130
segment_duration = 5.0

# Optional: 한글 깨짐 방지 (한글 폰트 설치 후 사용 가능)
# import matplotlib
# matplotlib.rcParams['font.family'] = 'NanumGothic'

# 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)
os.makedirs(visual_dir, exist_ok=True)

X, y = [], []

def convert_to_wav(file_path):
    """mp3/m4a/mp4 → wav 변환"""
    temp_wav = f"temp_{uuid.uuid4().hex[:8]}.wav"
    command = ['ffmpeg', '-y', '-i', file_path, '-ac', '1', '-ar', str(sr), temp_wav]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_wav

for label_name in ['quiet', 'loud']:
    label_path = os.path.join(base_dir, label_name)
    label = 0 if label_name == 'quiet' else 1
    save_label_dir = os.path.join(visual_dir, label_name)
    os.makedirs(save_label_dir, exist_ok=True)

    if not os.path.exists(label_path):
        print(f"⚠️ Folder not found: {label_path}")
        continue

    for fname in os.listdir(label_path):
        if not fname.lower().endswith(('.wav', '.mp3', '.m4a', '.mp4')):
            continue

        fpath = os.path.join(label_path, fname)

        # wav로 변환 (필요 시)
        if not fname.lower().endswith('.wav'):
            try:
                fpath = convert_to_wav(fpath)
            except Exception as e:
                print(f"❌ Failed to convert {fname} to wav:", e)
                continue

        try:
            y_audio, _ = librosa.load(fpath, sr=sr)
        except Exception as e:
            print(f"❌ Failed to load {fname}:", e)
            continue

        total_duration = librosa.get_duration(y=y_audio, sr=sr)
        segments = int(total_duration // segment_duration)

        for i in range(segments):
            start = int(i * segment_duration * sr)
            end = int((i + 1) * segment_duration * sr)
            segment = y_audio[start:end]

            if len(segment) < sr:
                continue  # 1초 미만 skip

            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y=segment, hop_length=hop_length)
            features = np.vstack([mfcc, zcr])  # (14, N)

            if features.shape[1] < max_len:
                features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
            else:
                features = features[:, :max_len]

            X.append(features.T)  # (130, 14)
            y.append(label)

            # 시각화 저장 (한글 제거, 충돌 방지)
            base_name = os.path.splitext(fname)[0]
            save_path = os.path.join(save_label_dir, f"{base_name}_seg{i+1}.png")
            plt.figure(figsize=(10, 4))
            plt.imshow(features, aspect='auto', origin='lower', cmap='coolwarm')
            plt.title(f"{label_name}_seg{i+1}")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

# 최종 저장
X = np.array(X)
y = np.array(y)
np.save(os.path.join(output_dir, 'X_lstm.npy'), X)
np.save(os.path.join(output_dir, 'y_lstm.npy'), y)
print("✅ 전처리 및 시각화 완료:", X.shape, y.shape)
