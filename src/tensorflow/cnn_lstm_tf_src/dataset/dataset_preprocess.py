import os
import librosa
import numpy as np
import subprocess
import matplotlib.pyplot as plt

# ✅ 설정
base_dir = '.gitignore/data' 
output_dir = 'outputs/cnn_lstm'
sr = 22050
n_mfcc = 13
hop_length = 512
max_len = 130
segment_duration = 5.0
save_visuals = True

X, y = [], []

def convert_to_wav(src_path, dst_path):
    if not os.path.exists(dst_path):
        command = ['ffmpeg', '-y', '-i', src_path, '-ac', '1', '-ar', str(sr), dst_path]
        subprocess.run(command, check=True)

# 🔧 출력 폴더 준비
os.makedirs(output_dir, exist_ok=True)
for label_name in ['quiet', 'loud']:
    os.makedirs(os.path.join(base_dir, label_name), exist_ok=True)
    if save_visuals:
        os.makedirs(os.path.join(output_dir, 'visuals', label_name), exist_ok=True)

# 🎧 데이터 처리
for label_name in ['quiet', 'loud']:
    folder_path = os.path.join(base_dir, label_name)
    label = 0 if label_name == 'quiet' else 1

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(('.wav', '.mp4', '.m4a')):
            continue

        file_path = os.path.join(folder_path, file_name)

        # mp4, m4a 변환
        if file_name.lower().endswith(('.mp4', '.m4a')):
            wav_name = os.path.splitext(file_name)[0] + '.wav'
            wav_path = os.path.join(folder_path, wav_name)
            convert_to_wav(file_path, wav_path)
            file_path = wav_path

        base_filename = os.path.splitext(file_name)[0]
        try:
            total_duration = librosa.get_duration(path=file_path)
        except:
            print(f"❗ duration 읽기 실패: {file_path}")
            continue

        segment_count = int(total_duration // segment_duration)

        for i in range(segment_count):
            offset = i * segment_duration
            try:
                y_audio, _ = librosa.load(file_path, sr=sr, offset=offset, duration=segment_duration)
            except:
                print(f"❗ load 실패: {file_path} (segment {i})")
                continue

            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
            features = np.vstack([mfcc, zcr])  # (14, N)

            if features.shape[1] < max_len:
                features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
            else:
                features = features[:, :max_len]

            X.append(features.T)  # (130, 14)
            y.append(label)

            # 📊 시각화
            if save_visuals:
                save_path = os.path.join(output_dir, 'visuals', label_name, f"{base_filename}_seg{i+1}.png")
                plt.figure(figsize=(10, 4))
                plt.imshow(features, aspect='auto', origin='lower', cmap='coolwarm')
                plt.title(f"{base_filename}_seg{i+1} - {label_name}")
                plt.xlabel("Frame")
                plt.ylabel("Feature Index (MFCC+ZCR)")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()

# 💾 저장
X = np.array(X)  # (샘플 수, 130, 14)
y = np.array(y)

np.save(os.path.join(output_dir, "X_lstm.npy"), X)
np.save(os.path.join(output_dir, "y_lstm.npy"), y)

print("✅ X shape (LSTM용):", X.shape)
print("✅ y shape:", y.shape)
print("📁 저장 완료:", output_dir)
